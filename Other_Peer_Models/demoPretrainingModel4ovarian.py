

# DemoPretrain4Ovarian


import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
from google.colab import drive
import math

# ==========================================
# --- 1. Setup ---
# ==========================================
def mountGoogleDrive(myFolder):
    root = '/content/drive'
    if not os.path.exists(root): drive.mount(root, force_remount=True)
    dest = root + '/My Drive' + myFolder
    if not os.path.exists(dest): os.makedirs(dest, exist_ok=True)
    os.chdir(dest)
    return dest

mountGoogleDrive('/CNN-Micro-SVM/')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"✅ Device: {device}")

# ==========================================
# --- 2. Data Loading ---
# ==========================================
X_train_df = pd.read_csv('ovarian_train_data.csv', header=0).fillna(0)
X_test_df  = pd.read_csv('ovarian_test_data.csv', header=0).fillna(0)
y_train_df = pd.read_csv('ovarian_train_labels.csv', header=None)
y_test_df  = pd.read_csv('ovarian_test_labels.csv', header=None)

def force_binary_labels(df):
    return pd.to_numeric(df.iloc[:, -1], errors='coerce').fillna(0).astype(int).values

y_train_raw = force_binary_labels(y_train_df)
y_test_raw  = force_binary_labels(y_test_df)
X_train = X_train_df.values
X_test  = X_test_df.values

def align_data(X, y):
    min_len = min(len(X), len(y))
    return X[:min_len], y[:min_len]

X_train, y_train_raw = align_data(X_train, y_train_raw)
X_test, y_test_raw   = align_data(X_test, y_test_raw)

# Initial Scaling (Required for Lasso)
scaler = StandardScaler()
X_train_norm = scaler.fit_transform(X_train)
X_test_norm  = scaler.transform(X_test)

print(f"Original Shape: {X_train_norm.shape}")

# ==========================================
# --- 3. LASSO FEATURE SELECTION (Top 300) ---
# ==========================================
print("\n🧬 STAGE 0: Lasso Feature Selection (Top 300 Genes)...")

# Train L1-regularized Logistic Regression
# C=0.1 increases regularization strength to force sparsity
lasso = LogisticRegression(penalty='l1', solver='liblinear', C=0.5, class_weight='balanced', random_state=42)
lasso.fit(X_train_norm, y_train_raw)

# Get coefficients and sort by absolute importance
importances = np.abs(lasso.coef_[0])
# Get indices of the top 300 features
top_300_indices = np.argsort(importances)[-300:]

# Subset the data
X_train_sel = X_train_norm[:, top_300_indices]
X_test_sel  = X_test_norm[:, top_300_indices]

print(f"Feature Selection Complete. New Shape: {X_train_sel.shape}")

# ==========================================
# --- 4. DIFFUSION MODEL AUGMENTATION ---
# ==========================================
print("\n🌊 STAGE 1: Training Tabular Diffusion for Minority Augmentation...")

# --- 4a. Diffusion Model Architecture ---
class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class TabularDiffusionNet(nn.Module):
    def __init__(self, input_dim, hidden_dim=256): # Reduced hidden dim for 300 inputs
        super().__init__()
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.mid_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim), # LayerNorm handles batch_size=1 safely
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.output_proj = nn.Linear(hidden_dim, input_dim)

    def forward(self, x, t):
        t_emb = self.time_mlp(t)
        x_emb = self.input_proj(x)
        h = x_emb + t_emb
        h = self.mid_layer(h)
        return self.output_proj(h)

# --- 4b. Diffusion Logic ---
class DiffusionManager:
    def __init__(self, input_dim, device, n_steps=100):
        self.model = TabularDiffusionNet(input_dim).to(device)
        self.device = device
        self.n_steps = n_steps
        self.beta = torch.linspace(1e-4, 0.02, n_steps).to(device)
        self.alpha = 1. - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)

    def train_on_minority(self, minority_data, epochs=500):
        optimizer = optim.AdamW(self.model.parameters(), lr=1e-3)
        loss_fn = nn.MSELoss()

        # Handle extreme scarcity
        if len(minority_data) < 32:
            repeat_factor = 32 // len(minority_data) + 1
            minority_data = np.tile(minority_data, (repeat_factor, 1))

        tensor_data = torch.tensor(minority_data).float().to(self.device)
        loader = DataLoader(TensorDataset(tensor_data), batch_size=32, shuffle=True)

        self.model.train()
        for epoch in range(epochs):
            epoch_loss = 0
            for (x,) in loader:
                optimizer.zero_grad()
                n = x.shape[0]
                t = torch.randint(0, self.n_steps, (n,)).to(self.device)
                noise = torch.randn_like(x).to(self.device)

                # Forward diffusion
                a_hat = self.alpha_hat[t][:, None]
                noisy_x = torch.sqrt(a_hat) * x + torch.sqrt(1 - a_hat) * noise

                # Predict noise
                noise_pred = self.model(noisy_x, t)
                loss = loss_fn(noise_pred, noise)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            if (epoch+1) % 100 == 0:
                print(f"   Diffusion Epoch {epoch+1}/{epochs} | Loss: {epoch_loss/len(loader):.4f}")

    def generate(self, n_samples):
        self.model.eval()
        with torch.no_grad():
            x = torch.randn(n_samples, self.model.input_proj.in_features).to(self.device)
            for i in reversed(range(self.n_steps)):
                t = (torch.ones(n_samples) * i).long().to(self.device)
                predicted_noise = self.model(x, t)

                alpha = self.alpha[t][:, None]
                alpha_hat = self.alpha_hat[t][:, None]
                beta = self.beta[t][:, None]

                if i > 0:
                    noise = torch.randn_like(x)
                else:
                    noise = torch.zeros_like(x)

                x = (1 / torch.sqrt(alpha)) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) + torch.sqrt(beta) * noise
        return x.cpu().numpy()

# --- 4c. Execute Augmentation ---
unique, counts = np.unique(y_train_raw, return_counts=True)
minority_class = unique[np.argmin(counts)]
majority_class = unique[np.argmax(counts)]
n_minority = min(counts)
n_majority = max(counts)
n_needed = n_majority - n_minority

print(f"Original Dist: {dict(zip(unique, counts))}. Generating {n_needed} synthetic samples for Class {minority_class}.")

# Extract Minority Data (using Lasso-selected features)
X_minority = X_train_sel[y_train_raw == minority_class]

# Train Diffusion
diff_manager = DiffusionManager(input_dim=X_train_sel.shape[1], device=device)
diff_manager.train_on_minority(X_minority)

# Generate
print("Generating synthetic data via Reverse Diffusion...")
X_synthetic = diff_manager.generate(n_needed)
y_synthetic = np.full(n_needed, minority_class)

# Combine
X_train_aug = np.vstack([X_train_sel, X_synthetic])
y_train_res = np.concatenate([y_train_raw, y_synthetic])

print(f"Augmented Train Shape: {X_train_aug.shape}, Class dist: {np.bincount(y_train_res)}")

# Prepare Tensors
train_X_tensor = torch.tensor(X_train_aug).float()
train_y_tensor = torch.tensor(y_train_res).long()
test_X_tensor  = torch.tensor(X_test_sel).float() # Use Lasso-selected test data
test_y_tensor  = torch.tensor(y_test_raw).long()

class_weights = compute_class_weight('balanced', classes=np.unique(y_train_res), y=y_train_res)
class_weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(device)

# ==========================================
# --- 5. PRE-TRAINING (Denoising Autoencoder) ---
# ==========================================
print("\n🟢 STAGE 2: PRE-TRAINING (Denoising Autoencoder)...")

input_dim = X_train_aug.shape[1] # Should be 300
latent_dim = 64

class DenoisingAutoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(DenoisingAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(256, latent_dim),
            nn.LeakyReLU(0.2)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, input_dim)
        )

    def forward(self, x):
        latent = self.encoder(x)
        recon = self.decoder(latent)
        return latent, recon

autoencoder = DenoisingAutoencoder(input_dim, latent_dim).to(device)
criterion_ae = nn.MSELoss()
optimizer_ae = optim.AdamW(autoencoder.parameters(), lr=0.001, weight_decay=1e-4)

ae_train_loader = DataLoader(TensorDataset(train_X_tensor, train_X_tensor), batch_size=32, shuffle=True)

for epoch in range(50):
    autoencoder.train()
    batch_loss = 0
    for bx, _ in ae_train_loader:
        bx = bx.to(device)
        optimizer_ae.zero_grad()
        # Add noise for DAE
        noise = torch.randn_like(bx) * 0.1
        noisy_bx = bx + noise

        _, recon = autoencoder(noisy_bx)
        loss = criterion_ae(recon, bx)
        loss.backward()
        optimizer_ae.step()
        batch_loss += loss.item()

    if (epoch+1) % 10 == 0:
        print(f"   AE Epoch {epoch+1}/50 | Loss: {batch_loss/len(ae_train_loader):.4f}")

# ==========================================
# --- 6. FINE-TUNING (MLP Classifier) ---
# ==========================================
print("\n🔵 STAGE 3: FINE-TUNING (MLP Classifier)...")

class RNASeqClassifier(nn.Module):
    def __init__(self, pretrained_encoder, latent_dim, num_classes):
        super(RNASeqClassifier, self).__init__()
        self.encoder = pretrained_encoder
        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.5),
            nn.Linear(32, num_classes)
        )
    def forward(self, x):
        latent = self.encoder(x)
        return self.classifier(latent)

model = RNASeqClassifier(autoencoder.encoder, latent_dim, num_classes=2).to(device)
criterion_clf = nn.CrossEntropyLoss(weight=class_weights_tensor)
optimizer_clf = optim.AdamW(model.parameters(), lr=0.0001, weight_decay=1e-3)

clf_train_loader = DataLoader(TensorDataset(train_X_tensor, train_y_tensor), batch_size=32, shuffle=True)
clf_test_loader  = DataLoader(TensorDataset(test_X_tensor, test_y_tensor), batch_size=64, shuffle=False)

for epoch in range(100):
    model.train()
    total_loss = 0
    for bx, by in clf_train_loader:
        bx, by = bx.to(device), by.to(device)
        optimizer_clf.zero_grad()
        output = model(bx)
        loss = criterion_clf(output, by)
        loss.backward()
        optimizer_clf.step()
        total_loss += loss.item()

    if (epoch+1) % 10 == 0:
        print(f"   Clf Epoch {epoch+1}/100 | Loss: {total_loss/len(clf_train_loader):.4f}")

# ==========================================
# --- 7. Comprehensive Evaluation ---
# ==========================================

def calculate_comprehensive_metrics(model, loader, device, title_prefix="Data"):
    model.eval()
    y_true = []
    y_pred = []

    with torch.no_grad():
        for bx, by in loader:
            bx = bx.to(device)
            outputs = model(bx)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            y_true.extend(by.numpy())
            y_pred.extend(preds)

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    accuracy = accuracy_score(y_true, y_pred)
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    precision = precision_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    d_index = np.log2(1 + accuracy) + np.log2(1 + (sensitivity + specificity) / 2)

    print("\n" + "="*40)
    print(f"📊 {title_prefix} Performance Metrics")
    print("="*40)
    print(f"0. D-index:     {d_index:.4f}")
    print(f"1. Accuracy:    {accuracy:.4f}")
    print(f"2. Sensitivity: {sensitivity:.4f}")
    print(f"3. Precision:   {precision:.4f}")
    print(f"4. F1 Score:    {f1:.4f}")
    print(f"   (Specificity): {specificity:.4f}")
    print("-" * 40)
    print(f"Confusion Matrix:\n{cm}")

    plt.figure(figsize=(4, 3))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['0', '1'], yticklabels=['0', '1'])
    plt.title(f'{title_prefix} Confusion Matrix')
    plt.show()

calculate_comprehensive_metrics(model, clf_test_loader, device, title_prefix="TEST")