
"""demoSAMCOVID.py
# DemoSAMCOVID
"""

import pandas as pd
import numpy as np
import torch
import os
import math
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from google.colab import drive

# --- 1. Setup Environment ---
def mountGoogleDrive(myFolder):
    root = '/content/drive'
    if not os.path.exists(root):
        drive.mount(root, force_remount=True)
    dest_folder = root + '/My Drive' + myFolder
    if not os.path.exists(dest_folder):
        try: os.makedirs(dest_folder, exist_ok=True)
        except: pass
    try: os.chdir(dest_folder)
    except: return '/content'
    return dest_folder

mountGoogleDrive('/CNN-Micro-SVM/')

# ==========================================
# 2. SAM Optimizer
# ==========================================
class SAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, rho=0.05, adaptive=False, **kwargs):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"
        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super(SAM, self).__init__(params, defaults)
        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)
            for p in group["params"]:
                if p.grad is None: continue
                self.state[p]["old_p"] = p.data.clone()
                e_w = (torch.pow(p, 2) if group["adaptive"] else 1.0) * p.grad * scale.to(p)
                p.add_(e_w)
        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                p.data = self.state[p]["old_p"]
        self.base_optimizer.step()
        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device
        norm = torch.norm(
                    torch.stack([
                        ((torch.abs(p) if group["adaptive"] else 1.0) * p.grad).norm(p=2).to(shared_device)
                        for group in self.param_groups for p in group["params"] if p.grad is not None
                    ]), p=2)
        return norm

# ==========================================
# 3. Model Definition
# ==========================================
class AudioClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(AudioClassifier, self).__init__()
        self.layer1 = nn.Linear(input_dim, 128)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.layer2 = nn.Linear(128, 64)
        self.layer3 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.dropout(x)
        x = self.relu(self.layer2(x))
        x = self.layer3(x)
        return x

# ==========================================
# 4. Metric Calculation Helper
# ==========================================
def calculate_metrics(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    sensitivity = recall_score(y_true, y_pred, average='weighted', zero_division=0) # Recall

    # Specificity Calculation
    cm = confusion_matrix(y_true, y_pred)
    FP = cm.sum(axis=0) - np.diag(cm)
    TN = cm.sum() - (cm.sum(axis=1) + FP)

    with np.errstate(divide='ignore', invalid='ignore'):
        class_specificity = TN / (TN + FP)
        class_specificity = np.nan_to_num(class_specificity)

    support = cm.sum(axis=1)
    specificity = np.sum(class_specificity * support) / np.sum(support)

    # d-index
    term1 = math.log2(1 + acc)
    term2 = math.log2(1 + (sensitivity + specificity) / 2)
    d_index = term1 + term2

    return {
        "Accuracy": acc,
        "Precision": precision,
        "Sensitivity": sensitivity,
        "Specificity": specificity,
        "F1-Score": f1,
        "d-index": d_index
    }

# ==========================================
# 5. Main Execution: 5-Fold CV
# ==========================================
if __name__ == "__main__":
    FILE_NAME = 'COVID.csv'
    N_FOLDS = 5
    BATCH_SIZE = 16
    EPOCHS = 50
    LEARNING_RATE = 0.005
    RHO = 0.05

    # Check for GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    try:
        # Load raw data
        df = pd.read_csv(FILE_NAME)
        X = df.iloc[:, 1:-1].values
        y = df.iloc[:, -1].values

        # Initialize Cross Validator
        skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

        # Store results for each fold
        fold_results = []

        print(f"\nStarting {N_FOLDS}-Fold Cross-Validation with SAM Optimizer...")
        print("="*60)

        for fold_idx, (train_index, test_index) in enumerate(skf.split(X, y)):
            print(f"Processing Fold {fold_idx + 1}/{N_FOLDS}...")

            # Split Data
            X_train_fold, X_test_fold = X[train_index], X[test_index]
            y_train_fold, y_test_fold = y[train_index], y[test_index]

            # Scale Data (Fit on Train, Transform Test) - Prevents leakage
            scaler = StandardScaler()
            X_train_fold = scaler.fit_transform(X_train_fold)
            X_test_fold = scaler.transform(X_test_fold)

            # Convert to Tensors and move to Device
            X_train_t = torch.tensor(X_train_fold, dtype=torch.float32).to(device)
            y_train_t = torch.tensor(y_train_fold, dtype=torch.long).to(device)
            X_test_t = torch.tensor(X_test_fold, dtype=torch.float32).to(device)
            y_test_t = torch.tensor(y_test_fold, dtype=torch.long).to(device)

            # DataLoader
            train_dataset = TensorDataset(X_train_t, y_train_t)
            train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

            # Initialize Model & Optimizer
            input_dim = X_train_fold.shape[1]
            num_classes = len(np.unique(y))

            model = AudioClassifier(input_dim, num_classes).to(device)
            optimizer = SAM(model.parameters(), torch.optim.SGD, rho=RHO, lr=LEARNING_RATE, momentum=0.9)
            criterion = nn.CrossEntropyLoss()

            # --- Training Loop for this Fold ---
            model.train()
            for epoch in range(EPOCHS):
                for batch_X, batch_y in train_loader:
                    # SAM Step 1
                    outputs = model(batch_X)
                    loss = criterion(outputs, batch_y)
                    loss.backward()
                    optimizer.first_step(zero_grad=True)

                    # SAM Step 2
                    criterion(model(batch_X), batch_y).backward()
                    optimizer.second_step(zero_grad=True)

            # --- Evaluation for this Fold ---
            model.eval()
            with torch.no_grad():
                outputs = model(X_test_t)
                _, predicted = torch.max(outputs.data, 1)

                # Move back to CPU for metric calculation
                y_true_np = y_test_t.cpu().numpy()
                y_pred_np = predicted.cpu().numpy()

                metrics = calculate_metrics(y_true_np, y_pred_np)
                fold_results.append(metrics)

                print(f"   > Fold {fold_idx+1} Accuracy: {metrics['Accuracy']:.4f} | d-index: {metrics['d-index']:.4f}")

        # ==========================================
        # 6. Aggregating Results
        # ==========================================
        print("\n" + "="*60)
        print(f"FINAL RESULTS ({N_FOLDS}-Fold CV Mean ± Std)")
        print("="*60)

        # Convert list of dicts to DataFrame for easy averaging
        df_results = pd.DataFrame(fold_results)

        # Calculate Mean and Std
        means = df_results.mean()
        stds = df_results.std()

        # Display formatted results
        for metric in means.index:
            print(f"{metric:15s}: {means[metric]:.4f} ± {stds[metric]:.4f}")

        print("="*60)

    except Exception as e:
        print(f"An error occurred: {e}")