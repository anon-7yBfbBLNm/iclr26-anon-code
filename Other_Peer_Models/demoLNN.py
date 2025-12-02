# -*- coding: utf-8 -*-
"""demoLNN.oy
# DemoLNN
# Data: IRMAS data 

"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import math
import os
from google.colab import drive

# =========================================================
# 1. SETUP & DRIVE MOUNTING
# =========================================================
def mountGoogleDrive(myFolder):
    root = '/content/drive'
    if not os.path.exists(root):
        drive.mount(root, force_remount=True)
    dest = root + '/My Drive' + myFolder
    if not os.path.exists(dest):
        os.makedirs(dest, exist_ok=True)
    os.chdir(dest)
    return dest

# Mount Google Drive (change folder name if you like)
mountGoogleDrive('/CNN-Micro-SVM/')

# Check device (Informational)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"✅ Device: {device}")
print(f"✅ Working Directory: {os.getcwd()}")

# ==========================================
# 1. Data Loading
# ==========================================

def load_data(filepath):
    df = pd.read_csv(filepath)
    # assumes first column is an ID / file_name, last column is label
    X = df.iloc[:, 1:-1].values
    y = df.iloc[:, -1].values
    return X, y

# ==========================================
# 2. Liquid Neural Network Definition
# ==========================================

class LiquidCell(nn.Module):
    """
    A Liquid Time-Constant (LTC) Cell.
    The decay rate (time constant) depends on the input, making the system 'liquid'.
    """
    def __init__(self, input_dim, hidden_dim):
        super(LiquidCell, self).__init__()
        self.hidden_dim = hidden_dim

        # Weight matrices for the input signal (not explicitly used in this variant,
        # but kept for extensibility)
        self.W_input = nn.Linear(input_dim, hidden_dim)

        # Weight matrices for the time-constant (tau)
        self.W_tau = nn.Linear(input_dim + hidden_dim, hidden_dim)

        # Weight matrices for the state update (driving force)
        self.W_update = nn.Linear(input_dim + hidden_dim, hidden_dim)

        self.act = nn.Tanh()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, h, dt=0.1):
        """
        x: Static input features [batch, input_dim]
        h: Current hidden state [batch, hidden_dim]
        dt: Integration step size (not explicitly used in this discrete version)
        """
        # Concatenate input and hidden state
        combined = torch.cat([x, h], dim=1)

        # 1. Compute Liquid Time Constant (tau)
        tau = self.sigmoid(self.W_tau(combined))

        # 2. Compute Driving Signal (S)
        signal = self.act(self.W_update(combined))

        # 3. LTC-style update
        h_new = (1 - tau) * h + tau * signal

        return h_new

class LiquidNeuralNetwork(nn.Module):
    def __init__(self, input_dim, num_classes, hidden_dim=64, steps=10):
        super(LiquidNeuralNetwork, self).__init__()
        self.hidden_dim = hidden_dim
        self.steps = steps  # Number of ODE integration steps

        # Projection to match dimensions
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        # The Liquid Cell
        self.cell = LiquidCell(hidden_dim, hidden_dim)

        # Classification Head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        # Project input features to hidden dimension
        x_embed = torch.relu(self.input_proj(x))

        # Initialize hidden state (e.g., zeros)
        batch_size = x.size(0)
        h = torch.zeros(batch_size, self.hidden_dim).to(x.device)

        # Evolve the system over 'time' (steps)
        for _ in range(self.steps):
            h = self.cell(x_embed, h)

        # Classify based on the final settled state
        logits = self.classifier(h)
        return logits

# ==========================================
# 3. Training & Helpers
# ==========================================

def train_model(model, train_loader, criterion, optimizer, epochs=50):
    model.train()
    for epoch in range(epochs):
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

def calculate_d_index(accuracy, sensitivity, specificity):
    term1 = math.log2(1 + accuracy)
    term2 = math.log2(1 + (sensitivity + specificity) / 2)
    return term1 + term2

def get_multiclass_specificity(cm):
    specificities = []
    for i in range(cm.shape[0]):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        tn = cm.sum() - tp - fp - fn
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0
        specificities.append(spec)
    return np.mean(specificities)

def get_predictions(model, loader):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    return np.array(all_labels), np.array(all_preds)

def plot_confusion_matrix(cm, title, class_names=None):
    plt.figure(figsize=(8, 6))
    if class_names is None:
        class_names = [f"Class {i}" for i in range(cm.shape[0])]
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues', cbar=False,
        xticklabels=class_names, yticklabels=class_names
    )
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(title)
    plt.tight_layout()
    plt.show()

# ==========================================
# 4. Main Execution (5-Fold CV) on IRMAS
# ==========================================

if __name__ == "__main__":
    try:
        # ---------- IRMAS data here ----------
        X, y_raw = load_data('IRMAS.csv')
    except FileNotFoundError:
        print("Error: 'IRMAS.csv' not found in the current working directory.")
        print("Make sure IRMAS.csv is in:", os.getcwd())
        exit()

    # Encode labels to 0 .. num_classes-1
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_raw)
    num_classes = len(label_encoder.classes_)
    print(f"✅ Number of classes (IRMAS): {num_classes}")
    print(f"✅ Original labels: {list(label_encoder.classes_)}")

    k_folds = 5
    skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)

    # Separate dicts for TRAIN and TEST metrics
    metrics_template = {
        'accuracy': [], 'sensitivity': [], 'specificity': [],
        'precision': [], 'f1': [], 'd_index': []
    }
    fold_metrics_train = {k: [] for k in metrics_template.keys()}
    fold_metrics_test = {k: [] for k in metrics_template.keys()}

    aggregated_cm_test = np.zeros((num_classes, num_classes), dtype=int)

    print(f"\nStarting {k_folds}-Fold CV with Liquid Neural Network on IRMAS...")

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        print(f"\n================ Fold {fold+1}/{k_folds} ================")

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # Build datasets & loaders
        train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))
        test_dataset = TensorDataset(torch.FloatTensor(X_test), torch.LongTensor(y_test))

        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
        # For evaluation on train set we use a non-shuffled loader
        train_eval_loader = DataLoader(train_dataset, batch_size=16, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

        # Initialize Liquid Neural Network with correct number of classes
        model = LiquidNeuralNetwork(
            input_dim=X.shape[1],
            num_classes=num_classes,
            steps=10
        ).to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        train_model(model, train_loader, criterion, optimizer, epochs=50)

        # ---------- TRAIN METRICS ----------
        y_train_true, y_train_pred = get_predictions(model, train_eval_loader)
        cm_train = confusion_matrix(y_train_true, y_train_pred, labels=range(num_classes))

        acc_train  = accuracy_score(y_train_true, y_train_pred)
        prec_train = precision_score(y_train_true, y_train_pred, average='macro', zero_division=0)
        sens_train = recall_score(y_train_true, y_train_pred, average='macro', zero_division=0)
        f1_train   = f1_score(y_train_true, y_train_pred, average='macro', zero_division=0)
        spec_train = get_multiclass_specificity(cm_train)
        d_idx_train = calculate_d_index(acc_train, sens_train, spec_train)

        fold_metrics_train['accuracy'].append(acc_train)
        fold_metrics_train['sensitivity'].append(sens_train)
        fold_metrics_train['specificity'].append(spec_train)
        fold_metrics_train['precision'].append(prec_train)
        fold_metrics_train['f1'].append(f1_train)
        fold_metrics_train['d_index'].append(d_idx_train)

        # ---------- TEST METRICS ----------
        y_true, y_pred = get_predictions(model, test_loader)
        cm_test = confusion_matrix(y_true, y_pred, labels=range(num_classes))
        aggregated_cm_test += cm_test

        acc  = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, average='macro', zero_division=0)
        sens = recall_score(y_true, y_pred, average='macro', zero_division=0)
        f1   = f1_score(y_true, y_pred, average='macro', zero_division=0)
        spec = get_multiclass_specificity(cm_test)
        d_idx = calculate_d_index(acc, sens, spec)

        fold_metrics_test['accuracy'].append(acc)
        fold_metrics_test['sensitivity'].append(sens)
        fold_metrics_test['specificity'].append(spec)
        fold_metrics_test['precision'].append(prec)
        fold_metrics_test['f1'].append(f1)
        fold_metrics_test['d_index'].append(d_idx)

        print(f"  TRAIN - Acc: {acc_train:.4f}, Sens: {sens_train:.4f}, Spec: {spec_train:.4f}, "
              f"Prec: {prec_train:.4f}, F1: {f1_train:.4f}, d-index: {d_idx_train:.4f}")
        print(f"  TEST  - Acc: {acc:.4f}, Sens: {sens:.4f}, Spec: {spec:.4f}, "
              f"Prec: {prec:.4f}, F1: {f1:.4f}, d-index: {d_idx:.4f}")

    # ==========================================
    # 5. Final Summary
    # ==========================================
    print("\n" + "="*50)
    print(f"FINAL RESULTS (TRAINING DATA, {k_folds}-FOLD CV AVERAGE)")
    print("="*50)
    for metric, values in fold_metrics_train.items():
        print(f"{metric.capitalize():<12}: {np.mean(values):.4f} (+/- {np.std(values):.4f})")

    print("\n" + "="*50)
    print(f"FINAL RESULTS (TEST DATA, {k_folds}-FOLD CV AVERAGE)")
    print("="*50)
    for metric, values in fold_metrics_test.items():
        print(f"{metric.capitalize():<12}: {np.mean(values):.4f} (+/- {np.std(values):.4f})")

    # Confusion matrix over all TEST folds
    class_names = [str(c) for c in label_encoder.classes_]
    plot_confusion_matrix(
        aggregated_cm_test,
        f"Aggregated Confusion Matrix ({k_folds}-Fold CV, Test Sets)",
        class_names=class_names
    )