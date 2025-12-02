# -*- coding: utf-8 -*-

"""demoPreTrainingModel4IRMAS.py
Autoendoer+ DNN FIne-tuning
"""

import pandas as pd
import numpy as np
import os
import torch
import sys
from google.colab import drive
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score, precision_recall_fscore_support
from tensorflow.keras import layers, models, callbacks, optimizers

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

# =========================================================
# 2. DATA LOADING & PREPROCESSING (IRMAS)
# =========================================================
csv_filename = 'IRMAS.csv'   # <<-- changed here

if not os.path.exists(csv_filename):
    print(f"❌ Error: '{csv_filename}' not found. Please ensure it is in the correct folder.")
    sys.exit()

print(f"✅ Loading {csv_filename}...")
df = pd.read_csv(csv_filename)

print(f"   Raw Shape: {df.shape}")
print("   Columns (first 10):", df.columns[:10].tolist())

# --- Features & Labels ---

# Drop metadata columns from features
feature_cols = [c for c in df.columns if c not in ['file_name', 'file_label']]
X = df[feature_cols].values.astype('float32')

# Map file_label from {1,..,11} -> {0,..,10} for sparse_categorical_crossentropy
original_labels = np.sort(df['file_label'].unique())
label_to_index = {lab: i for i, lab in enumerate(original_labels)}
index_to_label = {i: lab for lab, i in label_to_index.items()}  # optional, for later decoding

y = df['file_label'].map(label_to_index).values.astype('int64')

num_classes = len(original_labels)
print(f"   Number of classes: {num_classes}")
print("   Label mapping (original -> index):", label_to_index)

# Scaling: Critical for Autoencoders (Mean=0, Std=1)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Splitting: 80% Train, 20% Test
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

print(f"   Data Shape: {X_scaled.shape}")
print(f"   Train samples: {len(X_train)} | Test samples: {len(X_test)}")

# =========================================================
# 3. PHASE 1: UNSUPERVISED PRETRAINING (AUTOENCODER)
# =========================================================
input_dim = X_train.shape[1]
latent_dim = 16  # Dimension of the compressed representation

# --- Encoder Architecture ---
input_layer = layers.Input(shape=(input_dim,))
enc1 = layers.Dense(64, activation='relu')(input_layer)
enc2 = layers.Dense(32, activation='relu')(enc1)
latent = layers.Dense(latent_dim, activation='relu', name="latent_space")(enc2)

# --- Decoder Architecture ---
dec1 = layers.Dense(32, activation='relu')(latent)
dec2 = layers.Dense(64, activation='relu')(dec1)
output_layer = layers.Dense(input_dim, activation='linear')(dec2)

# --- Compile Autoencoder ---
autoencoder = models.Model(inputs=input_layer, outputs=output_layer)
autoencoder.compile(optimizer='adam', loss='mse')

print("\n🔵 [Phase 1] Pretraining Autoencoder (IRMAS)...")
history_ae = autoencoder.fit(
    X_train, X_train,
    epochs=60,
    batch_size=8,
    validation_split=0.1,
    verbose=0
)
print("✅ Autoencoder Pretraining Complete.")

# =========================================================
# 4. PHASE 2: SUPERVISED FINE-TUNING (DNN)
# =========================================================
# 1. Extract the Encoder part (reusing the learned weights)
encoder_model = models.Model(inputs=input_layer, outputs=latent)

# 2. Set Encoder to be Trainable
encoder_model.trainable = True

# 3. Add Classification Head
x = encoder_model(input_layer)
x = layers.Dense(32, activation='relu')(x)
x = layers.Dropout(0.2)(x)  # Regularization
class_output = layers.Dense(num_classes, activation='softmax')(x)

# 4. Compile Classifier
classifier = models.Model(inputs=input_layer, outputs=class_output)
classifier.compile(
    optimizer=optimizers.Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

print("\n🔵 [Phase 2] Fine-Tuning Classifier (IRMAS)...")
history_clf = classifier.fit(
    X_train, y_train,
    epochs=80,
    batch_size=8,
    validation_split=0.1,
    verbose=0
)
print("✅ Fine-Tuning Complete.")

# =========================================================
# 5. METRIC CALCULATION & EVALUATION
# =========================================================
def calculate_d_index(accuracy, sensitivity, specificity):
    # Formula: d-index = log2(1 + Acc) + log2(1 + (Sens + Spec)/2)
    return np.log2(1 + accuracy) + np.log2(1 + (sensitivity + specificity) / 2)

def evaluate_and_report(model, X, y_true, dataset_name="Test"):
    # Get Predictions
    y_pred_probs = model.predict(X, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)

    # 1. Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)

    # 2. Accuracy
    accuracy = accuracy_score(y_true, y_pred)

    # 3. Sensitivity (Recall), Precision, F1 (Macro Average)
    precision, sensitivity, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='macro', zero_division=0
    )

    # 4. Specificity (Macro Average)
    fp = cm.sum(axis=0) - np.diag(cm)
    fn = cm.sum(axis=1) - np.diag(cm)
    tp = np.diag(cm)
    tn = cm.sum() - (fp + fn + tp)

    with np.errstate(divide='ignore', invalid='ignore'):
        class_specificity = tn / (tn + fp)
        class_specificity = np.nan_to_num(class_specificity)

    specificity = np.mean(class_specificity)

    # 5. d-index
    d_idx = calculate_d_index(accuracy, sensitivity, specificity)

    # --- PRINT REPORT ---
    print(f"\n{'='*15} {dataset_name} SET RESULTS {'='*15}")
    print("Confusion Matrix:")
    print(cm)
    print("-" * 30)
    print(f"1. Accuracy:    {accuracy:.4f}")
    print(f"2. Sensitivity: {sensitivity:.4f}  (Macro Recall)")
    print(f"3. Precision:   {precision:.4f}  (Macro)")
    print(f"4. F1 Score:    {f1:.4f}  (Macro)")
    print(f"   Specificity: {specificity:.4f}  (Macro)")
    print("-" * 30)
    print(f"5. d-index:     {d_idx:.4f}")
    print("="*48)

# Evaluate on Training and Test sets
evaluate_and_report(classifier, X_train, y_train, "TRAINING")
evaluate_and_report(classifier, X_test, y_test, "TEST")