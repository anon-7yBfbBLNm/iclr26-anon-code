# Liquid Neural Networks, Pretraining, and SAM on Learning-Hard Datasets

This folder contains demos for training:

- **Liquid Neural Networks (LNN)** on the **IRMAS** dataset  
- **Autoencoder + DNN fine-tuning** on **IRMAS** and **OVARIAN** data  
- **Sharpness-Aware Minimization (SAM)** on **COVID** data  

The focus is on **learning-hard / quasi-linear** biomedical or tabular datasets (OVARIAN, COVID, IRMAS), assuming a **GPU (e.g., NVIDIA A100)** and a **Python + PyTorch/TensorFlow** environment.

All demos are designed to run in **Google Colab** and save work into a Google Drive folder (`/CNN-Micro-SVM/`).

---

## 1. Files and Roles

### 1.1 `demoLNN.py` — Liquid Neural Network on IRMAS

- Implements a **Liquid Time-Constant (LTC)**–style **Liquid Neural Network**:
  - `LiquidCell`: learns input-dependent time constants and updates the hidden state.
  - `LiquidNeuralNetwork`: projects static features into a hidden space, iteratively evolves them through the liquid cell, then classifies.
- Uses **5-fold Stratified Cross-Validation** on `IRMAS.csv`.
- Assumes:
  - First column = ID / filename
  - Last column = class label
- Pipeline:
  1. Load `IRMAS.csv` from the current working directory.
  2. Standardize features with `StandardScaler`.
  3. Encode labels with `LabelEncoder`.
  4. Train the LNN for each fold (PyTorch, Adam optimizer).
  5. Compute metrics on both **train** and **test** folds:
     - Accuracy, macro sensitivity/recall, macro specificity, macro precision, macro F1, **d-index**.
  6. Plot an aggregated confusion matrix over all test folds.

---

### 1.2 `demoPreTrainingModel4IRMAS.py` — Autoencoder + DNN on IRMAS

- Two-phase **TensorFlow/Keras** pipeline on `IRMAS.csv`:
  1. **Unsupervised pretraining**:  
     - Fully connected autoencoder with:
       - Input → 64 → 32 → latent (dim = 16) → 32 → 64 → reconstruction  
     - Trained with MSE loss to reconstruct standardized features.
  2. **Supervised fine-tuning**:
     - Reuses the **encoder** as feature extractor (trainable).
     - Adds a DNN classifier on top (Dense(32) + Dropout + softmax).
- Label handling:
  - Assumes column `file_label` with classes `{1, …, 11}`.
  - Remapped to `[0, …, 10]` for `sparse_categorical_crossentropy`.
- Data processing:
  - Drop metadata columns (`file_name`, `file_label`) from features.
  - Standardize features with `StandardScaler`.
  - Stratified 80/20 train/test split.
- Evaluation:
  - Confusion matrix.
  - Accuracy, macro sensitivity (recall), macro precision, macro F1, macro specificity.
  - **d-index** for overall discriminative ability.
  - Metrics reported separately for **training** and **test** sets.

---

### 1.3 `DemoPretrain4Ovarian.py` — Feature Selection, Diffusion Augmentation, AE + MLP on Ovarian Data

End-to-end pipeline for **OVARIAN** classification with several stages:

1. **Data loading**
   - Reads:
     - `ovarian_train_data.csv`, `ovarian_test_data.csv`
     - `ovarian_train_labels.csv`, `ovarian_test_labels.csv`
   - Labels coerced to binary integers (0/1).

2. **Initial scaling**
   - Standardize features (`StandardScaler`) → `X_train_norm`, `X_test_norm`.

3. **Stage 0 — L1 Logistic Regression (LASSO) Feature Selection**
   - Train `LogisticRegression(penalty='l1', solver='liblinear', class_weight='balanced')`.
   - Select **top 300 features** by absolute coefficient magnitude.
   - Reduce train & test to these 300 features.

4. **Stage 1 — Tabular Diffusion Model for Minority Class Augmentation**
   - Custom time-conditional **diffusion model**:
     - `SinusoidalPositionEmbeddings` for time.
     - `TabularDiffusionNet` with time embeddings + MLP + LayerNorm.
   - `DiffusionManager`:
     - Trains on **minority class** samples only.
     - Generates synthetic samples until class counts are balanced.
   - Result: augmented training set with realistic synthetic minority data.

5. **Stage 2 — Denoising Autoencoder (PyTorch)**
   - `DenoisingAutoencoder` with encoder/decoder MLP:
     - Encoder: Linear → BatchNorm → LeakyReLU → Dropout → Linear.
     - Decoder: Linear → BatchNorm → LeakyReLU → Linear.
   - Trained for reconstruction with added Gaussian noise (denoising behaviour).

6. **Stage 3 — Fine-Tuning Classifier**
   - `RNASeqClassifier`:
     - Uses pretrained encoder as feature extractor.
     - Small MLP head (latent → 32 → output logits).
   - Trained with **class-weighted** cross-entropy to handle imbalance.

7. **Evaluation**
   - Confusion matrix & metrics for **test set**:
     - Accuracy, sensitivity/recall, precision, F1, specificity.
     - **d-index**.
   - Confusion matrix plotted via seaborn heatmap.

---

### 1.4 `demoSAMCOVID.py` — SAM Optimizer on COVID Data

- Implements **Sharpness-Aware Minimization (SAM)** as a custom PyTorch optimizer:
  - Wraps a base optimizer (here, SGD with momentum).
  - Two-step update per batch:
    1. Perturb weights along gradient direction (first step).
    2. Recompute loss at perturbed weights, backprop again, then take actual optimizer step (second step).
- Model:
  - `AudioClassifier`: simple MLP
    - Input → 128 → 64 → output logits
    - ReLU activations, dropout between layers.
- Data:
  - Uses `COVID.csv`:
    - Assumes first column = ID, last column = label, middle columns = features.
- Training:
  - **5-fold Stratified Cross-Validation**.
  - For each fold:
    - Standardize train features and transform test features (no leakage).
    - Train with SAM + cross-entropy.
- Metrics:
  - Accuracy, precision, F1 (weighted), sensitivity/recall (weighted), specificity (weighted), **d-index**.
  - Mean ± std across all folds printed at the end.

---

## 2. Data Expectations

**IRMAS**

- File: `IRMAS.csv`
- Structure:
  - Column 0: ID or filename.
  - Columns 1 … (n-2): numeric features.
  - Column (n-1): label.
- Used by:
  - `demoLNN.py` (LNN, PyTorch).
  - `demoPreTrainingModel4IRMAS.py` (AE + DNN, TensorFlow).

**OVARIAN**

- Files:
  - `ovarian_train_data.csv`, `ovarian_test_data.csv`
  - `ovarian_train_labels.csv`, `ovarian_test_labels.csv`
- Labels:
  - Last column in label CSVs is coercible to integer 0/1.
- Used by:
  - `DemoPretrain4Ovarian.py`.

**COVID**

- File: `COVID.csv`
- Structure:
  - Same pattern: first column = ID, last = label, middle = features.
- Used by:
  - `demoSAMCOVID.py`.

> If your filenames or columns differ, adapt the corresponding loader sections in each script.

---

## 3. Environment & Dependencies

All scripts assume **Google Colab + Google Drive**:

- Each script calls:
  - `mountGoogleDrive('/CNN-Micro-SVM/')`  
  - Then changes working directory into that folder.

You can:
- Keep this as-is for Colab + Drive, and upload your CSVs into `/My Drive/CNN-Micro-SVM/`.
- Or comment out / modify the mount function if running locally.

### 3.1 Core Libraries

Across the demos you will need (at minimum):

- **Python ≥ 3.9**
- **PyTorch** (for LNN, diffusion, autoencoder, SAM classifier, ovarian classifier)
- **TensorFlow + Keras** (for `demoPreTrainingModel4IRMAS.py`)
- **Scientific stack**:
  - `numpy`
  - `pandas`
  - `scikit-learn`
- **Visualization**:
  - `matplotlib`
  - `seaborn`
- **Google Colab** helpers:
  - `google-colab` (for `drive` module, in Colab this is preinstalled)

Example (Colab cell):

```bash
pip install torch torchvision torchaudio tensorflow matplotlib seaborn scikit-learn
```

---

## 4. How to Run (Typical Colab Workflow)

1. **Start a GPU runtime**
   - `Runtime → Change runtime type → GPU` (preferably A100 if available).

2. **Upload scripts & data**
   - Either:
     - Upload this folder to your Google Drive under `/My Drive/CNN-Micro-SVM/`, or
     - Modify `mountGoogleDrive()` and paths accordingly.

3. **Run a demo**
   - For example, **LNN on IRMAS**:
     ```bash
     python demoLNN.py
     ```
   - **AE + DNN on IRMAS**:
     ```bash
     python demoPreTrainingModel4IRMAS.py
     ```
   - **Ovarian diffusion + pretraining pipeline**:
     ```bash
     python DemoPretrain4Ovarian.py
     ```
   - **SAM on COVID**:
     ```bash
     python demoSAMCOVID.py
     ```

Each script prints metrics to stdout and (where implemented) shows confusion-matrix plots.

---

## 5. Metrics & d-Index

All demos report standard classification metrics, plus **d-index**:

\[
d = \log_2(1 + \text{Accuracy}) + \log_2\left(1 + \frac{\text{Sensitivity} + \text{Specificity}}{2}\right)
\]

This combines **overall correctness (accuracy)** with **balanced sensitivity/specificity**, making it suitable for **imbalanced, learning-hard medical datasets**.

---

## 6. Notes & Extensions

- You can swap in:
  - Different network architectures (deeper LNNs, wider MLPs).
  - Alternative optimizers (AdamW, different SAM settings).
  - Different pretraining strategies (e.g., variational autoencoders).
- To adapt to non-Colab environments:
  - Remove the `mountGoogleDrive` calls.
  - Replace paths with local file paths.

---

## 7. License & Citation

- Add your chosen license file (e.g., `MIT`, `Apache-2.0`) at project root.
- If you publish results, consider citing:
  - Liquid Neural Networks / LTC.
  - SAM (Sharpness-Aware Minimization).
  - The specific OVARIAN, COVID, and IRMAS datasets you use.
