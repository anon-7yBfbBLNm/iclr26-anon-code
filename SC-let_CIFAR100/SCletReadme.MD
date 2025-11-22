# SC-let: Shared-Backbone SVM-micro-CNN-lets with NTC

This repository contains the **PyTorch implementation of SC-let (SVM-micro-CNN-let)**, a hybrid deep learning framework that bridges **deep representation learning** with **lazy-learning determinism**. On CIFAR-100, SC-let with Naive Traininglet Construction (NTC) reaches **≈80%+ test accuracy** while preserving the interpretability and stability of large-margin methods.

---

## 🔍 What is SC-let?

### SVM-micro-CNN-let (SC-let)

To endow Micro-inductive Learning (MiL) with representation learning while keeping the determinism of large-margin theory, we replace the notion of a static SVM node with an **SVM-micro-CNN-let (SC-let)**.

Each **SC-let**:

- Uses a deep CNN backbone to learn a feature map \(\phi_{\boldsymbol{\theta}}\).
- Feeds the learned representation into a **linear SVM head**.
- Acts as a *learninglet* that performs local, instance-specific induction rather than relying solely on a global classifier.

### Shared-Backbone Implementation

The theoretical framework allows a unique CNN per node, but this is computationally infeasible for CIFAR-100 scale. Instead, we instantiate a **single global SC-let backbone** and reuse it for all test points:

- **Backbone:** WideResNet-28-10
- **Role:** Learns the global manifold \(\phi_{\boldsymbol{\theta}}\).
- **Inference Time:** The backbone is shared dynamically to power **\(N_{\text{test}}\)** distinct SVM nodes.
- **Effect:** Every test point effectively obtains its **own local SVM-CNN-let**, which disentangles local patterns using a shared global representation.

Crucially, traininglets are constructed in the **pretrained feature space**, not pixel space, using **Naive Traininglet Construction (NTC)**. This ensures that nearest neighbors reflect **semantic similarity**, not raw pixel proximity.

---

## 🏗 Architecture: WideResNet-let

This implementation uses **WideResNet-28-10** as the shared feature extraction backbone.

### 1. Global Backbone \(\phi_{\boldsymbol{\theta}}\)

- **Quantity:** 1 (shared across all test points)
- **Model:** WideResNet-28-10
- **Depth:** 28 layers (3 groups of 4 residual blocks)
- **Widen factor:** \(k = 10\)  
  Channel expansion: \(16 \rightarrow 160 \rightarrow 320 \rightarrow 640\)
- **Output dimension:** **640** (semantic feature vector)
- **Training objective:** Multiclass hinge loss (global SVM primal view)

### 2. Local SVM Nodes \((w_i, b_i)\)

- **Quantity:** Up to the number of test images (e.g., **10,000** on CIFAR-100 test set)
- **Function:** For each test point, solve a **transient linear SVM** using only its local **traininglet**.
- **Outcome:** A tailored hyperplane that focuses on the local decision geometry around that test input.

---

## ⚙️ Methodology: NTC Workflow

SC-let operates in two main phases.

### Phase 1 — Global Representation Learning

1. Train WideResNet-28-10 on CIFAR-100 (50,000 images) with:
   - **Loss:** Multiclass hinge loss.
   - **Optimizer:** SGD with Nesterov momentum.
2. Learn the global mapping:
   \[
   \phi_{\boldsymbol{\theta}}: \mathcal{X} \to \mathbb{R}^{640}
   \]

At this stage, the network behaves like a standard global classifier, but in Phase 2 we **discard the global decision boundary**.

### Phase 2 — Naive Traininglet Construction (NTC)

For inference, we instead perform **local micro-induction** per test point:

1. **Semantic Search**  
   Embed the test point \(x_{\text{test}}\) into the feature space:
   \[
   z_{\text{test}} = \phi_{\boldsymbol{\theta}}(x_{\text{test}})
   \]
   and search for neighbors in \(\mathbb{R}^{640}\).

2. **Dual-Metric Neighbor Sets**
   - **Euclidean neighbors:**  
     \(S_{\text{euc}}\): closest points by \(L_2\) distance.
   - **Correlation neighbors:**  
     \(S_{\text{corr}}\): closest points by correlation distance \((1 - \rho)\).

3. **Intersection-Based Traininglet**
   - Define the final traininglet:
     \[
     T = S_{\text{euc}} \cap S_{\text{corr}}
     \]
   - This dual-metric intersection reduces **hubness** and filters out noisy neighbors.

4. **Micro-Induction via Local Linear SVM**
   - Train a **linear SVM** on the traininglet \(T\).
   - Use the resulting local hyperplane to predict the label of \(x_{\text{test}}\).

If the intersection is (rarely) empty, the system falls back to a purely Euclidean neighbor set to retain coverage.



## 📦 Installation

### Requirements

- Python **3.8+**
- **PyTorch 2.0+** (with CUDA for A100 or other GPUs)
- **torchvision** (for CIFAR-100 and AutoAugment)
- **scikit-learn** (for LinearSVC or other SVM solvers)
- `tqdm`, `numpy`, `matplotlib` (optional, for logging and visualization)

Install the core dependencies via:

```bash
pip install torch torchvision scikit-learn tqdm matplotlib
```

> Make sure your installed PyTorch build matches your CUDA version.

---

## ▶️ Usage

### 1. Training and Inference

This script should:

1. Train a WideResNet-28-10 backbone on CIFAR-100 with hinge loss.
2. Extract 640-D features for training and test sets.
3. Run **Naive Traininglet Construction (NTC)** for every test point.
4. Fit a local linear SVM on each traininglet and evaluate accuracy.

> See the `HParams` configuration in the script to control training and NTC behavior.

### 2. Key Hyperparameters (`HParams`)

Example configuration:

```python
from dataclasses import dataclass

@dataclass
class HParams:
    # Backbone training
    epochs: int   = 200      # Required for WideResNet convergence
    batch_size: int = 128
    lr: float     = 0.05     # Lower LR for hinge loss stability
    margin: float = 1.0      # SVM margin in hinge loss
    wd: float     = 5e-4     # Weight decay (critical for WRN)

    # NTC-related
    ntc_k: int    = 200      # Initial neighbor pool size for each metric
```

Adjust `epochs`, `lr`, and `ntc_k` depending on GPU memory and runtime budgets.


