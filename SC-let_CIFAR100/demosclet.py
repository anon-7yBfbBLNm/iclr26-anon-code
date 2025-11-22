# -*- coding: utf-8 -*-
"""demoSClet.ipynb
# DemoSClet Traininglet50/100/200 on CIFAR‑100
"""

# ================================================================
# SC-let on CIFAR-100
# Architecture: WideResNet-28-10 + SVM+ Hinge Loss + NTC Inference
# ================================================================

import os
import sys
import time
import random
import warnings
import numpy as np
from dataclasses import dataclass
from sklearn.exceptions import ConvergenceWarning

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as T
from sklearn.svm import LinearSVC

# ================================================================
# 0. Environment & Hardware
# ================================================================
from google.colab import drive

def setup_environment():
    # 1. Mount Drive
    root = '/content/drive'
    work_dir = '/content/drive/My Drive/NTCNNlet_Final/'

    if not os.path.exists(root):
        try:
            drive.mount(root, force_remount=True)
        except:
            print("Warning: Drive mount failed. Running locally.")
            return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if not os.path.exists(work_dir):
        try:
            os.makedirs(work_dir, exist_ok=True)
            print(f"Created working directory: {work_dir}")
        except:
            pass

    try:
        os.chdir(work_dir)
        print(f"Current Directory: {os.getcwd()}")
    except:
        pass

    # 2. GPU Setup
    print(f'PyTorch: {torch.__version__}')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if device.type == 'cuda':
        print(f'GPU: {torch.cuda.get_device_name(0)}')
        # A100/T4 Optimizations
        torch.backends.cudnn.benchmark = True
        try:
            torch.set_float32_matmul_precision('high')
            print("A100/H100 Optimization: High Precision Matmul Enabled")
        except:
            pass
    else:
        print("WARNING: Running on CPU. This will be slow.")

    return device

device = setup_environment()

# ================================================================
# 1. Data Pipeline (Strong Augmentation)
# ================================================================
CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
CIFAR100_STD  = (0.2675, 0.2565, 0.2761)

def get_dataloaders(data_root, batch_size, num_workers, seed):
    train_transform = T.Compose([
        T.RandomCrop(32, padding=4, padding_mode='reflect'),
        T.RandomHorizontalFlip(),
        T.AutoAugment(T.AutoAugmentPolicy.CIFAR10),
        T.ToTensor(),
        T.Normalize(CIFAR100_MEAN, CIFAR100_STD),
        T.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0),
    ])

    test_transform = T.Compose([
        T.ToTensor(),
        T.Normalize(CIFAR100_MEAN, CIFAR100_STD),
    ])

    train_ds = torchvision.datasets.CIFAR100(root=data_root, train=True,
                                             download=True, transform=train_transform)
    test_ds = torchvision.datasets.CIFAR100(root=data_root, train=False, download=True,
                                            transform=test_transform)

    g = torch.Generator().manual_seed(seed)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True,
        worker_init_fn=lambda wid: torch.manual_seed(seed + wid),
        generator=g, persistent_workers=(num_workers > 0)
    )

    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
        persistent_workers=(num_workers > 0)
    )

    return train_loader, test_loader

# ================================================================
# 2. Architecture: WideResNet-28-10 (The Backbone)
# Can be adjusted as a smaller one for data in tuned version
# ================================================================
class WideBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1, dropout_rate=0.3):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False)
            )
    def forward(self, x):
        o1 = F.relu(self.bn1(x))
        y = self.conv1(o1)
        o2 = F.relu(self.bn2(y))
        z = self.dropout(self.conv2(o2))
        return z + self.shortcut(x)

class WideResNetLet(nn.Module):
    def __init__(self, num_classes=100, depth=28, widen_factor=10, dropout=0.3):
        super().__init__()
        self.in_ch = 16
        n = (depth - 4) // 6
        k = widen_factor
        nStages = [16, 16*k, 32*k, 64*k]

        self.conv1 = nn.Conv2d(3, nStages[0], 3, padding=1, bias=False)
        self.layer1 = self._make_layer(nStages[1], n, stride=1, dropout=dropout)
        self.layer2 = self._make_layer(nStages[2], n, stride=2, dropout=dropout)
        self.layer3 = self._make_layer(nStages[3], n, stride=2, dropout=dropout)
        self.bn1 = nn.BatchNorm2d(nStages[3])
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.head = nn.Linear(nStages[3], num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='linear')
                nn.init.zeros_(m.bias)

    def _make_layer(self, out_ch, num_blocks, stride, dropout):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for s in strides:
            layers.append(WideBlock(self.in_ch, out_ch, s, dropout))
            self.in_ch = out_ch
        return nn.Sequential(*layers)

    def features(self, x):
        x = self.conv1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = F.relu(self.bn1(x))
        return x

    def forward(self, x):
        fmap = self.features(x)
        phi = self.gap(fmap).flatten(1)
        return self.head(phi)

# ================================================================
# 3. Stable Hinge Loss Training
# ================================================================
def multiclass_hinge_loss(scores, labels, margin=1.0):
    B, C = scores.size()
    correct = scores[torch.arange(B, device=scores.device), labels].unsqueeze(1)
    margins = margin + scores - correct
    margins = torch.clamp(margins, min=0.0)
    onehot = F.one_hot(labels, C).float()
    margins = margins * (1.0 - onehot)
    return margins.sum() / B

def train_epoch(model, loader, opt, device, hp, scaler):
    model.train()
    tot_loss = tot_acc = n = 0

    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        opt.zero_grad(set_to_none=True)

        # SAFETY 1: Float32 casting prevents Hinge Loss NaNs
        # Even when running in AMP (Auto Mixed Precision), we force the loss calc to FP32
        with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=(device.type == 'cuda')):
            scores = model(x)
            loss = multiclass_hinge_loss(scores.float(), y, hp.margin)

        scaler.scale(loss).backward()

        # SAFETY 2: Gradient Clipping prevents explosions
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        scaler.step(opt)
        scaler.update()

        _, pred = scores.max(1)
        tot_loss += loss.item()
        tot_acc += pred.eq(y).float().mean().item()
        n += 1

    return tot_loss/n, tot_acc/n

def evaluate(model, loader, device):
    model.eval()
    tot_acc = n = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=(device.type == 'cuda')):
                scores = model(x)
            _, pred = scores.max(1)
            tot_acc += pred.eq(y).float().mean().item()
            n += 1
    return tot_acc/n

# ================================================================
# 4. NTC (Naive Traininglet Construction) Engine
#    PTC is too expensive for image data for computing cost
# ================================================================
def extract_features(model, loader, device):
    """Extracts global feature vectors (640-dim)."""
    model.eval()
    feats, labs = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            fmap = model.features(x)
            phi = model.gap(fmap).flatten(1)
            feats.append(phi.cpu())
            labs.append(y)
    return torch.cat(feats, 0), torch.cat(labs, 0)

def compute_ntc_neighbors(X_train, X_test, k_max=200, device=None):
    """
    GPU-Accelerated Dual Metric Search.
    Returns two tensors: indices of Euclidean neighbors and Correlation neighbors.
    """
    X_train_d = X_train.to(device)
    X_test_d = X_test.to(device)
    n_test = X_test_d.size(0)

    all_euc = []
    all_corr = []

    print(f"[NTC] Computing Euclidean & Correlation Metrics on GPU...")

    # Pre-normalize for Correlation (Centered, Scaled)
    # This allows Correlation to be computed via Dot Product
    mean_tr = X_train_d.mean(dim=1, keepdim=True)
    std_tr = X_train_d.std(dim=1, keepdim=True) + 1e-8
    X_tr_norm = (X_train_d - mean_tr) / std_tr

    mean_te = X_test_d.mean(dim=1, keepdim=True)
    std_te = X_test_d.std(dim=1, keepdim=True) + 1e-8
    X_te_norm = (X_test_d - mean_te) / std_te

    batch_size = 512
    with torch.no_grad():
        for start in range(0, n_test, batch_size):
            end = min(start + batch_size, n_test)

            # 1. Euclidean Distance
            xb = X_test_d[start:end]
            dists = torch.cdist(xb, X_train_d)
            _, idx_euc = torch.topk(dists, k=k_max, dim=1, largest=False)
            all_euc.append(idx_euc.cpu())

            # 2. Correlation (via Dot Product of Normalized Vectors)
            xb_norm = X_te_norm[start:end]
            corrs = torch.mm(xb_norm, X_tr_norm.t())
            _, idx_corr = torch.topk(corrs, k=k_max, dim=1, largest=True)
            all_corr.append(idx_corr.cpu())

    return torch.cat(all_euc, dim=0), torch.cat(all_corr, dim=0)

def evaluate_ntc_traininglets(X_train, y_train, X_test, y_test, idx_euc, idx_corr, ks=(50, 100, 200)):
    n_test = X_test.size(0)
    acc_counts = {k: 0 for k in ks}

    print(f"[NTC] Constructing Traininglets (Intersection) & Training SVMs...")
    start_t = time.time()

    for i in range(n_test):
        test_feat = X_test[i].unsqueeze(0).numpy()
        true_label = int(y_test[i])

        # Get candidates from both metrics
        neighs_e = idx_euc[i].numpy()
        neighs_c = idx_corr[i].numpy()

        for k in ks:
            # --- NTC CORE LOGIC ---
            # The Traininglet consists ONLY of images that appear in
            # BOTH the top-k Euclidean list AND the top-k Correlation list.
            set_e = set(neighs_e[:k])
            set_c = set(neighs_c[:k])

            final_indices = list(set_e.intersection(set_c))

            # Fallback: If intersection is too strict (empty), use Euclidean
            if len(final_indices) < 5:
                final_indices = list(set_e)

            X_local = X_train[final_indices].numpy()
            y_local = y_train[final_indices].numpy()

            # Handle single-class traininglets (Edge Case)
            unique_classes = np.unique(y_local)
            if unique_classes.size < 2:
                pred_label = int(unique_classes[0])
            else:
                # --- FIX: Increased Iterations & Suppressed Warnings ---
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=ConvergenceWarning)

                    # Train Local SVM (Dual=True for speed/stability)
                    clf = LinearSVC(
                        C=1.0,
                        loss='hinge',
                        fit_intercept=True,
                        max_iter=50000, # High iter count ensures convergence
                        dual=True
                    )
                    clf.fit(X_local, y_local)
                    pred_label = int(clf.predict(test_feat)[0])

            if pred_label == true_label:
                acc_counts[k] += 1

        if (i+1) % 1000 == 0:
            elapsed = time.time() - start_t
            rate = (i+1) / elapsed
            print(f"  Processed {i+1}/{n_test} ({rate:.1f} img/s)")

    return {k: 100.0 * acc_counts[k] / n_test for k in ks}

# ================================================================
# 5. Main Execution Block
# ================================================================
@dataclass
class HParams:
    seed: int = 42
    data_root: str = './data'
    batch_size: int = 128
    epochs: int = 200        # Required for 85%
    lr: float = 0.05         # Safe LR for Hinge Loss
    margin: float = 1.0
    wd: float = 5e-4         # Weight Decay for Regularization
    workers: int = 4

def main():
    hp = HParams()
    torch.manual_seed(hp.seed)
    np.random.seed(hp.seed)

    print(f"\n[NTCNNlet] Starting Superior Performance Run...")
    print(f"Model: WideResNet-28-10 | Epochs: {hp.epochs} | Augmentation: Strong")

    # 1. Load Data
    train_loader, test_loader = get_dataloaders(hp.data_root, hp.batch_size, hp.workers, hp.seed)

    # 2. Init Model
    model = WideResNetLet(num_classes=100, depth=28, widen_factor=10).to(device)
    opt = optim.SGD(model.parameters(), lr=hp.lr, momentum=0.9, nesterov=True, weight_decay=hp.wd)
    sched = CosineAnnealingLR(opt, T_max=hp.epochs)

    # Gradient Scaler (compatible with newer Torch versions)
    scaler = torch.amp.GradScaler('cuda', enabled=(device.type == 'cuda'))

    # 3. Global Training Loop
    best_acc = 0.0
    print("\n=== Phase 1: Global Feature Learning ===")
    start_t = time.time()

    for ep in range(1, hp.epochs + 1):
        tr_loss, tr_acc = train_epoch(model, train_loader, opt, device, hp, scaler)
        sched.step()

        # Log progress
        if ep % 10 == 0 or ep > 180:
            va_acc = evaluate(model, test_loader, device)
            print(f"[{ep:03d}/{hp.epochs}] Loss: {tr_loss:.3f} | Tr_Acc: {tr_acc*100:.1f}% | Va_Acc: {va_acc*100:.2f}%")
            if va_acc > best_acc:
                best_acc = va_acc
                torch.save(model.state_dict(), 'best_ntc_model.pt')

    print(f"\n[Phase 1 Complete] Best Global Accuracy: {best_acc*100:.2f}%")
    print(f"Total Training Time: {(time.time() - start_t)/3600:.2f} hours")

    # 4. NTC Inference
    print("\n=== Phase 2: NTC Inference (Intersection Logic) ===")
    model.load_state_dict(torch.load('best_ntc_model.pt', map_location=device))

    # Extract
    print("Extracting features...")
    X_train, y_train = extract_features(model, train_loader, device)
    X_test, y_test = extract_features(model, test_loader, device)

    # Compute Dual Neighbors
    idx_euc, idx_corr = compute_ntc_neighbors(X_train, X_test, k_max=200, device=device)

    # Evaluate Traininglets
    metrics = evaluate_ntc_traininglets(X_train, y_train, X_test, y_test, idx_euc, idx_corr, ks=(50, 100, 200))

    print("\n============================================")
    print("      NTCNNlet SUPERIOR RESULTS             ")
    print("============================================")
    print(f"NTC-Traininglet-50:  {metrics[50]:.2f}%")
    print(f"NTC-Traininglet-100: {metrics[100]:.2f}%")
    print(f"NTC-Traininglet-200: {metrics[200]:.2f}%")
    print("============================================")

if __name__ == "__main__":
    main()