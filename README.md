# Anonymous ICLR 2026 Submission Code

This repository contains the implementation and datasets used for our ICLR 2026 anonymous submission.

---

## Repository Structure

```
iclr26-anon-code
├── Datasets
│   ├── CASIA
│   │   └── CASIA-1200.csv
│   ├── COVID
│   │   └── covid-data-2-3classes-cleaned.csv
│   ├── IRMAS
│   │   └── IRMAS_all_features.csv
│   ├── Ovarian
│   │   ├── ovarian_test_data.csv
│   │   ├── ovarian_test_labels.csv
│   │   ├── ovarian_train_data_resampled.csv
│   │   ├── ovarian_train_data.csv
│   │   ├── ovarian_train_labels_resampled.csv
│   │   └── ovarian_train_labels.csv
│   └── SAVEE
│       └── SAVEE-480.csv
│
├── GPU
│   └── ovarian_mil_gpu.py
│
├── MiL
│   ├── casia_mil.py
│   ├── covid19_mil.py
│   ├── irmas_mil.py
│   ├── ovarian_baseline.py
│   ├── ovarian_mil.py
│   └── savee_mil.py
│
├── ProtoNet_MAML
│   ├── casia_protonet_maml.py
│   ├── covid_protonet_maml.py
│   ├── irmas_protonet_maml.py
│   ├── ovarian_protonet_maml.py
│   └── savee_protonet_maml.py
│
├── SC-let (for CIFAR100)
│   ├── demosclet.py
│   └── SCletReadme.MD
│
└── README.md
```

---

## Components

### **1. Datasets/**
All datasets used in experiments are included in this repository.  
Each subfolder corresponds to one dataset:

- **CASIA**
- **COVID-19**
- **IRMAS**
- **Ovarian**
- **SAVEE**

All files are in CSV format for direct loading.

---

### **2. MiL/**
This directory contains the MiL code for all datasets:

- `casia_mil.py`
- `covid19_mil.py`
- `irmas_mil.py`
- `ovarian_mil.py`
- `savee_mil.py`

A baseline version for Ovarian data is also provided:

- `ovarian_baseline.py`

These scripts replicate the experimental setup used in the submission.

---

### **3. GPU/**
Contains GPU-accelerated MIL implementation:

- `ovarian_mil_gpu.py`

This version mirrors the CPU implementation but is optimized for faster execution on CUDA-enabled devices.

---

### **4. ProtoNet_MAML/**
This folder includes supplementary experiments:

- Prototypical Network
- Model-Agnostic Meta-Learning (MAML)

Files are dataset-specific:

- `*_protonet_maml.py`

These are not part of the primary method but included for completeness.

---

### **5. SC-let/**
This module includes the SC-let implementation for CIFAR100:

- **SC-let: Shared-Backbone SVM-micro-CNN-lets with NTC**
- Code: `demosclet.py`
- Details: `SCletReadme.MD`

---

## Environment & Reproducibility

- Python **3.12.11**
- All random seeds fixed for reproducibility
- CPU and GPU versions available

---

## Notes

- This repository is fully anonymized for double-blind review.
- All dataset files and scripts included are free of personal identifiers.
- Please refer to the corresponding subdirectories for further documentation (for example, SCletReadme.MD).

---
