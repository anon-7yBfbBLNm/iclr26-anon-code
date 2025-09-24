# Anonymous ICLR Submission Code

This repository contains the implementation and datasets used for our ICLR 2026 anonymous submission.

---

## Repository Structure

```
Datasets/
  CASIA/CASIA-1200.csv
  COVID/covid-data-2-3classes-cleaned.csv
  IRMAS/IRMAS_all_features.csv
  Ovarian/
    ovarian_train_data.csv
    ovarian_train_data_resampled.csv
    ovarian_train_labels.csv
    ovarian_train_labels_resampled.csv
    ovarian_test_data.csv
    ovarian_test_labels.csv
  SAVEE/SAVEE-480.csv

MiL/
  ovarian_baseline.py
  ovarian_mil.py
```

- **Datasets/**: CSV files used in the experiments.  
- **MiL/**: Source code for baseline and multiple instance learning (MIL) models.  


---

## Data

All datasets required for experiments are included under the `Datasets/` directory.  
Each subdirectory corresponds to a dataset used in evaluation.  

---

## Reproducibility

- Code tested with Python 3.12.11
- Random seeds fixed for reproducibility  
- Training and evaluation scripts can be executed directly as shown above  

---

## Notes

- This repository is anonymized for double-blind review.  
- No personal identifiers are included.  
