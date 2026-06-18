# Data Card — SecureScan AI

## Dataset 1: BigVul
- Source: https://www.kaggle.com/datasets/kaggler10240/msr-data
- License: Unknown
- Size: 1.47 GB
- Samples: 188,636
- Vulnerable: 10,900
- Safe: 177,736
- Language: C/C++
- Known issues: Class imbalance (94% safe)

## Dataset 2: DiverseVul
- Source: https://www.kaggle.com/datasets/ahmedtabib/diversevul-cleaned
- License: Unknown
- Size: 110 MB
- Samples: 319,523
- Vulnerable: 17,879
- Safe: 301,644
- Language: C/C++
- Known issues: Class imbalance (94% safe)

## Dataset 3: FormAI
- Source: https://github.com/FormAI-Dataset/FormAI-dataset
- License: MIT
- Size: ~500 MB
- Samples: 246,549
- Vulnerable: 201,274
- Safe: 45,275
- Language: C (AI-generated)
- Known issues: Majority vulnerable (opposite of BigVul)

## Combined Dataset
- Total samples: 749,794
- Training samples: 420,240
- Validation samples: 89,760
- Test samples: 90,000
- Balancing strategy: 3:1 (safe:vulnerable)

## Verification
- BigVul row count: 188,636
- DiverseVul row count: 319,523
- FormAI row count: 246,549
