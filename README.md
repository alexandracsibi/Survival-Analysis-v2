## Project Structure

```text
survival-analysis-v2/
│
├── README.md                         # Project overview, setup instructions, data format, and experiment guide
├── requirements.txt                  # Python package list for pip installation
├── environment.yml                   # Optional Conda environment for reproducibility
├── run_synthetic_experiments.sh      # Runs the main synthetic baseline, GNN, and SSL experiments
│
├── configs/                          # Experiment configuration files
│   ├── baselines/                    # Supervised non-graph survival baselines
│   │   ├── deepsurv/                 # DeepSurv / Cox-based configs
│   │   └── deephit/                  # DeepHit discrete-time configs
│   │
│   ├── gnn/                          # Supervised graph survival experiments
│   │   ├── cox/                      # GraphSAGE encoder + Cox survival head
│   │   └── deephit/                  # GraphSAGE encoder + DeepHit head
│   │
│   └── ssl/                          # Semi-supervised graph survival experiments
│       └── gnn_deephit/              # GNN + DeepHit semi-supervised experiments
│           └── static_teacher/        # Static teacher pseudo-labeling experiments
│
├── data/                             # Prepared survival-ready datasets, usually gitignored
│   ├── synthetic_binary/             # Synthetic single-event survival dataset
│   │   └── graphs/                   # Cached kNN graph edge_index files
│   │
│   ├── synthetic_competing/           # Synthetic competing-risk survival dataset
│   │   └── graphs/                   # Cached kNN graph edge_index files
│   │
│   ├── SEER/                         # Prepared healthcare survival dataset
│   └── MNB/                          # Prepared financial / mortgage-style survival dataset
│
├── src/                              # Main source code
│   ├── datasets/                     # Dataset loading and graph dataset wrappers
│   │   ├── loaders.py                # Loads train/val/test arrays into a unified survival format
│   │   └── graph_dataset.py          # Builds PyG graph data with features, labels, times, and split masks
│   │
│   ├── models/                       # Model definitions
│   │   ├── baselines.py              # DeepSurv and DeepHit baseline model classes
│   │   ├── encoders.py               # GraphSAGE and neural network encoders
│   │   ├── heads.py                  # Cox and DeepHit prediction heads
│   │   └── gnn_models.py             # GraphSAGECox and GraphSAGEDeepHit model wrappers
│   │
│   ├── graphs/                       # Graph construction utilities
│   │   └── builders.py               # Builds kNN feature-similarity graphs for survival samples
│   │
│   ├── ssl/                          # Semi-supervised learning utilities
│   │   └── pseudo_labeling.py        # Teacher-student pseudo-label generation and confidence filtering
│   │
│   ├── training/                     # Training logic
│   │   ├── baseline_runner.py        # Training and prediction logic for DeepSurv and DeepHit
│   │   ├── gnn_runner.py             # Training and prediction logic for GraphSAGE survival models
│   │   └── ssl_runner.py             # Static-teacher pseudo-labeling training logic
│   │
│   ├── losses.py                     # Cox loss, DeepHit likelihood/ranking loss, and pseudo-label losses
│   ├── evaluation.py                 # C-index, time-dependent AUC, Brier score, and IBS utilities
│   └── utils.py                      # Seed setting, config loading, device selection, and helpers
│
├── scripts/                          # Command-line entry points
│   ├── build_graphs.py               # Builds and saves graph edge_index files before GNN/SSL training
│   ├── train.py                      # Trains one experiment from a YAML config
│   └── evaluate.py                   # Evaluates a saved checkpoint and writes metric tables
│
├── notebooks/                        # Debugging and inspection notebooks
│   ├── 00_dataset_checks.ipynb       # Dataset shape, feature, event-rate, and split checks
│   ├── 01_baseline_debug.ipynb       # DeepSurv and DeepHit debugging notebook
│   └── 02_gnn_ssl_debug.ipynb        # GNN, graph, and SSL debugging notebook
│
└── results/                          # Experiment outputs, usually gitignored
    ├── checkpoints/                  # Saved model weights
    ├── figures/                      # Generated plots for thesis results
    └── tables/                       # Histories, configs, metadata, and metric CSV files
```