survival-analysis-v2/
│
├── README.md                   # Project overview, setup, data format, and how to run experiments
├── requirements.txt            # Python package list for pip installation
├── environment.yml             # Conda environment for reproducibility
│
├── configs/                    # Experiment configuration files
│   ├── seer_baseline.yaml      # SEER config for DeepSurv / DeepHit baselines
│   ├── seer_gnn.yaml           # SEER config for supervised GNN survival model
│   ├── seer_ssl.yaml           # SEER config for GNN + SSL experiments
│   ├── mnb_baseline.yaml       # MNB config for DeepSurv / DeepHit baselines
│   ├── mnb_gnn.yaml            # MNB config for supervised GNN survival model
│   ├── mnb_ssl.yaml            # MNB config for GNN + SSL experiments
│   ├── synthetic_baseline.yaml # Synthetic config for baseline sanity checks
│   ├── synthetic_gnn.yaml      # Synthetic config for GNN sanity checks
│   └── synthetic_ssl.yaml      # Synthetic config for SSL sanity checks
│
├── data/                       # Prepared survival-ready datasets
│   ├── SEER/                   # Prepared SEER train/val/test files
│   ├── MNB/                    # Prepared MNB train/val/test files
│   └── synthetic/              # Prepared synthetic train/val/test files
│
├── src/                        # Main source code
│   ├── datasets/               # Dataset loading and graph dataset wrappers
│   │   ├── loaders.py          # Loads prepared train/val/test files into a unified format
│   │   └── graph_dataset.py    # Wraps node features, survival targets, and graph edges for GNNs
│   │
│   ├── models/                 # Model definitions
│   │   ├── baselines.py        # DeepSurv and DeepHit baseline models
│   │   ├── encoders.py         # MLP, GCN, and GraphSAGE encoders
│   │   └── heads.py            # Cox and DeepHit prediction heads
│   │
│   ├── graphs/                 # Graph construction and graph utilities
│   │   ├── builders.py         # Builds kNN, feature-similarity, and hybrid graphs
│   │   └── utils.py            # Edge weights, graph stats, normalization, and graph helpers
│   │
│   ├── ssl/                    # Semi-supervised learning methods
│   │   ├── pseudo_labeling.py  # Generates and filters pseudo-labels by confidence
│   │   └── graph_propagation.py# Propagates labels or risk information through the graph
│   │
│   ├── training/               # Training logic for each experiment type
│   │   ├── base_trainer.py     # Shared epoch loop, validation, early stopping, logging, checkpoints
│   │   ├── baseline_runner.py  # Training logic for DeepSurv and DeepHit
│   │   ├── gnn_runner.py       # Training logic for GNN encoder + survival head
│   │   └── ssl_runner.py       # Training logic for pseudo-labeling and graph-based SSL
│   │
│   ├── losses.py               # Cox loss, DeepHit loss, pseudo-label loss, graph SSL losses
│   ├── evaluation.py           # C-index, Brier/IBS, plots, calibration, result summaries
│   └── utils.py                # Seeds, config loading, device handling, saving/loading helpers
│
├── scripts/                    # Command-line entry points
│   ├── build_graphs.py         # Builds and caches graphs before GNN/SSL training
│   ├── train.py                # Runs one experiment from a config file
│   └── evaluate.py             # Evaluates saved models and creates tables/figures
│
├── notebooks/                  # Debugging and inspection notebooks
│   ├── 00_dataset_checks.ipynb # Checks dataset shapes, event rates, time ranges, and splits
│   ├── 01_baseline_debug.ipynb # Quick debugging for DeepSurv and DeepHit
│   └── 02_gnn_ssl_debug.ipynb  # Quick debugging for GNN, graph propagation, and pseudo-labeling
│
└── results/                    # Experiment outputs, usually gitignored
    ├── checkpoints/            # Saved model weights
    ├── figures/                # Generated plots for thesis/results
    └── tables/                 # Metric tables and comparison summaries