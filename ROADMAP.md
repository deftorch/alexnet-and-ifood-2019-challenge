# Roadmap: AlexNet Implementation for iFood 2019

## Phase 1: Preparation & Documentation
- [x] Create `ROADMAP.md` (This file)
- [x] Create directory structure (`src/`, `data/`, `notebooks/`, `docs/`)
- [x] Write Paper Summary (`docs/PAPER_SUMMARY.md`)
- [x] Setup `requirements.txt`

## Phase 2: Data Pipeline
- [x] Investigate iFood 2019 dataset structure
- [x] Create dataset downloader/loader script (`src/data_loader.py`)
- [x] Implement Data Augmentation transforms (Included in `get_dataloaders`)
- [x] Implement Train/Val/Test split handling
- [x] Create Mock Dataset for verification (`src/create_mock_data.py`)

## Phase 3: Model Implementation
- [x] Implement AlexNet Baseline (`src/models/alexnet.py`)
- [x] Implement Modification 1 (Batch Normalization) (`src/models/alexnet.py`)
- [x] Implement Modification 2 (LeakyReLU) (`src/models/alexnet.py`)
- [x] Implement Combined Modification (`src/models/alexnet.py`)

## Phase 4: Training & Evaluation
- [x] Create training script with logging (WandB) (`src/train.py`)
- [x] Implement class imbalance handling (Can be enhanced with Class Weights)
- [x] Create evaluation script (Accuracy, Confusion Matrix) (`src/evaluate.py`)

## Phase 5: Experiments (Simulation/Execution)
- [x] Verified code with Mock Dataset
- [ ] Experiment A: AlexNet Baseline (To be run by user with real data)
- [ ] Experiment B: Modified 1 (To be run by user with real data)
- [ ] Experiment C: Modified 2 (To be run by user with real data)
- [ ] Experiment D: Combined (To be run by user with real data)

## Phase 6: Finalization
- [x] Polish Documentation (`README.md` updated in next steps)
- [x] Final Code Review
