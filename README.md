# AMPICIP

**AMPICIP: Adaptive Multi-Pocket Induction and Contextualized Interaction Propagation for Drug–Target Affinity Prediction**

This repository contains the implementation of AMPICIP, a drug–target affinity (DTA) prediction framework that combines adaptive local target-structure induction, cross-graph atom–residue interaction modeling, topology-aware propagation, and global target-context fusion.

> The code is provided for research and reproducibility purposes. Citation information will be added after publication.

## Overview

AMPICIP consists of four main components:

1. **AMPIS (Adaptive Multi-Pocket Induced Subgraph)** — constructs compact, potentially multi-pocket target subgraphs from residue-level binding-site probabilities and the residue-contact graph.
2. **DUCGA (Dual Unidirectional Cross-Graph Attention)** — models fine-grained cross-graph interactions between drug atoms and target residues.
3. **CIP (Cross-Injected Propagation)** — propagates cross-injected atom and residue representations along the original drug chemical-bond and target residue-contact topologies.
4. **GGTF (Global-Guided Target Fusion)** — complements the AMPIS-derived local target representation with an ESM2-derived global target representation through gated residual fusion.

The overall pipeline is:

```text
Drug SMILES -----------------> Drug atom graph --------\
                                                       \
                                                        DUCGA -> CIP -> Drug representation
                                                       /
Target structure + LABind --> AMPIS residue graph ----/

Target sequence --> ESM2 global representation --> GGTF
                                                  |
                                                  v
                                   Final target representation
                                                  |
                       Drug + target representations
                                                  |
                                                  v
                                      Affinity prediction
```

## Repository Structure

```text
AMPICIP/
├── layers.py
├── metrics.py
├── model.py
├── precompute_esm2.py
├── precompute_residue_global.py
├── residue_graph.py
├── single_mol_atom_graph.py
├── train.py
├── train_coldstart.py
├── losses.py                  # required by the training scripts
├── hu122-DPI.yml
├── dataset.zip
├── splits.zip
└── README.md
```

### Code files

| File | Description |
|---|---|
| `layers.py` | Implements the main neural-network building blocks used by AMPICIP, including graph encoders, DUCGA, CIP, graph readout, and global–local target fusion modules. |
| `model.py` | Defines the complete AMPICIP model and connects the drug encoder, target encoder, DUCGA, CIP, GGTF, and final affinity-prediction head. |
| `residue_graph.py` | Builds target residue-contact graphs and implements AMPIS construction using LABind-derived residue binding-site probabilities, connected-component selection, probability-mass filtering, and local pocket expansion. |
| `single_mol_atom_graph.py` | Converts a drug SMILES string into a PyTorch Geometric atom–bond graph. Atom and bond features are generated with RDKit. |
| `precompute_residue_global.py` | Precomputes full target residue-contact graphs, residue distance matrices, and residue feature matrices for efficient AMPIS construction during training. |
| `precompute_esm2.py` | Precomputes ESM2-based global target representations so that ESM2 does not need to be executed repeatedly during DTA training. |
| `metrics.py` | Implements regression evaluation metrics, including MSE, concordance index (CI), Spearman correlation, and Pearson correlation. |
| `train.py` | Main training script for the **warm-start** setting. It performs 5-fold training/evaluation using the predefined warm-start splits. |
| `train_coldstart.py` | Main training script for **drug cold-start** and **target cold-start** settings. It uses predefined entity-level splits and checks for entity leakage between training/validation/test sets. |
| `losses.py` | Implements the MSE objective and the HSIC auxiliary regularization used during AMPICIP training. |
| `hu122-DPI.yml` | Conda environment specification used for the experiments. |

## Data

Two archives are provided.

### `dataset.zip`

Contains the processed Davis and KIBA data and target-side resources used by AMPICIP.

After extraction, the expected layout is approximately:

```text
dataset/
├── davis/
│   ├── updated_full_with_seq_ligid.csv
│   ├── RESULT_all.csv
│   ├── PDB/
│   └── Esm2/
└── kiba/
    ├── updated_full_with_seq_ligid.csv
    ├── RESULT_all.csv
    ├── PDB/
    └── Esm2/
```

`RESULT_all.csv` contains the residue-level binding-site probabilities used by AMPIS.

### `splits.zip`

Contains the predefined 5-fold splits used in the experiments.

```text
splits/
├── davis_label_stratified_seed0/
├── kiba_label_stratified_seed0/
├── davis_drug_coldstart_sampleval_multibal_seed0/
├── davis_target_coldstart_sampleval_multibal_seed0/
├── davis_drug_coldstart_strictval_seed0/
├── davis_target_coldstart_strictval_seed0/
├── kiba_drug_coldstart_sampleval_multibal_seed0/
└── kiba_target_coldstart_sampleval_multibal_seed0/
```

Each fold directory contains `split_train.csv`, `split_val.csv`, and `split_test.csv`.

## Environment Setup

The experiments were run with Python 3.9, PyTorch 2.5.1, PyTorch Geometric 2.6.1, RDKit, ESM2, and CUDA 12.1-compatible packages.

Create the Conda environment with:

```bash
conda env create -f hu122-DPI.yml
conda activate hu122-DPI
```

For portability across machines, remove the machine-specific `prefix:` line at the end of `hu122-DPI.yml` before recreating the environment on another system.

## Preparation

### 1. Extract the data

Extract both archives in the repository root:

```text
dataset.zip -> ./dataset/
splits.zip  -> ./splits/
```

### 2. Precompute residue-side structural information

`precompute_residue_global.py` precomputes the complete residue-contact graph, residue features, and distance matrix used during AMPIS construction.

```bash
python precompute_residue_global.py
```

The current script contains dataset-specific paths near the beginning of the file. Check that they point to the dataset you want to process before running it.

The generated files are stored under `precomp_residue/davis/` or `precomp_residue/kiba/`, depending on the selected dataset.

### 3. Precompute ESM2 target representations

If ESM2 representations are not already included in the extracted dataset, run:

```bash
python precompute_esm2.py
```

The resulting per-target representations should be placed under `dataset/davis/Esm2/` and/or `dataset/kiba/Esm2/`.

## Training

The dataset and experimental setting are selected inside the training scripts rather than through command-line arguments.

### Warm-start

Use `train.py`.

Before running, set:

```python
ACTIVE_DATASET = "davis"   # or "kiba"
EXPERIMENT_SETTING = "warm_start"
```

Then run:

```bash
python train.py
```

### Cold-start

Use `train_coldstart.py`.

Select the dataset:

```python
ACTIVE_DATASET = "davis"   # or "kiba"
```

and choose the desired setting, for example:

```python
EXPERIMENT_SETTING = "drug_coldstart_sampleval"
```

or

```python
EXPERIMENT_SETTING = "target_coldstart_sampleval"
```

For Davis, strict-validation split variants are also provided.

Then run:

```bash
python train_coldstart.py
```

The cold-start script checks that held-out drug or target entities do not leak into the training/test or validation/test partitions.

## Training Outputs

Outputs are written under:

```text
checkpoints_kfold/
```

Each fold stores files such as:

```text
fold1/
├── best.pt
├── last.pt
├── metrics.json
├── test_predictions.csv
├── split_train.csv
├── split_val.csv
└── split_test.csv
```

Training uses AdamW optimization, warmup followed by cosine learning-rate decay, early stopping based on validation MSE, MSE as the main objective, an HSIC-based auxiliary regularization term, and automatic mixed precision when CUDA is available.

## Evaluation Metrics

The implementation provides:

- **MSE** — Mean Squared Error
- **Pearson correlation**
- **Concordance Index (CI)**
- **Spearman correlation**

The main manuscript reports MSE, Pearson correlation, and CI for the benchmark experiments.

## Notes on Reproducibility

- Use the supplied `splits.zip` to reproduce the reported 5-fold experiments.
- Run `train.py` for warm-start experiments.
- Run `train_coldstart.py` for drug and target cold-start experiments.
- AMPIS requires residue-level binding-site probabilities and target structural information.
- GGTF uses precomputed ESM2 representations as global target context.
- The training scripts import `losses.py`; make sure this file is included in the repository.
- Check dataset-selection variables at the beginning of preprocessing and training scripts before execution.

## Citation

If you use this repository in your research, please cite:

```text
AMPICIP: Adaptive Multi-Pocket Induction and Contextualized Interaction
Propagation for Drug–Target Affinity Prediction.
```

Full citation information will be added after publication.

## Contact

For questions regarding the code or data, please open an issue in this repository.
