# AMPICIP

**AMPICIP: Adaptive Multi-Pocket Induction and Contextualized Interaction Propagation for Drug–Target Affinity Prediction**

This repository contains the implementation of AMPICIP, a drug–target affinity (DTA) prediction framework that combines adaptive local target-structure induction, cross-graph atom–residue interaction modeling, topology-aware propagation, and global target-context fusion.

> The code is provided for research and reproducibility purposes. Citation information will be added after publication.

## Overview

AMPICIP consists of four main components:

1. **AMPIS (Adaptive Multi-Pocket Induced Subgraph)** — constructs compact, potentially multi-pocket target subgraphs from residue-level binding-site probabilities and the residue-contact graph.
2. **DUCGA (Dual Unidirectional Cross-Graph Attention)** — models fine-grained cross-graph interactions between drug atoms and target residues through two opposite unidirectional attention paths.
3. **CIP (Cross-Injected Propagation)** — propagates cross-injected atom and residue representations along the original drug chemical-bond and target residue-contact topologies.
4. **GGTF (Global-Guided Target Fusion)** — complements the AMPIS-derived local target representation with an ESM2-derived global target representation through gated residual fusion.

## Repository Structure

```text
AMPICIP/
├── layers.py
├── losses.py
├── metrics.py
├── model.py
├── precompute_esm2.py
├── precompute_residue_global.py
├── residue_graph.py
├── single_mol_atom_graph.py
├── train.py
├── train_coldstart.py
├── hu122-DPI.yml
├── .gitignore
└── README.md
```

Large reproducibility data are distributed through the
[GitHub Releases page](https://github.com/WentaoXie1998/AMPICIP/releases/latest).

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
| `losses.py` | Implements the MSE objective and the HSIC auxiliary regularization used during AMPICIP training. |
| `train.py` | Main training script for the **warm-start** setting. It performs 5-fold training/evaluation using the predefined warm-start splits. |
| `train_coldstart.py` | Main training script for **drug cold-start** and **target cold-start** settings. It uses predefined entity-level splits and checks for entity leakage between training/validation/test sets. |
| `hu122-DPI.yml` | Conda environment specification used for the experiments. |

## Data and Predefined Splits

Because the processed datasets and predefined split archives exceed GitHub's
regular repository file-size limits, they are provided as multi-volume 7-Zip
archives in the latest GitHub Release:

**Download:**  
https://github.com/WentaoXie1998/AMPICIP/releases/latest

The current data release is tagged **`v1.0.0-data`**.

### Dataset

Download all dataset volumes into the same directory:

```text
dataset.7z.001
dataset.7z.002
dataset.7z.003
```

Reconstruct the original archive with 7-Zip:

```bash
7z x dataset.7z.001
```

This produces `dataset.zip`. Extract it in the repository root:

```bash
7z x dataset.zip
```

The expected layout is approximately:

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

### Predefined 5-fold splits

Download all split volumes into the same directory:

```text
splits.7z.001
splits.7z.002
splits.7z.003
splits.7z.004
```

Reconstruct the original archive:

```bash
7z x splits.7z.001
```

This produces `splits.zip`. Extract it in the repository root:

```bash
7z x splits.zip
```

The extracted split directory contains the predefined 5-fold partitions used
for warm-start, drug cold-start, and target cold-start experiments:

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

The experimental environment is provided in `hu122-DPI.yml`.

Create the Conda environment with:

```bash
conda env create -f hu122-DPI.yml
conda activate hu122-DPI
```

For better portability across machines, remove the machine-specific `prefix:`
line at the end of `hu122-DPI.yml` if necessary before recreating the
environment on another system.

## Preparation

After downloading and extracting the release assets, the repository root
should contain the extracted `dataset/` and `splits/` directories.

### Optional: precompute residue-side structural information

`precompute_residue_global.py` precomputes the complete residue-contact graph,
residue features, and distance matrix used during AMPIS construction.

```bash
python precompute_residue_global.py
```

Check the dataset-specific paths near the beginning of the script before
running it. The generated files are stored under `precomp_residue/davis/` or
`precomp_residue/kiba/`, depending on the selected dataset.

### Optional: precompute ESM2 target representations

If ESM2 representations need to be regenerated, run:

```bash
python precompute_esm2.py
```

The resulting per-target representations should be placed under:

```text
dataset/davis/Esm2/
dataset/kiba/Esm2/
```

## Training

The dataset and experimental setting are selected inside the training scripts
rather than through command-line arguments.

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

and choose the desired cold-start setting, for example:

```python
EXPERIMENT_SETTING = "drug_coldstart_sampleval"
```

or:

```python
EXPERIMENT_SETTING = "target_coldstart_sampleval"
```

For Davis, strict-validation split variants are also provided.

Then run:

```bash
python train_coldstart.py
```

The cold-start script checks that held-out drug or target entities do not leak
into the training/test or validation/test partitions.

## Training Outputs

Training outputs are written under:

```text
checkpoints_kfold/
```

For each fold, the script saves files such as:

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

## Evaluation Metrics

The implementation provides:

- **MSE** — Mean Squared Error
- **Pearson correlation**
- **Concordance Index (CI)**
- **Spearman correlation**

The main manuscript reports MSE, Pearson correlation, and CI for the benchmark experiments.

## Notes on Reproducibility

- Download the data and predefined splits from the [GitHub Releases page](https://github.com/WentaoXie1998/AMPICIP/releases/latest).
- Download **all volumes** of an archive before extraction.
- Use the supplied predefined splits to reproduce the reported 5-fold experiments.
- Run `train.py` for warm-start experiments.
- Run `train_coldstart.py` for drug and target cold-start experiments.
- AMPIS requires residue-level binding-site probabilities and target structural information.
- GGTF uses precomputed ESM2 representations as global target context.
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
