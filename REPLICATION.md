# REPLICATION.md

# Reproducing the Experiments

This document explains how to reproduce the experiments reported in this repository.

## Overview

The project consists of three stages:

```text
Data & Activation Generation
            ↓
Joint NLA Training
            ↓
Results & Analysis
```

The corresponding notebooks are:

```text
01_Phi3_Data_and_Activation_Generation.ipynb
                ↓
02_Joint_128D_NLA_Training.ipynb
                ↓
03_NLA_Results_and_Analysis.ipynb
```

---

# Repository Structure

```text
.
├── notebooks/
│   ├── 01_Phi3_Data_and_Activation_Generation.ipynb
│   ├── 02_Joint_128D_NLA_Training.ipynb
│   └── 03_NLA_Results_and_Analysis.ipynb
│
├── results/
│   ├── wikitext/
│   └── khanacademy/
│
├── scripts/
│   └── generate_descriptions_and_activations.py
│
├── README.md
└── REPLICATION.md
```

---

# Data Availability

The processed activation files are not included directly in this repository because they exceed GitHub storage limits.

The complete experiment data can be downloaded from:

```text
https://drive.google.com/drive/folders/1a4TGWksomddc2p_n2pHj106K4ASohhle?usp=sharing
```

The download contains:

```text
data/
models/
results/
```

After downloading, place them in the repository root:

```text
Natural Language Autoencoder/
│
├── data/
├── models/
├── results/
├── notebooks/
└── README.md
```

---

# Option 1: Reproduce Results Using Provided Data

This is the fastest way to reproduce the reported results.

## Step 1

Download the shared Google Drive folder.

## Step 2

Place:

```text
data/
models/
results/
```

inside the repository root.

## Step 3

Open:

```text
03_NLA_Results_and_Analysis.ipynb
```

and run all cells.

This notebook reproduces:

* Layer-wise FVE analysis
* Qualitative analysis
* Best/Worst sample analysis
* Figures used in the report

---

# Option 2: Retrain Existing Models

If you would like to retrain the NLA models using the provided activations:

## Step 1

Download the shared data folder.

## Step 2

Open:

```text
02_Joint_128D_NLA_Training.ipynb
```

## Step 3

Select:

```python
DATASET_NAME = "wikitext"
ACTIVATION_LAYER = 20
```

(or any other available layer)

## Step 4

Run the notebook.

The notebook will:

* Load activations
* Create MiniLM semantic targets
* Train the joint bottleneck model
* Save checkpoints
* Compute FVE scores
* Generate evaluation plots

---

# Option 3: Generate Activations From Scratch

The repository also contains the script used to generate semantic descriptions and transformer activations.

```text
scripts/generate_descriptions_and_activations.py
```

This script:

1. Loads raw text samples.
2. Generates semantic descriptions using Phi-3-mini.
3. Extracts transformer activations.
4. Saves activations for selected layers.

This stage is the most computationally expensive part of the pipeline.

The final experiments reported in this repository were generated using access to a DGX system (Thanks to my friend who gave me access to his hardware), which allowed activations to be extracted across multiple transformer layers and larger datasets.

Example usage:

```bash
python generate_descriptions_and_activations.py
```

Please refer to the script configuration for dataset paths, layer selection, and output directories.

---

# Datasets

Two datasets were used:

## WikiText

General encyclopedic and factual text.

Approximately:

```text
5000 samples
```

## Khan Academy (Cosmopedia)

Educational content covering mathematics, science, and related topics.

Approximately:

```text
5000 samples
```

---

# Hardware Notes

The project was intentionally designed around an open-source model (Phi-3-mini) and relatively modest computational requirements.

However, activation extraction across multiple layers can still be time-consuming.

For this reason:

* Training can be reproduced on a standard GPU.
* Large-scale activation generation is substantially faster on multi-GPU systems such as DGX.

---

# Expected Results

The strongest results reported in the README are:

| Dataset      | Best Layer | Mean FVE |
| ------------ | ---------: | -------: |
| WikiText     |   Layer 20 |    0.597 |
| Khan Academy |    Layer 8 |    0.718 |

Minor variation may occur due to differences in hardware, software versions, or random seeds.

---

# Notes

This project is a simplified reimplementation of Anthropic's Natural Language Autoencoder (NLA).

Several deliberate simplifications were made:

* Phi-3-mini instead of Claude
* Dense 128-dimensional bottleneck
* MiniLM semantic supervision
* No Sparse Autoencoders
* No reinforcement-learning-based optimization

The goal was not to exactly reproduce Anthropic's system, but to investigate whether similar semantic bottleneck behavior emerges in a smaller and more accessible setting.
