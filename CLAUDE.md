# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an EEG seizure detection project using the CHB-MIT scalp EEG dataset. The goal is to create a binary classifier that can distinguish between seizure and non-seizure states in pediatric EEG data using time-series data analysis, digital signal processing, and machine learning.

## Development Setup

```bash
# Install dependencies and set up the project
uv sync

# Run any project command
uv run <command>
```

## Data Structure

The project expects EEG data to be located in a `data/` directory (gitignored). Data can be downloaded using:
```bash
# Method 1: wget
wget -r -N -c -np https://physionet.org/files/chbmit/1.0.0/

# Method 2: AWS S3 (faster)
aws s3 sync --no-sign-request s3://physionet-open/chbmit/1.0.0/ data/
```

## Project Architecture

### Core Modules

- **`extraction/`**: Main processing pipeline and utilities
  - `pipeline.py`: Core data processing pipeline for EEG analysis
  - `tools.py`: Helper functions for data labeling, feature extraction, and bandpower calculation
  - `parse_summaries.py`: Parses seizure summary files to extract onset/offset times
  - `logger.py`: Logging utilities
  - `LT.py`: Additional processing utilities

- **`analysis/`**: Jupyter notebooks for exploratory data analysis
  - `eda.ipynb`: Exploratory data analysis of the dataset
  - `feature_extraction.ipynb`: Feature extraction experiments
  - `feature_selection.ipynb`: Feature selection analysis

- **`models/`**: Machine learning models and experiments
  - `NN.ipynb`: Neural network implementation
  - `SVM.ipynb`: Support Vector Machine experiments
  - `svm.py`: SVM model implementation

## Key Features and Functionality

### Data Processing Pipeline
- Loads EEG data from `.edf` files using MNE library
- Handles bipolar channel configurations (each channel represents difference between two electrodes)
- Segments data with configurable window size and overlap
- Annotates data with seizure onset/duration information

### Feature Extraction
The project implements both time-domain and frequency-domain features:

**Time-domain features:**
- Mean, variance, mean absolute value (MAV), skewness

**Frequency-domain features:**
- Absolute Band Power (ABP) and Relative Band Power for standard EEG frequency bands:
  - Delta (<4 Hz), Theta (4-8 Hz), Alpha (8-12 Hz), Beta (13-30 Hz), Gamma (30-45+ Hz)
- Spectral Entropy (complexity measure of frequency distribution)
- Peak Frequency identification

### Data Organization
- Uses `all_summary_data.json` for seizure timing information
- Patient-specific file organization (files prefixed with patient ID)
- Train/test splitting by subject (not by time) to prevent data leakage

## Running Jupyter Lab

For remote development:
```bash
uv run jupyter lab --no-browser --ip=<your_server_ip> --port=8080 > logs/jupyter.log 2>&1 &
```

## Key Dependencies

- **MNE**: EEG/MEG data processing library (primary tool for EEG analysis)
- **NumPy/SciPy**: Numerical computing and signal processing
- **scikit-learn**: Machine learning algorithms
- **Pandas**: Data manipulation
- **Matplotlib/Seaborn**: Data visualization
- **antropy**: Entropy measures for time series

## Domain Knowledge Reference

See `nomenclature.md` for detailed explanations of:
- Brain wave frequency bands and their significance
- EEG signal processing concepts (PSD, spectral entropy, bandpower)
- Seizure terminology (ictal, interictal, preictal, postictal)
- Bipolar channel configurations
- Different PSD estimation methods (periodogram, Welch, multitaper)