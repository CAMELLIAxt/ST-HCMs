# Supplementary Code and Data for ST-HCMs

This repository contains the implementation code and data for the simulation studies and real-world data experiments presented in Paper **Spatio-Temporal Hierarchical Causal Models**, which proposes a novel statistical framework for estimating treatment effects in spatiotemporal hierarchical data.

## Repository Structure

### Code

The codebase is organized into two main directories:

#### 1. Simulation Studies (`/codes/simulation/`)
- `data_simulator.py`: Generates synthetic data with specified spatiotemporal dependencies
- `posterior_analyzer_parallel.py`: Implements posterior analysis for unbiasedness validation
- `consistency_experiment.py`: Evaluates estimator consistency with varying numbers of subunits
- `run_experiments.py`: Executes comparison experiments with baseline methods
- `robustness_data_simulator.py`: Generates data for robustness checks
- `run_robustness_analysis.py`: Tests model performance under assumption violations
- `baselines.py`: Implements baseline methods (ST-LMM, T-HCM, etc.)
- `gp_estimator.py`: Core implementation of the proposed method

#### 2. Real-world Data Analysis (`/codes/real_world/`)
- `prepare_data.py`: Preprocesses traffic and crash data
- `run_model_GBM.py`: Implements GBM-based comparative analysis
- `run_final_analysis.py`: Executes main analysis on Chicago traffic data


### Data

#### Simulation Data
- Generated programmatically using simulation code
- Configurable parameters for different experimental scenarios

#### Real-world Dataset
- `Chi_GIS/`: Chicago regional GIS files
- `Crashes_25secW.csv`: Processed crash records
- `segment_example.xlsx`: Sample traffic segment data
- Full dataset available at: [https://data.cityofchicago.org/Transportation/Chicago-Traffic-Tracker-Historical-Congestion-Esti/sxs8-h27x/about_data]

## Requirements
- Python 3.8+
- PyTorch
- GPyTorch
- pandas
- numpy
- scikit-learn
- lightgbm
- geopandas

## Usage
1. Install required packages
2. For simulation experiments:
   ```bash
   cd codes/simulation/
   python run_experiments.py
   ```
3. For real-world analysis:
   ```bash
   cd codes/real_world/
   python run_final_analysis.py
   ```


Note: Some data files have been partially provided due to size constraints. Complete datasets are available upon reasonable request.
