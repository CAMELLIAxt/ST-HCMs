# file: posterior_analyzer_parallel.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import torch
import gpytorch
import random
import concurrent.futures
import os
from functools import partial

# Reuse our existing, well-tested modules
from data_simulator import SyntheticDataGenerator
from gp_estimator import STHCM_Estimator


def single_sample_worker(sample_index: int, estimator: STHCM_Estimator, data_df: pd.DataFrame) -> float:
    unit_avg_outcome = data_df.groupby(['unit_id', 'time'])['outcome'].mean().reset_index()
    
    posterior_outcomes = {0: np.zeros(estimator.N), 1: np.zeros(estimator.N)}

    for a_star in [0, 1]:
        for i in range(estimator.N):
            model, likelihood, (x_scaler, y_scaler) = estimator.models_[i], estimator.likelihoods_[i], estimator.scalers_[i]
            model.eval()
            likelihood.eval()
            
            sim_outcomes = np.zeros(estimator.T)
            for t in range(estimator.T):
                outcome_lag = sim_outcomes[t-1] if t > 0 else 0
                spatial_lag = 0
                if t > 0 and estimator.neighbor_map_[i]:
                    neighbor_histories = unit_avg_outcome[
                        (unit_avg_outcome['unit_id'].isin(estimator.neighbor_map_[i])) &
                        (unit_avg_outcome['time'] == t-1)
                    ]['outcome']
                    if not neighbor_histories.empty:
                        spatial_lag = neighbor_histories.mean()

                features_t = np.array([[a_star, outcome_lag, spatial_lag, t]])
                features_t_torch = torch.tensor(x_scaler.transform(features_t), dtype=torch.float32)
                
                with torch.no_grad(), gpytorch.settings.fast_pred_var():
                    predictive_dist = likelihood(model(features_t_torch))
                    y_pred_scaled_sample = predictive_dist.sample().numpy()

                y_pred_sample = y_scaler.inverse_transform(y_pred_scaled_sample.reshape(-1, 1)).flatten()[0]
                sim_outcomes[t] = y_pred_sample
            
            posterior_outcomes[a_star][i] = sim_outcomes[-1]

    mean_effect_1 = np.mean(posterior_outcomes[1])
    mean_effect_0 = np.mean(posterior_outcomes[0])
    
    return mean_effect_1 - mean_effect_0

if __name__ == '__main__':
    SEED = 42
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    random.seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    print("Configuring experiment for posterior analysis...")
    SCENARIO_CONFIG = {
        "N_UNITS": 16, "M_SUBUNITS": 100, "T_STEPS": 8, "TRUE_ATE": 5.0,
        "CONFOUNDING_STRENGTH": 3.0, "SPATIAL_SPILLOVER": 0.5,
        "SPATIAL_STRUCTURE": 'grid', "SCENARIO": 'interactive'
    }
    N_SAMPLES = 800 

    print("Generating a single dataset...")
    generator_params = {
        'N': SCENARIO_CONFIG["N_UNITS"], 'm': SCENARIO_CONFIG["M_SUBUNITS"], 'T': SCENARIO_CONFIG["T_STEPS"],
        'treatment_effect': SCENARIO_CONFIG["TRUE_ATE"], 'confounding_strength': SCENARIO_CONFIG["CONFOUNDING_STRENGTH"],
        'spatial_spillover_strength': SCENARIO_CONFIG["SPATIAL_SPILLOVER"], 'spatial_structure': SCENARIO_CONFIG["SPATIAL_STRUCTURE"]
    }
    generator = SyntheticDataGenerator(**generator_params)
    if SCENARIO_CONFIG["SCENARIO"] == 'interactive':
        data_df, neighbor_map = generator.generate_v2()
    else:
        data_df, neighbor_map = generator.generate()

    print("Fitting the ST-HCM (GP) model (this may take a while)...")
    estimator = STHCM_Estimator(training_iterations=100)
    estimator.fit(data_df, neighbor_map)
    print("Fitting complete.")

    print(f"Estimating the posterior distribution with {N_SAMPLES} samples in parallel...")
    
    worker_fn = partial(single_sample_worker, estimator=estimator, data_df=data_df)
    
    tasks = range(N_SAMPLES)
    ate_posterior_samples = []
    
    n_jobs = max(1, os.cpu_count() - 1)
    with concurrent.futures.ProcessPoolExecutor(max_workers=n_jobs) as executor:
        results_iterator = executor.map(worker_fn, tasks)
        for result in tqdm(results_iterator, total=N_SAMPLES, desc="Drawing Posterior Samples"):
            ate_posterior_samples.append(result)
    
    ate_posterior_samples = np.array(ate_posterior_samples)