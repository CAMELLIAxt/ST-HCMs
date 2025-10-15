# file: run_experiments.py (Robust & Memory-Efficient Version)

import pandas as pd
import numpy as np
from tqdm import tqdm
import itertools
import concurrent.futures
import os
import random
import torch
import inspect
import gc

from baselines import (
    estimate_ate_st_lmm,
    estimate_ate_aggregated_st_lmm,
    estimate_ate_panel_model_no_space 
)
from data_simulator import SyntheticDataGenerator

# --- Central Configuration ---
CONFIG = {
    "N_UNITS": 16,
    "M_SUBUNITS": 50,
    "T_STEPS": 8,
    "BASE_TREATMENT_EFFECT": 5.0,
    "SPATIAL_STRUCTURE": 'grid',
    "N_TRIALS": 20,
    "GP_TRAINING_ITER": 100,
    "N_JOBS": max(1, os.cpu_count() - 2)
}

MODEL_MAPPING = {
    "ST-HCM (LMM)": estimate_ate_st_lmm,
    "T-HCM (LMM)": estimate_ate_panel_model_no_space,
    "Aggregated ST-LMM": estimate_ate_aggregated_st_lmm
}

def run_single_trial(params: dict) -> dict:
    trial_num, conf_strength, spillover_strength = params

    # 1. Generate Data
    generator = SyntheticDataGenerator(
        N=CONFIG["N_UNITS"], m=CONFIG["M_SUBUNITS"], T=CONFIG["T_STEPS"],
        treatment_effect=CONFIG["BASE_TREATMENT_EFFECT"],
        confounding_strength=conf_strength,
        spatial_spillover_strength=spillover_strength,
        spatial_structure=CONFIG["SPATIAL_STRUCTURE"]
    )
    data_df, neighbor_map = generator.generate_v4()
    true_ate = generator.true_ate_v4

    # --- Run all models ---
    trial_results = {
        "trial": trial_num,
        "confounding_strength": conf_strength,
        "spillover_strength": spillover_strength,
        "true_ate": true_ate
    }

    for model_name, estimator_handle in MODEL_MAPPING.items():
        try:
            ate = np.nan 
            if inspect.isclass(estimator_handle):
                estimator = estimator_handle(training_iterations=CONFIG["GP_TRAINING_ITER"])
                estimator.fit(data_df, neighbor_map)

                ate_method = getattr(estimator, 'estimate_ate')
                sig = inspect.signature(ate_method)
                
                if 'neighbor_map' in sig.parameters:
                    ate = ate_method(data_df, neighbor_map)
                else:
                    ate = ate_method(data_df)
               
            else: 
                sig = inspect.signature(estimator_handle)
                if 'neighbor_map' in sig.parameters:
                    ate = estimator_handle(data_df, neighbor_map)
                else:
                    ate = estimator_handle(data_df) 
            trial_results[model_name] = ate
        except Exception as e:
            print(f"Error in trial {trial_num} with model {model_name}: {e}")
            trial_results[model_name] = np.nan
            
    del data_df, neighbor_map, generator
    gc.collect()
    
    return trial_results

def worker_wrapper(params):
    trial_num, _, _ = params
    seed = 42 + trial_num
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    return run_single_trial(params)

if __name__ == '__main__':
    RESULTS_FILENAME = "simulation_results_unified.csv"

    # --- Define the Full Parameter Grid ---
    confounding_levels = [0.0, 1.0, 2.0, 3.0, 4.0]
    spillover_levels = [0.0, 0.5, 1.0, 1.5, 2.0]
    
    param_grid = list(itertools.product(
        range(CONFIG["N_TRIALS"]),
        confounding_levels,
        spillover_levels
    ))
    tasks = [(p[0], p[1], p[2]) for p in param_grid]
    
    completed_tasks = set()
    if os.path.exists(RESULTS_FILENAME):
        print(f"Found existing results file: {RESULTS_FILENAME}. Checking for completed tasks.")
        try:
            df_existing = pd.read_csv(RESULTS_FILENAME)
            for index, row in df_existing.iterrows():
                completed_tasks.add((
                    int(row['trial']), 
                    float(row['confounding_strength']), 
                    float(row['spillover_strength'])
                ))
            print(f"Found {len(completed_tasks)} completed tasks. They will be skipped.")
        except (pd.errors.EmptyDataError, KeyError):
            print("Results file is empty or corrupted. Starting from scratch.")
            completed_tasks = set()
    
    tasks_to_run = [t for t in tasks if t not in completed_tasks]
    if not tasks_to_run:
        print("All experiments are already complete!")
    else:
        print(f"Starting experiment with {len(tasks_to_run)} remaining trials out of {len(tasks)} total.")
    
        with open(RESULTS_FILENAME, 'a', newline='') as f:
            header_written = os.path.getsize(RESULTS_FILENAME) > 0 if os.path.exists(RESULTS_FILENAME) else False
            sample_keys = list(MODEL_MAPPING.keys())
            header = ["trial", "confounding_strength", "spillover_strength", "true_ate"] + sample_keys
            if not header_written:
                f.write(','.join(header) + '\n')
            
            with concurrent.futures.ProcessPoolExecutor(max_workers=CONFIG["N_JOBS"]) as executor:
                
                futures = {executor.submit(worker_wrapper, task): task for task in tasks_to_run}
                
                for future in tqdm(concurrent.futures.as_completed(futures), total=len(tasks_to_run), desc="Running All Trials"):
                    try:
                        result = future.result()
                        if result:
                            values = [result.get(key, '') for key in header]
                            f.write(','.join(map(str, values)) + '\n')
                            f.flush() 
                    except Exception as e:
                        task = futures[future]
                        print(f"A trial with params {task} generated an exception: {e}")

    print("\nAll trials complete.")