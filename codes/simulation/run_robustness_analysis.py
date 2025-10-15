# file: run_sensitivity_analysis.py

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
from codes.simulation.robustness_data_simulator import SensitivityDataGenerator

from baselines import (
    estimate_ate_st_lmm,
    estimate_ate_aggregated_st_lmm,
    estimate_ate_panel_model_no_space  # This is T-HCM (LMM)
)


CONFIG = {
    "N_UNITS": 16,
    "M_SUBUNITS": 50,
    "T_STEPS": 8,
    "BASE_TREATMENT_EFFECT": 5.0,
    "BASE_CONFOUNDING": 2.0,       # Fixed level of base confounding
    "BASE_SPILLOVER": 1.5,         # Fixed level of base spillover
    "SPATIAL_STRUCTURE": 'grid',
    "N_TRIALS": 20,
    "N_JOBS": max(1, os.cpu_count() - 2)
}

MODEL_MAPPING_LMM = {
    "ST-HCM (LMM)": estimate_ate_st_lmm,
    "T-HCM (LMM)": estimate_ate_panel_model_no_space,
    "Aggregated ST-LMM": estimate_ate_aggregated_st_lmm
}

def run_single_sensitivity_trial(params: dict) -> dict:
    # Unpack parameters
    trial_num = params['trial_num']
    violation_type = params['violation_type']
    violation_strength = params['violation_strength']

    # 1. Initialize the special data generator
    generator = SensitivityDataGenerator(
        N=CONFIG["N_UNITS"], m=CONFIG["M_SUBUNITS"], T=CONFIG["T_STEPS"],
        treatment_effect=CONFIG["BASE_TREATMENT_EFFECT"],
        base_confounding_strength=CONFIG["BASE_CONFOUNDING"],
        base_spatial_spillover_strength=CONFIG["BASE_SPILLOVER"],
        spatial_structure=CONFIG["SPATIAL_STRUCTURE"]
    )
    
    # 2. Generate data based on the violation type
    if violation_type == 'spatial_ordering':
        data_df, neighbor_map = generator.generate_violating_spatial_ordering(
            cyclicity_strength=violation_strength
        )
    elif violation_type == 'time_invariance':
        data_df, neighbor_map = generator.generate_violating_time_invariance(
            confounder_drift_strength=violation_strength
        )
    else:
        raise ValueError(f"Unknown violation type: {violation_type}")

    true_ate = generator.true_ate

    # 3. Run all specified LMM models
    trial_results = {
        "trial": trial_num,
        "violation_type": violation_type,
        "violation_strength": violation_strength,
        "true_ate": true_ate
    }

    for model_name, estimator_handle in MODEL_MAPPING_LMM.items():
        try:
            ate = estimator_handle(data_df, neighbor_map)
            trial_results[model_name] = ate
        except Exception as e:
            print(f"Error in trial {trial_num} ({violation_type}) with model {model_name}: {e}")
            trial_results[model_name] = np.nan
            
    del data_df, neighbor_map, generator
    gc.collect()
    
    return trial_results

def worker_wrapper(params):
    seed = 42 + params['trial_num']
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    return run_single_sensitivity_trial(params)

def run_analysis(violation_type: str, violation_strengths: list, results_filename: str):
    """
    Main function to set up and run a specific sensitivity analysis.
    """
    print(f"\n--- Starting Sensitivity Analysis for: {violation_type.upper()} ---")

    # Create the list of all tasks to run for this analysis
    tasks = []
    trial_counter = 0
    for strength in violation_strengths:
        for i in range(CONFIG['N_TRIALS']):
            tasks.append({
                'trial_num': trial_counter,
                'violation_type': violation_type,
                'violation_strength': strength
            })
            trial_counter += 1
            
    # --- Check for completed tasks for checkpointing ---
    completed_tasks = set()
    if os.path.exists(results_filename):
        print(f"Found existing results file: {results_filename}. Checking for completed tasks.")
        try:
            df_existing = pd.read_csv(results_filename)
            # Filter for the current violation type
            df_existing = df_existing[df_existing['violation_type'] == violation_type]
            for _, row in df_existing.iterrows():
                completed_tasks.add(float(row['violation_strength']))
        except (pd.errors.EmptyDataError, KeyError):
            pass 
    

    strengths_done = {s for s in completed_tasks}
    tasks_to_run = [t for t in tasks if t['violation_strength'] not in strengths_done]

    if not tasks_to_run:
        print("All experiments for this analysis are already complete!")
        return
        
    print(f"Total trials to run for this analysis: {len(tasks_to_run)}")

    # Define the header here, outside the conditional block
    header = ["trial", "violation_type", "violation_strength", "true_ate"] + list(MODEL_MAPPING_LMM.keys())
    # --- Execute in Parallel ---
    with open(results_filename, 'a', newline='') as f:
        # Write header only if the file is new/empty
        if os.path.getsize(results_filename) == 0:
            header = ["trial", "violation_type", "violation_strength", "true_ate"] + list(MODEL_MAPPING_LMM.keys())
            f.write(','.join(header) + '\n')
            
        with concurrent.futures.ProcessPoolExecutor(max_workers=CONFIG["N_JOBS"]) as executor:
            futures = {executor.submit(worker_wrapper, task): task for task in tasks_to_run}
            
            for future in tqdm(concurrent.futures.as_completed(futures), total=len(tasks_to_run), desc=f"Running {violation_type}"):
                try:
                    result = future.result()
                    if result:
                        values = [result.get(key, '') for key in header]
                        f.write(','.join(map(str, values)) + '\n')
                        f.flush()
                except Exception as e:
                    task = futures[future]
                    print(f"A trial with params {task} generated an exception in the main loop: {e}")

if __name__ == '__main__':
    RESULTS_FILENAME = "ass_sensitivity_analysis_results.csv"
    
    # --- Experiment 1: Violate Spatial Ordering ---
    spatial_violation_strengths = [0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1]
    run_analysis(
        violation_type='spatial_ordering', 
        violation_strengths=spatial_violation_strengths,
        results_filename=RESULTS_FILENAME
    )
    
    # --- Experiment 2: Violate Time Invariance ---
    time_violation_strengths = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
    run_analysis(
        violation_type='time_invariance',
        violation_strengths=time_violation_strengths,
        results_filename=RESULTS_FILENAME
    )

    # --- Final Post-processing ---
    print("\nAll sensitivity analyses complete. Post-processing the final results file...")
    try:
        final_df = pd.read_csv(RESULTS_FILENAME)
        for model_name in MODEL_MAPPING_LMM.keys():
            if model_name in final_df.columns:
                 final_df[f'error_{model_name}'] = np.abs(final_df[model_name] - final_df['true_ate'])
        
        final_df.to_csv(RESULTS_FILENAME, index=False)
        print(f"Successfully added error columns to {RESULTS_FILENAME}.")
    except Exception as e:
        print(f"Could not post-process the results file. Error: {e}")