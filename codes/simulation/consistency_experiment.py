# file: robust_consistency_experiment.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import time
import os
import concurrent.futures
from functools import partial
import gc # Garbage Collector interface

from data_simulator import SyntheticDataGenerator
from gp_estimator import STHCM_Estimator

def run_single_experiment_worker(params: dict):
    N, m, T = params['N'], params['m'], params['T']
    true_ate = params['true_ate']
    run_index = params['run_index']

    time.sleep(run_index * 0.1)

    try:
        generator = SyntheticDataGenerator(
            N=N, m=m, T=T, treatment_effect=true_ate,
            confounding_strength=2.0, spatial_spillover_strength=0.5,
            spatial_structure='grid' if N > 1 else 'line'
        )
        data_df, neighbor_map = generator.generate_v2()
        
        estimator = STHCM_Estimator(training_iterations=50)
        estimator.fit(data_df, neighbor_map)
        
        del data_df 
        gc.collect()

        data_df_est, _ = generator.generate_v2()
        estimated_ate = estimator.estimate_ate(data_df_est)

        error = abs(estimated_ate - true_ate)
        
        return {
            'variable': 'm', 'value': m, 'error': error, 
            'run_index': run_index, 'true_ate': true_ate,
            'estimated_ate': estimated_ate
        }
    except Exception as e:
        print(f"--- ERROR in worker for m={m}, run={run_index}: {e} ---")
        return {
            'variable': 'm', 'value': m, 'error': np.nan, 
            'run_index': run_index, 'true_ate': np.nan,
            'estimated_ate': np.nan
        }

def plot_from_csv(csv_filepath="consistency_results_robust.csv"):
    if not os.path.exists(csv_filepath):
        print(f"Error: Results file not found at {csv_filepath}")
        return

    results_df = pd.read_csv(csv_filepath)
    if results_df.empty:
        print("Warning: The results file is empty. Nothing to plot.")
        return
    
    sns.set_theme(style="whitegrid", palette="colorblind", font_scale=1.4)
    plt.figure(figsize=(10, 7))

    sns.lineplot(data=results_df, x='value', y='error', marker='o', 
                 markersize=10, linewidth=3, errorbar='sd', err_style="band")

    plt.title("Estimator Consistency vs. Number of Subunits (m)", fontsize=18, pad=20)
    plt.xlabel("Number of Subunits per Unit (m)", fontsize=16)
    plt.ylabel("Absolute Estimation Error of ATE", fontsize=16)
    plt.xscale('log')
    # Use the unique sorted values from the 'value' column for ticks
    unique_m_values = sorted(results_df['value'].unique())
    plt.xticks(unique_m_values, labels=unique_m_values) 
    plt.grid(True, which="both", ls="--", linewidth=0.5)
    plt.tight_layout()
    
    plot_filename = "consistency_plot_robust.pdf"
    plt.savefig(plot_filename)
    print(f"\nConsistency plot saved to {plot_filename}")
    plt.show()

if __name__ == '__main__':
    M_VALUES = [1000]   
    FIXED_N = 16
    FIXED_T = 8
    TRUE_ATE_V2 = 5.0
    N_REPEATS = 5 
    CSV_FILENAME = "consistency_results_robust.csv"

    if not os.path.exists(CSV_FILENAME):
        pd.DataFrame(columns=[
            'variable', 'value', 'error', 'run_index', 
            'true_ate', 'estimated_ate'
        ]).to_csv(CSV_FILENAME, index=False)
        print(f"Created new results file: {CSV_FILENAME}")

    completed_df = pd.read_csv(CSV_FILENAME)
    completed_tasks = set(tuple(row) for row in completed_df[['value', 'run_index']].to_numpy())
    print(f"Found {len(completed_tasks)} completed tasks from previous runs.")

    tasks_to_run = []
    for m_val in M_VALUES:
        for i in range(N_REPEATS):
            if (m_val, i) not in completed_tasks:
                tasks_to_run.append({
                    'N': FIXED_N, 'm': m_val, 'T': FIXED_T,
                    'true_ate': TRUE_ATE_V2, 'run_index': i
                })
    
    if not tasks_to_run:
        print("All experiments are already complete!")
    else:
        print(f"Starting robust parallel experiment. Tasks to run: {len(tasks_to_run)}")  
        start_time = time.time()
        MAX_CONCURRENT_PROCESSES = 10 
        
        with concurrent.futures.ProcessPoolExecutor(max_workers=MAX_CONCURRENT_PROCESSES) as executor:
            futures = {executor.submit(run_single_experiment_worker, task): task for task in tasks_to_run}
            
            for future in tqdm(concurrent.futures.as_completed(futures), total=len(tasks_to_run), desc="Running Experiments"):
                result = future.result()
                if result:
                    pd.DataFrame([result]).to_csv(CSV_FILENAME, mode='a', header=False, index=False)

        end_time = time.time()
        print(f"\nExperiment execution finished in {(end_time - start_time) / 60:.2f} minutes.")