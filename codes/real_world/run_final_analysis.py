import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from tqdm import tqdm
import json
import os
import logging
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
import seaborn as sns

def create_ate_plot(results_dict, estimator_type, output_path):
    df_list = []
    for model_name, ate_dist in results_dict.items():
        if ate_dist:
            base_model = model_name.split(' (')[0]
            df_list.append(pd.DataFrame({'ATE': ate_dist, 'Base Model': base_model}))
    
    if not df_list: return
    
    plot_data = pd.concat(df_list)
    base_model_order = ['Aggregated', 'T-HCM', 'ST-HCM']
    models_present = [m for m in base_model_order if m in plot_data['Base Model'].unique()]
    color_map = {model: color for model, color in zip(base_model_order, sns.color_palette("deep", len(base_model_order)))}

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 7))

    sns.kdeplot(data=plot_data, x='ATE', hue='Base Model', hue_order=models_present,
                palette={m: color_map[m] for m in models_present},
                fill=True, alpha=0.2, linewidth=2, ax=ax, legend=False)

    legend_handles, legend_labels = [], []
    for base_model_name in models_present:
        mean_ate = plot_data[plot_data['Base Model'] == base_model_name]['ATE'].mean()
        ax.axvline(x=mean_ate, color=color_map[base_model_name], linestyle='--', linewidth=2.5)
        handle = plt.Rectangle((0,0), 1, 1, color=color_map[base_model_name], alpha=0.4)
        legend_handles.append(handle)
        legend_labels.append(f'{base_model_name} (Mean = {mean_ate:.3f})')
        
    ax.legend(legend_handles, legend_labels, fontsize=12, title='Model Structure', title_fontsize='14')
    ax.set_title(f'ATE Distributions Estimated by {estimator_type} Models', fontsize=18, pad=20)
    ax.set_xlabel('Average Treatment Effect (ATE) of Crashes on Traffic Speed', fontsize=14)
    ax.set_ylabel('Density', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.show()

    
PANEL_DATA_PATH = 'chicago_panel_final.csv'
NEIGHBOR_FEATURES_PATH = 'chicago_panel_with_features.csv' 
RESULTS_OUTPUT_PATH = 'ate_results_gbm_newfrature.json' 
LOG_FILE_PATH = 'experiment__gbm_newfrature.log'      
ATE_FIGURE_PATH = 'ate_comparison__gbm_newfrature.png'  

N_BOOTSTRAP_RUNS = 100    
SUBSAMPLE_FRACTION = 0.2 
MAIN_RANDOM_SEED = 42 
N_JOBS = 8

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE_PATH, mode='w'), 
        logging.StreamHandler()
    ]
)

def run_single_bootstrap_iteration(data, model_config, iteration_seed):
    np.random.seed(iteration_seed)
    
    if model_config.get('bootstrap_by', 'unit') == 'unit':
        unique_units = data[model_config['unit_col']].unique()
        bootstrap_units = np.random.choice(unique_units, size=len(unique_units), replace=True)
        bootstrap_sample = data[data[model_config['unit_col']].isin(bootstrap_units)]
    else: 
        bootstrap_sample = data.sample(n=len(data), replace=True, random_state=iteration_seed)

    try:
        features = model_config.get('features')
        treatment_col = model_config.get('treatment_col')
        
        X = bootstrap_sample[features]
        y = bootstrap_sample[model_config['target']]
        
        numeric_features = X.select_dtypes(include=np.number).columns.tolist()
        
        preprocessor = ColumnTransformer(transformers=[('num', StandardScaler(), numeric_features)], remainder='passthrough')
        
        gbm_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                       ('regressor', lgb.LGBMRegressor(random_state=iteration_seed, n_jobs=1, verbose=-1))])
        gbm_pipeline.fit(X, y)
        predict_fn = gbm_pipeline.predict

        df_fact_0 = bootstrap_sample.copy(); df_fact_0[treatment_col] = 0
        df_fact_1 = bootstrap_sample.copy(); df_fact_1[treatment_col] = 1

        pred_0 = predict_fn(df_fact_0[features])
        pred_1 = predict_fn(df_fact_1[features])
            
        return np.mean(pred_1) - np.mean(pred_0)

    except Exception as e:
        logging.warning(f"fail (seed: {iteration_seed}): {e}")
        return None

if __name__ == "__main__":
    df_panel = pd.read_csv(PANEL_DATA_PATH)
    df_panel['timestamp'] = pd.to_datetime(df_panel['timestamp'])

    unit_treatment_map = df_panel.groupby(['unit', 'timestamp'])['treatment'].max().reset_index()
    unit_treatment_map.rename(columns={'treatment': 'unit_treatment'}, inplace=True)
    df_featured = pd.merge(df_panel, unit_treatment_map, on=['unit', 'timestamp'], how='left')
    
    df_featured = df_featured.sort_values(by=['unit', 'subunit', 'timestamp'])
    df_featured['speed_lag1'] = df_featured.groupby('subunit')['speed'].shift(1)
    df_featured['unit_treatment_lag1'] = df_featured.groupby('unit')['unit_treatment'].shift(1)
    
    df_old_features = pd.read_csv(NEIGHBOR_FEATURES_PATH, usecols=['unit', 'timestamp', 'avg_neighbor_speed_lag1', 'avg_neighbor_treatment_lag1']).drop_duplicates()
    df_old_features['timestamp'] = pd.to_datetime(df_old_features['timestamp'])
    df_featured = pd.merge(df_featured, df_old_features, on=['unit', 'timestamp'], how='left')

    df_featured['hour'] = df_featured['timestamp'].dt.hour
    df_featured['dayofweek'] = df_featured['timestamp'].dt.dayofweek
    df_featured.dropna(subset=['speed_lag1', 'unit_treatment_lag1'], inplace=True)
    df_featured.fillna(0, inplace=True)
    
    for col in ['unit', 'subunit', 'hour', 'dayofweek']:
        df_featured[col] = df_featured[col].astype('category')
    
    if SUBSAMPLE_FRACTION < 1.0:
        np.random.seed(MAIN_RANDOM_SEED)
        subunits_to_keep = df_featured.groupby('unit')['subunit'].unique().apply(
            lambda x: np.random.choice(x, size=max(1, int(len(x) * SUBSAMPLE_FRACTION)), replace=False)
        ).explode()
        df_sampled = df_featured[df_featured['subunit'].isin(subunits_to_keep)].copy()
    else:
        df_sampled = df_featured
    
    # d) 准备聚合数据
    df_agg = df_sampled.groupby(['unit', 'timestamp'], observed=False).agg(
        speed=('speed', 'mean'), unit_treatment=('unit_treatment', 'max'), hour=('hour', 'first'),
        dayofweek=('dayofweek', 'first'), avg_neighbor_speed_lag1=('avg_neighbor_speed_lag1', 'first'),
        avg_neighbor_treatment_lag1=('avg_neighbor_treatment_lag1', 'first')
    ).reset_index()
    df_agg = df_agg.sort_values(['unit', 'timestamp'])
    df_agg['agg_speed_lag1'] = df_agg.groupby('unit', observed=False)['speed'].shift(1)
    df_agg['agg_unit_treatment_lag1'] = df_agg.groupby('unit', observed=False)['unit_treatment'].shift(1)
    df_agg.dropna(inplace=True)

    T_HCM_FEATURES = ['unit_treatment', 'speed_lag1', 'unit_treatment_lag1', 'hour', 'dayofweek']
    ST_HCM_FEATURES = T_HCM_FEATURES + ['avg_neighbor_speed_lag1', 'avg_neighbor_treatment_lag1']
    AGG_FEATURES = ['unit_treatment', 'agg_speed_lag1', 'agg_unit_treatment_lag1', 'hour', 'dayofweek', 'avg_neighbor_speed_lag1']
    
    models_to_run = {
        "Aggregated (GBM)": {"type": "GBM", "features": AGG_FEATURES + ['unit'], "target": "speed", "data": df_agg, "unit_col": "unit", "bootstrap_by": "unit", "treatment_col": "unit_treatment"},
        "T-HCM (GBM)": {"type": "GBM", "features": T_HCM_FEATURES + ['unit'], "target": "speed", "data": df_sampled, "unit_col": "unit", "treatment_col": "unit_treatment"},
        "ST-HCM (GBM)": {"type": "GBM", "features": ST_HCM_FEATURES + ['unit'], "target": "speed", "data": df_sampled, "unit_col": "unit", "treatment_col": "unit_treatment"}
    }

    if os.path.exists(RESULTS_OUTPUT_PATH):
        with open(RESULTS_OUTPUT_PATH, 'r') as f: all_results = json.load(f)
    else: all_results = {}

    for name, config in models_to_run.items():
        existing_results = all_results.get(name, [])
        num_runs_needed = N_BOOTSTRAP_RUNS - len(existing_results)
        if num_runs_needed <= 0:
            continue
        iteration_seeds = [MAIN_RANDOM_SEED + i for i in range(len(existing_results), N_BOOTSTRAP_RUNS)]
        results_list = Parallel(n_jobs=N_JOBS)(
            delayed(run_single_bootstrap_iteration)(config['data'], config, seed) 
            for seed in tqdm(iteration_seeds, desc=f"Bootstrap for {name}")
        )
        new_valid_results = [res for res in results_list if res is not None and np.isfinite(res)]
        all_results[name] = existing_results + new_valid_results
        with open(RESULTS_OUTPUT_PATH, 'w') as f: json.dump(all_results, f, indent=4)


    create_ate_plot(all_results, 'GBM', ATE_FIGURE_PATH)
    


def create_ate_plot(results_dict, estimator_type, output_path):
    df_list = []
    for model_name, ate_dist in results_dict.items():
        if ate_dist:
            base_model = model_name.split(' (')[0]
            df_list.append(pd.DataFrame({'ATE': ate_dist, 'Base Model': base_model}))
    
    if not df_list: return
    
    plot_data = pd.concat(df_list)
    base_model_order = ['Aggregated', 'T-HCM', 'ST-HCM']
    models_present = [m for m in base_model_order if m in plot_data['Base Model'].unique()]
    color_map = {model: color for model, color in zip(base_model_order, sns.color_palette("deep", len(base_model_order)))}

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 7))

    sns.kdeplot(data=plot_data, x='ATE', hue='Base Model', hue_order=models_present,
                palette={m: color_map[m] for m in models_present},
                fill=True, alpha=0.2, linewidth=2, ax=ax, legend=False)

    legend_handles, legend_labels = [], []
    for base_model_name in models_present:
        mean_ate = plot_data[plot_data['Base Model'] == base_model_name]['ATE'].mean()
        ax.axvline(x=mean_ate, color=color_map[base_model_name], linestyle='--', linewidth=2.5)
        handle = plt.Rectangle((0,0), 1, 1, color=color_map[base_model_name], alpha=0.4)
        legend_handles.append(handle)
        legend_labels.append(f'{base_model_name} (Mean = {mean_ate:.3f})')
        
    ax.legend(legend_handles, legend_labels, fontsize=12, title='Model Structure', title_fontsize='14')
    ax.set_title(f'ATE Distributions Estimated by {estimator_type} Models', fontsize=18, pad=20)
    ax.set_xlabel('Average Treatment Effect (ATE) of Crashes on Traffic Speed', fontsize=14)
    ax.set_ylabel('Density', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.show()
