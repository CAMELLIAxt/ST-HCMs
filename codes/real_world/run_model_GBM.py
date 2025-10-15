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

FEATURED_DATA_PATH = 'chicago_panel_with_features.csv'
RESULTS_OUTPUT_PATH = 'ate_results_gbm.json'
LOG_FILE_PATH = 'experiment_gbm.log'
ATE_FIGURE_PATH = 'ate_comparison_GBM.png'


N_BOOTSTRAP_RUNS = 100
SUBSAMPLE_FRACTION = 0.4
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
        X = bootstrap_sample[features]
        y = bootstrap_sample[model_config['target']]
        
        numeric_features = X.select_dtypes(include=np.number).columns.tolist()
        
        preprocessor = ColumnTransformer(transformers=[
            ('num', StandardScaler(), numeric_features)],
            remainder='passthrough'
        )
        
        gbm_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                       ('regressor', lgb.LGBMRegressor(random_state=iteration_seed, n_jobs=1, verbose=-1))])
        gbm_pipeline.fit(X, y)
        predict_fn = gbm_pipeline.predict

        df_fact_0 = bootstrap_sample.copy(); df_fact_0['treatment'] = 0
        df_fact_1 = bootstrap_sample.copy(); df_fact_1['treatment'] = 1

        pred_0 = predict_fn(df_fact_0[features])
        pred_1 = predict_fn(df_fact_1[features])
            
        return np.mean(pred_1) - np.mean(pred_0)

    except Exception as e:
        logging.warning(f"fail (seed: {iteration_seed}): {e}")
        return None

if __name__ == "__main__":
    df_full = pd.read_csv(FEATURED_DATA_PATH)
    df_full['timestamp'] = pd.to_datetime(df_full['timestamp'])
    for col in ['unit', 'subunit', 'hour', 'dayofweek']:
        df_full[col] = df_full[col].astype('category')

    if SUBSAMPLE_FRACTION < 1.0:
        np.random.seed(MAIN_RANDOM_SEED)
        subunits_to_keep = df_full.groupby('unit')['subunit'].unique().apply(
            lambda x: np.random.choice(x, size=max(1, int(len(x) * SUBSAMPLE_FRACTION)), replace=False)
        ).explode()
        df_sampled = df_full[df_full['subunit'].isin(subunits_to_keep)].copy()
    else:
        df_sampled = df_full
    
    df_agg = df_sampled.groupby(['unit', 'timestamp'], observed=False).agg(
        speed=('speed', 'mean'), treatment=('treatment', 'max'), hour=('hour', 'first'),
        dayofweek=('dayofweek', 'first'), avg_neighbor_speed_lag1=('avg_neighbor_speed_lag1', 'first')
    ).reset_index()
    df_agg = df_agg.sort_values(['unit', 'timestamp'])
    df_agg['agg_speed_lag1'] = df_agg.groupby('unit', observed=False)['speed'].shift(1)
    df_agg.dropna(inplace=True)
    
    T_HCM_FEATURES = ['treatment', 'speed_lag1', 'treatment_lag1', 'hour', 'dayofweek']
    ST_HCM_FEATURES = T_HCM_FEATURES + ['avg_neighbor_speed_lag1', 'avg_neighbor_treatment_lag1']
    AGG_FEATURES = ['treatment', 'agg_speed_lag1', 'hour', 'dayofweek', 'avg_neighbor_speed_lag1']
    
    models_to_run = {
        "Aggregated (GBM)": {"type": "GBM", "features": AGG_FEATURES + ['unit'], "target": "speed", "data": df_agg, "unit_col": "unit", "bootstrap_by": "unit"},
        "T-HCM (GBM)": {"type": "GBM", "features": T_HCM_FEATURES + ['unit'], "target": "speed", "data": df_sampled, "unit_col": "unit"},
        "ST-HCM (GBM)": {"type": "GBM", "features": ST_HCM_FEATURES + ['unit'], "target": "speed", "data": df_sampled, "unit_col": "unit"}
    }
    
    all_results = {}
    for name, config in models_to_run.items():
        logging.info(f"Start running the model: {name}")
        iteration_seeds = [MAIN_RANDOM_SEED + i for i in range(N_BOOTSTRAP_RUNS)]
        results_list = Parallel(n_jobs=N_JOBS)(
            delayed(run_single_bootstrap_iteration)(config['data'], config, seed) 
            for seed in tqdm(iteration_seeds, desc=f"Bootstrap for {name}")
        )
        all_results[name] = [res for res in results_list if res is not None and np.isfinite(res)]

    with open(RESULTS_OUTPUT_PATH, 'w') as f: json.dump(all_results, f, indent=4)