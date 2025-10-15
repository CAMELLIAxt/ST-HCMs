# file: baselines.py 
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from tqdm import tqdm
import torch
import gpytorch
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Tuple
from gp_estimator import GPModel

def create_spatiotemporal_features(df: pd.DataFrame, neighbor_map: dict, aggregate: bool) -> pd.DataFrame:
    featured_df = df.copy()
    if aggregate:
        unit_df = featured_df.groupby(['unit_id', 'time']).mean().reset_index()
        unit_df['outcome_lag'] = unit_df.groupby('unit_id')['outcome'].shift(1)
        base_df_for_spatial = unit_df
        final_df_to_merge = unit_df
    else:
        featured_df['outcome_lag'] = featured_df.groupby(['unit_id', 'subunit_id'])['outcome'].shift(1)
        base_df_for_spatial = featured_df.groupby(['unit_id', 'time'])['outcome'].mean().reset_index()
        final_df_to_merge = featured_df
        
    spatial_lag_data = []
    base_df_for_spatial = base_df_for_spatial.set_index(['unit_id', 'time'])

    unique_units = df['unit_id'].unique()
    unique_times = df['time'].unique()

    for i in unique_units:
        for t in unique_times:
            spatial_lag = 0.0
            if t > 0 and neighbor_map.get(i):
                neighbors = neighbor_map[i]
                neighbor_indices = [(n, t - 1) for n in neighbors]
            
                valid_indices = [idx for idx in neighbor_indices if idx in base_df_for_spatial.index]
                
                if valid_indices:
                    neighbor_outcomes = base_df_for_spatial.loc[valid_indices, 'outcome'].values
                    
                    if not np.all(np.isnan(neighbor_outcomes)):
                        spatial_lag = np.nanmean(neighbor_outcomes)
            
            spatial_lag_data.append({'unit_id': i, 'time': t, 'spatial_lag': spatial_lag})

    spatial_lag_df = pd.DataFrame(spatial_lag_data)
    
    # Merge spatial lag and clean up
    final_df = pd.merge(final_df_to_merge, spatial_lag_df, on=['unit_id', 'time'])
    final_df = final_df.dropna(subset=['outcome_lag'])
    
    return final_df


def estimate_ate_aggregated_st_lmm(df: pd.DataFrame, neighbor_map: dict) -> float:
    featured_df = create_spatiotemporal_features(df, neighbor_map, aggregate=True)
    model = smf.ols('outcome ~ treatment + outcome_lag + spatial_lag', data=featured_df).fit()
    return model.params['treatment']

def estimate_ate_panel_model_no_space(df: pd.DataFrame, neighbor_map: dict) -> float:
    featured_df = df.copy()
    featured_df['outcome_lag'] = featured_df.groupby(['unit_id', 'subunit_id'])['outcome'].shift(1)
    featured_df = featured_df.dropna()
    model = smf.ols('outcome ~ treatment + outcome_lag + C(unit_id)', data=featured_df).fit()
    return model.params['treatment']

def estimate_ate_st_lmm(df: pd.DataFrame, neighbor_map: dict) -> float:
    featured_df = create_spatiotemporal_features(df, neighbor_map, aggregate=False)
    model = smf.ols('outcome ~ treatment + outcome_lag + spatial_lag + C(unit_id)', data=featured_df).fit()
    return model.params['treatment']

class GPModel_NoSpace(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(GPModel_NoSpace, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        # ARD on 2 dims: treatment, outcome_lag
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims=2))
        self.time_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(nu=2.5))

    def forward(self, x):
        covar_x = x[:, :-1]
        time_x = x[:, -1].view(-1, 1)
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(covar_x)
        covar_t = self.time_module(time_x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x * covar_t)

class Independent_THCM_Estimator:
    def __init__(self, training_iterations: int = 100):
        self.training_iter = training_iterations
        self.models_: Dict[int, GPModel_NoSpace] = {}
        self.likelihoods_: Dict[int, gpytorch.likelihoods.GaussianLikelihood] = {}
        self.scalers_: Dict[int, Tuple[StandardScaler, StandardScaler]] = {}
        self.N, self.T = 0, 0

    def fit(self, df: pd.DataFrame, neighbor_map: dict = None):
        self.N, self.T = df['unit_id'].nunique(), df['time'].nunique()
        featured_df = df.copy()
        featured_df['outcome_lag'] = featured_df.groupby(['unit_id', 'subunit_id'])['outcome'].shift(1)
        featured_df = featured_df.dropna()
        
        for i in range(self.N):
            unit_data = featured_df[featured_df['unit_id'] == i]
            features = ['treatment', 'outcome_lag', 'time']
            target = 'outcome'
            X_train, y_train = unit_data[features].values, unit_data[target].values
            
            x_scaler, y_scaler = StandardScaler(), StandardScaler()
            X_train_scaled = x_scaler.fit_transform(X_train)
            y_train_scaled = y_scaler.fit_transform(y_train.reshape(-1, 1)).flatten()
            self.scalers_[i] = (x_scaler, y_scaler)
            X_train_torch = torch.tensor(X_train_scaled, dtype=torch.float32)
            y_train_torch = torch.tensor(y_train_scaled, dtype=torch.float32)

            likelihood = gpytorch.likelihoods.GaussianLikelihood(
                noise_constraint=gpytorch.constraints.GreaterThan(1e-6)
            )
            model = GPModel_NoSpace(X_train_torch, y_train_torch, likelihood)
            self.models_[i], self.likelihoods_[i] = model, likelihood
            
            model.train(); likelihood.train()
            optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
            mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
            
            for _ in range(self.training_iter):
                with gpytorch.settings.cholesky_jitter(1e-6):
                    optimizer.zero_grad()
                    output = model(X_train_torch)
                    loss = -mll(output, y_train_torch)
                    if torch.isnan(loss): break
                    loss.backward()
                    optimizer.step()

    def estimate_ate(self, df: pd.DataFrame, neighbor_map: dict = None) -> float:
        final_effects = {0: [], 1: []}
        for a_star in [0, 1]:
            for i in range(self.N):
                model, likelihood = self.models_[i], self.likelihoods_[i]
                x_scaler, y_scaler = self.scalers_[i]
                model.eval(); likelihood.eval()
                
                sim_outcomes = np.zeros(self.T)
                for t in range(self.T):
                    outcome_lag = sim_outcomes[t-1] if t > 0 else 0
                    features_t = np.array([[a_star, outcome_lag, t]])
                    features_t_scaled = x_scaler.transform(features_t)
                    features_t_torch = torch.tensor(features_t_scaled, dtype=torch.float32)
                    
                    with torch.no_grad(), gpytorch.settings.fast_pred_var():
                        prediction = likelihood(model(features_t_torch))
                        y_pred_scaled = prediction.mean.numpy()
                    
                    y_pred = y_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()[0]
                    sim_outcomes[t] = y_pred
                
                final_effects[a_star].append(sim_outcomes[-1])

        return np.mean(final_effects[1]) - np.mean(final_effects[0])

class Aggregated_STGP_Estimator:
    def __init__(self, training_iterations: int = 100):
        self.training_iter = training_iterations
        self.model_: GPModel = None
        self.likelihood_: gpytorch.likelihoods.GaussianLikelihood = None
        self.scaler_: Tuple[StandardScaler, StandardScaler] = None
        self.neighbor_map_: dict = None
        self.N, self.T = 0, 0

    def fit(self, df: pd.DataFrame, neighbor_map: dict):
        self.neighbor_map_ = neighbor_map
        self.N, self.T = df['unit_id'].nunique(), df['time'].nunique()
        
        unit_df = create_spatiotemporal_features(df, neighbor_map, aggregate=True)

        features = ['treatment', 'outcome_lag', 'spatial_lag', 'time']
        target = 'outcome'
        X_train, y_train = unit_df[features].values, unit_df[target].values
        
        x_scaler, y_scaler = StandardScaler(), StandardScaler()
        X_train_scaled = x_scaler.fit_transform(X_train)
        y_train_scaled = y_scaler.fit_transform(y_train.reshape(-1, 1)).flatten()
        self.scaler_ = (x_scaler, y_scaler)
        X_train_torch = torch.tensor(X_train_scaled, dtype=torch.float32)
        y_train_torch = torch.tensor(y_train_scaled, dtype=torch.float32)
        
        self.likelihood_ = gpytorch.likelihoods.GaussianLikelihood(
            noise_constraint=gpytorch.constraints.GreaterThan(1e-6)
        )
        self.model_ = GPModel(X_train_torch, y_train_torch, self.likelihood_)
        
        self.model_.train(); self.likelihood_.train()
        optimizer = torch.optim.Adam(self.model_.parameters(), lr=0.1)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood_, self.model_)
        
        for _ in tqdm(range(self.training_iter), desc="Fitting Aggregated ST-GP", leave=False):
            with gpytorch.settings.cholesky_jitter(1e-6):
                optimizer.zero_grad()
                output = self.model_(X_train_torch)
                loss = -mll(output, y_train_torch)
                if torch.isnan(loss): break
                loss.backward()
                optimizer.step()

    def estimate_ate(self, df: pd.DataFrame, neighbor_map: dict) -> float:
        if self.model_ is None:
            raise RuntimeError("The fit method must be called before estimate_ate.")
            
        final_effects = {0: [], 1: []}
        x_scaler, y_scaler = self.scaler_
        
        self.model_.eval(); self.likelihood_.eval()
        
        for a_star in [0, 1]:
            sim_unit_outcomes = np.zeros((self.N, self.T))
            for t in range(self.T):
                for i in range(self.N):
                    outcome_lag = sim_unit_outcomes[i, t-1] if t > 0 else 0
                    
                    spatial_lag = 0.0
                    if t > 0 and self.neighbor_map_.get(i):
                        # Ensure neighbors exist before indexing
                        valid_neighbors = [n for n in self.neighbor_map_[i] if n < self.N]
                        if valid_neighbors:
                            neighbor_lags = sim_unit_outcomes[valid_neighbors, t-1]
                            spatial_lag = np.mean(neighbor_lags)
                        
                    features_t = np.array([[a_star, outcome_lag, spatial_lag, t]])
                    features_t_scaled = x_scaler.transform(features_t)
                    features_t_torch = torch.tensor(features_t_scaled, dtype=torch.float32)
                    
                    with torch.no_grad(), gpytorch.settings.fast_pred_var():
                        prediction = self.likelihood_(self.model_(features_t_torch))
                        y_pred_scaled = prediction.mean.numpy()
                    
                    y_pred = y_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()[0]
                    sim_unit_outcomes[i, t] = y_pred
            
            final_effects[a_star] = np.mean(sim_unit_outcomes[:, -1])

        return final_effects[1] - final_effects[0]