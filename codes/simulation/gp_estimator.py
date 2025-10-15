# file: gp_estimator.py
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
import torch
import gpytorch
from sklearn.preprocessing import StandardScaler
from data_simulator import SyntheticDataGenerator
import warnings

class GPModel(gpytorch.models.ExactGP):
    """A Gaussian Process model for spatio-temporal data."""
    def __init__(self, train_x, train_y, likelihood):
        super(GPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(ard_num_dims=3) 
        )
        self.time_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.MaternKernel(nu=2.5)
        )

    def forward(self, x):
        covar_x = x[:, :-1]
        time_x = x[:, -1].view(-1, 1)
        
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(covar_x)
        covar_t = self.time_module(time_x)
        covar = covar_x * covar_t
        return gpytorch.distributions.MultivariateNormal(mean_x, covar)

class STHCM_Estimator:
    """
    Implements the two-stage estimation algorithm for ST-HCMs.
    
    Stage 1: Learns unit-specific conditional dynamics using a GP model.
    Stage 2: Performs counterfactual simulation via G-Computation to estimate the ATE.
    """
    def __init__(self, training_iterations: int = 100):
        self.training_iter = training_iterations
        self.models_: Dict[int, GPModel] = {}
        self.likelihoods_: Dict[int, gpytorch.likelihoods.GaussianLikelihood] = {}
        self.scalers_: Dict[int, Tuple[StandardScaler, StandardScaler]] = {}
        self.neighbors_: Dict[int, List[int]] = {}
        self.N = 0
        self.T = 0
  
    def _get_spatial_lag_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Helper to compute spatial lags based on the stored neighbor map.
        [CORRECTED VERSION]
        """
        unit_avg_outcome = df.groupby(['unit_id', 'time'])['outcome'].mean().reset_index()

        spatial_lag_data = []
        for i in range(self.N):
            for t in range(self.T):
                spatial_lag = 0.0 
                # Spatial lag is defined by neighbors' outcomes at t-1
                if t > 0 and self.neighbor_map_.get(i):
                    neighbors = self.neighbor_map_[i]
                    # Efficiently filter for all neighbors at the specific time t-1
                    neighbor_outcomes_t_minus_1 = unit_avg_outcome[
                        (unit_avg_outcome['unit_id'].isin(neighbors)) &
                        (unit_avg_outcome['time'] == t - 1)
                    ]['outcome'].values
                    
                    if len(neighbor_outcomes_t_minus_1) > 0:
                        spatial_lag = np.mean(neighbor_outcomes_t_minus_1)

                spatial_lag_data.append({'unit_id': i, 'time': t, 'spatial_lag': spatial_lag})
        
        return pd.DataFrame(spatial_lag_data)
    
    def fit(self, df: pd.DataFrame, neighbor_map: Dict[int, List[int]]):
        """
        Stage 1: Train a unit-specific GP model for each unit.
        
        Args:
            df: The training data.
            neighbor_map: A dictionary defining the spatial structure.
        """
        self.N = df['unit_id'].nunique()
        self.T = df['time'].nunique()
        self.neighbor_map_ = neighbor_map # Store the correct neighbor map

        # Create lagged features
        featured_df = df.copy()
        featured_df['outcome_lag'] = featured_df.groupby(['unit_id', 'subunit_id'])['outcome'].shift(1)
        
        # Create and merge spatial lag features
        spatial_lag_df = self._get_spatial_lag_df(df)
        featured_df = pd.merge(featured_df, spatial_lag_df, on=['unit_id', 'time'])
        featured_df = featured_df.dropna()
        
        for i in range(self.N):
            print(f"Fitting model for unit {i+1}/{self.N}...")
            unit_data = featured_df[featured_df['unit_id'] == i]
            
            features = ['treatment', 'outcome_lag', 'spatial_lag', 'time']
            target = 'outcome'
            
            X_train = unit_data[features].values
            y_train = unit_data[target].values
            
            x_scaler, y_scaler = StandardScaler(), StandardScaler()
            X_train_scaled = x_scaler.fit_transform(X_train)
            y_train_scaled = y_scaler.fit_transform(y_train.reshape(-1, 1)).flatten()
            
            self.scalers_[i] = (x_scaler, y_scaler)
            
            X_train_torch = torch.tensor(X_train_scaled, dtype=torch.float32)
            y_train_torch = torch.tensor(y_train_scaled, dtype=torch.float32)
            
            likelihood = gpytorch.likelihoods.GaussianLikelihood()
            model = GPModel(X_train_torch, y_train_torch, likelihood)
            self.models_[i] = model
            self.likelihoods_[i] = likelihood
            
            # Training loop
            model.train()
            likelihood.train()
            optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
            mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
            
            for iter_ in range(self.training_iter):
                optimizer.zero_grad()
                output = model(X_train_torch)
                loss = -mll(output, y_train_torch)
                loss.backward()
                optimizer.step()


    def estimate_ate(self, df: pd.DataFrame) -> float:
        """
        Stage 2: Perform G-Computation via simulation and aggregate results.
        """
        final_effects = {0: [], 1: []}
        
        # Pre-calculate real-world neighbor histories for simulation.
        # This is a key methodological choice: we simulate the counterfactual
        # for unit `i` given the observed evolution of its neighbors.
        unit_avg_outcome = df.groupby(['unit_id', 'time'])['outcome'].mean().reset_index()

        for a_star in [0, 1]:
            print(f"\nSimulating policy do(A = {a_star})...")
            for i in range(self.N):
                # ... (The rest of the simulation loop is identical to your original code)
                model = self.models_[i]
                likelihood = self.likelihoods_[i]
                x_scaler, y_scaler = self.scalers_[i]
                
                model.eval()
                likelihood.eval()
                
                sim_outcomes = np.zeros(self.T)
                
                for t in range(self.T):
                    # Get real neighbor history
                    spatial_lag = 0
                    if t > 0 and self.neighbor_map_[i]:
                        neighbor_lags = []
                        for neighbor in self.neighbor_map_[i]:
                            val = unit_avg_outcome.loc[
                                (unit_avg_outcome['unit_id'] == neighbor) & 
                                (unit_avg_outcome['time'] == t-1), 
                                'outcome'
                            ].values
                            if len(val) > 0:
                                neighbor_lags.append(val[0])
                        spatial_lag = np.mean(neighbor_lags) if neighbor_lags else 0
                    
                    outcome_lag = sim_outcomes[t-1] if t > 0 else 0
                    
                    features_t = np.array([[a_star, outcome_lag, spatial_lag, t]])
                    features_t_scaled = x_scaler.transform(features_t)
                    features_t_torch = torch.tensor(features_t_scaled, dtype=torch.float32)
                    
                    with torch.no_grad(), gpytorch.settings.fast_pred_var():
                        prediction = likelihood(model(features_t_torch))
                        y_pred_scaled = prediction.mean.numpy()
                    
                    y_pred = y_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()[0]
                    sim_outcomes[t] = y_pred
                
                final_effects[a_star].append(sim_outcomes[-1])

        mean_effect_1 = np.mean(final_effects[1])
        mean_effect_0 = np.mean(final_effects[0])
        
        ate_T = mean_effect_1 - mean_effect_0
        return ate_T

if __name__ == '__main__':
    N_UNITS = 25 
    M_SUBUNITS = 50
    T_STEPS = 10
    TRUE_ATE = 5.0
    CONFOUNDING = 2.0
    SPILLOVER = 0.5

    print("Generating synthetic data...")
    generator = SyntheticDataGenerator(
        N=N_UNITS, m=M_SUBUNITS, T=T_STEPS,
        treatment_effect=TRUE_ATE,
        confounding_strength=CONFOUNDING,
        spatial_spillover_strength=SPILLOVER,
        spatial_structure='grid' 
    )
    data_df, neighbor_map = generator.generate() 
    print("Data generated.")

    print("\nInitializing and fitting the ST-HCM Estimator...")
    estimator = STHCM_Estimator(training_iterations=100)
    estimator.fit(data_df, neighbor_map) 
    print("Fitting complete.")

    print("\nEstimating ATE using G-Computation...")
    estimated_ate = estimator.estimate_ate(data_df)