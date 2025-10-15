# file: data_simulator.py
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
import warnings

class SyntheticDataGenerator:
    """
    Generates synthetic spatio-temporal hierarchical data.
    """
    def __init__(self, N: int, m: int, T: int, confounding_strength: float = 1.0, 
                 spatial_spillover_strength: float = 0.5, treatment_effect: float = 5.0,
                 noise_std: float = 2.0,
                 spatial_structure: str = 'line'):
        self.N = N
        self.m = m
        self.T = T
        self.confounding_strength = confounding_strength
        self.spatial_spillover_strength = spatial_spillover_strength
        self.treatment_effect = treatment_effect
        self.noise_std = noise_std
        if spatial_structure == 'line':
            self.neighbors = self._create_1d_line_grid()
        elif spatial_structure == 'grid':
            self.neighbors = self._create_2d_grid()
        else:
            raise ValueError("spatial_structure must be 'line' or 'grid'")
        # The unobserved confounder U_i. This will NOT be in the final output.
        self.U = np.random.randn(N)

    def _create_1d_line_grid(self) -> Dict[int, List[int]]:
        """Creates a simple 1D grid spatial structure (e.g., units on a line)."""
        neighbors = {}
        for i in range(self.N):
            neighbors[i] = []
            if i > 0:
                neighbors[i].append(i - 1)
            if i < self.N - 1:
                neighbors[i].append(i + 1)
        return neighbors

    def _create_2d_grid(self) -> Dict[int, List[int]]:
        """Creates a 2D grid spatial structure."""
        side_len = int(np.sqrt(self.N))
        if side_len * side_len != self.N:
            raise ValueError("For 2D grid, N must be a perfect square.")
        
        neighbors = {}
        for i in range(self.N):
            neighbors[i] = []
            row, col = divmod(i, side_len)
            # Up
            if row > 0: neighbors[i].append(i - side_len)
            # Down
            if row < side_len - 1: neighbors[i].append(i + side_len)
            # Left
            if col > 0: neighbors[i].append(i - 1)
            # Right
            if col < side_len - 1: neighbors[i].append(i + 1)
        return neighbors
    
    def _sigmoid(self, x: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-x))


    def generate(self) -> Tuple[pd.DataFrame, Dict[int, List[int]]]:
        records = []
        unit_avg_outcomes = np.zeros((self.N, self.T))

        for t in range(self.T):
            for i in range(self.N):
                treatment_prob = self._sigmoid(self.confounding_strength * self.U[i] - 0.5)
                treatments = np.random.binomial(1, treatment_prob, self.m)

                temporal_lag = unit_avg_outcomes[i, t-1] if t > 0 else 0
                spatial_lag = 0
                if t > 0 and self.neighbors[i]:
                    neighbor_outcomes = [unit_avg_outcomes[k, t-1] for k in self.neighbors[i]]
                    spatial_lag = np.mean(neighbor_outcomes)

                base_outcome = (self.confounding_strength * np.tanh(self.U[i]) + 
                                0.5 * temporal_lag + 
                                self.spatial_spillover_strength * spatial_lag)
                
                noise = np.random.randn(self.m) * self.noise_std
                outcomes = (base_outcome + 
                            self.treatment_effect * treatments + 
                            noise)
                
                unit_avg_outcomes[i, t] = np.mean(outcomes)

                for j in range(self.m):
                    records.append({
                        'unit_id': i,
                        'subunit_id': j,
                        'time': t,
                        'treatment': treatments[j],
                        'outcome': outcomes[j]
                    })
        
        return pd.DataFrame(records), self.neighbors
    
    def generate_v5(self):
        records = []
        unit_avg_outcomes = np.zeros((self.N, self.T))
        self.true_ate_v5 = self.treatment_effect

        for t in range(self.T):
            for i in range(self.N):
                treatment_prob = self._sigmoid(self.confounding_strength * self.U[i] - 0.5)
                treatments = np.random.binomial(1, treatment_prob, self.m)

                temporal_lag = unit_avg_outcomes[i, t-1] if t > 0 else 0
                spatial_lag = 0
                if t > 0 and self.neighbors[i]:
                    neighbor_outcomes = [unit_avg_outcomes[k, t-1] for k in self.neighbors[i]]
                    spatial_lag = np.mean(neighbor_outcomes)

                linear_dynamic_effect = 0.5 * temporal_lag + self.spatial_spillover_strength * spatial_lag

                base_outcome = self.confounding_strength * self.U[i]

                noise = np.random.randn(self.m) * self.noise_std
                outcomes = (base_outcome +
                            linear_dynamic_effect +
                            self.treatment_effect * treatments +
                            noise)

                unit_avg_outcomes[i, t] = np.mean(outcomes)

                for j in range(self.m):
                    records.append({
                        'unit_id': i, 'subunit_id': j, 'time': t,
                        'treatment': treatments[j], 'outcome': outcomes[j]
                    })

        return pd.DataFrame(records), self.neighbors
    

    def generate_v6(self):
        records = []
        unit_avg_outcomes = np.zeros((self.N, self.T))
        self.true_ate_v6 = self.treatment_effect 

        for t in range(self.T):
            for i in range(self.N):
                treatment_prob = self._sigmoid(self.confounding_strength * self.U[i] - 0.5)
                treatments = np.random.binomial(1, treatment_prob, self.m)

                temporal_lag = unit_avg_outcomes[i, t-1] if t > 0 else 0
                spatial_lag = 0
                if t > 0 and self.neighbors[i]:
                    neighbor_outcomes = [unit_avg_outcomes[k, t-1] for k in self.neighbors[i]]
                    spatial_lag = np.mean(neighbor_outcomes)
                
                non_linear_dynamic_effect = 0.5 * np.sin(temporal_lag) + self.spatial_spillover_strength * np.tanh(spatial_lag / 5.0)
                interaction_effect = 2.0 * spatial_lag * treatments
                
                cate_i = self.treatment_effect + self.confounding_strength * np.exp(self.U[i] / 4.0)
                heterogeneous_treatment_effect = cate_i * treatments
                
                base_outcome = (self.confounding_strength * np.tanh(self.U[i])) * (1 + 0.2 * non_linear_dynamic_effect)
                noise = np.random.randn(self.m) * self.noise_std
                outcomes = base_outcome + heterogeneous_treatment_effect + non_linear_dynamic_effect + interaction_effect + noise
                
                unit_avg_outcomes[i, t] = np.mean(outcomes)

                for j in range(self.m):
                    records.append({
                        'unit_id': i, 'subunit_id': j, 'time': t,
                        'treatment': treatments[j], 'outcome': outcomes[j]
                    })
        
        return pd.DataFrame(records), self.neighbors