# file: robustness_data_simulator.py

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple

class SensitivityDataGenerator:
    def __init__(self, N: int, m: int, T: int, 
                 base_confounding_strength: float = 2.0, 
                 base_spatial_spillover_strength: float = 1.5,
                 treatment_effect: float = 5.0,
                 noise_std: float = 2.0,
                 spatial_structure: str = 'line'):
        
        self.N = N
        self.m = m
        self.T = T
        self.base_confounding_strength = base_confounding_strength
        self.base_spatial_spillover_strength = base_spatial_spillover_strength
        self.treatment_effect = treatment_effect
        self.noise_std = noise_std
        
        if spatial_structure == 'line':
            self.neighbors = self._create_1d_line_grid()
        elif spatial_structure == 'grid':
            # For grid, we need a spatial ordering for the acyclic part
            side_len = int(np.sqrt(N))
            if side_len * side_len != N:
                raise ValueError("For 2D grid, N must be a perfect square.")
            self.neighbors, self.ordered_units = self._create_2d_grid_with_ordering()
        else:
            raise ValueError("spatial_structure must be 'line' or 'grid'")
            
        # Static part of the confounder
        self.U_static = np.random.randn(N)
        self.true_ate = self.treatment_effect

    def _create_1d_line_grid(self):
    
        neighbors = {}
        for i in range(self.N):
            neighbors[i] = [n for n in [i-1, i+1] if 0 <= n < self.N]
        return neighbors

    def _create_2d_grid_with_ordering(self):
        side_len = int(np.sqrt(self.N))
        neighbors = {}
        for i in range(self.N):
            neighbors[i] = []
            row, col = divmod(i, side_len)
            if row > 0: neighbors[i].append(i - side_len)
            if row < side_len - 1: neighbors[i].append(i + side_len)
            if col > 0: neighbors[i].append(i - 1)
            if col < side_len - 1: neighbors[i].append(i + 1)
        ordered_units = list(range(self.N))
        return neighbors, ordered_units

    def _sigmoid(self, x: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-x))

    def generate_violating_spatial_ordering_v0(self, cyclicity_strength: float):
        records = []
        # We need to store subunit outcomes for the simultaneous effects
        subunit_outcomes = np.zeros((self.N, self.m, self.T))
        unit_avg_outcomes = np.zeros((self.N, self.T))

        for t in range(self.T):
            # --- First pass: Acyclic part (as before) ---
            for i in self.ordered_units: # Iterate in causal order
                treatment_prob = self._sigmoid(self.base_confounding_strength * self.U_static[i] - 0.5)
                treatments = np.random.binomial(1, treatment_prob, self.m)
                
                temporal_lag = unit_avg_outcomes[i, t-1] if t > 0 else 0
                
                # Acyclic spatial lag (from neighbors j where j < i)
                acyclic_spatial_lag = 0
                if t > 0:
                    acyclic_neighbors = [n for n in self.neighbors.get(i, []) if n < i]
                    if acyclic_neighbors:
                        # Use already computed t-1 average outcomes
                        neighbor_outcomes = [unit_avg_outcomes[k, t-1] for k in acyclic_neighbors]
                        acyclic_spatial_lag = np.mean(neighbor_outcomes)
                
                dynamic_effect = 0.5 * temporal_lag + self.base_spatial_spillover_strength * acyclic_spatial_lag
                base_outcome = self.base_confounding_strength * self.U_static[i]
                noise = np.random.randn(self.m) * self.noise_std
                
                # Store treatments for the second pass
                for j in range(self.m):
                    records.append({'unit_id': i, 'subunit_id': j, 'time': t, 'treatment': treatments[j]})
                
                outcomes_pass1 = base_outcome + dynamic_effect + self.treatment_effect * treatments + noise
                subunit_outcomes[i, :, t] = outcomes_pass1

            # --- Second pass: Add cyclic/feedback part ---
            # This simulates the simultaneous influence
            final_outcomes_t = subunit_outcomes[:, :, t].copy()
            for i in self.ordered_units:
                # Cyclic spatial influence (from neighbors j where j > i)
                cyclic_spatial_influence = 0
                cyclic_neighbors = [n for n in self.neighbors.get(i, []) if n > i]
                if cyclic_neighbors:
                    # Use the outcomes from the first pass of these "later" units
                    neighbor_outcomes_pass1 = subunit_outcomes[cyclic_neighbors, :, t].mean() # 这里有问题！
                    cyclic_spatial_influence = neighbor_outcomes_pass1

                final_outcomes_t[i, :] += cyclicity_strength * cyclic_spatial_influence

            # Update final outcomes and averages for the next time step
            unit_avg_outcomes[:, t] = final_outcomes_t.mean(axis=1)
            # Add final outcomes to records
            for rec_idx in range(len(records) - self.N * self.m, len(records)):
                rec = records[rec_idx]
                rec['outcome'] = final_outcomes_t[rec['unit_id'], rec['subunit_id']]

        return pd.DataFrame(records), self.neighbors

    def generate_violating_spatial_ordering_v1(self, cyclicity_strength: float):
        """
        Generates data that violates the spatial causal ordering assumption.
        [CORRECTED VERSION]
        """
        records = []
        unit_avg_outcomes = np.zeros((self.N, self.T))

        treatments_cube = np.zeros((self.N, self.m, self.T), dtype=int)

        for t in range(self.T):
            C = np.zeros((self.N, self.N))
            if cyclicity_strength > 0:
                for i in range(self.N):
                    feedback_neighbors = [n for n in self.neighbors.get(i, []) if n > i]
                    if feedback_neighbors:
                        for neighbor in feedback_neighbors:
                            C[i, neighbor] = cyclicity_strength / len(feedback_neighbors)

            I = np.identity(self.N)
            if np.linalg.det(I - C) == 0:
                print(f"Warning: Simultaneous system is unstable for cyclicity={cyclicity_strength}. Capping strength.")
                C = C * 0.9 / np.max(np.abs(np.linalg.eigvals(C))) if np.max(np.abs(np.linalg.eigvals(C))) > 0 else C

            inv_I_minus_C = np.linalg.inv(I - C)

            base_unit_outcomes_t = np.zeros(self.N)
            current_U = self.U_static 
            
            for i in range(self.N):
                treatment_prob = self._sigmoid(self.base_confounding_strength * current_U[i] - 0.5)
                treatments = np.random.binomial(1, treatment_prob, self.m)
                treatments_cube[i, :, t] = treatments
                
                temporal_lag = unit_avg_outcomes[i, t-1] if t > 0 else 0
                
                acyclic_spatial_lag = 0
                if t > 0:
                    acyclic_neighbors = [n for n in self.neighbors.get(i, []) if n < i]
                    if acyclic_neighbors:
                        acyclic_spatial_lag = np.mean(unit_avg_outcomes[acyclic_neighbors, t-1])

                dynamic_effect = 0.5 * temporal_lag + self.base_spatial_spillover_strength * acyclic_spatial_lag
                base_confounding_effect = self.base_confounding_strength * current_U[i]
                
                treatment_component = self.treatment_effect * np.mean(treatments)
                
                base_unit_outcomes_t[i] = base_confounding_effect + dynamic_effect + treatment_component

            final_unit_avg_outcomes_t = inv_I_minus_C @ base_unit_outcomes_t
            unit_avg_outcomes[:, t] = final_unit_avg_outcomes_t
            
            for i in range(self.N):
                noise = np.random.randn(self.m) * self.noise_std
                
                indiv_treatment_effect = self.treatment_effect * treatments_cube[i, :, t]
                mean_treatment_effect = np.mean(indiv_treatment_effect)
                
                outcomes = final_unit_avg_outcomes_t[i] + (indiv_treatment_effect - mean_treatment_effect) + noise
                
                for j in range(self.m):
                    records.append({
                        'unit_id': i, 'subunit_id': j, 'time': t,
                        'treatment': treatments_cube[i, j, t],
                        'outcome': outcomes[j]
                    })
        
        return pd.DataFrame(records), self.neighbors
    
    def generate_violating_spatial_ordering(self, cyclicity_strength: float):
        records = []
        unit_avg_outcomes = np.zeros((self.N, self.T))

        for t in range(self.T):
            subunit_noise_t = np.random.randn(self.N, self.m) * self.noise_std
            
            correlated_noise_t = subunit_noise_t.copy()
            if cyclicity_strength > 0:
                for i in range(self.N):
                    all_neighbors = self.neighbors.get(i, [])
                    if all_neighbors:
                        neighbor_noise = subunit_noise_t[all_neighbors, :].mean(axis=0)
                        correlated_noise_t[i, :] += cyclicity_strength * neighbor_noise

            for i in range(self.N):
                treatment_prob = self._sigmoid(self.base_confounding_strength * self.U_static[i] - 0.5)
                treatments = np.random.binomial(1, treatment_prob, self.m)

                temporal_lag = unit_avg_outcomes[i, t-1] if t > 0 else 0
                spatial_lag = 0
                if t > 0 and self.neighbors[i]:
                    neighbor_outcomes = [unit_avg_outcomes[k, t-1] for k in self.neighbors[i]]
                    spatial_lag = np.mean(neighbor_outcomes)
                
                linear_dynamic_effect = 0.5 * temporal_lag + self.base_spatial_spillover_strength * spatial_lag
                base_confounding_effect = self.base_confounding_strength * self.U_static[i]
                
                noise_for_this_unit = correlated_noise_t[i, :]

                outcomes = (base_confounding_effect +
                            linear_dynamic_effect +
                            self.treatment_effect * treatments +
                            noise_for_this_unit) 

                unit_avg_outcomes[i, t] = np.mean(outcomes)
                for j in range(self.m):
                    records.append({
                        'unit_id': i, 'subunit_id': j, 'time': t,
                        'treatment': treatments[j], 'outcome': outcomes[j]
                    })

        return pd.DataFrame(records), self.neighbors
    
    def generate_violating_time_invariance(self, confounder_drift_strength: float):
        records = []
        unit_avg_outcomes = np.zeros((self.N, self.T))
        
        U_t = np.zeros((self.N, self.T))
        U_t[:, 0] = self.U_static

        for t in range(self.T):
            if t > 0:
                U_t[:, t] = U_t[:, t-1] + confounder_drift_strength * np.random.randn(self.N)

            for i in range(self.N):
                treatment_prob = self._sigmoid(self.base_confounding_strength * U_t[i, t] - 0.5)
                treatments = np.random.binomial(1, treatment_prob, self.m)

                temporal_lag = unit_avg_outcomes[i, t-1] if t > 0 else 0
                spatial_lag = 0
                if t > 0 and self.neighbors[i]:
                    neighbor_outcomes = [unit_avg_outcomes[k, t-1] for k in self.neighbors[i]]
                    spatial_lag = np.mean(neighbor_outcomes)
                
                dynamic_effect = 0.5 * temporal_lag + self.base_spatial_spillover_strength * spatial_lag
                
                base_outcome = self.base_confounding_strength * U_t[i, t]
                
                noise = np.random.randn(self.m) * self.noise_std
                outcomes = base_outcome + dynamic_effect + self.treatment_effect * treatments + noise
                
                unit_avg_outcomes[i, t] = np.mean(outcomes)
                for j in range(self.m):
                    records.append({
                        'unit_id': i, 'subunit_id': j, 'time': t,
                        'treatment': treatments[j], 'outcome': outcomes[j]
                    })

        return pd.DataFrame(records), self.neighbors