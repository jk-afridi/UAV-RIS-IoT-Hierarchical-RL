#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AoI-Aware UAV-RIS Environment for IoT Data Collection

Extends the SmartCityRISEnvironment with:
  1. UAV mobility (3D trajectory with velocity constraints)
  2. Per-device Age of Information (AoI) tracking
  3. Rician fading channel model (position-dependent LoS)
  4. Uplink IoT data collection (reversed link direction)
  5. UAV propulsion energy model (Zeng & Zhang, IEEE TWC 2019)
"""

import gym
from gym import spaces
import numpy as np
from .smart_city_environment import SmartCityRISEnvironment


class AoIUAVRISEnvironment(SmartCityRISEnvironment):
    """
    AoI-aware UAV-mounted RIS environment for IoT uplink data collection.

    The UAV flies between DBSCAN-clustered IoT device groups, collecting
    sensor data via RIS-enhanced uplink. The objective is to minimize
    average Age of Information across all devices.
    """

    def __init__(self,
                 num_antennas=4,
                 num_RIS_elements=64,
                 num_users=4,
                 mismatch=False,
                 channel_est_error=False,
                 cascaded_channels=False,
                 beta_min=0.6,
                 theta_bar=0.0,
                 kappa_bar=1.5,
                 AWGN_var=1e-2,
                 channel_noise_var=1e-2,
                 seed=0,
                 city_size=100,
                 bs_height=15,
                 ris_height=15,
                 max_service_angle=60,
                 # --- New AoI parameters ---
                 v_max=10.0,
                 h_min=10.0,
                 h_max=50.0,
                 energy_budget=500.0,
                 episode_length=200,
                 delta_t=1.0,
                 r_min=0.5,
                 k_rice=10.0,
                 iot_tx_power_dbm=10.0,
                 enable_3d_trajectory=False,
                 layout_seed=42,
                 k_rice_bs=None):
        """
        Parameters (new ones only):
            v_max (float): Maximum UAV speed (m/s)
            h_min (float): Minimum UAV altitude (m)
            h_max (float): Maximum UAV altitude (m)
            energy_budget (float): Total UAV energy budget (Joules)
            episode_length (int): Number of timesteps per episode
            delta_t (float): Duration of each timestep (seconds)
            r_min (float): Minimum uplink rate for successful decode (bps/Hz)
            k_rice (float): Rician K-factor (linear) of the device<->RIS links
            k_rice_bs (float|None): Rician K-factor of the BS<->RIS link (default: k_rice).
                A strongly LoS (large K) BS link makes H1 ~rank-1, which makes
                simultaneous multi-device (SDMA) separation impossible; set a small
                value to model a scattering-rich BS side.
            iot_tx_power_dbm (float): IoT device transmit power (dBm)
            enable_3d_trajectory (bool): If True, UAV altitude is variable
        """
        # Store AoI-specific params before parent init
        self.v_max = v_max
        self.h_min = h_min
        self.h_max = h_max
        self.energy_budget = energy_budget
        self.episode_length = episode_length
        self.delta_t = delta_t
        self.r_min = r_min
        self.k_rice = k_rice
        self.k_rice_bs = k_rice if k_rice_bs is None else k_rice_bs
        self.iot_tx_power_dbm = iot_tx_power_dbm
        self.iot_tx_power = 10 ** (iot_tx_power_dbm / 10) * 1e-3  # Convert to Watts
        self.enable_3d = enable_3d_trajectory

        # UAV propulsion model constants (rotary-wing, Zeng & Zhang TWC 2019)
        self.P0 = 79.86        # Blade profile power (W)
        self.P_i = 88.63       # Induced power (W)
        self.U_tip = 120.0     # Tip speed of rotor blade (m/s)
        self.v0 = 4.03         # Mean rotor induced velocity (m/s)
        self.d0 = 0.6          # Fuselage drag ratio
        self.rho = 1.225       # Air density (kg/m^3)
        self.s_rotor = 0.05    # Rotor solidity
        self.A_rotor = 0.503   # Rotor disc area (m^2)

        # Initialize parent (SmartCityRIS → RIS_MISO_PDA)
        super().__init__(
            num_antennas=num_antennas,
            num_RIS_elements=num_RIS_elements,
            num_users=num_users,
            mismatch=mismatch,
            channel_est_error=channel_est_error,
            cascaded_channels=cascaded_channels,
            beta_min=beta_min,
            theta_bar=theta_bar,
            kappa_bar=kappa_bar,
            AWGN_var=AWGN_var,
            channel_noise_var=channel_noise_var,
            seed=seed,
            city_size=city_size,
            bs_height=bs_height,
            ris_height=ris_height,
            max_service_angle=max_service_angle,
            layout_seed=layout_seed
        )

        # --- AoI state variables ---
        self.K_total = self.total_users  # Total IoT devices
        self.aoi = np.ones(self.K_total)  # AoI counters (init to 1)

        # --- UAV state ---
        self.uav_pos = np.array([city_size / 2, city_size / 2, ris_height],
                                dtype=np.float64)
        self.energy_remaining = energy_budget
        self.timestep = 0

        # --- High-level state/action spaces ---
        # High-level state: [uav_x, uav_y, uav_z, aoi_1..aoi_K, e_remain]
        self.high_state_dim = 3 + self.K_total + 1
        # High-level action: [v_x, v_y, (v_z if 3D), cluster_id]
        vel_dim = 3 if self.enable_3d else 2
        self.high_action_dim = vel_dim  # cluster_id is discrete, handled separately

        self.high_observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self.high_state_dim,), dtype=np.float64
        )
        self.high_action_space = spaces.Box(
            low=-1.0, high=1.0,
            shape=(self.high_action_dim,), dtype=np.float64
        )

    # ============================
    # UAV Mobility
    # ============================

    def _move_uav(self, velocity):
        """Move UAV by velocity vector, clipping to speed and boundary constraints."""
        velocity = np.array(velocity, dtype=np.float64)

        # Clip speed to v_max
        speed = np.linalg.norm(velocity)
        if speed > self.v_max:
            velocity = velocity * (self.v_max / speed)
            speed = self.v_max

        # Update position
        new_pos = self.uav_pos.copy()
        new_pos[:len(velocity)] += velocity * self.delta_t

        # Clip to city boundaries
        new_pos[0] = np.clip(new_pos[0], 0, self.city_size)
        new_pos[1] = np.clip(new_pos[1], 0, self.city_size)

        # Clip altitude
        if self.enable_3d and len(velocity) == 3:
            new_pos[2] = np.clip(new_pos[2], self.h_min, self.h_max)

        self.uav_pos = new_pos
        return speed

    def _compute_propulsion_energy(self, speed):
        """
        Compute UAV propulsion energy for one timestep.
        Rotary-wing model from Zeng & Zhang, IEEE TWC 2019, Eq. (12).
        """
        v = max(speed, 0.01)  # Avoid division by zero

        # Blade profile power
        P_blade = self.P0 * (1 + 3 * v**2 / self.U_tip**2)

        # Induced power
        P_induced = self.P_i * np.sqrt(
            np.sqrt(1 + v**4 / (4 * self.v0**4)) - v**2 / (2 * self.v0**2)
        )

        # Parasite power (fuselage drag)
        P_parasite = 0.5 * self.d0 * self.rho * self.s_rotor * self.A_rotor * v**3

        total_power = P_blade + P_induced + P_parasite
        return total_power * self.delta_t

    # ============================
    # Channel Model (Rician)
    # ============================

    def _ris_steering(self, direction):
        """LoS array response of the RIS (square UPA, half-wavelength spacing,
        in the x-y plane) towards unit vector `direction`."""
        nx = int(np.ceil(np.sqrt(self.L)))
        idx = np.arange(self.L)
        m, n = idx // nx, idx % nx
        return np.exp(1j * np.pi * (m * direction[0] + n * direction[1]))

    def _bs_steering(self, direction):
        """LoS array response of the BS ULA (along x, half-wavelength spacing)."""
        return np.exp(1j * np.pi * np.arange(self.M) * direction[0])

    def _compute_rician_channel(self, pos1, pos2, shape, los=None, k_rice=None):
        """
        Rician fading channel between two positions.
        H = sqrt(K/(1+K)) * H_LoS + sqrt(1/(1+K)) * H_NLoS
        `los` (optional, shape `shape`) is a unit-modulus geometric LoS
        response; if omitted the LoS term is all-ones (legacy behaviour).
        """
        distance = max(np.linalg.norm(pos1 - pos2), 0.1)

        # Free-space path loss (amplitude)
        freq = 2.4e9
        c = 3e8
        pl_db = 20 * np.log10(distance) + 20 * np.log10(freq) + 20 * np.log10(4 * np.pi / c)
        pl_linear = 10 ** (-pl_db / 20)

        H_LoS = (np.ones(shape, dtype=complex) if los is None else los) * pl_linear
        H_NLoS = (np.random.normal(0, np.sqrt(0.5), shape) +
                  1j * np.random.normal(0, np.sqrt(0.5), shape)) * pl_linear

        K = self.k_rice if k_rice is None else k_rice
        return np.sqrt(K / (1 + K)) * H_LoS + np.sqrt(1 / (1 + K)) * H_NLoS

    def _update_channels_for_position(self):
        """Recompute H_1 / H_2 from the CURRENT UAV (= RIS) position."""
        ris = self.uav_pos  # RIS is mounted on the UAV

        # H1: RIS <-> BS (L x M), geometric LoS
        d_bs = self.bs_position - ris
        d_bs = d_bs / max(np.linalg.norm(d_bs), 1e-9)
        los1 = np.outer(self._ris_steering(d_bs), self._bs_steering(-d_bs))
        self.H_1 = self._compute_rician_channel(self.bs_position, ris, (self.L, self.M), los1,
                                             k_rice=self.k_rice_bs)

        # H2: devices <-> RIS (L x K), geometric LoS per device
        cols = []
        for user_pos in self.get_current_users():
            d_u = user_pos - ris
            d_u = d_u / max(np.linalg.norm(d_u), 1e-9)
            los2 = self._ris_steering(d_u).reshape(-1, 1)
            cols.append(self._compute_rician_channel(ris, user_pos, (self.L, 1), los2)[:, 0])
        self.H_2 = np.column_stack(cols)

    # ============================
    # AoI Logic
    # ============================

    def _update_aoi(self, served_cluster_id, achieved_rates):
        """
        Update AoI for all devices.
        - Devices in served cluster with rate >= R_min: AoI resets to 1
        - All other devices: AoI increments by 1
        """
        # Increment all
        self.aoi += 1

        # Reset successfully served devices
        start_idx = served_cluster_id * self.users_per_ris
        end_idx = start_idx + self.users_per_ris
        for i, rate in enumerate(achieved_rates):
            if rate >= self.r_min:
                self.aoi[start_idx + i] = 1

    def get_average_aoi(self):
        """Return average AoI across all devices."""
        return np.mean(self.aoi)

    def get_peak_aoi(self):
        """Return maximum AoI across all devices."""
        return np.max(self.aoi)

    # ============================
    # High-Level Interface
    # ============================

    def get_high_level_state(self):
        """Construct the high-level state vector."""
        state = np.concatenate([
            self.uav_pos,                              # 3
            self.aoi / self.episode_length,             # K_total (normalized)
            [self.energy_remaining / self.energy_budget] # 1 (normalized)
        ])
        return state

    def high_level_reset(self):
        """Reset the entire episode."""
        self.timestep = 0
        self.energy_remaining = self.energy_budget
        self.aoi = np.ones(self.K_total)
        self.uav_pos = np.array([self.city_size / 2, self.city_size / 2,
                                 self.ris_height], dtype=np.float64)
        return self.get_high_level_state()

    def high_level_step(self, velocity, cluster_id):
        """
        Execute one high-level timestep:
          1. Move UAV
          2. Update channels
          3. Run low-level RIS optimization (external)
          4. Update AoI
          5. Compute reward

        Args:
            velocity: UAV velocity vector [v_x, v_y] or [v_x, v_y, v_z]
            cluster_id: Which cluster to serve (0 to num_ris_groups-1)

        Returns:
            high_state, reward, done, info
        """
        self.timestep += 1

        # 1. Move UAV
        speed = self._move_uav(velocity)

        # 2. Consume energy
        energy_used = self._compute_propulsion_energy(speed)
        self.energy_remaining -= energy_used

        # 3. Update channels for new position
        cluster_id = int(np.clip(cluster_id, 0, self.num_ris_groups - 1))
        self.set_current_ris_group(cluster_id)
        self._update_channels_for_position()

        # 4. Low-level state from the position-aware channels computed above.
        #    (Previously this called self.reset(), which redrew Rayleigh channels at
        #    the *static* cluster position and silently discarded the UAV position.)
        self.episode_t = 0
        self.info["true reward"] = 0
        low_state = self._build_initial_state()

        # 5. Compute achieved rates (placeholder — will be filled by low-level)
        # For now, return the low_state so the external loop can run SAC
        info = {
            'low_state': low_state,
            'cluster_id': cluster_id,
            'uav_pos': self.uav_pos.copy(),
            'energy_remaining': self.energy_remaining,
            'speed': speed,
            'energy_used': energy_used,
            'timestep': self.timestep,
        }

        # Check termination
        done = (self.timestep >= self.episode_length or
                self.energy_remaining <= 0)

        return self.get_high_level_state(), done, info

    def finalize_step(self, cluster_id, achieved_rates):
        """
        Called after low-level SAC produces rates for the served cluster.
        Updates AoI and returns the high-level reward.

        Args:
            cluster_id: Which cluster was served
            achieved_rates: Array of rates for the 4 users in the cluster

        Returns:
            reward (float): Negative average AoI (higher is better for RL)
        """
        self._update_aoi(cluster_id, achieved_rates)

        avg_aoi = self.get_average_aoi()
        reward = -avg_aoi  # Negative because we want to minimize AoI

        return reward

    # ============================
    # Utility
    # ============================

    def compute_rates_from_action(self, action):
        """
        Given a low-level action (beamforming + phase shifts),
        compute the uplink rates for the currently served cluster.
        """
        action = action.reshape(1, -1)

        # Extract beamforming and phase
        G_real = action[:, :self.M ** 2]
        G_imag = action[:, self.M ** 2:2 * self.M ** 2]
        Phi_real = action[:, -2 * self.L:-self.L]
        Phi_imag = action[:, -self.L:]

        G = G_real.reshape(self.M, self.K) + 1j * G_imag.reshape(self.M, self.K)
        angles = np.arctan2(Phi_real.flatten(), Phi_imag.flatten())
        betas = self._compute_PDA(angles)
        Phi = np.diag(np.exp(1j * angles) * betas)

        # Vectorized effective channel: (K, K) matrix
        # H_2.T is (K, L), Phi is (L, L), H_1 is (L, M), G is (M, K)
        effective_channel = self.H_2.T @ Phi @ self.H_1 @ G
        
        rates = []
        for k in range(self.K):
            # Signal power: diagonal elements (user k through combiner k)
            signal = np.abs(effective_channel[k, k]) ** 2 * self.iot_tx_power

            # Interference: all users j through combiner k, minus the signal
            interference = np.sum(np.abs(effective_channel[:, k]) ** 2) * self.iot_tx_power - signal

            # Noise must be scaled by the combiner norm
            g_k = G[:, k]
            combiner_norm_sq = np.linalg.norm(g_k) ** 2
            noise = self.awgn_var * combiner_norm_sq
            
            # Avoid division by zero
            if noise < 1e-16:
                noise = 1e-16

            sinr = signal / (interference + noise)
            rate = np.log2(1 + sinr)
            rates.append(rate)

        return np.array(rates)

    def get_env_info(self):
        """Return a summary dict of the environment configuration."""
        return {
            'K_total': self.K_total,
            'num_clusters': self.num_ris_groups,
            'users_per_cluster': self.users_per_ris,
            'L': self.L,
            'M': self.M,
            'v_max': self.v_max,
            'energy_budget': self.energy_budget,
            'episode_length': self.episode_length,
            'high_state_dim': self.high_state_dim,
            'high_action_dim': self.high_action_dim,
            'low_state_dim': self.state_dim,
            'low_action_dim': self.action_dim,
            'enable_3d': self.enable_3d,
            'r_min': self.r_min,
            'k_rice': self.k_rice,
        }
