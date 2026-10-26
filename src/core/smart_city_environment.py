#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Smart City RIS Environment
Includes realistic physical location constraints and service coverage limits
"""

import gym
from gym import spaces
import numpy as np
import matplotlib.pyplot as plt
from .environment import RIS_MISO_PDA

class SmartCityRISEnvironment(RIS_MISO_PDA):
    """
    RIS Environment for Smart City IoT Deployment
    
    Environment setup:
    - 100x100 meter urban area
    - Base Station located at center (50, 50), height 15m
    - RIS height 15m, service coverage limited to 60 degrees
    - 10 RIS positions, each serving 4 users
    """
    
    def __init__(self, num_antennas=4,
                 num_RIS_elements=16,
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
                 layout_seed=42):
        """
        Initialize Smart City RIS Environment
        
        Args:
            city_size (float): City size (meters)
            bs_height (float): Base Station height (meters)
            ris_height (float): RIS height (meters)
            max_service_angle (float): Maximum service angle (degrees)
        """
        
        # City environment parameters
        self.city_size = city_size
        self.bs_height = bs_height
        self.ris_height = ris_height
        self.max_service_angle = np.radians(max_service_angle)  # Convert to radians
        
        # Base Station position (city center)
        self.bs_position = np.array([city_size/2, city_size/2, bs_height])
        
        # Initialize RIS and user positions
        self.num_ris_groups = 10  # 10 RIS groups
        self.users_per_ris = 4    # Each RIS serves 4 users
        self.total_users = self.num_ris_groups * self.users_per_ris  # 40 total users
        
        # Layout RNG is local: it no longer re-seeds the GLOBAL numpy RNG (previous
        # code called np.random.seed(42) here, which silently gave every run the
        # same city layout and overrode the caller's seed).
        self.layout_seed = layout_seed
        self._layout_rng = np.random.RandomState(layout_seed)

        # Generate RIS positions and corresponding user positions
        self._generate_ris_positions()
        self._generate_user_positions()
        
        # Currently served RIS group ID
        self.current_ris_group = 0
        
        # Call parent class initialization
        super().__init__(
            num_antennas=num_antennas,
            num_RIS_elements=num_RIS_elements,
            num_users=num_users,  # 每次只服務4個用戶
            mismatch=mismatch,
            channel_est_error=channel_est_error,
            cascaded_channels=cascaded_channels,
            beta_min=beta_min,
            theta_bar=theta_bar,
            kappa_bar=kappa_bar,
            AWGN_var=AWGN_var,
            channel_noise_var=channel_noise_var,
            seed=seed
        )
        
    def _generate_ris_positions(self):
        """Generate RIS positions across the city."""
        rng = self._layout_rng

        # Randomly distribute RIS positions, avoiding edges
        margin = 10  # Edge margin

        self.ris_positions = []
        for i in range(self.num_ris_groups):
            x = rng.uniform(margin, self.city_size - margin)
            y = rng.uniform(margin, self.city_size - margin)
            z = self.ris_height
            self.ris_positions.append(np.array([x, y, z]))
        
        self.ris_positions = np.array(self.ris_positions)
        
    def _generate_user_positions(self):
        """Generate user positions within each RIS service area."""
        self.user_positions = []
        self.user_ris_mapping = {}  # User-to-RIS mapping
        
        for ris_idx, ris_pos in enumerate(self.ris_positions):
            ris_users = []
            
            # Calculate RIS service radius
            max_service_radius = self.ris_height / np.tan(self.max_service_angle)
            
            for user_idx in range(self.users_per_ris):
                # Randomly generate user position within service area
                attempts = 0
                while attempts < 100:  # Max 100 attempts
                    # Random position within circular service area
                    angle = self._layout_rng.uniform(0, 2*np.pi)
                    radius = self._layout_rng.uniform(1, max_service_radius * 0.9)  # Leave margin
                    
                    user_x = ris_pos[0] + radius * np.cos(angle)
                    user_y = ris_pos[1] + radius * np.sin(angle)
                    user_z = 1.5  # User height 1.5 meters
                    
                    # Check if within city boundaries
                    if (0 <= user_x <= self.city_size and 
                        0 <= user_y <= self.city_size):
                        
                        user_pos = np.array([user_x, user_y, user_z])
                        
                        # Verify service angle
                        if self._check_service_angle(ris_pos, user_pos):
                            ris_users.append(user_pos)
                            global_user_idx = ris_idx * self.users_per_ris + user_idx
                            self.user_ris_mapping[global_user_idx] = ris_idx
                            break
                    
                    attempts += 1
                
                if attempts >= 100:
                    # If no suitable position found, use default position
                    print(f"WARNING: Could not find valid position for RIS {ris_idx}, User {user_idx}")
                    user_pos = np.array([ris_pos[0] + 5, ris_pos[1] + 5, 1.5])
                    ris_users.append(user_pos)
                    global_user_idx = ris_idx * self.users_per_ris + user_idx
                    self.user_ris_mapping[global_user_idx] = ris_idx
            
            self.user_positions.extend(ris_users)
        
        self.user_positions = np.array(self.user_positions)
        
    def _check_service_angle(self, ris_pos, user_pos):
        """Check if the angle from RIS to user is within service range."""
        # Horizontal distance
        horizontal_dist = np.sqrt((ris_pos[0] - user_pos[0])**2 + 
                                 (ris_pos[1] - user_pos[1])**2)
        
        # Vertical distance
        vertical_dist = ris_pos[2] - user_pos[2]
        
        # Compute angle from vertical boresight
        if vertical_dist <= 0:
            return False
        angle_from_vertical = np.arctan(horizontal_dist / vertical_dist)
        
        return angle_from_vertical <= self.max_service_angle
        
    def _compute_channel_with_distance(self, pos1, pos2):
        """Compute channel matrix based on distance between two positions."""
        distance = np.linalg.norm(pos1 - pos2)
        
        # Path loss model (free-space path loss)
        frequency = 2.4e9  # 2.4 GHz
        c = 3e8  # Speed of light
        path_loss_db = 20 * np.log10(distance) + 20 * np.log10(frequency) + 20 * np.log10(4*np.pi/c)
        path_loss_linear = 10**(-path_loss_db/20)
        
        # Generate Rayleigh fading channel
        channel_real = np.random.normal(0, np.sqrt(0.5), (self.L, self.M)) * path_loss_linear
        channel_imag = np.random.normal(0, np.sqrt(0.5), (self.L, self.M)) * path_loss_linear
        
        return channel_real + 1j * channel_imag
        
    def set_current_ris_group(self, ris_group_id):
        """Set the currently active RIS group."""
        if 0 <= ris_group_id < self.num_ris_groups:
            self.current_ris_group = ris_group_id
        else:
            raise ValueError(f"RIS group ID must be between 0 and {self.num_ris_groups-1}")
            
    def get_current_users(self):
        """Get the users served by the currently active RIS group."""
        start_idx = self.current_ris_group * self.users_per_ris
        end_idx = start_idx + self.users_per_ris
        return self.user_positions[start_idx:end_idx]
        
    def reset(self):
        """Reset the environment."""
        self.episode_t = 0
        self.info["true reward"] = 0
        
        # Get current RIS position
        current_ris_pos = self.ris_positions[self.current_ris_group]
        current_users = self.get_current_users()
        
        # Compute channel matrices based on physical positions
        # H1: Base Station to RIS channel
        self.H_1 = self._compute_channel_with_distance(self.bs_position, current_ris_pos)
        
        # H2: RIS to Users channel
        h2_channels = []
        for user_pos in current_users:
            h2_user = self._compute_channel_with_distance(current_ris_pos, user_pos)
            h2_channels.append(h2_user[:, 0])  # Each user has a single antenna
        
        self.H_2 = np.column_stack(h2_channels)
        
        return self._build_initial_state()

    def _build_initial_state(self):
        """Build the low-level state from the CURRENT self.H_1 / self.H_2.

        Split out of reset() so that subclasses (e.g. the UAV environment) can
        supply their own, position-dependent channels without reset() redrawing
        and overwriting them.
        """
        # Initialize other parameters
        self.G = np.eye(self.M, dtype=complex)
        self.Phi = np.eye(self.L, dtype=complex)
        self.Phi_mismatch = np.eye(self.L, dtype=complex)
        
        # Construct initial state
        init_action_G = np.hstack((np.real(self.G.reshape(1, -1)), 
                                  np.imag(self.G.reshape(1, -1))))
        init_action_Phi = np.hstack((np.real(np.diag(self.Phi)).reshape(1, -1), 
                                   np.imag(np.diag(self.Phi)).reshape(1, -1)))
        init_action = np.hstack((init_action_G, init_action_Phi))
        
        # Compute power and channel information
        power_t = np.real(np.diag(self.G.conjugate().T @ self.G)).reshape(1, -1) ** 2
        
        if self.cascaded_channels:
            D = self._compute_D()
            H_2_tilde = self._compute_H_2_tilde(D)
            power_r = np.linalg.norm(H_2_tilde, axis=0).reshape(1, -1) ** 2
            D_real, D_imag = np.real(D).reshape(1, -1), np.imag(D).reshape(1, -1)
            self.state = np.hstack((init_action, power_t, power_r, D_real, D_imag))
        else:
            H_2_tilde = self.H_2.T @ self.Phi @ self.H_1 @ self.G
            power_r = np.linalg.norm(H_2_tilde, axis=0).reshape(1, -1) ** 2
            H_1_real, H_1_imag = np.real(self.H_1).reshape(1, -1), np.imag(self.H_1).reshape(1, -1)
            H_2_real, H_2_imag = np.real(self.H_2).reshape(1, -1), np.imag(self.H_2).reshape(1, -1)
            self.state = np.hstack((init_action, power_t, power_r, H_1_real, H_1_imag, H_2_real, H_2_imag))
        
        return self.state
        
    def visualize_environment(self, save_path=None):
        """Visualize the environment layout."""
        fig, ax = plt.subplots(1, 1, figsize=(12, 10))
        
        # Font settings
        plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        
        # Draw city boundary
        ax.add_patch(plt.Rectangle((0, 0), self.city_size, self.city_size, 
                                  fill=False, edgecolor='black', linewidth=2))
        
        # Draw Base Station
        ax.scatter(self.bs_position[0], self.bs_position[1], 
                  c='red', marker='^', s=200, label=f'BS (height: {self.bs_height}m)')
        
        # Draw RIS positions and service areas
        colors = plt.cm.tab10(np.linspace(0, 1, self.num_ris_groups))
        
        for i, (ris_pos, color) in enumerate(zip(self.ris_positions, colors)):
            # Draw RIS position
            ax.scatter(ris_pos[0], ris_pos[1], c=[color], marker='s', s=150, 
                      label=f'RIS {i+1}' if i < 5 else '')  # 只顯示前5個標籤避免擁擠
            
            # Draw service coverage circle
            max_radius = self.ris_height / np.tan(self.max_service_angle)
            circle = plt.Circle((ris_pos[0], ris_pos[1]), max_radius, 
                              fill=False, color=color, alpha=0.3, linestyle='--')
            ax.add_patch(circle)
            
            # Draw corresponding users
            start_idx = i * self.users_per_ris
            end_idx = start_idx + self.users_per_ris
            user_group = self.user_positions[start_idx:end_idx]
            
            ax.scatter(user_group[:, 0], user_group[:, 1], 
                      c=[color], marker='o', s=50, alpha=0.7)
            
            # Draw RIS-to-user connections
            for user_pos in user_group:
                ax.plot([ris_pos[0], user_pos[0]], [ris_pos[1], user_pos[1]], 
                       color=color, alpha=0.3, linewidth=1)
        
        # Set chart properties
        ax.set_xlim(-5, self.city_size + 5)
        ax.set_ylim(-5, self.city_size + 5)
        ax.set_xlabel('X Coordinate (meters)', fontsize=12)
        ax.set_ylabel('Y Coordinate (meters)', fontsize=12)
        ax.set_title('Smart City RIS Layout\n' + 
                    f'City size: {self.city_size}x{self.city_size}m, ' +
                    f'RIS count: {self.num_ris_groups}, ' +
                    f'Total users: {self.total_users}', fontsize=14)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Environment layout saved to: {save_path}")
        
        plt.show()
        
        return fig, ax
    
    def get_environment_info(self):
        """Get environment configuration info."""
        info = {
            'city_size': self.city_size,
            'bs_position': self.bs_position.tolist(),
            'bs_height': self.bs_height,
            'ris_height': self.ris_height,
            'max_service_angle_deg': np.degrees(self.max_service_angle),
            'num_ris_groups': self.num_ris_groups,
            'users_per_ris': self.users_per_ris,
            'total_users': self.total_users,
            'ris_positions': self.ris_positions.tolist(),
            'user_positions': self.user_positions.tolist(),
            'user_ris_mapping': self.user_ris_mapping
        }
        return info 
