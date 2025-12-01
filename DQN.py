import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from shapely.geometry import Polygon
from shapely.ops import cascaded_union
import time
start_time = time.time()

# ----- Channel geometry helpers (for UPA + channel model) -----

def direction_cosines_from_az_el(az, el):
    thx = np.cos(el) * np.cos(az)
    thy = np.cos(el) * np.sin(az)
    return thx, thy


def az_el_from_vector(v):
    vx, vy, vz = v
    r = np.linalg.norm(v) + 1e-12
    az = np.arctan2(vy, vx)
    el = np.arcsin(vz / r)
    return az, el


def upa_steering_vector(Nx, Ny, thx, thy):
    ax = np.exp(-1j * np.pi * thx * np.arange(Nx)) / np.sqrt(Nx)
    ay = np.exp(-1j * np.pi * thy * np.arange(Ny)) / np.sqrt(Ny)
    a = np.kron(ax, ay)
    return a



class BeamEnv:
    def __init__(self, N_sats, beam_radius, beam_sep, UT_position,
                 fc=12e9,
                 Gt_dBi=35.0,
                 Gr_dBi=10.0,
                 Nx=8, Ny=8,
                 noise_figure_dB=5.0,
                 bandwidth=20e6,
                 temperature=290.0,
                 altitude_km=550.0,
                 n_interferers=3,
                 snr_db_clip=(-10.0, 30.0)):
        self.N_sats = N_sats
        self.beam_radius = beam_radius
        self.beam_sep = beam_sep
        self.UT_position = np.array(UT_position, dtype=float)
        # Channel parameters
        self.c = 299792458.0
        self.fc = fc
        self.lam = self.c / self.fc
        self.Gt = 10**(Gt_dBi/10.0)
        self.Gr = 10**(Gr_dBi/10.0)
        self.Nx = Nx
        self.Ny = Ny
        self.noise_figure_dB = noise_figure_dB
        self.bandwidth = bandwidth
        self.temperature = temperature
        self.kB = 1.380649e-23
        self.altitude_km = altitude_km
        self.altitude_m = altitude_km * 1000.0
        self.n_interferers = n_interferers
        self.snr_db_clip = snr_db_clip
        # caches for logging
        self._last_sinrs_lin = np.ones(self.N_sats)
        self._last_pseudoranges_m = np.zeros(self.N_sats)
        self.reset()
        
    def reset(self):
        self.detected_beams = self._generate_beam_positions()
        self.undetected_beams = self._generate_beam_positions()
        self.state = self._compute_state()
        return self.state
    
    def _generate_beam_positions(self):
        center_offset = np.random.uniform(-10, 10, (self.N_sats, 2))
        start_points = self.UT_position + center_offset + np.random.uniform(-5, 5, (self.N_sats, 2))
        end_points = start_points + np.random.uniform(-3, 3, (self.N_sats, 2))
        return np.stack([start_points, end_points], axis=1)
    
    # ---------- Channel-related helpers ----------
    def _thermal_noise_watts(self):
        N0 = self.kB * self.temperature * self.bandwidth
        NF = 10**(self.noise_figure_dB/10.0)
        return N0 * NF

    def _free_space_path_loss_linear(self, d_m):
        return (4.0*np.pi*self.fc*d_m/self.c)**2

    def _upa_channel_vector(self, sat_xyz_km, ut_xyz_km):
        v_km = ut_xyz_km - sat_xyz_km
        d_km = np.linalg.norm(v_km) + 1e-12
        d_m = d_km * 1000.0
        az, el = az_el_from_vector(v_km)
        thx, thy = direction_cosines_from_az_el(az, el)
        a = upa_steering_vector(self.Nx, self.Ny, thx, thy)
        L = self._free_space_path_loss_linear(d_m)
        beta = np.sqrt(self.Gt * self.Gr / L)
        phi = np.random.uniform(0.0, 2.0*np.pi)
        g = beta * np.exp(1j*phi) * a
        return g, d_m, az, el

    def _random_interferer(self, sat_xyz_km, ut_xyz_km):
        az = np.random.uniform(-np.pi, np.pi)
        el = np.random.uniform(0.0, np.deg2rad(60.0))
        thx, thy = direction_cosines_from_az_el(az, el)
        a = upa_steering_vector(self.Nx, self.Ny, thx, thy)
        v_km = ut_xyz_km - sat_xyz_km
        d_km = np.linalg.norm(v_km) + 1e-12
        d_m = d_km * 1000.0
        L = self._free_space_path_loss_linear(d_m)
        beta = np.sqrt(self.Gt * self.Gr / L)
        return beta * a

    def _beam_channel_features(self, beam_center_xy_km):
        ut_xyz_km = np.array([self.UT_position[0], self.UT_position[1], 0.0])
        sat_xyz_km = np.array([beam_center_xy_km[0], beam_center_xy_km[1], self.altitude_km])
        g, d_m, az, el = self._upa_channel_vector(sat_xyz_km, ut_xyz_km)
        thx, thy = direction_cosines_from_az_el(az, el)
        f_des = upa_steering_vector(self.Nx, self.Ny, thx, thy)
        f_des = f_des / (np.linalg.norm(f_des) + 1e-12)
        interf_powers = 0.0
        for _ in range(self.n_interferers):
            f_k = self._random_interferer(sat_xyz_km, ut_xyz_km)
            interf_powers += np.abs(np.vdot(g, f_k))**2
        signal_power = np.abs(np.vdot(g, f_des))**2
        noise_power = self._thermal_noise_watts()
        sinr_lin = signal_power / (interf_powers + noise_power + 1e-18)
        sinr_db = 10.0 * np.log10(max(sinr_lin, 1e-18))
        d_km = d_m / 1000.0
        norm_distance = np.clip(d_km / 1000.0, 0.0, 1.0)
        norm_az = (az + np.pi) / (2.0*np.pi)
        norm_el = (el + (np.pi/2.0)) / np.pi
        min_db, max_db = self.snr_db_clip
        sinr_db_clipped = np.clip(sinr_db, min_db, max_db)
        norm_sinr = (sinr_db_clipped - min_db) / (max_db - min_db + 1e-12)
        sigma_m = 30.0 / np.sqrt(1.0 + sinr_lin)
        pseudo_range_m = d_m + np.random.normal(0.0, sigma_m)
        norm_range = np.clip(pseudo_range_m / (self.altitude_m + 1e-12), 0.0, 1.0)
        phase = (pseudo_range_m % self.lam) / self.lam
        feat = np.array([norm_distance, norm_az, norm_el, norm_sinr, norm_range, phase], dtype=float)
        return feat, sinr_lin, pseudo_range_m

    def _compute_state(self):
        state = []
        sinrs = []
        pranges = []
        for i in range(self.N_sats):
            beam = self.detected_beams[i]
            beam_center = self._compute_beam_center(beam)
            feat, s_lin, pr_m = self._beam_channel_features(beam_center)
            state.extend(feat.tolist())
            sinrs.append(s_lin)
            pranges.append(pr_m)
        self._last_sinrs_lin = np.array(sinrs)
        self._last_pseudoranges_m = np.array(pranges)
        return np.array(state)
    
    def _compute_beam_center(self, beam):
        return (beam[0] + beam[1]) / 2
    
    def _create_circle_polygon(self, center, radius, n_points=32):
        angles = np.linspace(0, 2*np.pi, n_points)
        circle_points = np.array([(center[0] + radius*np.cos(theta),
                                 center[1] + radius*np.sin(theta)) for theta in angles])
        return Polygon(circle_points)
    
    def _compute_intersection_centroid(self, detected_beams, undetected_beams=None):
        beam_polygons = [self._create_circle_polygon(self._compute_beam_center(beam), 
                                                   self.beam_radius/5) for beam in detected_beams]
        intersection = cascaded_union(beam_polygons)
        
        if undetected_beams is not None:
            undetected_polygons = [self._create_circle_polygon(self._compute_beam_center(beam), 
                                                             self.beam_radius/5) for beam in undetected_beams]
            for polygon in undetected_polygons:
                if intersection.intersects(polygon):
                    intersection = intersection.difference(polygon)
        
        if not intersection.is_empty:
            centroid = intersection.centroid
            return np.array([centroid.x, centroid.y])
        return self.UT_position

    def _compute_error(self, weights):
        weighted_position = np.zeros(2)
        total_weight = 1e-10
        for i in range(self.N_sats):
            beam_center = self._compute_beam_center(self.detected_beams[i])
            weight = weights[i]
            weighted_position += beam_center * weight
            total_weight += weight
        predicted_position = weighted_position / total_weight
        position_error = np.linalg.norm(predicted_position - self.UT_position)
        weight_regularization = np.abs(np.sum(weights) - 1.0)
        return position_error + 0.1 * weight_regularization
    
    def step(self, action):
        weights = np.clip(action, 0, 1)
        error = self._compute_error(weights)
        reward = -np.exp(error / 10.0) + 1
        self.state = self._compute_state()
        done = error < 1.0 or np.random.rand() > 0.95
        return self.state, reward, done, {'error': error}
    
    def visualize_beams(self, weights=None, episode=None, final=False):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # --- ALG-B ---
        detected_polygons = [self._create_circle_polygon(self._compute_beam_center(beam), 
                                                    self.beam_radius/5) for beam in self.detected_beams]
        intersection = cascaded_union(detected_polygons)
        
        if undetected_polygons := [self._create_circle_polygon(self._compute_beam_center(beam), 
                                                            self.beam_radius/5) for beam in self.undetected_beams]:
            for polygon in undetected_polygons:
                if intersection.intersects(polygon):
                    intersection = intersection.difference(polygon)
        
        if not intersection.is_empty:
            if intersection.geom_type == 'Polygon':
                x, y = intersection.exterior.xy
                ax1.fill(x, y, alpha=0.3, fc='gray', ec='none')
            elif intersection.geom_type == 'MultiPolygon':
                for geom in intersection.geoms:
                    x, y = geom.exterior.xy
                    ax1.fill(x, y, alpha=0.3, fc='gray', ec='none')
        
        predicted_position_algb = self._compute_intersection_centroid(self.detected_beams, self.undetected_beams)
        ax1.plot(predicted_position_algb[0], predicted_position_algb[1], '*', 
                color='yellow', markersize=20, label='Predicted Position')
        distance_algb = np.linalg.norm(predicted_position_algb - self.UT_position)
        ax1.text(0.02, 1.12, f'Distance Error: {distance_algb*1000:.2f} m', 
                transform=ax1.transAxes, verticalalignment='top', fontsize=18)

        # --- DQN ---
        if weights is not None and final:
            weighted_position = np.zeros(2)
            total_weight = 1e-10
            for i in range(self.N_sats):
                beam_center = self._compute_beam_center(self.detected_beams[i])
                weight = weights[i]
                weighted_position += beam_center * weight
                total_weight += weight
            predicted_position_dqn = weighted_position / total_weight
            ax2.plot(predicted_position_dqn[0], predicted_position_dqn[1], '*', 
                    color='yellow', markersize=20, label='Predicted Position')
            distance_dqn = np.linalg.norm(predicted_position_dqn - self.UT_position)
            ax2.text(0.02,1.12, f'Distance Error: {distance_dqn*1000:.2f} m', 
                    transform=ax2.transAxes, verticalalignment='top', fontsize=18)
        
        # --- Draw beams ---
        for i in range(self.N_sats):
            beam = self.detected_beams[i]
            beam_center = self._compute_beam_center(beam)
            weight = weights[i] if weights is not None else 1.0
            linewidth = 1.5 + 2.5 * weight if weights is not None else 1.5
            for ax in [ax1, ax2]:
                circle = Circle(beam_center, self.beam_radius/5,
                            color='blue', alpha=0.6, fill=False, linewidth=linewidth)
                ax.add_patch(circle)
            
            beam = self.undetected_beams[i]
            beam_center = self._compute_beam_center(beam)
            inverse_weight = 1 - weight if weights is not None else 0.0
            linewidth = 1.5 + 2.5 * inverse_weight if weights is not None else 1.5
            for ax in [ax1, ax2]:
                circle = Circle(beam_center, self.beam_radius/5,
                            color='red', alpha=0.6, fill=False, linewidth=linewidth)
                ax.add_patch(circle)
        
        # --- Common settings ---
        for ax in [ax1, ax2]:
            ax.plot(self.UT_position[0], self.UT_position[1], 'ko', markerfacecolor='none', markersize=10)
            ax.add_patch(Circle(self.UT_position, self.beam_radius/10, color='gray', alpha=0.3))
            ax.set_xlim(-20, 20)
            ax.set_ylim(-20, 20)
            ax.set_xlabel('X (m)', fontsize=18)
            ax.set_ylabel('Y (m)', fontsize=18)
            ax.grid(True, linestyle='--', alpha=0.7)

        ax1.text(0.5, -0.18, '(a) ALG-B', transform=ax1.transAxes,
                ha='center', va='center', fontsize=18, fontweight='bold')
        ax2.text(0.5, -0.18, '(b) DQN', transform=ax2.transAxes,
                ha='center', va='center', fontsize=18, fontweight='bold')

        legend_elements = [
            plt.Line2D([0], [0], marker='o', color='blue', linestyle='none', label='Detected Beams', markersize=10, markerfacecolor='none'),
            plt.Line2D([0], [0], marker='o', color='red', linestyle='none', label='Undetected Beams', markersize=10, markerfacecolor='none'),
            plt.Line2D([0], [0], marker='o', color='k', linestyle='none', markerfacecolor='none', markersize=10, label='Actual UT Position'),
            plt.Line2D([0], [0], marker='*', color='yellow', linestyle='none', label='Predicted Position', markersize=15)
        ]
        ax2.legend(handles=legend_elements, bbox_to_anchor=(0.99, 1), loc='upper left', fontsize=12)

        plt.tight_layout(pad=3.0)
        plt.show()

class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_dim)
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return torch.sigmoid(self.fc3(x))

def train_dqn(visualize_interval=1000):
    state_dim, action_dim = 60, 10
    gamma, epsilon, epsilon_decay, min_epsilon = 0.99, 1.0, 0.995, 0.01
    replay_buffer = deque(maxlen=10000)
    batch_size, num_episodes = 64, 1000
    dqn, target_dqn = DQN(state_dim, action_dim), DQN(state_dim, action_dim)
    target_dqn.load_state_dict(dqn.state_dict())
    optimizer = optim.Adam(dqn.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()
    env = BeamEnv(N_sats=10, beam_radius=50, beam_sep=75, UT_position=[0, 0])
    rewards_per_episode, errors_per_episode, final_weights = [], [], None

    for episode in range(num_episodes):
        state, total_reward, episode_errors = env.reset(), 0, []
        if episode % visualize_interval == 0 and final_weights is not None:
            env.visualize_beams(weights=final_weights, episode=episode, final=True)

        for _ in range(1000):
            if random.random() < epsilon:
                action = np.random.rand(action_dim)
            else:
                with torch.no_grad():
                    action = dqn(torch.FloatTensor(state)).numpy()
            next_state, reward, done, info = env.step(action)
            total_reward += reward
            episode_errors.append(info['error'])
            replay_buffer.append((state, action, reward, next_state, done))
            state = next_state

            if len(replay_buffer) > batch_size:
                batch = random.sample(replay_buffer, batch_size)
                states, actions, rewards, next_states, dones = zip(*batch)
                states, actions, rewards, next_states, dones = map(torch.FloatTensor,
                    [states, actions, rewards, next_states, dones])
                with torch.no_grad():
                    target_q = rewards + gamma * (1 - dones) * torch.max(target_dqn(next_states), dim=1)[0]
                current_q = torch.sum(dqn(states) * actions, dim=1)
                loss = loss_fn(current_q, target_q)
                optimizer.zero_grad(); loss.backward(); optimizer.step()
            if done: break

        epsilon = max(min_epsilon, epsilon * epsilon_decay)
        if episode % 10 == 0:
            target_dqn.load_state_dict(dqn.state_dict())
        rewards_per_episode.append(total_reward)
        errors_per_episode.append(np.mean(episode_errors))
        if episode == num_episodes - 1:
            final_weights = action
            env.visualize_beams(weights=final_weights, episode=episode, final=True)
        if episode % 50 == 0:
            print(f"訓練回合 {episode}, 總獎勵: {total_reward:.2f}, 平均誤差: {np.mean(episode_errors):.2f}")
        torch.save(dqn.state_dict(), "trained_dqn_weights.pth")
    return rewards_per_episode, errors_per_episode, final_weights, env

print("Starting training...")
rewards, errors, final_weights, env = train_dqn(visualize_interval=10)
end_time = time.time()
print(f"程式總執行時間: {end_time - start_time:.2f} 秒")

plt.figure(figsize=(12, 6))
window_size = 50
plt.subplot(1, 2, 1)
plt.plot(rewards, label='Raw Reward', color='steelblue')
if len(rewards) >= window_size:
    ma = np.convolve(rewards, np.ones(window_size)/window_size, mode='valid')
    plt.plot(range(window_size - 1, len(rewards)), ma, color='orange', label=f'Smoothed Reward')
plt.xlabel("Episodes", fontsize=18)
plt.ylabel("Reward", fontsize=18)
plt.legend(fontsize=14)
plt.grid(True, linestyle='--', alpha=0.7)

plt.subplot(1, 2, 2)
errors_m = np.array(errors) * 1000
plt.plot(errors_m, label='Raw Error', color='steelblue')
if len(errors_m) >= window_size:
    ma = np.convolve(errors_m, np.ones(window_size)/window_size, mode='valid')
    plt.plot(range(window_size - 1, len(errors_m)), ma, color='orange', label=f'Smoothed Error')
plt.xlabel("Episodes", fontsize=18)
plt.ylabel("Error (m)", fontsize=18)
plt.legend(fontsize=14)
plt.grid(True, linestyle='--', alpha=0.7)

plt.tight_layout(pad=3.0)
plt.show()

plt.figure(figsize=(8, 4))
normalized_weights = final_weights / np.sum(final_weights)
plt.bar(range(len(normalized_weights)), normalized_weights)
plt.xlabel("Beam Index",fontsize=18); plt.ylabel("Weight",fontsize=18)
plt.grid(); plt.show()
print("Training completed!")