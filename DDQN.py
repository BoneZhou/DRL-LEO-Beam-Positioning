import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from shapely.geometry import Polygon
from shapely.ops import cascaded_union

# ==============================
# 通道工具（UPA + 幾何）
# ==============================

def direction_cosines_from_az_el(az, el):
    # az: [-pi, pi], el: [-pi/2, pi/2]
    # 回傳用在 UPA steering vector 的 direction cosines (theta_x, theta_y)
    thx = np.cos(el) * np.cos(az)
    thy = np.cos(el) * np.sin(az)
    return thx, thy

def az_el_from_vector(v):
    # v: 3D 向量 (from sat to UT) in km
    vx, vy, vz = v
    r = np.linalg.norm(v) + 1e-12
    az = np.arctan2(vy, vx)
    el = np.arcsin(vz / r)
    return az, el

def upa_steering_vector(Nx, Ny, thx, thy):
    # UPA steering vector，元素間距預設 0.5λ（隱含在 pi 因子）
    # 回傳長度 Nx*Ny 的複數向量
    ax = np.exp(-1j * np.pi * thx * np.arange(Nx)) / np.sqrt(Nx)
    ay = np.exp(-1j * np.pi * thy * np.arange(Ny)) / np.sqrt(Ny)
    a = np.kron(ax, ay)
    return a

# ==============================
# 環境：幾何 ALG-B + 通道模型 + 都普勒
# ==============================

class BeamEnv:
    def __init__(
        self,
        N_sats=10,
        beam_radius=50.0,           # 圖上的波束半徑 (km)
        UT_position=(0.0, 0.0),     # km
        # 通道參數
        fc=12e9,                    # carrier: 12 GHz
        Gt_dBi=35.0,                # Tx gain (dBi)
        Gr_dBi=10.0,                # Rx gain (dBi)
        Nx=8, Ny=8,                 # UPA 大小
        noise_figure_dB=5.0,
        bandwidth=20e6,             # Hz
        temperature=290.0,          # K
        altitude_km=550.0,          # 衛星高度 550 km
        n_interferers=3,
        snr_db_clip=(-10.0, 30.0),
    ):
        self.N_sats = N_sats
        self.beam_radius = float(beam_radius)          # km
        self.UT_position = np.array(UT_position, dtype=np.float32)  # km

        # 通道參數
        self.c = 299_792_458.0
        self.fc = fc
        self.lam = self.c / self.fc                    # m
        self.Gt = 10 ** (Gt_dBi / 10.0)
        self.Gr = 10 ** (Gr_dBi / 10.0)
        self.Nx, self.Ny = Nx, Ny
        self.noise_figure_dB = noise_figure_dB
        self.bandwidth = bandwidth
        self.temperature = temperature
        self.kB = 1.380649e-23
        self.altitude_km = float(altitude_km)
        self.altitude_m = self.altitude_km * 1000.0
        self.n_interferers = n_interferers
        self.snr_db_clip = snr_db_clip
        
        # [New] LEO 衛星速度 (m/s)
        self.SAT_VELOCITY_MAG = 7560.0

        # 緩存 SINR / 偽距 (m)
        self._last_sinrs_lin = np.ones(self.N_sats, dtype=np.float32)
        self._last_pseudoranges_m = np.zeros(self.N_sats, dtype=np.float32)

        self.reset()

    # ---------- 通道相關 ----------

    def _thermal_noise_watts(self):
        # 噪聲功率：kTB * NF
        N0 = self.kB * self.temperature * self.bandwidth
        NF = 10 ** (self.noise_figure_dB / 10.0)
        return N0 * NF

    def _free_space_path_loss_linear(self, d_m):
        # 自由空間路損 (線性)：L = (4*pi*f_c*d/c)^2
        return (4.0 * np.pi * self.fc * d_m / self.c) ** 2

    def _upa_channel_vector(self, sat_xyz_km, ut_xyz_km):
        # 輸入 sat / UT 位置 (km)，回傳：
        # g: 通道向量 (complex, Nx*Ny)
        # d_m: sat-UT 斜距 (m)
        # az, el: 方位 / 仰角 (rad)
        v_km = ut_xyz_km - sat_xyz_km
        d_km = np.linalg.norm(v_km) + 1e-12
        d_m = d_km * 1000.0

        az, el = az_el_from_vector(v_km)
        thx, thy = direction_cosines_from_az_el(az, el)
        a = upa_steering_vector(self.Nx, self.Ny, thx, thy)

        L = self._free_space_path_loss_linear(d_m)
        beta = np.sqrt(self.Gt * self.Gr / L)
        phi = np.random.uniform(0.0, 2.0 * np.pi)
        g = beta * np.exp(1j * phi) * a
        return g, d_m, az, el

    def _random_interferer(self, sat_xyz_km, ut_xyz_km):
        # 隨機干擾波束
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

    # ---------- 幾何部分 ----------

    def reset(self):
        # 如果你要每個 episode 隨機 UT 位置，可以改這裡
        # self.UT_position = np.random.uniform(-20.0, 20.0, size=2).astype(np.float32)
        self.detected_beams = self._gen_beams()
        self.undetected_beams = self._gen_beams()
        
        # [New] 重置時產生隨機衛星速度向量 (3D, m/s)
        self.sat_velocities = self._gen_velocities()
        
        self.state = self._build_state()
        return self.state

    def _gen_beams(self):
        off = np.random.uniform(-10, 10, (self.N_sats, 2)).astype(np.float32)
        s = self.UT_position + off + np.random.uniform(-5, 5, (self.N_sats, 2)).astype(np.float32)
        e = s + np.random.uniform(-3, 3, (self.N_sats, 2)).astype(np.float32)
        return np.stack([s, e], axis=1)
    
    # [New] 產生隨機衛星速度 (假設衛星主要在水平面運動)
    def _gen_velocities(self):
        angles = np.random.uniform(0, 2*np.pi, self.N_sats)
        # Vx, Vy, Vz (m/s)
        vx = self.SAT_VELOCITY_MAG * np.cos(angles)
        vy = self.SAT_VELOCITY_MAG * np.sin(angles)
        vz = np.zeros(self.N_sats) # 簡化假設垂直速度為 0
        return np.stack([vx, vy, vz], axis=1).astype(np.float32)

    def _center(self, b):
        return (b[0] + b[1]) / 2.0

    def _circle_poly(self, c, r, n=64):
        th = np.linspace(0, 2 * np.pi, n)
        pts = np.stack([c[0] + r * np.cos(th), c[1] + r * np.sin(th)], axis=1)
        return Polygon(pts)

    def _intersection_centroid(self):
        det = [self._circle_poly(self._center(b), self.beam_radius / 5.0) for b in self.detected_beams]
        inter = cascaded_union(det)
        und = [self._circle_poly(self._center(b), self.beam_radius / 5.0) for b in self.undetected_beams]
        for p in und:
            if inter.is_empty:
                break
            if inter.intersects(p):
                inter = inter.difference(p)
        if not inter.is_empty:
            c = inter.centroid
            return np.array([c.x, c.y], dtype=np.float32)
        return self.UT_position.copy()

    def _weighted_cog(self, w):
        centers = np.array([self._center(b) for b in self.detected_beams], dtype=np.float32)
        w = np.clip(w, 1e-8, 1.0)
        w = w / (np.sum(w) + 1e-8)
        return np.sum(centers * w[:, None], axis=0)

    def _err_km(self, p):
        return float(np.linalg.norm(p - self.UT_position))

    # ---------- 通道特徵 (每 beam 7 維: 增加 Doppler) ----------

    def _beam_channel_features(self, beam_center_xy_km, sat_velocity_ms):
        # 給單一 beam 的通道特徵：
        # [norm_dist, norm_az, norm_el, norm_sinr, norm_range, norm_doppler, phase]
        
        ut_xyz_km = np.array([self.UT_position[0], self.UT_position[1], 0.0], dtype=np.float32)
        sat_xyz_km = np.array([beam_center_xy_km[0], beam_center_xy_km[1], self.altitude_km], dtype=np.float32)

        # LoS 通道
        g, d_m, az, el = self._upa_channel_vector(sat_xyz_km, ut_xyz_km)

        # Beamformer 指向該 AoD
        thx, thy = direction_cosines_from_az_el(az, el)
        f_des = upa_steering_vector(self.Nx, self.Ny, thx, thy)
        f_des = f_des / (np.linalg.norm(f_des) + 1e-12)

        # 干擾
        interf_powers = 0.0
        for _ in range(self.n_interferers):
            f_k = self._random_interferer(sat_xyz_km, ut_xyz_km)
            interf_powers += np.abs(np.vdot(g, f_k)) ** 2

        signal_power = np.abs(np.vdot(g, f_des)) ** 2
        noise_power = self._thermal_noise_watts()
        sinr_lin = signal_power / (interf_powers + noise_power + 1e-18)
        sinr_db = 10.0 * np.log10(max(sinr_lin, 1e-18))

        # ---- 特徵正規化 ----
        # 1. 距離 (km) -> [0,1]
        d_km = d_m / 1000.0
        norm_distance = np.clip(d_km / 1000.0, 0.0, 1.0)

        # 2. Angle: az, el → [0,1]
        norm_az = (az + np.pi) / (2.0 * np.pi)
        norm_el = (el + (np.pi / 2.0)) / np.pi

        # 3. SINR dB clip → [0,1]
        min_db, max_db = self.snr_db_clip
        sinr_db_clipped = np.clip(sinr_db, min_db, max_db)
        norm_sinr = (sinr_db_clipped - min_db) / (max_db - min_db + 1e-12)

        # 4. 偽距：d_m + 雜訊
        sigma_m = 30.0 / np.sqrt(1.0 + sinr_lin)
        pseudo_range_m = d_m + np.random.normal(0.0, sigma_m)
        norm_range = np.clip(pseudo_range_m / (self.altitude_m + 1e-12), 0.0, 1.0)
        
        # 5. [New] Doppler (Range Rate)
        # 計算視線向量 (Unit Vector Sat -> UT)
        vec_sat_ut_m = (ut_xyz_km - sat_xyz_km) * 1000.0
        dist_m = np.linalg.norm(vec_sat_ut_m) + 1e-12
        u_los = vec_sat_ut_m / dist_m
        
        # True Range Rate = dot(V_sat, u_los) (單位: m/s)
        true_range_rate = np.dot(sat_velocity_ms, u_los)
        
        # 加入雜訊 N(0, 5) from Table
        doppler_noise = np.random.normal(0, 5.0)
        measured_doppler = true_range_rate + doppler_noise
        
        # 正規化: 範圍約 [-7600, 7600] -> [0, 1]
        norm_doppler = (measured_doppler / (self.SAT_VELOCITY_MAG + 100.0) + 1.0) / 2.0
        norm_doppler = np.clip(norm_doppler, 0.0, 1.0)

        # 6. 相位 (0~1)
        phase = (pseudo_range_m % self.lam) / self.lam

        feat = np.array(
            [norm_distance, norm_az, norm_el, norm_sinr, norm_range, norm_doppler, phase],
            dtype=np.float32,
        )
        return feat, sinr_lin, pseudo_range_m

    def _build_state(self):
        # state: 7 features x N_sats = 70 維
        feats = []
        sinrs = []
        pranges = []

        for i in range(self.N_sats):
            c = self._center(self.detected_beams[i])  # km
            v = self.sat_velocities[i] # m/s
            
            f, s_lin, pr_m = self._beam_channel_features(c, v)
            
            feats.append(f)
            sinrs.append(s_lin)
            pranges.append(pr_m)

        self._last_sinrs_lin = np.array(sinrs, dtype=np.float32)
        self._last_pseudoranges_m = np.array(pranges, dtype=np.float32)
        feats = np.stack(feats, axis=0).reshape(-1)  # 7*N_sats
        return feats.astype(np.float32)

# ==============================
# DDQN 模型與 Replay Buffer
# ==============================

class QNet(nn.Module):
    def __init__(self, state_dim, action_dim, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, action_dim),
        )

    def forward(self, x):
        return self.net(x)

class ReplayBuffer:
    def __init__(self, capacity=200_000):
        self.capacity = capacity
        self.buffer = []
        self.idx = 0

    def push(self, s, a, r, s2, d):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.idx] = (
            s.astype(np.float32),
            int(a),
            float(r),
            s2.astype(np.float32),
            float(d),
        )
        self.idx = (self.idx + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        s, a, r, s2, d = zip(*batch)
        return np.array(s), np.array(a), np.array(r), np.array(s2), np.array(d)

    def __len__(self):
        return len(self.buffer)

def softmax_weights(q, temp=1.0):
    z = (q - np.max(q)) / max(temp, 1e-6)
    e = np.exp(z)
    return e / (np.sum(e) + 1e-8)

# ==============================
# DDQN 訓練流程
# ==============================

def train_ddqn(
    total_episodes=1000,
    rollout_steps=512,
    batch_size=128,
    gamma=0.99,
    lr=3e-4,
    eps_start=1.0,
    eps_end=0.05,
    eps_decay=0.995,
    target_sync=500,
    device="cpu",
):
    start_time = time.time()
    env = BeamEnv()
    action_dim = env.N_sats
    # [Modified] 7 features per beam * 10 sats = 70 dims
    state_dim = 7 * env.N_sats

    q_net = QNet(state_dim, action_dim).to(device)
    target_net = QNet(state_dim, action_dim).to(device)
    target_net.load_state_dict(q_net.state_dict())
    optimizer = optim.Adam(q_net.parameters(), lr=lr)
    buffer = ReplayBuffer(200_000)

    epsilon = eps_start
    steps = 0
    rewards_per_episode = []
    errors_per_episode = []
    weights_history = []

    last_weights = np.ones(action_dim, dtype=np.float32) / action_dim

    for ep in range(total_episodes):
        state = env.reset()
        total_reward = 0.0
        ep_errors_m = []

        for t in range(rollout_steps):
            # epsilon-greedy 選 beam index
            with torch.no_grad():
                q_values = q_net(torch.as_tensor(state, dtype=torch.float32, device=device)).cpu().numpy()
            if random.random() < epsilon:
                action = random.randrange(action_dim)
            else:
                action = int(np.argmax(q_values))

            # 將 Q 值 softmax 成權重，做 weighted COG 定位誤差
            w = softmax_weights(q_values)
            last_weights = w.copy()
            pos = env._weighted_cog(w)
            err_km = env._err_km(pos)
            err_m = err_km * 1000.0

            # reward：跟原版一樣
            reward = -np.exp(err_km / 10.0) + 1.0

            # 每步都重抽幾何（與你原本邏輯一致）
            next_state = env.reset()
            done = 1.0 if (err_km < 0.001 or random.random() > 0.98) else 0.0

            buffer.push(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            ep_errors_m.append(err_m)

            # DDQN 更新
            if len(buffer) >= batch_size:
                s_batch, a_batch, r_batch, s2_batch, d_batch = buffer.sample(batch_size)
                s_batch = torch.as_tensor(s_batch, dtype=torch.float32, device=device)
                a_batch = torch.as_tensor(a_batch, dtype=torch.long, device=device).unsqueeze(-1)
                r_batch = torch.as_tensor(r_batch, dtype=torch.float32, device=device).unsqueeze(-1)
                s2_batch = torch.as_tensor(s2_batch, dtype=torch.float32, device=device)
                d_batch = torch.as_tensor(d_batch, dtype=torch.float32, device=device).unsqueeze(-1)

                with torch.no_grad():
                    q_next_online = q_net(s2_batch)
                    best_actions = torch.argmax(q_next_online, dim=1, keepdim=True)
                    q_next_target = target_net(s2_batch).gather(1, best_actions)
                    y = r_batch + (1.0 - d_batch) * gamma * q_next_target

                q_sa = q_net(s_batch).gather(1, a_batch)
                loss = nn.MSELoss()(q_sa, y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                steps += 1
                if steps % target_sync == 0:
                    target_net.load_state_dict(q_net.state_dict())

            if done > 0.5:
                break

        epsilon = max(eps_end, epsilon * eps_decay)
        rewards_per_episode.append(total_reward)
        errors_per_episode.append(float(np.mean(ep_errors_m) if len(ep_errors_m) > 0 else np.nan))
        weights_history.append(last_weights / (np.sum(last_weights) + 1e-8))

        if ep % 50 == 0:
            print(f"[Episode {ep}] Reward={total_reward:.2f}, Error={errors_per_episode[-1]:.2f} m, eps={epsilon:.3f}")

    # ==============================
    # Final beams + ALG-B vs DDQN 定位圖
    # ==============================
    final_w = weights_history[-1] if len(weights_history) > 0 else (np.ones(action_dim) / action_dim)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    det_polys = [env._circle_poly(env._center(b), env.beam_radius / 5.0) for b in env.detected_beams]
    inter = cascaded_union(det_polys)
    und_polys = [env._circle_poly(env._center(b), env.beam_radius / 5.0) for b in env.undetected_beams]
    for p in und_polys:
        if inter.is_empty:
            break
        if inter.intersects(p):
            inter = inter.difference(p)
    if not inter.is_empty:
        if inter.geom_type == "Polygon":
            x, y = inter.exterior.xy
            ax1.fill(x, y, alpha=0.3, fc="gray", ec="none")
        else:
            for g in inter.geoms:
                x, y = g.exterior.xy
                ax1.fill(x, y, alpha=0.3, fc="gray", ec="none")

    base = env._intersection_centroid()
    base_err_m = env._err_km(base) * 1000.0
    ax1.plot(base[0], base[1], "*", color="yellow", markersize=18)
    ax1.text(0.02, 0.98, f"Distance Error: {base_err_m:.2f} m",
             transform=ax1.transAxes, va="top")
    ax1.set_title("ALG.B", bbox=dict(facecolor="white", edgecolor="black", pad=5))

    pos_ddqn = env._weighted_cog(final_w)
    err_ddqn_m = env._err_km(pos_ddqn) * 1000.0
    ax2.plot(pos_ddqn[0], pos_ddqn[1], "*", color="yellow", markersize=18)
    ax2.text(0.02, 0.98, f"Distance Error: {err_ddqn_m:.2f} m",
             transform=ax2.transAxes, va="top")
    ax2.set_title("DDQN + Doppler", bbox=dict(facecolor="white", edgecolor="black", pad=5))

    for i in range(env.N_sats):
        c_det = env._center(env.detected_beams[i])
        c_und = env._center(env.undetected_beams[i])
        w_i = final_w[i]
        lw_det = 1.5 + 2.5 * w_i
        lw_und = 1.5 + 2.5 * (1.0 - w_i)
        for ax in (ax1, ax2):
            ax.add_patch(
                Circle(c_det, env.beam_radius / 5.0, color="blue", alpha=0.6, fill=False, linewidth=lw_det)
            )
            ax.add_patch(
                Circle(c_und, env.beam_radius / 5.0, color="red", alpha=0.6, fill=False, linewidth=lw_und)
            )
    for ax in (ax1, ax2):
        ax.plot(env.UT_position[0], env.UT_position[1], "ko", markerfacecolor="none", markersize=10)
        ax.set_xlim(-20, 20)
        ax.set_ylim(-20, 20)
        ax.set_xlabel("X (km)")
        ax.set_ylabel("Y (km)")
        ax.grid(True, linestyle="--", alpha=0.7)

    plt.suptitle(f"Training Episode {total_episodes - 1}")
    plt.tight_layout()
    plt.show()

    # ==============================
    # 訓練摘要 + 曲線 + 權重圖
    # ==============================
    elapsed = time.time() - start_time
    q_flops = 2 * state_dim * 128 + 2 * 128 * 128 + 2 * 128 * action_dim
    rollout_flops = total_episodes * rollout_steps * q_flops
    updates = total_episodes * rollout_steps
    update_flops = 3 * updates * q_flops
    total_flops = rollout_flops + update_flops

    if len(errors_per_episode) > 0 and not np.isnan(errors_per_episode[-1]):
        print("========== DDQN 訓練摘要 ==========")
        print(f"[Model FLOPs] QNet 每次 forward 約 {q_flops/1e6:.2f} M FLOPs")
        print(f"[Training FLOPs] Rollout 累計 ≈ {rollout_flops/1e9:.2f} GFLOPs")
        print(f"[Training FLOPs] Update 累計 ≈ {update_flops/1e9:.2f} GFLOPs")
        print(f"[TOTAL FLOPs] 模型×訓練總次數 ≈ {total_flops/1e9:.2f} GFLOPs")
        print(f"[Training Time] 總耗時: {elapsed:.2f} 秒")
        print(f"[Final Error] 定位誤差距離: {errors_per_episode[-1]:.2f} m")

    # Reward / Error 曲線
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(rewards_per_episode, label="Raw Reward")
    if len(rewards_per_episode) > 5:
        w = max(5, len(rewards_per_episode) // 20)
        ma = np.convolve(rewards_per_episode, np.ones(w) / w, mode="valid")
        plt.plot(
            np.arange(w - 1, w - 1 + len(ma)),
            ma,
            linewidth=2,
            label="Smoothed Reward",
        )
    plt.xlabel("Episodes", fontsize=14)
    plt.ylabel("Reward", fontsize=14)
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.7)

    plt.subplot(1, 2, 2)
    plt.plot(errors_per_episode, label="Raw Error")
    if len(errors_per_episode) > 5:
        w2 = max(5, len(errors_per_episode) // 20)
        ma2 = np.convolve(errors_per_episode, np.ones(w2) / w2, mode="valid")
        plt.plot(
            np.arange(w2 - 1, w2 - 1 + len(ma2)),
            ma2,
            linewidth=2,
            label="Smoothed Error",
        )
    plt.xlabel("Episodes", fontsize=14)
    plt.ylabel("Error (m)", fontsize=14)
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.7)

    plt.tight_layout()
    plt.show()

    # Final weight bar
    plt.figure(figsize=(8, 4))
    fw = final_w
    plt.bar(range(len(fw)), fw)
    plt.xlabel("Beam")
    plt.ylabel("Weight")
    plt.title("Final Beam Weight Distribution (DDQN)")
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.show()

    # Weights heatmap
    if len(weights_history) > 0:
        W = np.array(weights_history)
        plt.figure(figsize=(12, 4.5))
        im = plt.imshow(W.T, aspect="auto", origin="lower")
        plt.colorbar(im, fraction=0.046, pad=0.04, label="Weight")
        plt.yticks(range(W.shape[1]), [f"Beam {i}" for i in range(W.shape[1])])
        plt.xlabel("Episode")
        plt.title("Beam Weights over Training (Heatmap)")
        plt.tight_layout()
        plt.show()

    return rewards_per_episode, errors_per_episode

if __name__ == "__main__":
    train_ddqn(total_episodes=1000, rollout_steps=512, device="cpu")