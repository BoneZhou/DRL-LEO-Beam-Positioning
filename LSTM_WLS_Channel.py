
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from matplotlib.patches import Circle
from shapely.geometry import Polygon
from shapely.ops import cascaded_union
import time

start_time = time.time()

# ==============================
# Channel / UPA helper functions
# ==============================

def direction_cosines_from_az_el(az, el):
    """Compute direction cosines (theta_x, theta_y) from azimuth/elevation."""
    thx = np.cos(el) * np.cos(az)
    thy = np.cos(el) * np.sin(az)
    return thx, thy


def az_el_from_vector(v):
    """Get azimuth / elevation from a 3D vector."""
    vx, vy, vz = v
    r = np.linalg.norm(v) + 1e-12
    az = np.arctan2(vy, vx)
    el = np.arcsin(vz / r)
    return az, el


def upa_steering_vector(Nx, Ny, thx, thy):
    """
    UPA steering vector, assuming half-wavelength spacing (embedded in pi factor).
    Returns a complex vector of length Nx*Ny.
    """
    ax = np.exp(-1j * np.pi * thx * np.arange(Nx)) / np.sqrt(Nx)
    ay = np.exp(-1j * np.pi * thy * np.arange(Ny)) / np.sqrt(Ny)
    a = np.kron(ax, ay)
    return a


# ==============================
# Beam environment with channel model
# ==============================

class BeamEnv:
    """
    Beam environment (channel-based):
    - N_sats beams around a fixed UT
    - 通道模型：UPA + FSPL + SINR + 偽距 + 相位
    - 每個 beam 提供 pseudo-range，用來構建殘差矩陣
    """

    def __init__(self,
                 N_sats=10,
                 beam_radius=50.0,           # km (for drawing)
                 UT_position=(0.0, 0.0),     # km
                 # Channel parameters
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
        self.UT_position = np.array(UT_position, dtype=np.float32)

        # Channel-related
        self.c = 299792458.0
        self.fc = fc
        self.lam = self.c / self.fc
        self.Gt = 10 ** (Gt_dBi / 10.0)
        self.Gr = 10 ** (Gr_dBi / 10.0)
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

        # Caches
        self._last_sinrs_lin = np.ones(self.N_sats, dtype=np.float32)
        self._last_pseudoranges_m = np.zeros(self.N_sats, dtype=np.float32)

        self.reset()

    # ---------- Geometry ----------
    def _generate_beam_positions(self):
        """
        Generate random beams around UT, like DQN/PPO env.
        """
        center_offset = np.random.uniform(-10, 10, (self.N_sats, 2))
        start_points = self.UT_position + center_offset + np.random.uniform(-5, 5, (self.N_sats, 2))
        end_points = start_points + np.random.uniform(-3, 3, (self.N_sats, 2))
        return np.stack([start_points, end_points], axis=1)  # shape: (N_sats, 2, 2)

    def reset(self):
        self.detected_beams = self._generate_beam_positions()
        self.undetected_beams = self._generate_beam_positions()
        # 會在 _compute_state 裡更新 pseudo-ranges
        _ = self._compute_state()
        return self._compute_residual_matrix()

    def _beam_center(self, beam):
        return (beam[0] + beam[1]) / 2.0

    # ---------- Channel helpers ----------
    def _thermal_noise_watts(self):
        N0 = self.kB * self.temperature * self.bandwidth
        NF = 10 ** (self.noise_figure_dB / 10.0)
        return N0 * NF

    def _free_space_path_loss_linear(self, d_m):
        return (4.0 * np.pi * self.fc * d_m / self.c) ** 2

    def _upa_channel_vector(self, sat_xyz_km, ut_xyz_km):
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
        """
        For one beam, compute 6-dim channel feature & pseudo-range.
        """
        ut_xyz_km = np.array([self.UT_position[0], self.UT_position[1], 0.0], dtype=np.float32)
        sat_xyz_km = np.array([beam_center_xy_km[0], beam_center_xy_km[1], self.altitude_km], dtype=np.float32)

        # LoS channel
        g, d_m, az, el = self._upa_channel_vector(sat_xyz_km, ut_xyz_km)

        # Desired beamformer
        thx, thy = direction_cosines_from_az_el(az, el)
        f_des = upa_steering_vector(self.Nx, self.Ny, thx, thy)
        f_des = f_des / (np.linalg.norm(f_des) + 1e-12)

        # Interferers
        interf_powers = 0.0
        for _ in range(self.n_interferers):
            f_k = self._random_interferer(sat_xyz_km, ut_xyz_km)
            interf_powers += np.abs(np.vdot(g, f_k)) ** 2

        signal_power = np.abs(np.vdot(g, f_des)) ** 2
        noise_power = self._thermal_noise_watts()
        sinr_lin = signal_power / (interf_powers + noise_power + 1e-18)
        sinr_db = 10.0 * np.log10(max(sinr_lin, 1e-18))

        # distance normalization
        d_km = d_m / 1000.0
        norm_distance = np.clip(d_km / 1000.0, 0.0, 1.0)

        # az/el normalization
        norm_az = (az + np.pi) / (2.0 * np.pi)
        norm_el = (el + (np.pi / 2.0)) / np.pi

        # SINR normalization
        min_db, max_db = self.snr_db_clip
        sinr_db_clipped = np.clip(sinr_db, min_db, max_db)
        norm_sinr = (sinr_db_clipped - min_db) / (max_db - min_db + 1e-12)

        # pseudo-range
        sigma_m = 30.0 / np.sqrt(1.0 + sinr_lin)
        pseudo_range_m = d_m + np.random.normal(0.0, sigma_m)
        norm_range = np.clip(pseudo_range_m / (self.altitude_m + 1e-12), 0.0, 1.0)

        # phase
        phase = (pseudo_range_m % self.lam) / self.lam

        feat = np.array(
            [norm_distance, norm_az, norm_el, norm_sinr, norm_range, phase],
            dtype=np.float32,
        )
        return feat, sinr_lin, pseudo_range_m

    def _compute_state(self):
        """
        Compute per-beam channel features (但這裡我們主要是要 pseudo-range).
        """
        feats = []
        sinrs = []
        pranges = []
        for i in range(self.N_sats):
            c = self._beam_center(self.detected_beams[i])
            f, s_lin, pr_m = self._beam_channel_features(c)
            feats.extend(f.tolist())
            sinrs.append(s_lin)
            pranges.append(pr_m)
        self._last_sinrs_lin = np.array(sinrs, dtype=np.float32)
        self._last_pseudoranges_m = np.array(pranges, dtype=np.float32)
        return np.array(feats, dtype=np.float32)

    def _compute_residual_matrix(self):
        """
        模仿論文的「殘差矩陣」概念，但較簡化：
        - 先拿到每顆 beam 的 pseudo-range r_i
        - 定義 centered residual: e_i = r_i - mean(r)
        - 構造 NxN 矩陣 M：M[i,j] = |e_i - e_j|
        - 再做 normalize 到 [0,1]
        """
        r = self._last_pseudoranges_m  # shape (N_sats,)
        e = r - np.mean(r)
        M = np.abs(e[:, None] - e[None, :])  # pairwise residual difference
        max_val = np.max(M) + 1e-12
        M_norm = M / max_val
        return M_norm.astype(np.float32)  # (N_sats, N_sats)

    # ---------- Weighted CoG & error ----------
    def weighted_position(self, weights):
        """
        Weighted center-of-gravity using detected beams & given weights.
        """
        weights = np.clip(weights, 1e-8, None)
        weights = weights / (np.sum(weights) + 1e-12)
        pos = np.zeros(2, dtype=np.float32)
        for i in range(self.N_sats):
            c = self._beam_center(self.detected_beams[i])
            pos += c * weights[i]
        return pos

    def error_m(self, weights):
        est_pos = self.weighted_position(weights)
        err_km = np.linalg.norm(est_pos - self.UT_position)
        return float(err_km * 1000.0)

    # ---------- For visualization ----------
    def visualize_case(self, weights=None, title="LSTM Residual-Channel Positioning"):
        fig, ax = plt.subplots(figsize=(6, 6))
        # draw beams
        for i in range(self.N_sats):
            c_det = self._beam_center(self.detected_beams[i])
            w_i = weights[i] if weights is not None else 1.0 / self.N_sats
            lw = 1.5 + 3.0 * w_i
            circ = Circle(c_det, self.beam_radius / 5.0, color="blue", alpha=0.5, fill=False, linewidth=lw)
            ax.add_patch(circ)

        # UT and estimate
        if weights is not None:
            est = self.weighted_position(weights)
            err_m = self.error_m(weights)
            ax.plot(est[0], est[1], "r*", label=f"Est (err={err_m:.1f} m)", markersize=12)

        ax.plot(self.UT_position[0], self.UT_position[1], "ko", label="True UT", markerfacecolor="none", markersize=8)
        ax.set_xlim(-20, 20)
        ax.set_ylim(-20, 20)
        ax.set_xlabel("X (km)")
        ax.set_ylabel("Y (km)")
        ax.set_title(title)
        ax.grid(True, linestyle="--", alpha=0.7)
        ax.legend()
        plt.tight_layout()
        plt.show()


# ==============================
# Dataset generation for LSTM (Residual matrix)
# ==============================

def generate_lstm_residual_dataset(num_samples=2000, N_sats=10):
    """
    產生 LSTM 訓練資料：
    - X: (num_samples, N_sats, N_sats) 殘差矩陣 (channel-based pseudo-range residual)
    - y: (num_samples, N_sats) 目標權重（用幾何距離設計的“理想加權”）
    """
    env = BeamEnv(N_sats=N_sats)
    X_list = []
    y_list = []

    for _ in range(num_samples):
        # 新的幾何 + 通道
        env.reset()
        M = env._compute_residual_matrix()  # (N_sats, N_sats)

        # 理想權重：距離越近權重越大 → w_i ∝ exp(-distance_i / tau)
        centers = np.array([env._beam_center(env.detected_beams[i]) for i in range(N_sats)], dtype=np.float32)
        dists_km = np.linalg.norm(centers - env.UT_position[None, :], axis=1)  # km
        tau = 10.0  # km scale
        raw_w = np.exp(-dists_km / tau)
        weights = raw_w / (np.sum(raw_w) + 1e-12)  # normalize to sum=1

        X_list.append(M)
        y_list.append(weights.astype(np.float32))

    X = np.stack(X_list, axis=0)  # (num_samples, N_sats, N_sats)
    y = np.stack(y_list, axis=0)  # (num_samples, N_sats)
    return X, y


# ==============================
# LSTM model (Residual matrix as sequence)
# ==============================

def build_lstm_residual_model(input_shape=(10, 10), output_dim=10):
    """
    LSTM 輸入：每一個 time step 是殘差矩陣的一列（長度 N_sats），
    序列長度 = N_sats。也就是 input_shape=(N_sats, N_sats)。
    """
    model = Sequential()
    model.add(LSTM(64, input_shape=input_shape, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(64, activation="relu"))
    # 輸出 10 維權重，用 softmax 統一成加總 = 1
    model.add(Dense(output_dim, activation="softmax"))
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                  loss="mse",
                  metrics=["mae"])
    return model


# ==============================
# Training & evaluation
# ==============================

def train_lstm_channel_residual(num_samples=2000,
                                N_sats=10,
                                epochs=30,
                                batch_size=32):
    print("Generating residual-based channel dataset ...")
    X, y = generate_lstm_residual_dataset(num_samples=num_samples, N_sats=N_sats)
    # train/val split
    split = int(0.8 * num_samples)
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]

    print("Building LSTM model (residual-based, channel-aware) ...")
    model = build_lstm_residual_model(input_shape=(N_sats, N_sats), output_dim=N_sats)

    print("Start training LSTM on residual matrices (state ~ N×N) ...")
    hist = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        shuffle=True,
        verbose=1
    )

    # 紀錄訓練時間
    elapsed = time.time() - start_time
    print(f"Training finished, elapsed time: {elapsed:.2f} s")

    # 用驗證集做定位誤差統計（重新抽樣新的幾何）
    env = BeamEnv(N_sats=N_sats)
    num_test = min(200, X_val.shape[0])
    errors_m = []

    for _ in range(num_test):
        _ = env.reset()
        M = env._compute_residual_matrix()[None, ...]  # shape (1, N_sats, N_sats)
        pred_w = model.predict(M, verbose=0)[0]  # (N_sats,)
        err_m = env.error_m(pred_w)
        errors_m.append(err_m)

    errors_m = np.array(errors_m)
    print(f"[LSTM Residual + Channel] 平均定位誤差: {np.mean(errors_m):.2f} m,  標準差: {np.std(errors_m):.2f} m")

    # 畫訓練曲線 (Loss)
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(hist.history["loss"], label="Train Loss")
    plt.plot(hist.history["val_loss"], label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("LSTM Training Loss (Residual-Channel)")
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.hist(errors_m, bins=20, alpha=0.8)
    plt.xlabel("Error (m)")
    plt.ylabel("Count")
    plt.title("LSTM Residual-Channel Localization Error (m)")
    plt.grid(True, linestyle="--", alpha=0.7)

    plt.tight_layout()
    plt.show()

    # 挑一個案例畫 2D beam 圖 + UT + estimated position
    env = BeamEnv(N_sats=N_sats)
    _ = env.reset()
    M_case = env._compute_residual_matrix()[None, ...]
    pred_w_case = model.predict(M_case, verbose=0)[0]
    env.visualize_case(weights=pred_w_case,
                       title="LSTM Residual-Channel Beam Positioning (Single Case)")

    # 存成 npz 給之後跟 DQN/PPO 比較用
    np.savez("lstm_channel_residual_logs.npz",
             train_loss=np.array(hist.history["loss"], dtype=np.float32),
             val_loss=np.array(hist.history["val_loss"], dtype=np.float32),
             errors_m=errors_m)

    return model, hist, errors_m


if __name__ == "__main__":
    model, hist, errors_m = train_lstm_channel_residual(
        num_samples=2000,
        N_sats=10,
        epochs=30,
        batch_size=32
    )
