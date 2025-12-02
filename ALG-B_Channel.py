import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Point
from shapely.ops import unary_union

# =========================
#      幾何 / 波束參數
# =========================

N_sats = 10      # 衛星數量
beam_radius = 50 # 波束半徑 (km)
beam_sep = 75    # 波束中心距離 (km)
grid_size = 5    # 每個衛星的波束網格尺寸 (5x5)
N_samples = 1    # 要跑幾個隨機樣本（1 就是單張圖）

# UT 真實位置 (地面座標, km)
UT_true_pos = np.array([0.0, 0.0])

# =========================
#      通道模型參數
# =========================
# 3D 距離 + FSPL + log-normal shadowing + 小尺度衰落 + 熱雜訊 → SNR

c = 3e8                 # 光速 (m/s)
fc = 2e9                # 載波頻率 2 GHz
lam = c / fc            # 波長 (m)

sat_alt_km = 600.0      # 衛星高度 (km)
P_tx_dBm = 40.0         # 每個波束的等效發射功率 (dBm)
G_tx_dBi = 20.0         # 衛星天線增益 (dBi)
G_rx_dBi = 0.0          # UT 天線增益 (dBi)

BW_Hz = 10e6            # 有效通道頻寬 10 MHz
NF_dB = 5.0             # 接收機雜訊指數 (Noise Figure)
kT_dBm_per_Hz = -174.0  # 熱雜訊功率譜密度 (dBm/Hz)

# 雜訊底 (noise floor)
noise_floor_dBm = kT_dBm_per_Hz + 10 * np.log10(BW_Hz) + NF_dB  # dBm

# Shadowing / fading 參數 (以 dB 做隨機擾動)
shadow_sigma_dB = 4.0   # log-normal shadowing 標準差
fading_sigma_dB = 2.0   # 小尺度衰落(簡化成高斯在 dB) 標準差

# 偵測門檻 (可以用 SNR 也可以用接收功率)
snr_det_threshold_dB = 0.0   # SNR >= 0 dB 視為「有成功解出 beam ID」


# =========================
#      通道模型函式
# =========================

def compute_snr_and_prx(beam_center_km, ut_pos_km):
    """
    計算某個波束中心到 UT 的接收功率與 SNR (單一 snapshot)
    - beam_center_km: 波束中心在地面的 (x, y) (km)
    - ut_pos_km: UT 在地面的 (x, y) (km)
    """
    # 地面水平距離 (km)
    horiz_dist_km = np.linalg.norm(beam_center_km - ut_pos_km)  # km

    # 3D 斜距離 (km → m)
    slant_dist_km = np.sqrt(sat_alt_km**2 + horiz_dist_km**2)
    d_m = slant_dist_km * 1000.0

    # 自由空間路徑損耗 FSPL (dB)：20 log10(4πd/λ)
    import math
    fspl_dB = 20.0 * math.log10(4.0 * math.pi * d_m / lam)

    # log-normal shadowing + 小尺度衰落 (用 dB 隨機加減)
    shadowing_dB = np.random.normal(0.0, shadow_sigma_dB)
    fading_dB = np.random.normal(0.0, fading_sigma_dB)

    # 接收功率 (dBm)
    P_rx_dBm = P_tx_dBm + G_tx_dBi + G_rx_dBi - fspl_dB + shadowing_dB + fading_dB

    # SNR (dB)
    snr_dB = P_rx_dBm - noise_floor_dBm

    return snr_dB, P_rx_dBm


# =========================
#      畫圖初始化
# =========================

fig, ax = plt.subplots(1, 2, figsize=(12, 6))

for i in range(2):
    ax[i].set_aspect('equal')
    ax[i].set_xlim([-20, 20])
    ax[i].set_ylim([-20, 20])
    ax[i].set_xlabel('X (km)')
    ax[i].set_ylabel('Y (km)')

ax[0].set_title('Positioning Results using ALG. A (with channel)')
ax[1].set_title('Positioning Results using ALG. B (with channel)')


# =========================
#      開始模擬
# =========================

errors_A_km = []  # 儲存每個樣本的誤差 (ALG A)
errors_B_km = []  # 儲存每個樣本的誤差 (ALG B)

last_CoG_A = None
last_CoG_B = None

for sample in range(N_samples):
    # 隨機位移衛星 (簡化)
    sat_positions = beam_sep * (np.random.rand(N_sats, 2) - 0.5)

    detected_beams = []     # 偵測到的波束中心 (ALG A/B 都會用到)
    undetected_beams = []   # R<d<=3R 且沒被偵測到的波束中心 (ALG B 用)
    beam_shapes = []        # Shapely 圓形，用於取交集

    # 為每一顆衛星產生波束
    for i in range(N_sats):
        for row in range(-2, 3):
            for col in range(-2, 3):
                beam_center = sat_positions[i] + np.array([row, col]) * beam_sep
                dist_ground = np.linalg.norm(beam_center - UT_true_pos)  # km

                # 太遠的 beam 直接略過
                if dist_ground > 3 * beam_radius:
                    continue

                # 計算通道 SNR & P_rx
                snr_dB, P_rx_dBm = compute_snr_and_prx(beam_center, UT_true_pos)

                # -----------------------------
                # ① footprint 內 (d <= R)
                #    -> 若 SNR 過門檻, 當作「偵測到的 beam」
                #    -> 若 SNR 太低, 直接無視 (不當作 undetected)
                # -----------------------------
                if dist_ground <= beam_radius:
                    if snr_dB >= snr_det_threshold_dB:
                        detected_beams.append(beam_center)

                        # 在 ALG A 圖畫偵測 beam
                        circleA = plt.Circle(tuple(beam_center),
                                             beam_radius,
                                             fill=False)
                        ax[0].add_patch(circleA)

                        # 在 ALG B 圖畫偵測 beam
                        circleB = plt.Circle(tuple(beam_center),
                                             beam_radius,
                                             fill=False)
                        ax[1].add_patch(circleB)

                        # shapely polygon (km 座標)
                        beam_shapes.append(
                            Point(beam_center[0], beam_center[1]).buffer(beam_radius)
                        )

                # -----------------------------
                # ② R < d <= 3R 的鄰近波束
                #    -> 若 SNR < 門檻, 視為「未偵測 beam」，ALG B 用來扣掉
                # -----------------------------
                elif beam_radius < dist_ground <= 3 * beam_radius:
                    if snr_dB < snr_det_threshold_dB:
                        undetected_beams.append(beam_center)

                        # 未偵測 beam 僅畫在 ALG B（右圖）
                        circleB = plt.Circle(tuple(beam_center),
                                             beam_radius,
                                             fill=False)
                        ax[1].add_patch(circleB)
                    else:
                        # 若側邊也偵測到，可選擇畫出 (這裡只是畫線，不影響演算法)
                        circleB = plt.Circle(tuple(beam_center),
                                             beam_radius,
                                             fill=False)
                        ax[1].add_patch(circleB)

    # ========== ALG A：偵測到的波束交集 ==========
    if beam_shapes:
        from functools import reduce
        intersection_A = reduce(lambda a, b: a.intersection(b), beam_shapes)

        if intersection_A.is_valid and (not intersection_A.is_empty) and intersection_A.area > 0:
            centroidA = intersection_A.centroid
            CoG_A = np.array([centroidA.x, centroidA.y])
            last_CoG_A = CoG_A

            # 這個樣本的誤差 (km)
            err_A_km = np.linalg.norm(CoG_A - UT_true_pos)
            errors_A_km.append(err_A_km)

            # 標示 ALG A 的 CoG
            ax[0].plot(CoG_A[0], CoG_A[1],
                       'o', markersize=8,
                       markerfacecolor='none')
        else:
            print('Warning: ALG A intersection is invalid or empty.')
            intersection_A = None
    else:
        print('Warning: No detected beams → ALG. A 無法定位')
        intersection_A = None

    # ========== ALG B：從 ALG A 信心區扣掉「未偵測波束」 ==========
    if intersection_A is not None:
        intersection_B = intersection_A
        for beam in undetected_beams:
            beam_shape = Point(beam[0], beam[1]).buffer(beam_radius)
            # 只有真的有交集才扣，避免多餘運算
            if intersection_B.intersects(beam_shape):
                intersection_B = intersection_B.difference(beam_shape)
                # 若扣到整個沒了，就跳出
                if intersection_B.is_empty:
                    break

        if (intersection_B is not None) and (not intersection_B.is_empty) and intersection_B.area > 0:
            centroidB = intersection_B.centroid
            CoG_B = np.array([centroidB.x, centroidB.y])
            last_CoG_B = CoG_B

            # 這個樣本的誤差 (km)
            err_B_km = np.linalg.norm(CoG_B - UT_true_pos)
            errors_B_km.append(err_B_km)

            # 標示 ALG B 的 CoG
            ax[1].plot(CoG_B[0], CoG_B[1],
                       'o', markersize=8,
                       markerfacecolor='none')
        else:
            print('Warning: Intersection in ALG. B is invalid or empty.')
    else:
        print('ALG B: 因為 ALG A 沒有信心區，所以也無法計算')

# =========================
#      畫出 UT 真實位置
# =========================

for i in range(2):
    ax[i].plot(UT_true_pos[0], UT_true_pos[1],
               '^k', markersize=8)

# 若只有 1 個樣本，在圖上直接寫誤差
if N_samples == 1:
    if last_CoG_A is not None and len(errors_A_km) > 0:
        ax[0].text(0.02, 0.95,
                   f'Error = {errors_A_km[0]*1000:.2f} m',
                   transform=ax[0].transAxes,
                   ha='left', va='top')
    if last_CoG_B is not None and len(errors_B_km) > 0:
        ax[1].text(0.02, 0.95,
                   f'Error = {errors_B_km[0]*1000:.2f} m',
                   transform=ax[1].transAxes,
                   ha='left', va='top')

plt.tight_layout()
plt.show()

# =========================
#      在終端輸出誤差統計
# =========================

def summarize_errors(name, errs_km):
    if not errs_km:
        print(f'{name}: 無有效樣本')
        return
    errs_km = np.array(errs_km)
    mae_km = np.mean(np.abs(errs_km))
    rmse_km = np.sqrt(np.mean(errs_km**2))
    max_km = np.max(np.abs(errs_km))
    print(f'[{name}] N = {len(errs_km)}')
    print(f'  MAE  = {mae_km*1000:.2f} m')
    print(f'  RMSE = {rmse_km*1000:.2f} m')
    print(f'  MAX  = {max_km*1000:.2f} m')

print('\n===== Positioning Error (with channel) =====')
summarize_errors('ALG A', errors_A_km)
summarize_errors('ALG B', errors_B_km)
