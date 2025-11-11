import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Point, Polygon
from shapely.ops import unary_union

# 初始化參數
N_sats = 10  # 衛星數量
beam_radius = 50  # 波束半徑 (km)
beam_sep = 75  # 波束中心距離 (km)
grid_size = 5  # 每個衛星的波束網格尺寸 (5x5)
N_samples = 1  # 只展示一個樣本的圖表

# 初始化圖表
fig, ax = plt.subplots(1, 2, figsize=(12, 6))
for i in range(2):
    ax[i].set_aspect('equal')
    ax[i].set_xlim([-20, 20])
    ax[i].set_ylim([-20, 20])
    ax[i].set_xlabel('X (km)')
    ax[i].set_ylabel('Y (km)')
ax[0].set_title('Positioning Results using ALG. A')
ax[1].set_title('Positioning Results using ALG. B')

# 開始模擬
for sample in range(N_samples):
    # 隨機旋轉和位移衛星
    sat_positions = beam_sep * (np.random.rand(N_sats, 2) - 0.5)

    # 儲存偵測到的波束中心座標
    detected_beams = []
    undetected_beams = []
    beam_shapes = []

    # 隨機生成波束覆蓋圖案，計算與波束中心的距離
    for i in range(N_sats):
        for row in range(-2, 3):
            for col in range(-2, 3):
                beam_center = sat_positions[i] + np.array([row, col]) * beam_sep
                dist = np.linalg.norm(beam_center)  # 假設UT位置在原點(0, 0)

                # 偵測到的波束
                if dist <= beam_radius:
                    detected_beams.append(beam_center)
                    ax[0].add_patch(plt.Circle(tuple(beam_center), beam_radius, color='blue', fill=False))
                    ax[1].add_patch(plt.Circle(tuple(beam_center), beam_radius, color='blue', fill=False))
                    # 繪製偵測到的波束形狀
                    beam_shapes.append(Point(beam_center).buffer(beam_radius))  # 用圓形表示波束範圍
                elif dist <= 3 * beam_radius:  # ALG B 未偵測到的波束
                    undetected_beams.append(beam_center)
                    ax[1].add_patch(plt.Circle(tuple(beam_center), beam_radius, color='red', fill=False))

    # ALG A：計算交集區域並著色
    if beam_shapes:
        intersection_A = beam_shapes[0]
        for shape in beam_shapes[1:]:
            intersection_A = intersection_A.intersection(shape)

        if intersection_A.is_valid and intersection_A.area > 0:
            # 計算並標示質心
            CoG_A = np.array(intersection_A.centroid.coords)[0]
            ax[0].plot(CoG_A[0], CoG_A[1], 'ob', markersize=7, linewidth=1)
            UT_position_est_A = CoG_A  # 這是 ALG A 的 UT 預估位置
        else:
            print('Warning: Intersection in ALG. A is invalid or empty.')

    # ALG B：從 ALG A 的交集區域中扣除未偵測到的波束
    if undetected_beams:
        intersection_B = intersection_A
        for beam in undetected_beams:
            beam_shape = Point(beam).buffer(beam_radius)
            intersection_B = intersection_B.difference(beam_shape)

        if intersection_B.is_valid and intersection_B.area > 0:
            # 計算並標示質心
            CoG_B = np.array(intersection_B.centroid.coords)[0]
            ax[1].plot(CoG_B[0], CoG_B[1], 'or', markersize=7, linewidth=1)
            UT_position_est_B = CoG_B  # 這是 ALG B 的 UT 預估位置
        else:
            print('Warning: Intersection in ALG. B is invalid or empty.')

# 繪製預估UT的位置 (在兩個子圖中都顯示)
ax[0].plot(UT_position_est_A[0], UT_position_est_A[1], 'ko', markersize=8, linewidth=2)
ax[1].plot(UT_position_est_B[0], UT_position_est_B[1], 'ko', markersize=8, linewidth=2)

# 添加圖例
ax[0].legend(['Detected Beams', 'CoG (ALG. A)', 'Estimated UT Position'], loc='best')
ax[1].legend(['Detected Beams', 'Undetected Beams', 'CoG (ALG. B)', 'Estimated UT Position'], loc='best')

plt.tight_layout()
plt.show()
