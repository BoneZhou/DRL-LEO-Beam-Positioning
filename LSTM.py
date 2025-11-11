import numpy as np
import random
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, BatchNormalization, Dropout
import time
start_time = time.time()

def generate_gnss_data(num_epochs=1000, num_satellites=60, bias_ratio=0.1):
    """
    生成模擬的 GNSS 伏距數據，並進行數據正規化。
    """
    receiver_pos = np.array([0, 0, 0])
    satellite_positions = np.random.uniform(-20_000, 20_000, (num_satellites, 3))
    true_ranges = np.linalg.norm(satellite_positions - receiver_pos, axis=1)
    
    # 使用較小的噪聲範圍
    alpha = 0.9
    sigma = 2  # 降低高斯噪聲的標準差
    lambda_exp = 1 / 50  # 降低指數分佈的期望值
    
    noise = np.where(np.random.rand(num_satellites) < alpha, 
                     np.random.normal(0, sigma, num_satellites), 
                     np.random.exponential(lambda_exp, num_satellites))
    
    num_biased = int(num_satellites * bias_ratio)
    biased_indices = random.sample(range(num_satellites), num_biased)
    bias_magnitude = np.random.uniform(20, 50, num_biased)  # 降低偏差範圍
    noise[biased_indices] += bias_magnitude
    
    pseudo_ranges = true_ranges + noise
    
    # 正規化偽距
    pseudo_ranges = (pseudo_ranges - np.mean(pseudo_ranges)) / np.std(pseudo_ranges)
    return satellite_positions, pseudo_ranges, biased_indices, receiver_pos

def compute_residual_matrix(satellite_positions, pseudo_ranges):
    """
    計算正規化的殘差矩陣。
    """
    num_satellites = len(satellite_positions)
    residual_matrix = np.zeros((num_satellites, num_satellites))
    
    for i in range(num_satellites):
        subset_indices = [j for j in range(num_satellites) if j != i]
        subset_positions = satellite_positions[subset_indices]
        estimated_position = np.average(subset_positions, axis=0)
        
        for j in range(num_satellites):
            if j == i:
                residual_matrix[i, j] = 0  # 改用0替代無窮大
            else:
                predicted_range = np.linalg.norm(satellite_positions[j] - estimated_position)
                residual_matrix[i, j] = pseudo_ranges[j] - predicted_range
    
    # 正規化殘差矩陣
    residual_matrix = (residual_matrix - np.mean(residual_matrix)) / (np.std(residual_matrix) + 1e-8)
    return residual_matrix

def weighted_least_squares(satellite_positions, pseudo_ranges, weights):
    """
    加權最小二乘 (WLS) 定位。
    """
    W = np.diag(weights)
    A = np.hstack([satellite_positions, np.ones((len(satellite_positions), 1))])
    b = pseudo_ranges.reshape(-1, 1)
    
    x_hat = np.linalg.inv(A.T @ W @ A) @ (A.T @ W @ b)
    return x_hat[:3].flatten()

def build_lstm_model(input_shape):
    """
    建立改進的 LSTM 模型。
    """
    model = Sequential([
        # 第一個 LSTM 層
        LSTM(256, return_sequences=True, input_shape=input_shape,
             kernel_initializer='he_normal',
             recurrent_initializer='orthogonal'),
        BatchNormalization(),
        Dropout(0.3),
        
        # 第二個 LSTM 層
        LSTM(128, return_sequences=False,
             kernel_initializer='he_normal',
             recurrent_initializer='orthogonal'),
        BatchNormalization(),
        Dropout(0.3),
        
        # 全連接層
        Dense(128, activation='relu', kernel_initializer='he_normal'),
        BatchNormalization(),
        Dropout(0.3),
        
        # 輸出層
        Dense(60, activation='sigmoid')  # 改用sigmoid作為輸出層
    ])
    
    # 使用二元交叉熵作為損失函數
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model

# 生成訓練數據
num_samples = 2000  # 增加訓練樣本
X_train = []
y_train = []

for _ in range(num_samples):
    sat_positions, pseudo_ranges, biased_sats, receiver_pos = generate_gnss_data()
    residual_matrix = compute_residual_matrix(sat_positions, pseudo_ranges)
    X_train.append(residual_matrix)
    
    # 創建二元標籤
    target_weights = np.ones(60)
    target_weights[biased_sats] = 0
    y_train.append(target_weights)

X_train = np.array(X_train).reshape((num_samples, 60, 60))
y_train = np.array(y_train)

# 建立和訓練模型
lstm_model = build_lstm_model((60, 60))
history = lstm_model.fit(
    X_train, y_train,
    epochs=20,
    batch_size=32,
    validation_split=0.2,
    shuffle=True,
    verbose=1
)

# 測試模型性能
sat_positions, pseudo_ranges, biased_sats, receiver_pos = generate_gnss_data()
residual_matrix = compute_residual_matrix(sat_positions, pseudo_ranges)
X_test = residual_matrix.reshape((1, 60, 60))
predicted_weights = lstm_model.predict(X_test).flatten()
predicted_weights = (predicted_weights - np.min(predicted_weights)) / (np.max(predicted_weights) - np.min(predicted_weights))  # 重新正規化

# 使用預測權重進行定位
estimated_position = weighted_least_squares(sat_positions, pseudo_ranges, predicted_weights)
error_distance = np.linalg.norm(estimated_position - receiver_pos)
print(f"\n估算位置: {estimated_position}")
print(f"定位誤差距離: {error_distance * 1000:.2f} m")

end_time = time.time()
print(f"程式總執行時間: {end_time - start_time:.2f} 秒")
# 視覺化結果
plt.figure(figsize=(15, 5))

# 繪製訓練歷史
plt.subplot(121)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss During Training')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# 繪製3D定位結果
ax = plt.subplot(122, projection='3d')
ax.scatter(sat_positions[:, 0], sat_positions[:, 1], sat_positions[:, 2], 
          c=predicted_weights, cmap='coolwarm', label='Satellites')
ax.scatter(0, 0, 0, color='green', marker='x', s=100, label='True Position')
ax.scatter(estimated_position[0], estimated_position[1], estimated_position[2], 
          color='red', marker='o', s=100, label='Estimated Position')
ax.set_xlabel('X (m)')
ax.set_ylabel('Y (m)')
ax.set_zlabel('Z (m)')
ax.legend()

plt.tight_layout()
plt.show()
