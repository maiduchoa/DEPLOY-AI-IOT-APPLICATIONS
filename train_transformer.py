import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, Callback
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os
from scipy import ndimage
import random

# Callback để xử lý việc dừng training an toàn
class SafeInterruptCallback(Callback):
    def __init__(self):
        super().__init__()
        self.interrupted = False
        
    def on_epoch_end(self, epoch, logs=None):
        try:
            if self.interrupted:
                print('\nĐang dừng training một cách an toàn...')
                self.model.stop_training = True
        except KeyboardInterrupt:
            self.interrupted = True
            print('\nNhấn Ctrl+C lần nữa để dừng ngay lập tức.')

def add_noise(data, noise_factor=0.05):
    """Add random noise to sequences"""
    noise = np.random.normal(0, noise_factor, data.shape)
    return data + noise

def time_warp(data, sigma=0.2):
    """Apply time warping to sequences"""
    warped = np.zeros_like(data)
    for i in range(data.shape[0]):
        warped[i] = ndimage.gaussian_filter1d(data[i], sigma=sigma)
    return warped

def magnitude_scale(data, sigma=0.1):
    """Apply random scaling to sequences"""
    scale = np.random.normal(1, sigma, size=(data.shape[0], 1, 1))
    return data * scale

def create_sequences(data, seq_length=32, stride=16):
    """Create sequences with overlap"""
    sequences = []
    for i in range(0, len(data) - seq_length + 1, stride):
        sequences.append(data[i:i + seq_length])
    return np.array(sequences)

def augment_data(X, y):
    """Augment sequence data with multiple techniques"""
    print("Augmenting data...")
    X_aug = []
    y_aug = []
    
    # Original data
    X_aug.append(X)
    y_aug.append(y)
    
    # Noisy data
    X_noise = add_noise(X.copy())
    X_aug.append(X_noise)
    y_aug.append(y)
    
    # Time warped data
    X_warp = time_warp(X.copy())
    X_aug.append(X_warp)
    y_aug.append(y)
    
    # Magnitude scaled data
    X_scale = magnitude_scale(X.copy())
    X_aug.append(X_scale)
    y_aug.append(y)
    
    # Combine augmentations
    X_combined = np.concatenate(X_aug)
    y_combined = np.concatenate(y_aug)
    
    # Shuffle
    indices = np.arange(len(X_combined))
    np.random.shuffle(indices)
    
    return X_combined[indices], y_combined[indices]

class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim
        )
        self.ffn = tf.keras.Sequential([
            tf.keras.layers.Dense(ff_dim, activation="relu"),
            tf.keras.layers.Dense(embed_dim),
        ])
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)
        
    def call(self, inputs, training=False):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

def build_transformer_model(input_shape, num_classes):
    """Build Transformer model for time series classification"""
    inputs = tf.keras.Input(shape=input_shape)
    
    # Position encoding
    positions = tf.range(start=0, limit=input_shape[0], delta=1)
    pos_encoding = tf.keras.layers.Embedding(input_dim=input_shape[0], output_dim=input_shape[1])(positions)
    x = inputs + pos_encoding
    
    # Transformer blocks
    transformer_block1 = TransformerBlock(input_shape[1], 4, 32)
    transformer_block2 = TransformerBlock(input_shape[1], 4, 32)
    
    x = transformer_block1(x)
    x = transformer_block2(x)
    
    # Global average pooling
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    
    # Dense layers
    x = tf.keras.layers.Dense(64, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)
    
    return tf.keras.Model(inputs=inputs, outputs=outputs)

# Đọc và xử lý dữ liệu
print("Đang đọc dữ liệu...")
PROCESSED_DATA_FOLDER = "processed_data"
processed_data = []

for file in os.listdir(PROCESSED_DATA_FOLDER):
    if file.startswith("processed_") and file.endswith(".csv"):
        file_path = os.path.join(PROCESSED_DATA_FOLDER, file)
        df = pd.read_csv(file_path)
        processed_data.append(df)
        print(f"Đã đọc file: {file} ({len(df)} mẫu)")

# Gộp dữ liệu
df = pd.concat(processed_data, ignore_index=True)
print(f"\nTổng số mẫu: {len(df)}")

print("\nPhân bố ActivityLabel:")
print(df['ActivityLabel'].value_counts())

# Chuẩn bị dữ liệu
X = df[['AccelX', 'AccelY', 'AccelZ', 'GyroX', 'GyroY', 'GyroZ']].values
y = df['ActivityLabel'].values

# Chuẩn hóa dữ liệu
print("\nChuẩn hóa dữ liệu...")
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Tạo sequences
seq_length = 32  # Độ dài chuỗi
stride = 16  # Stride cho việc tạo sequences
print(f"\nTạo sequences (độ dài = {seq_length}, stride = {stride})...")

# Tạo sequences cho dữ liệu X
sequences = []
labels = []
for i in range(0, len(X) - seq_length + 1, stride):
    sequences.append(X[i:i + seq_length])
    labels.append(y[i + seq_length - 1])  # Lấy nhãn của điểm cuối cùng trong sequence

X = np.array(sequences)
y = np.array(labels)

print(f"Shape của dữ liệu sau khi tạo sequences: X={X.shape}, y={y.shape}")

# Chia dữ liệu
print("\nChia dữ liệu...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Augment dữ liệu training
print("\nAugment dữ liệu training...")
X_train_aug, y_train_aug = augment_data(X_train, y_train)
print(f"Shape của dữ liệu train sau khi augment: {X_train_aug.shape}")

# Xây dựng và biên dịch mô hình
print("\nXây dựng mô hình Transformer...")
model = build_transformer_model(
    input_shape=(seq_length, 6),
    num_classes=len(np.unique(y))
)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Callbacks
early_stop = EarlyStopping(
    monitor='val_accuracy',
    patience=15,
    restore_best_weights=True,
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_accuracy',
    factor=0.5,
    patience=5,
    min_lr=1e-6,
    verbose=1
)

safe_interrupt = SafeInterruptCallback()

# Huấn luyện mô hình
print("\nBắt đầu huấn luyện... (Nhấn Ctrl+C để dừng an toàn)")
try:
    history = model.fit(
        X_train_aug, y_train_aug,
        validation_data=(X_test, y_test),
        epochs=50,
        batch_size=64,
        callbacks=[early_stop, reduce_lr, safe_interrupt],
        verbose=1
    )
    
    print("\nLưu mô hình...")
    model.save('processed_data/transformer_model.h5')
    print("✅ Đã lưu mô hình!")
    
except KeyboardInterrupt:
    print("\nĐã dừng training.")
    
    if len(history.history['loss']) > 0:
        print("Lưu mô hình trạng thái cuối cùng...")
        model.save('processed_data/transformer_model2.h5')
        print("✅ Đã lưu mô hình!")

# Đánh giá mô hình
print("\nĐánh giá mô hình...")
test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"Test accuracy: {test_accuracy:.4f}")

# In ma trận nhầm lẫn và báo cáo phân loại
from sklearn.metrics import confusion_matrix, classification_report
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)

print("\nMa trận nhầm lẫn:")
print(confusion_matrix(y_test, y_pred_classes))

print("\nBáo cáo phân loại chi tiết:")
print(classification_report(y_test, y_pred_classes))

# In thông tin về các nhãn
activity_mapping = pd.read_csv(os.path.join(PROCESSED_DATA_FOLDER, 'activity_mapping.csv'))
print("\nMapping các hoạt động:")
print(activity_mapping) 