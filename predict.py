import socket
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Layer
from sklearn.preprocessing import StandardScaler
import json
import threading
import queue
import time
import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from collections import deque
import pandas as pd

class TransformerBlock(Layer):
    def __init__(self, embed_dim=6, num_heads=4, ff_dim=32, rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.rate = rate
        
        self.att = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = tf.keras.Sequential([
            tf.keras.layers.Dense(ff_dim, activation="relu"),
            tf.keras.layers.Dense(embed_dim),
        ])
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)
        
    def call(self, inputs, training=None):
        attn_output = self.att(inputs, inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)
        
    def get_config(self):
        config = super().get_config()
        config.update({
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "ff_dim": self.ff_dim,
            "rate": self.rate
        })
        return config

def build_transformer_model(input_shape, num_classes):
    inputs = tf.keras.Input(shape=input_shape)
    positions = tf.range(start=0, limit=input_shape[0], delta=1)
    pos_encoding = tf.keras.layers.Embedding(input_dim=input_shape[0], output_dim=input_shape[1])(positions)
    x = inputs + pos_encoding
    
    transformer_block1 = TransformerBlock(input_shape[1], 4, 32)
    transformer_block2 = TransformerBlock(input_shape[1], 4, 32)
    
    x = transformer_block1(x)
    x = transformer_block2(x)
    
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    x = tf.keras.layers.Dense(64, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)
    
    return tf.keras.Model(inputs=inputs, outputs=outputs)

class ActivityPredictor:
    def __init__(self, host='192.168.52.147', port=8080, buffer_size=128):
        self.running = True
        self.connected = False
        self.server_socket = None
        self.host = host
        self.port = port
        self.buffer_size = buffer_size
        
        self.sequence_length = 32
        self.stride = 8
        self.min_confidence = 0.3
        self.prediction_threshold = 0.3
        self.smoothing_window = 7
        self.samples_per_second = 50
        self.prediction_window = 2
        self.samples_needed = self.samples_per_second * self.prediction_window
        
        self.raw_buffer = deque(maxlen=self.buffer_size)
        self.sequence_buffer = deque(maxlen=self.sequence_length + self.stride)
        self.prediction_buffer = deque(maxlen=self.smoothing_window)
        self.data_queue = queue.Queue(maxsize=10)
        self.last_prediction_time = 0
        
        self.inference_times = deque(maxlen=50)
        
        self._init_socket()
        
        self.time_data = np.arange(self.buffer_size)
        self.accel_data = np.zeros((self.buffer_size, 3))
        self.gyro_data = np.zeros((self.buffer_size, 3))
        
        self._load_activity_mapping()
        self._load_scaler()
        self._load_model()
        
        self.setup_gui()
        
        self.receiver_thread = threading.Thread(target=self.receive_data)
        self.predictor_thread = threading.Thread(target=self.predict_activity)
    
    def _init_socket(self):
        # Sử dụng UDP socket thay vì TCP
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        for port in range(8080, 8090):
            try:
                self.port = port
                self.server_socket.bind((self.host, port))
                print(f"Server UDP đang lắng nghe tại {self.host}:{port}")
                break
            except Exception as e:
                if port == 8089:
                    raise Exception("Không thể tìm thấy port khả dụng")
                continue
    
    def _load_model(self):
        print("Đang load model...")
        try:
            input_shape = (self.sequence_length, 6)
            self.model = build_transformer_model(input_shape, self.num_classes)
            self.model.load_weights('processed_data/transformer_model.h5')
            print("Warm up model...")
            dummy_input = np.zeros((1, self.sequence_length, 6))
            self.model.predict(dummy_input, verbose=0)
            print("Đã load model thành công!")
            self.model.summary()
        except Exception as e:
            print(f"Lỗi khi load model: {e}, sử dụng model mặc định cho test")
            self.model = build_transformer_model((self.sequence_length, 6), self.num_classes)

    def _load_activity_mapping(self):
        print("Đang load activity mapping...")
        try:
            activity_df = pd.read_csv('processed_data/activity_mapping.csv')
            activity_df = activity_df.sort_values('numeric_label')
            self.activities = activity_df['activity'].tolist()
            self.num_classes = len(self.activities)
            print("Các hoạt động được nhận dạng:")
            for i, activity in enumerate(self.activities):
                print(f"{i}: {activity}")
            if self.num_classes != 6:
                raise ValueError(f"Expected 6 classes, but got {self.num_classes}")
        except Exception as e:
            print(f"Lỗi khi đọc activity mapping: {e}")
            self.activities = ['standing', 'sitting', 'walking', 'jogging', 'falling', 'jumping']
            self.num_classes = 6
    
    def _load_scaler(self):
        try:
            scaler_data = np.load('processed_data/scaler_params.npz')
            self.scaler = StandardScaler()
            self.scaler.mean_ = scaler_data['mean']
            self.scaler.scale_ = scaler_data['scale']
            print("Đã load scaler từ file")
        except:
            print("Không tìm thấy scaler, khởi tạo scaler mặc định")
            self.scaler = StandardScaler()
            dummy_data = np.random.randn(1000, 6)
            self.scaler.fit(dummy_data)
            self.need_fit_scaler = False
    
    def setup_gui(self):
        self.root = tk.Tk()
        self.root.title("Dự đoán hoạt động theo thời gian thực")
        self.root.geometry("800x900")
        
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        ttk.Label(main_frame, text="Hoạt động hiện tại:", font=('Arial', 14)).grid(row=0, column=0, pady=10)
        self.activity_label = ttk.Label(main_frame, text="Đang chờ...", font=('Arial', 20, 'bold'))
        self.activity_label.grid(row=1, column=0, pady=10)
        
        graph_frame = ttk.Frame(main_frame)
        graph_frame.grid(row=2, column=0, pady=20)
        
        self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(8, 6))
        self.canvas = FigureCanvasTkAgg(self.fig, master=graph_frame)
        self.canvas.get_tk_widget().pack()
        
        self.accel_lines = []
        for i, color in enumerate(['r', 'g', 'b']):
            line, = self.ax1.plot(self.time_data, self.accel_data[:, i], color, label=f'Axis {i}')
            self.accel_lines.append(line)
        self.ax1.set_title('Gia tốc')
        self.ax1.set_ylabel('m/s²')
        self.ax1.legend(['X', 'Y', 'Z'])
        self.ax1.grid(True)
        
        self.gyro_lines = []
        for i, color in enumerate(['r', 'g', 'b']):
            line, = self.ax2.plot(self.time_data, self.gyro_data[:, i], color, label=f'Axis {i}')
            self.gyro_lines.append(line)
        self.ax2.set_title('Gyro')
        self.ax2.set_ylabel('rad/s')
        self.ax2.set_xlabel('Thời gian')
        self.ax2.legend(['X', 'Y', 'Z'])
        self.ax2.grid(True)
        
        control_frame = ttk.Frame(main_frame)
        control_frame.grid(row=3, column=0, pady=10)
        
        self.start_button = ttk.Button(control_frame, text="Bắt đầu", command=self.start)
        self.start_button.pack(side=tk.LEFT, padx=5)
        
        self.stop_button = ttk.Button(control_frame, text="Dừng", command=self.stop)
        self.stop_button.pack(side=tk.LEFT, padx=5)
        self.stop_button.state(['disabled'])
        
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.update_gui()
    
    def update_plots(self):
        try:
            for i, line in enumerate(self.accel_lines):
                line.set_ydata(self.accel_data[:, i])
            self.ax1.relim()
            self.ax1.autoscale_view()
            
            for i, line in enumerate(self.gyro_lines):
                line.set_ydata(self.gyro_data[:, i])
            self.ax2.relim()
            self.ax2.autoscale_view()
            
            self.canvas.draw()
        except Exception as e:
            print(f"Lỗi khi cập nhật biểu đồ: {e}")
    
    def update_gui(self):
        if self.running:
            try:
                self.update_plots()
                self.root.after(100, self.update_gui)
            except Exception as e:
                print(f"Lỗi khi cập nhật GUI: {e}")
    
    def receive_data(self):
        print("Server UDP đang chờ dữ liệu từ ESP32...")
        buffer = ""
        data_points_collected = 0
        collection_start_time = time.time()
        
        try:
            while self.running:
                try:
                    # Nhận dữ liệu UDP
                    data, addr = self.server_socket.recvfrom(1024)  # Không cần accept() như TCP
                    data = data.decode()
                    if not data:
                        continue
                    
                    print(f"Dữ liệu thô nhận được từ {addr}: {data}")
                    buffer += data
                    
                    while '\n' in buffer:
                        line, buffer = buffer.split('\n', 1)
                        try:
                            sensor_data = json.loads(line)
                            data_point = [
                                sensor_data['ax'], sensor_data['ay'], sensor_data['az'],
                                sensor_data['gx'], sensor_data['gy'], sensor_data['gz']
                            ]
                            print(f"Raw sensor data: {data_point}")
                            self.raw_buffer.append(data_point)
                            data_points_collected += 1
                            
                            if len(self.raw_buffer) == self.buffer_size:
                                data_array = np.array(self.raw_buffer)
                                self.accel_data = data_array[:, :3]
                                self.gyro_data = data_array[:, 3:]
                            
                            if data_points_collected >= self.samples_needed:
                                data_array = np.array(list(self.raw_buffer)[-self.samples_needed:])
                                self.data_queue.put(data_array)
                                data_points_collected = 0
                                collection_start_time = time.time()
                                
                        except json.JSONDecodeError:
                            print("Lỗi khi đọc dữ liệu JSON")
                            continue
                
                except Exception as e:
                    print(f"Lỗi khi nhận dữ liệu: {e}")
                    # Không thoát vòng lặp vì UDP không cần kết nối liên tục
                    
            self.server_socket.close()
        
        except Exception as e:
            print(f"Không nhận được dữ liệu, mô phỏng dữ liệu: {e}")
            while self.running:
                fake_data = {
                    "ax": np.random.randn(), "ay": np.random.randn(), "az": np.random.randn(),
                    "gx": np.random.randn(), "gy": np.random.randn(), "gz": np.random.randn()
                }
                line = json.dumps(fake_data) + '\n'
                buffer = line
                try:
                    sensor_data = json.loads(line)
                    data_point = [
                        sensor_data['ax'], sensor_data['ay'], sensor_data['az'],
                        sensor_data['gx'], sensor_data['gy'], sensor_data['gz']
                    ]
                    self.raw_buffer.append(data_point)
                    if len(self.raw_buffer) >= self.samples_needed:
                        data_array = np.array(list(self.raw_buffer)[-self.samples_needed:])
                        self.data_queue.put(data_array)
                        self.accel_data = data_array[:, :3]
                        self.gyro_data = data_array[:, 3:]
                except Exception as e:
                    print(f"Lỗi mô phỏng: {e}")
                time.sleep(0.02)
    
    def predict_activity(self):
        batch_size = 4
        while self.running:
            try:
                if not self.data_queue.empty():
                    data = self.data_queue.get()
                    print(f"Received data shape: {data.shape}")
                    
                    for point in data:
                        self.sequence_buffer.append(point)
                    
                    if len(self.sequence_buffer) >= self.sequence_length:
                        start_time = time.time()
                        sequences = []
                        for i in range(0, len(self.sequence_buffer) - self.sequence_length + 1, self.stride):
                            seq = np.array(list(self.sequence_buffer)[i:i + self.sequence_length])
                            sequences.append(seq)
                        
                        if sequences:
                            sequences = np.array(sequences)
                            sequences_scaled = self.scaler.transform(
                                sequences.reshape(-1, 6)
                            ).reshape(-1, self.sequence_length, 6)
                            print(f"Input shape to model: {sequences_scaled.shape}")
                            
                            predictions = []
                            confidences = []
                            weights = np.linspace(0.5, 1.0, len(sequences_scaled))
                            
                            for i in range(0, len(sequences_scaled), batch_size):
                                batch = sequences_scaled[i:i + batch_size]
                                preds = self.model.predict(batch, verbose=0)
                                batch_weights = weights[i:i + batch_size]
                                
                                for pred, weight in zip(preds, batch_weights):
                                    activity_idx = np.argmax(pred)
                                    confidence = pred[activity_idx] * weight
                                    predictions.append(activity_idx)
                                    confidences.append(confidence)
                            
                            final_pred = predictions[-1]
                            final_conf = confidences[-1]
                            
                            inference_time = time.time() - start_time
                            self.inference_times.append(inference_time)
                            
                            self.prediction_buffer.append((final_pred, final_conf))
                            
                            if len(self.prediction_buffer) >= self.smoothing_window:
                                pred_counts = {}
                                conf_sums = {}
                                total_weight = 0
                                
                                weights = np.linspace(0.6, 1.0, len(self.prediction_buffer))
                                for (pred, conf), weight in zip(self.prediction_buffer, weights):
                                    pred_counts[pred] = pred_counts.get(pred, 0) + weight
                                    conf_sums[pred] = conf_sums.get(pred, 0) + (conf * weight)
                                    total_weight += weight
                                
                                for pred in pred_counts:
                                    pred_counts[pred] /= total_weight
                                
                                max_count = max(pred_counts.values())
                                if max_count >= self.prediction_threshold:
                                    max_classes = [c for c, count in pred_counts.items() 
                                                 if count >= max_count * 0.9]
                                    final_class = max(max_classes, 
                                                     key=lambda c: conf_sums[c]/pred_counts[c])
                                    avg_confidence = conf_sums[final_class] / (pred_counts[final_class] * total_weight)
                                    
                                    if avg_confidence >= self.min_confidence:
                                        avg_infer_time = np.mean(self.inference_times) * 1000
                                        activity_name = self.activities[final_class]
                                        stability = max_count * avg_confidence
                                        stability_text = "Cao" if stability > 0.8 else "Trung bình" if stability > 0.6 else "Thấp"
                                        
                                        print(f"\033[92mHoạt động: {activity_name} "
                                              f"(độ tin cậy: {avg_confidence*100:.1f}%, "
                                              f"độ ổn định: {stability_text}, "
                                              f"inference: {avg_infer_time:.1f}ms)\033[0m")
                                        
                                        self.activity_label.config(
                                            text=f"{activity_name}\n"
                                                 f"Confidence: {avg_confidence*100:.1f}%\n"
                                                 f"Độ ổn định: {stability_text}\n"
                                                 f"Inference: {avg_infer_time:.1f}ms"
                                        )
                                        self.root.update_idletasks()
                                    else:
                                        print(f"\033[93mKhông chắc chắn "
                                              f"(độ tin cậy: {avg_confidence*100:.1f}%)\033[0m")
                                        self.activity_label.config(text="Không chắc chắn")
                                        self.root.update_idletasks()
                
                time.sleep(0.01)
            
            except Exception as e:
                print(f"Lỗi khi dự đoán: {e}")
                continue
    
    def start(self):
        try:
            self.running = True
            self.receiver_thread = threading.Thread(target=self.receive_data)
            self.predictor_thread = threading.Thread(target=self.predict_activity)
            self.receiver_thread.start()
            self.predictor_thread.start()
            self.start_button.state(['disabled'])
            self.stop_button.state(['!disabled'])
        except Exception as e:
            print(f"Lỗi khi bắt đầu: {e}")
    
    def stop(self):
        try:
            self.running = False
            self.start_button.state(['!disabled'])
            self.stop_button.state(['disabled'])
        except Exception as e:
            print(f"Lỗi khi dừng: {e}")
    
    def on_closing(self):
        try:
            self.running = False
            self.server_socket.close()
            self.root.quit()
            self.root.destroy()
        except Exception as e:
            print(f"Lỗi khi đóng ứng dụng: {e}")
            
    def run(self):
        try:
            self.root.mainloop()
        except Exception as e:
            print(f"Lỗi khi chạy ứng dụng: {e}")

if __name__ == "__main__":
    try:
        predictor = ActivityPredictor()
        predictor.run()
    except Exception as e:
        print(f"Lỗi chính: {e}")