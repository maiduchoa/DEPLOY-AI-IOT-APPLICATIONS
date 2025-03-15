import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, classification_report
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime
import time
import json
import logging
from tensorflow.keras.models import load_model
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
from tensorflow.keras.layers import Layer

# Thiết lập logging
def setup_logging(results_dir):
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f'{results_dir}/model_evaluation.log'),
            logging.StreamHandler()
        ]
    )

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

def create_sequences(data, seq_length=32, stride=16):
    if len(data) < seq_length:
        raise ValueError(f"Dữ liệu quá ngắn ({len(data)}) so với seq_length ({seq_length})")
        
    sequences = []
    for i in range(0, len(data) - seq_length + 1, stride):
        sequences.append(data[i:i + seq_length])
    result = np.array(sequences)
    logging.info(f"Đã tạo {len(result)} sequences với shape {result.shape}")
    return result

def load_and_preprocess_data():
    if not os.path.exists('processed_data'):
        raise FileNotFoundError("Thư mục 'processed_data' không tồn tại")
        
    processed_data = []
    total_samples = 0
    
    files_found = False
    for file in os.listdir('processed_data'):
        if file.startswith('processed_') and file.endswith('.csv'):
            files_found = True
            try:
                df = pd.read_csv(f'processed_data/{file}')
                total_samples += len(df)
                logging.info(f"Đã đọc file: {file} ({len(df):,} mẫu)")
                processed_data.append(df)
            except Exception as e:
                logging.error(f"Lỗi khi đọc file {file}: {str(e)}")
    
    if not files_found:
        raise FileNotFoundError("Không tìm thấy file dữ liệu phù hợp trong processed_data")
        
    if not processed_data:
        raise ValueError("Không có dữ liệu nào được tải thành công")
    
    df = pd.concat(processed_data, ignore_index=True)
    logging.info(f"Tổng số mẫu: {total_samples:,}")
    
    features = df[['AccelX', 'AccelY', 'AccelZ', 'GyroX', 'GyroY', 'GyroZ']].values
    labels = df['ActivityLabel'].values
    
    logging.info("Chuẩn hóa dữ liệu...")
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    logging.info("Tạo sequences...")
    X = create_sequences(features_scaled)
    y = []
    for i in range(0, len(labels) - 32 + 1, 16):
        y.append(labels[i + 32 - 1])
    y = np.array(y)
    
    logging.info(f"Shape của dữ liệu sau khi tạo sequences: X={X.shape}, y={y.shape}")
    return X, y

def calculate_flops(model, input_shape):
    try:
        concrete = tf.function(lambda x: model(x))
        concrete_func = concrete.get_concrete_function(tf.TensorSpec(input_shape, tf.float32))
        frozen_func = convert_variables_to_constants_v2(concrete_func)
        graph_def = frozen_func.graph.as_graph_def()
        
        with tf.Graph().as_default() as graph:
            tf.graph_util.import_graph_def(graph_def, name='')
            run_meta = tf.compat.v1.RunMetadata()
            opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()
            flops = tf.compat.v1.profiler.profile(graph=graph, run_meta=run_meta, cmd="scope", options=opts)
            return flops.total_float_ops
    except Exception as e:
        logging.error(f"Không thể tính FLOPS: {str(e)}")
        return None

def measure_inference_time(model, X_test, num_runs=100, batch_size=32):
    logging.info("Đo thời gian inference...")
    sample_data = X_test[:batch_size]
    
    logging.info("Warmup...")
    _ = model.predict(sample_data, batch_size=batch_size, verbose=0)
    
    logging.info(f"Đang đo {num_runs} lần với batch_size={batch_size}...")
    times = []
    for i in range(num_runs):
        start_time = time.time()
        _ = model.predict(sample_data, batch_size=batch_size, verbose=0)
        end_time = time.time()
        times.append((end_time - start_time) * 1000 / batch_size)  # ms per sample
    
    return {
        'average_ms_per_sample': np.mean(times),
        'std_ms': np.std(times),
        'min_ms': np.min(times),
        'max_ms': np.max(times),
        'median_ms': np.median(times),
        'batch_size': batch_size
    }

def plot_confusion_matrix(y_true, y_pred, class_names, save_path):
    plt.figure(figsize=(12, 10))
    cm = confusion_matrix(y_true, y_pred)
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    
    sns.heatmap(cm_percent, annot=True, fmt='.1f', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    
    plt.title('Ma trận nhầm lẫn (%)')
    plt.xlabel('Dự đoán')
    plt.ylabel('Thực tế')
    plt.xticks(rotation=45)
    plt.yticks(rotation=45)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    return cm, cm_percent

def plot_tsne(features, labels, class_names, save_path):
    logging.info("Đang tính toán t-SNE...")
    tsne = TSNE(n_components=2, random_state=42, n_jobs=-1)
    features_2d = tsne.fit_transform(features)
    
    plt.figure(figsize=(12, 10))
    for i, label in enumerate(class_names):
        mask = labels == i
        plt.scatter(features_2d[mask, 0], features_2d[mask, 1], 
                   label=label, alpha=0.6, s=50)
    
    plt.title('Biểu đồ t-SNE')
    plt.xlabel('t-SNE 1')
    plt.ylabel('t-SNE 2')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    return features_2d

def get_model_parameters(model):
    try:
        trainable_params = np.sum([np.prod(v.shape) for v in model.trainable_variables])
        non_trainable_params = np.sum([np.prod(v.shape) for v in model.non_trainable_variables])
        return {
            'total_parameters': int(trainable_params + non_trainable_params),
            'trainable_parameters': int(trainable_params),
            'non_trainable_parameters': int(non_trainable_params)
        }
    except Exception as e:
        logging.error(f"Không thể tính số tham số của model: {str(e)}")
        return {'total_parameters': 0, 'trainable_parameters': 0, 'non_trainable_parameters': 0}

def save_model_metrics():
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_dir = f'model_metrics_{timestamp}'
        os.makedirs(results_dir, exist_ok=True)
        
        setup_logging(results_dir)
        logging.info(f"Bắt đầu đánh giá model - Kết quả lưu tại: {results_dir}")
        
        logging.info("1. Đang load và tiền xử lý dữ liệu...")
        X_test, y_test = load_and_preprocess_data()
        
        logging.info("2. Xây dựng model...")
        input_shape = (32, 6)
        num_classes = len(np.unique(y_test))
        weights_path = 'processed_data/transformer_model.h5'
        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"Không tìm thấy file weights tại {weights_path}")
            
        model = build_transformer_model(input_shape, num_classes)
        model.load_weights(weights_path)
        
        logging.info("3. Thực hiện dự đoán...")
        y_pred = model.predict(X_test, batch_size=32, verbose=1)
        y_pred_classes = np.argmax(y_pred, axis=1)
        
        mapping_path = 'processed_data/activity_mapping.csv'
        if not os.path.exists(mapping_path):
            raise FileNotFoundError(f"Không tìm thấy file activity_mapping tại {mapping_path}")
        activity_mapping = pd.read_csv(mapping_path)
        class_names = activity_mapping['activity'].tolist()
        
        logging.info("5. Tính toán metrics...")
        metrics = {}
        accuracy = float(np.mean(y_test == y_pred_classes))
        precision, recall, f1, support = precision_recall_fscore_support(y_test, y_pred_classes, average='weighted')
        precision_per_class, recall_per_class, f1_per_class, support_per_class = \
            precision_recall_fscore_support(y_test, y_pred_classes, average=None)
        classification_rep = classification_report(y_test, y_pred_classes, target_names=class_names, output_dict=True)
        cm = confusion_matrix(y_test, y_pred_classes)
        cm_percent = (cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100).tolist()
        
        metrics.update({
            'accuracy': float(accuracy),
            'precision_weighted': float(precision),
            'recall_weighted': float(recall),
            'f1_score_weighted': float(f1),
            'per_class_metrics': {
                class_names[i]: {
                    'precision': float(precision_per_class[i]),
                    'recall': float(recall_per_class[i]),
                    'f1_score': float(f1_per_class[i]),
                    'support': int(support_per_class[i])
                } for i in range(len(class_names))
            },
            'confusion_matrix': {'raw': cm.tolist(), 'percentage': cm_percent},
            'classification_report': classification_rep
        })
        
        logging.info("6. Đo thời gian inference...")
        inference_times = measure_inference_time(model, X_test)
        metrics['inference_time'] = inference_times
        
        logging.info("7. Tính toán FLOPS...")
        flops = calculate_flops(model, (None, 32, 6))
        metrics['flops'] = int(flops) if flops is not None else "Không thể tính toán"
        
        logging.info("8. Lấy thông số model...")
        model_params = get_model_parameters(model)
        metrics['model_parameters'] = model_params
        
        logging.info("9. Tạo visualizations...")
        plot_confusion_matrix(y_test, y_pred_classes, class_names, f'{results_dir}/confusion_matrix.png')
        if len(X_test) < 10000:
            plot_tsne(X_test.reshape(X_test.shape[0], -1), y_test, class_names, f'{results_dir}/tsne_plot.png')
        
        logging.info("10. Lưu báo cáo...")
        metrics['data_shape'] = {'X_test': X_test.shape, 'y_test': y_test.shape}
        metrics['class_names'] = class_names
        
        with open(f'{results_dir}/metrics.json', 'w', encoding='utf-8') as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)
        
        logging.info("\n=== Tóm tắt kết quả chính ===")
        logging.info(f"Accuracy: {metrics['accuracy']:.4f}")
        logging.info(f"Precision (weighted): {metrics['precision_weighted']:.4f}")
        logging.info(f"Recall (weighted): {metrics['recall_weighted']:.4f}")
        logging.info(f"F1-score (weighted): {metrics['f1_score_weighted']:.4f}")
        logging.info(f"Thời gian inference trung bình: {metrics['inference_time']['average_ms_per_sample']:.2f}ms/sample")
        logging.info(f"Số tham số model: {metrics['model_parameters']['total_parameters']:,}")
        
        return metrics
        
    except Exception as e:
        logging.error(f"Lỗi tổng thể trong quá trình đánh giá: {str(e)}")
        raise

if __name__ == "__main__":
    try:
        metrics = save_model_metrics()
    except Exception as e:
        print(f"Đánh giá model thất bại: {str(e)}")