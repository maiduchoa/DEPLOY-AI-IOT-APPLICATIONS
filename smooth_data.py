import os
import pandas as pd
import numpy as np
from scipy.signal import savgol_filter

# Thư mục chứa các file CSV
RAW_DATA_FOLDER = "raw_data"
PROCESSED_DATA_FOLDER = "processed_data"
os.makedirs(PROCESSED_DATA_FOLDER, exist_ok=True)

# Định nghĩa mapping cho các hoạt động
ACTIVITY_MAPPING = {
    'standing': 0,
    'sitting': 1,
    'walking': 2,
    'jogging': 3,
    'falling': 4,
    'jumping': 5
}

# Hàm làm mịn dữ liệu
def smooth_data(data, window_size=5, poly_order=2):
    """
    Dùng Savitzky-Golay filter để làm mịn dữ liệu.
    :param data: Dữ liệu cần làm mịn.
    :param window_size: Kích thước cửa sổ (odd number).
    :param poly_order: Bậc của đa thức.
    :return: Dữ liệu đã được làm mịn.
    """
    return savgol_filter(data, window_size, poly_order)

# In ra mapping hoạt động
print("\nActivity mapping:")
for activity, value in ACTIVITY_MAPPING.items():
    print(f"{activity}: {value}")

# Lưu mapping hoạt động vào file để tham khảo sau này
activity_mapping_df = pd.DataFrame({
    'activity': list(ACTIVITY_MAPPING.keys()),
    'numeric_label': list(ACTIVITY_MAPPING.values())
})
mapping_file = os.path.join(PROCESSED_DATA_FOLDER, 'activity_mapping.csv')
activity_mapping_df.to_csv(mapping_file, index=False)
print(f"\n✅ Đã lưu mapping hoạt động vào: {mapping_file}")

# Xử lý từng file
print("\nĐang xử lý các file...")
for file in os.listdir(RAW_DATA_FOLDER):
    if file.endswith(".csv"):
        # Đọc dữ liệu từ file CSV
        df = pd.read_csv(os.path.join(RAW_DATA_FOLDER, file))
        
        # Kiểm tra các cột cần thiết
        if 'AccelX' in df.columns and 'AccelY' in df.columns and 'AccelZ' in df.columns and \
           'GyroX' in df.columns and 'GyroY' in df.columns and 'GyroZ' in df.columns and \
           'ActivityLabel' in df.columns:
            
            # Làm mịn các cột gia tốc và con quay hồi chuyển
            df['AccelX'] = smooth_data(df['AccelX'].values)
            df['AccelY'] = smooth_data(df['AccelY'].values)
            df['AccelZ'] = smooth_data(df['AccelZ'].values)
            df['GyroX'] = smooth_data(df['GyroX'].values)
            df['GyroY'] = smooth_data(df['GyroY'].values)
            df['GyroZ'] = smooth_data(df['GyroZ'].values)

            # Chuyển đổi nhãn hoạt động thành số và thay thế trực tiếp
            df['ActivityLabel'] = df['ActivityLabel'].map(ACTIVITY_MAPPING)
            
            # Lưu file đã xử lý
            output_file = os.path.join(PROCESSED_DATA_FOLDER, f"processed_{file}")
            df.to_csv(output_file, index=False)
            print(f"✅ Đã xử lý file: {file}")
            print(f"   Đã lưu kết quả vào: {output_file}")
            print(f"   Số mẫu: {len(df)}")
        else:
            print(f"⚠️ File {file} không có đầy đủ các cột cần thiết")

