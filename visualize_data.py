import os
import pandas as pd
import matplotlib.pyplot as plt

# Thư mục chứa các file CSV
RAW_DATA_FOLDER = "processed_data"
TARGET_FILE = "processed_HoangHa-rawdata.csv"

def plot_activity_data(df, activity):
    """
    Vẽ biểu đồ cho một hoạt động.
    """
    # Lọc dữ liệu cho hoạt động
    activity_data = df[df['ActivityLabel'] == activity]
    
    # Tạo figure mới cho hoạt động này
    plt.figure(figsize=(15, 8))
    
    # Vẽ dữ liệu gia tốc
    plt.subplot(2, 1, 1)
    plt.plot(activity_data['AccelX'], label='AccelX', color='r')
    plt.plot(activity_data['AccelY'], label='AccelY', color='g')
    plt.plot(activity_data['AccelZ'], label='AccelZ', color='b')
    plt.title(f"{activity} - Accelerometer Data")
    plt.xlabel('Time')
    plt.ylabel('Acceleration')
    plt.legend()
    plt.grid(True)
    
    # Vẽ dữ liệu con quay hồi chuyển
    plt.subplot(2, 1, 2)
    plt.plot(activity_data['GyroX'], label='GyroX', color='r')
    plt.plot(activity_data['GyroY'], label='GyroY', color='g')
    plt.plot(activity_data['GyroZ'], label='GyroZ', color='b')
    plt.title(f"{activity} - Gyroscope Data")
    plt.xlabel('Time')
    plt.ylabel('Gyroscope')
    plt.legend()
    plt.grid(True)
    
    # Điều chỉnh layout
    plt.tight_layout()
    plt.show()

# Đọc dữ liệu từ file HoangHa
file_path = os.path.join(RAW_DATA_FOLDER, TARGET_FILE)
if os.path.exists(file_path):
    print(f"\nĐang xử lý file: {TARGET_FILE}")
    df = pd.read_csv(file_path)
    
    # Kiểm tra các cột cần thiết
    required_columns = ['AccelX', 'AccelY', 'AccelZ', 'GyroX', 'GyroY', 'GyroZ', 'ActivityLabel']
    if all(col in df.columns for col in required_columns):
        # Lấy danh sách các hoạt động trong file
        activities = df['ActivityLabel'].unique()
        print(f"\nCác hoạt động trong file:")
        for activity in activities:
            print(f"- {activity}")
            # Vẽ biểu đồ cho từng hoạt động
            plot_activity_data(df, activity)
    else:
        print(f"⚠️ File không có đầy đủ các cột cần thiết")
        missing_cols = [col for col in required_columns if col not in df.columns]
        print(f"Thiếu các cột: {missing_cols}")
else:
    print(f"❌ Không tìm thấy file {TARGET_FILE} trong thư mục {RAW_DATA_FOLDER}")
