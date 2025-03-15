#include <WiFi.h>
#include <Wire.h>
#include <Adafruit_MPU6050.h>
#include <Adafruit_Sensor.h>

// Cấu hình WiFi
const char* ssid = "YOUR_WIFI_NAME";      // Thay YOUR_WIFI_NAME bằng tên WiFi của bạn
const char* password = "YOUR_WIFI_PASS";   // Thay YOUR_WIFI_PASS bằng mật khẩu WiFi của bạn

// Cấu hình Server
const char* server_ip = "192.168.1.xxx";   // Thay xxx bằng địa chỉ IP của máy tính chạy Python
const int server_port = 8080;

// Thông số kết nối
const int MAX_RETRIES = 5;                // Số lần thử kết nối tối đa
const int RETRY_DELAY = 5000;             // Thời gian chờ giữa các lần thử (5 giây)
const int SEND_INTERVAL = 50;             // Thời gian giữa các lần gửi dữ liệu (50ms = 20Hz)

// Khởi tạo đối tượng
Adafruit_MPU6050 mpu;
WiFiClient client;

// Biến theo dõi thời gian
unsigned long lastSendTime = 0;
unsigned long lastConnectTime = 0;

void setup() {
    // Khởi tạo Serial để debug
    Serial.begin(115200);
    while (!Serial) {
        delay(10);  // Đợi Serial port được mở
    }
    
    // Khởi tạo I2C
    Wire.begin();
    
    // Khởi tạo MPU6050
    initMPU6050();
    
    // Kết nối WiFi
    connectWiFi();
    
    // Kết nối server lần đầu
    connectServer();
}

void initMPU6050() {
    Serial.println("Khởi tạo MPU6050...");
    if (!mpu.begin()) {
        Serial.println("Không tìm thấy MPU6050. Kiểm tra lại kết nối!");
        while (1) {
            delay(10);
        }
    }
    Serial.println("MPU6050 đã sẵn sàng!");

    // Cấu hình MPU6050
    mpu.setAccelerometerRange(MPU6050_RANGE_8_G);
    mpu.setGyroRange(MPU6050_RANGE_500_DEG);
    mpu.setFilterBandwidth(MPU6050_BAND_21_HZ);
}

void connectWiFi() {
    Serial.println("\nĐang kết nối WiFi...");
    WiFi.begin(ssid, password);
    
    int retries = 0;
    while (WiFi.status() != WL_CONNECTED && retries < MAX_RETRIES) {
        delay(1000);
        Serial.print(".");
        retries++;
    }
    
    if (WiFi.status() == WL_CONNECTED) {
        Serial.println("\nĐã kết nối WiFi!");
        Serial.print("Địa chỉ IP của ESP32: ");
        Serial.println(WiFi.localIP());
    } else {
        Serial.println("\nKhông thể kết nối WiFi! Khởi động lại...");
        ESP.restart();
    }
}

bool connectServer() {
    if (!client.connected()) {
        // Đóng kết nối cũ nếu còn
        client.stop();
        delay(100);
        
        Serial.print("\nKết nối tới server ");
        Serial.print(server_ip);
        Serial.print(":");
        Serial.println(server_port);
        
        if (client.connect(server_ip, server_port)) {
            Serial.println("Đã kết nối tới server!");
            return true;
        } else {
            Serial.println("Không thể kết nối tới server!");
            return false;
        }
    }
    return true;
}

void sendSensorData() {
    // Đọc dữ liệu từ MPU6050
    sensors_event_t a, g, temp;
    mpu.getEvent(&a, &g, &temp);
    
    // Tạo chuỗi JSON
    String json = "{";
    json += "\"ax\":" + String(a.acceleration.x, 6) + ",";  // Tăng độ chính xác
    json += "\"ay\":" + String(a.acceleration.y, 6) + ",";
    json += "\"az\":" + String(a.acceleration.z, 6) + ",";
    json += "\"gx\":" + String(g.gyro.x, 6) + ",";
    json += "\"gy\":" + String(g.gyro.y, 6) + ",";
    json += "\"gz\":" + String(g.gyro.z, 6);
    json += "}\n";
    
    // Gửi dữ liệu
    client.print(json);
    
    // In ra Serial để debug
    // Serial.print(json);  // Bỏ comment nếu muốn xem dữ liệu gửi
}

void loop() {
    // Kiểm tra kết nối WiFi
    if (WiFi.status() != WL_CONNECTED) {
        Serial.println("Mất kết nối WiFi! Đang kết nối lại...");
        connectWiFi();
        return;
    }
    
    // Kiểm tra và thử kết nối lại với server
    unsigned long now = millis();
    if (!client.connected()) {
        if (now - lastConnectTime >= RETRY_DELAY) {
            lastConnectTime = now;
            if (!connectServer()) {
                Serial.println("Thử lại sau " + String(RETRY_DELAY/1000) + " giây...");
                return;
            }
        }
        return;
    }
    
    // Gửi dữ liệu theo chu kỳ
    if (now - lastSendTime >= SEND_INTERVAL) {
        lastSendTime = now;
        sendSensorData();
    }
} 