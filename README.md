# LSM-YOLO: Phát hiện vùng quan tâm trong ảnh y tế

## 📋 Giới thiệu

**LSM-YOLO** là một project nghiên cứu và phát triển mô hình phát hiện vùng quan tâm (Region of Interest - RoI) trong ảnh y tế. Project sử dụng kỹ thuật Knowledge Distillation để tạo ra một mô hình nhỏ gọn LSM-YOLO từ các mô hình teacher như YOLOv9c, YOLOv8s, đảm bảo hiệu suất cao trong việc phát hiện các vùng quan trọng trên ảnh y tế.

## 🎯 Mục tiêu Project

- **Nén mô hình**: Thực hiện Knowledge Distillation từ các mô hình YOLO lớn (YOLOv9c, YOLOv8s) sang mô hình nhỏ gọn LSM-YOLO
- **Duy trì hiệu suất**: Đảm bảo độ chính xác phát hiện xấp xỉ với mô hình teacher ban đầu
- **Đánh giá và trực quan hóa**: Cung cấp các công cụ đánh giá và trực quan hóa kết quả
- **Triển khai API**: Xây dựng API REST sử dụng FastAPI để phục vụ mô hình
- **Containerization**: Đóng gói toàn bộ hệ thống thành Docker Image để dễ dàng triển khai

## 🔧 Công nghệ sử dụng

- **Computer Vision**: Xử lý ảnh và phát hiện đối tượng
- **Knowledge Distillation**: Kỹ thuật nén mô hình
- **FastAPI**: Framework xây dựng API REST
- **Docker**: Containerization và triển khai
- **Python**: Ngôn ngữ lập trình chính

## 🚀 Cài đặt và Sử dụng

### Yêu cầu hệ thống
- Python 3.8+
- Docker (tùy chọn)
- GPU (khuyến nghị cho training)

### Cài đặt từ source

```bash
# Clone repository
git clone <repository-url>
cd Project_DL

# Cài đặt dependencies
pip install -r requirements.txt
```

### Chạy API

```bash
# Chạy API server
python app/main.py

# Hoặc sử dụng uvicorn
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### Sử dụng Docker

```bash
# Build Docker image
docker build -t lsm-yolo-api .

# Chạy container
docker run -p 8000:8000 lsm-yolo-api
```

## 📡 API Endpoints

### `POST /detect_image/`
Phát hiện vùng quan tâm trong ảnh y tế

**Input**: File ảnh (JPEG, PNG, etc.)
**Output**: 
- Ảnh với bounding boxes được vẽ (nếu có phát hiện)
- JSON response với thông tin detections

**Ví dụ sử dụng**:
```bash
curl -X POST "http://localhost:8000/detect_image/" \
     -H "accept: application/json" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@medical_image.jpg"
```

### `GET /`
Endpoint kiểm tra trạng thái API

## 🔬 Phương pháp Knowledge Distillation

Project sử dụng kỹ thuật Knowledge Distillation để:

1. **Teacher Models**: YOLOv9c, YOLOv8s - các mô hình lớn với hiệu suất cao
2. **Student Model**: LSM-YOLO - mô hình nhỏ gọn, tối ưu cho deployment
3. **Transfer Learning**: Chuyển giao kiến thức từ teacher sang student model
4. **Optimization**: Tối ưu hóa student model để duy trì độ chính xác

## 📊 Đánh giá và Kết quả

- **Metrics**: mAP (mean Average Precision), FPS, Model Size
- **Visualization**: Confusion matrix, PR curves, Detection examples
- **Comparison**: So sánh hiệu suất giữa teacher và student models
