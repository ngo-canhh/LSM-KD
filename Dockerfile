FROM python:3.10-slim

# Install system dependencies required for OpenCV
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Thiết lập thư mục làm việc trong container
WORKDIR /code

# Sao chép file requirements.txt vào thư mục làm việc
COPY ./requirements.txt /code/requirements.txt
COPY ./LSM-YOLO /code/LSM-YOLO

# Cài đặt các dependencies từ requirements.txt
# --no-cache-dir để giảm kích thước image
# --default-timeout=100 để tránh lỗi timeout khi download package lớn
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r /code/requirements.txt

RUN pip install --no-cache-dir /code/LSM-YOLO

# Sao chép toàn bộ thư mục app (chứa code và model) vào thư mục làm việc /code/app
COPY ./app /code/app


# Mở cổng mà FastAPI sẽ chạy (ví dụ 8000)
EXPOSE 8000

# Lệnh để chạy ứng dụng FastAPI khi container khởi động
# Chạy uvicorn, trỏ đến object 'app' trong file 'main.py' bên trong thư mục 'app'
# --host 0.0.0.0 để có thể truy cập từ bên ngoài container
# --port 8000 để chạy trên cổng đã expose
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]