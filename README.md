# Traffic Sign Recognition with CNN

## Giới thiệu

Nhận dạng biển báo giao thông Đức bằng mô hình Convolutional Neural Network (CNN). Dự án hướng đến xây dựng hệ thống có khả năng phân loại chính xác các loại biển báo giao thông theo chuẩn GTSRB (German Traffic Sign Recognition Benchmark).

## Mục lục

* [Tính năng](#tính-năng)
* [Kiến trúc & Công nghệ](#kiến-trúc--công-nghệ)
* [Yêu cầu hệ thống & Cài đặt](#yêu-cầu-hệ-thống--cài-đặt)
* [Cách sử dụng](#cách-sử-dụng)
* [Dataset](#dataset)
* [Cấu trúc thư mục](#cấu-trúc-thư-mục)
* [Ví dụ kết quả](#ví-dụ-kết-quả)
* [Hướng phát triển](#hướng-phát-triển)
* [Đóng góp](#đóng-góp)
* [Liên hệ](#liên-hệ)

## Tính năng

* Đào tạo mô hình CNN để phân loại biển báo giao thông Đức.
* Hỗ trợ train (đào tạo) và inference (dự đoán) trên ảnh đầu vào.
* Lưu trữ và nạp lại trọng số mô hình.

## Kiến trúc & Công nghệ

* **Ngôn ngữ:** Python 3.8+
* **Thư viện chính:** TensorFlow / Keras (hoặc PyTorch), OpenCV, scikit-learn
* **Công cụ hỗ trợ:** Jupyter Notebook, Matplotlib,...

## Yêu cầu hệ thống & Cài đặt

1. Clone repository:

   ```bash
   git clone https://github.com/hungdeniubeo/bienbao.git
   cd bienbao
   ```
2. Tạo virtual environment và cài đặt dependencies:

   ```bash
   python3 -m venv venv
   source venv/bin/activate  # macOS/Linux
   venv\Scripts\activate     # Windows
   pip install -r requirements.txt
   ```

## Cách sử dụng:

1. **Chuẩn bị dataset:**

   * Tải GTSRB dataset từ Kaggle về thư mục `data/`.
2. **Đào tạo mô hình:**

   ```bash
   python src/train.py \
     --data_dir data/ \
     --epochs 50 \
     --batch_size 32 \
     --model_output models/traffic_sign_cnn.h5
   ```
3. **Dự đoán (Inference):**

   ```bash
   python src/predict.py \
     --model_path models/traffic_sign_cnn.h5 \
     --image_path sample_images/00001_00000.ppm
   ```
4. **Xem kết quả:**
   Kết quả dự đoán sẽ hiển thị tên biển báo và độ tin cậy.

## Dataset

* Sử dụng GTSRB (German Traffic Sign Recognition Benchmark) từ Kaggle: [https://www.kaggle.com/datasets/johndesq/german-traffic-signs](https://www.kaggle.com/datasets/johndesq/german-traffic-signs)
* Giải nén vào `data/` theo cấu trúc:

  ```text
  data/
  ├── Train/
  └── Test/
  ```

## Cấu trúc thư mục

```
bienbao/
├── data/                  # Dataset GTSRB
├── models/                # Lưu checkpoint & model đã train
├── src/                   # Code chính
│   ├── train.py           # Script đào tạo mô hình
│   ├── predict.py         # Script dự đoán ảnh
│   └── utils.py           # Hàm hỗ trợ (tiền xử lý, load data)
├── requirements.txt       # Thư viện cần cài
├── docs/                  # Hình ảnh demo, báo cáo
└── README.md
```

## Ví dụ kết quả

![Demo kết quả](docs/demo_result.png)

## Hướng phát triển

* Thử nghiệm data augmentation để cải thiện độ chính xác.
* Chuyển sang mô hình sâu hơn hoặc tận dụng Transfer Learning (MobileNet, ResNet).
* Triển khai API với Flask hoặc FastAPI.
* Xây dựng giao diện web để upload ảnh và nhận diện real-time.

## Đóng góp

1. Fork repository
2. Tạo branch feature mới
3. Tạo Pull Request rõ ràng với mô tả thay đổi


## Liên hệ

* GitHub: [hungdeniubeo](https://github.com/hungdeniubeo)
* Email: [phihung3922@example.com](mailto:phihung3922@gmail.com) 
