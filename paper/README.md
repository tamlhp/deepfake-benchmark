# dfd_benchmark

# Phương pháp giả mạo ảnh
## styleGAN
https://github.com/NVlabs/stylegan2

[paper v1](1812.04948.pdf)

[paper v1](1912.04958.pdf)


## starGAN
https://github.com/clovaai/stargan-v2

[paper v1](1711.09020.pdf)

[paper v2](1912.01865.pdf)




# Phương pháp phát hiện ảnh giả mạo
## Classic ML
### Head pose 
[paper](1811.00661.pdf)

Exposing deep fakes using inconsistent head poses

Trong quá trình chuyển từ mặt người này sang người khác, 
hướng xoay của khuôn mặt bị thay đổi. Vì vậy so sánh sự khác biệt 
giữa hướng xoay của phần trung tâm khuôn mặt và phần rìa khuôn mặt 
có thể xác phân biệt được ảnh thật giả.


### Visual Arrtifact
[paper](Exploiting visual artifacts to expose deepfakes and face manipulations-annotated.pdf)

Exploiting Visual Artifacts to Expose Deepfakes and Face Manipulations

Tính toán những đặc điểm trên khuôn mặt:

* Độ khác biệt về màu sắc giữa hai mắt
* Sự biến thiên màu sắc và độ tương phản trên da
* Chi tiết trên khuôn mặt, đặc biệt là mắt và vùng rìa khuôn mặt


### frequency domain 
[paper](1911.00686.pdf)

Unmasking DeepFakes with simple Features

Thuật toán Fourier được sử dụng để tính toán miền tần số trong ảnh. Kết quả 
của phép biến đổi Fourier là một dữ liệu có kích thước bằng kích thước 
ảnh ảnh ban đầu. Để chuyển kết quả này thành vector một chiều, sử dụng 
thuật toán tính trung bình Azimuthal trên thông tin phổ năng lượng. Thuật 
toán này tính trung bình các giá trị có cùng khoảng cách tới tâm của ảnh.


## Deep learning
### Mesonet
[paper](1809.00888.pdf)

MesoNet: a compact facial video forgery detection network

Các mạng DNN được thiết kế sâu để trích xuất thông tin đặc trưng ở 
tầng cao có thể dẫn đến không phân biệt được ảnh khuôn mặt thật hoặc giả.
 Mạng Meso được thiết kế để có thể trích xuất những đặc trưng tầm trung, 
 thuận lợi cho việc phân biệt các ảnh thật và giả. 

### Capsule
[paper](Capsule-forensics Using Capsule Networks to Detect Forged Images and Videos-annotated.pdfs)

Capsule-forensics: Using capsule networks to detect forged images and videos.

Một lớp capsule network được thiết kế phía sau mạng VGG-19 để trích xuất ra
 những đặc trưng quan trọng. Mạng capsule sử dụng lớp CNN cuối của mạng VGG 
 và tính toán ra tập các vector đặc trưng của mỗi ảnh.



### xception
[paper](1901.08971-annotated.pdf)

Faceforensics++: Learning to detect manipulated facial images

Sử dụng mạng đã đào tạo sẵn từ bộ ImageNet, thay thế lớp fully với 
1 output. Mạng xception với ý tưởng chính là thay vì sử dụng lớp 
tích chập truyền thống, mô hình sử dụng lớp có tên là Depthwise 
separable convolutions nhằm mục đích giảm thiểu độ phức tạp tính 
toán và tăng hiệu quả học. Phương pháp này tiết kiệm tài nguyên 
tính toán so với lớp tích chập thông thường.


## Figerprint

[paper](1811.08180.pdf)


