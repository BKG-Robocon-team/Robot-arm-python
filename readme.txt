Circular Hough Transform - Thuần XLA 

* How to run :
- step 1 : mở cmd, cd path folder có chứa file CircularHoughTransform.py
- step 2 : type "python CircularHoughTransform.py <image_name>"
// có thể change name của CircularHoughTransform.py để type dễ hơn kkkkk

* Done shits :
- Detect được vật tròn trong ảnh ( sử dụng hàm cv2.HoughCircles)
- tìm được tọa độ center và bán kính => in ra file text => tính được diện tích của bóng 

* Undone shits :
- code lại thuật toán CHT bị sai ( ở đâu đó ). Nhưng nếu gọi thẳng hàm cv2.HoughCircles thì đã detect được đường tròn khá ổn.
- chưa xong lọc màu



