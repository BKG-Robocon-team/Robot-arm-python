#Google Colab

############################################# block code 1 #############################################

from google.colab import files

# Tải lên ảnh từ máy tính
uploaded = files.upload()

# Lưu ảnh đã tải lên vào Colab
for filename in uploaded.keys():
    print('File "{name}" with length {length} bytes'.format(name=filename, length=len(uploaded[filename])))



############################################# block code 2 #####################################################

pip install opencv-python

############################################# block code 3 #####################################################

import cv2
import numpy as np
from google.colab.patches import cv2_imshow
from google.colab import files

def increase_brightness(image, alpha):
  # Convert the image to float32 for calculations
  image = image.astype(np.float32)

  # Increase brightness using multiplication with alpha
  brightened_image = image * alpha

  # Clip pixel values to the valid range (0-255)
  brightened_image = np.clip(brightened_image, 0, 255)

  # Convert back to uint8 for further processing
  return brightened_image.astype(np.uint8)


image = cv2.imread('football-ball.jpg')

####################################################################################
# Resize img
new_width = 600
new_height = int(image.shape[0] * (new_width / image.shape[1]))
image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
# Apply Gaussian blur
# 7 7
blurred_image = cv2.GaussianBlur(image, (7,7), 0)
cv2_imshow(blurred_image)
# Increase brightness by a factor of 1.2 (adjust as needed)
brightened_image = increase_brightness(blurred_image, 2.6)
cv2_imshow(brightened_image)
# Convert the image to HSV color space
hsv_image = cv2.cvtColor(brightened_image, cv2.COLOR_BGR2HLS)
cv2_imshow(hsv_image)  # Corrected: Use blurred_image
# Define the lower and upper threshold bounds for HSV channels
# Adjust these values based on the desired color range you want to extract
#hue: màu 0-360     saturation: độ tinh khiết (0-đục, 255- đầy đủ)    lightness: độ sáng tổi (0-đen, 255-trắng)
'''Hue (°)   | Color
--------------------
0-30      | Red
30-90     | Yellow
90-150    | Green
150-210   | Cyan
210-270   | Blue
270-330   | Magenta
330-360   | Red (again) '''
#lower_bounds = np.array([80,100,0])  # Hue, Saturation, Lightness (adjust as needed)
#upper_bounds = np.array([180,255,255])  # Hue, Saturation, Lightness (adjust as needed)
lower_bounds = np.array([80,0,0])  # Hue, Saturation, Lightness (adjust as needed)
upper_bounds = np.array([180,255,255])  # Hue, Saturation, Lightness (adjust as needed)


# Apply thresholding using inRange faunction
mask = cv2.inRange(hsv_image, lower_bounds, upper_bounds)

# Apply the mask to the original image to extract the desired color region
result = cv2.bitwise_and(brightened_image,brightened_image, mask=mask)


# Show the original image, mask, and thresholded image
#cv2_imshow(brightened_image)
cv2_imshow(mask)
cv2_imshow(result)

####################################################################################

# Đọc ảnh và chuyển đổi sang ảnh xám
#image = cv2.imread('hsv.png')
gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
cv2_imshow(gray)

# Áp dụng Gaussian Blur để làm mờ ảnh, giúp loại bỏ nhiễu
blur = cv2.GaussianBlur(gray, (9, 9), 0)
#blur = cv2.GaussianBlur(gray, (5, 5), 0)

cv2_imshow(blur)

# Phát hiện cạnh bằng phương pháp Canny
edges = cv2.Canny(blur, 10, 150)
cv2_imshow(edges)

# Áp dụng Hough Circle Transform
circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, dp=1, minDist=150, param1=40, param2=30, minRadius=0, maxRadius=0)
#goc: circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, dp=1, minDist=150, param1=50, param2=30, minRadius=0, maxRadius=0)
#ổn 3: 40 30


# Vẽ các hình tròn được phát hiện
if circles is not None:
    circles = np.uint16(np.around(circles))
    for i in circles[0, :]:
        center = (i[0], i[1])
        radius = i[2]
        # Vẽ hình tròn và tâm
        cv2.circle(image, center, radius, (0, 255, 0), 2)
        cv2.circle(image, center, 2, (0, 0, 255), 3)

# Hiển thị ảnh với các hình tròn đã phát hiện
cv2_imshow(image)
