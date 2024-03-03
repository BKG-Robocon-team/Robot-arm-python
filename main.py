

# PATH : C:\Users\minhc\anaconda3\envs\ROBOCON-ball-detection

import os
import argparse
import cv2
import numpy as np
import math
import matplotlib.pyplot as plt
from collections import defaultdict




# def find_hough_circles(image, edge_image, r_min, r_max, delta_r, num_thetas, bin_threshold, post_process = True):
#   #image size
#   img_height, img_width = edge_image.shape[:2]
  
#   # R and Theta ranges
#   dtheta = int(360 / num_thetas)
  
#   ## Thetas is bins created from 0 to 360 degree with increment of the dtheta
#   thetas = np.arange(0, 360, step=dtheta)
  
#   ## Radius ranges from r_min to r_max 
#   rs = np.arange(r_min, r_max, step=delta_r)
  
#   # Calculate Cos(theta) and Sin(theta) it will be required later
#   cos_thetas = np.cos(np.deg2rad(thetas))
#   sin_thetas = np.sin(np.deg2rad(thetas))
  
#   # Evaluate and keep ready the candidate circles dx and dy for different delta radius
#   # based on the the parametric equation of circle.
#   # x = x_center + r * cos(t) and y = y_center + r * sin(t),  
#   # where (x_center,y_center) is Center of candidate circle with radius r. t in range of [0,2PI)
#   circle_candidates = []
#   for r in rs:
#     for t in range(num_thetas):
#       #instead of using pre-calculated cos and sin theta values you can calculate here itself by following
#       #circle_candidates.append((r, int(r*cos(2*pi*t/num_thetas)), int(r*sin(2*pi*t/num_thetas))))
#       #but its better to pre-calculate and use it here.
#       circle_candidates.append((r, int(r * cos_thetas[t]), int(r * sin_thetas[t])))
  
#   # Hough Accumulator, we are using defaultdic instead of standard dict as this will initialize for key which is not 
#   # aready present in the dictionary instead of throwing exception.
#   accumulator = defaultdict(int)
  
#   for y in range(img_height):
#     for x in range(img_width):
#       if edge_image[y][x] != 0: #white pixel
#         # Found an edge pixel so now find and vote for circle from the candidate circles passing through this pixel.
#         for r, rcos_t, rsin_t in circle_candidates:
#           x_center = x - rcos_t
#           y_center = y - rsin_t
#           accumulator[(x_center, y_center, r)] += 1 #vote for current candidate
  
#   # Output image with detected lines drawn
#   output_img = image.copy()
#   # Output list of detected circles. A single circle would be a tuple of (x,y,r,threshold) 
#   out_circles = []
  
#   # Sort the accumulator based on the votes for the candidate circles 
#   for candidate_circle, votes in sorted(accumulator.items(), key=lambda i: -i[1]):
#     x, y, r = candidate_circle
#     current_vote_percentage = votes / num_thetas
#     if current_vote_percentage >= bin_threshold: 
#       # Shortlist the circle for final result
#       out_circles.append((x, y, r, current_vote_percentage))
#       print(x, y, r, current_vote_percentage)
      
  
#   # Post process the results, can add more post processing later.
#   if post_process :
#     pixel_threshold = 10
#     postprocess_circles = []
#     for x, y, r, v in out_circles:
#       # Exclude circles that are too close of each other
#       # all((x - xc) ** 2 + (y - yc) ** 2 > rc ** 2 for xc, yc, rc, v in postprocess_circles)
#       # Remove nearby duplicate circles based on pixel_threshold
#       if all(abs(x - xc) > pixel_threshold or abs(y - yc) > pixel_threshold or abs(r - rc) > pixel_threshold for xc, yc, rc, v in postprocess_circles):
#         postprocess_circles.append((x, y, r, v))
#     out_circles = postprocess_circles
  
    
#   # Draw shortlisted circles on the output image
#   for x, y, r, v in out_circles:
#     output_img = cv2.circle(output_img, (x,y), r, (0,255,0), 2)
  
#   return output_img, out_circles

def increase_brightness(image, alpha):
  # Convert the image to float32 for calculations
  image = image.astype(np.float32)

  # Increase brightness using multiplication with alpha
  brightened_image = image * alpha

  # Clip pixel values to the valid range (0-255)
  brightened_image = np.clip(brightened_image, 0, 255)

  # Convert back to uint8 for further processing
  return brightened_image.astype(np.uint8)



def main():
    
    parser = argparse.ArgumentParser(description='Find Hough circles from the image.')
    parser.add_argument('image_path', type=str, help='Full path of the input image.')
    parser.add_argument('--r_min', type=float, help='Min radius circle to detect. Default is 5.')
    parser.add_argument('--r_max', type=float, help='Max radius circle to detect.')
    parser.add_argument('--delta_r', type=float, help='Delta change in radius from r_min to r_max. Default is 1.')
    parser.add_argument('--num_thetas', type=float, help='Number of steps for theta from 0 to 2PI. Default is 100.')
    parser.add_argument('--bin_threshold', type=int, help='Thresholding value to shortlist candidate for circle. Default is 0.4 i.e. 40%.')
    parser.add_argument('--min_edge_threshold', type=int, help='Minimum threshold value for edge detection. Default 100.')
    parser.add_argument('--max_edge_threshold', type=int, help='Maximum threshold value for edge detection. Default 200.')
    
    args = parser.parse_args()
    
    img_path = args.image_path
    r_min = 10
    r_max = 200
    delta_r = 1
    num_thetas = 100
    bin_threshold = 0.4
    min_edge_threshold = 100
    max_edge_threshold = 200
    
    if args.r_min:
        r_min = args.r_min
    
    if args.r_max:
        r_max = args.r_max
        
    if args.delta_r:
        delta_r = args.delta_r
    
    if args.num_thetas:
        num_thetas = args.num_thetas
    
    if args.bin_threshold:
        bin_threshold = args.bin_threshold
        
    if args.min_edge_threshold:
        min_edge_threshold = args.min_edge_threshold
        
    if args.max_edge_threshold:
        max_edge_threshold = args.max_edge_threshold
    
    image = cv2.imread(img_path)
    
    # Resize img 
    new_width = 600
    new_height = int(image.shape[0] * (new_width / image.shape[1]))
    input_img = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
    
    # Apply Gaussian blur
    blurred_image = cv2.GaussianBlur(input_img, (7,7), 0)
    cv2.imshow("Blurred Image", blurred_image)
    cv2.waitKey(0)


    # OR SHOULD I REDUCE THE BRIGHTNESS ? 
    # Increase brightness by a factor of 2.6 adjust as needed) 
    
    brightened_image = increase_brightness(blurred_image, 2.6)

    # Display the original and brightened images
    cv2.imshow("Brightened Image", brightened_image)
    cv2.waitKey(0)


    # Convert the image to HSL color space
    hsv_image = cv2.cvtColor(brightened_image, cv2.COLOR_BGR2HLS)  # Corrected: Use blurred_image
    # cv2.imshow(hsv_image)

    # cv2.waitKey(0)



    ## Define the lower and upper threshold bounds for HSV channels
    # Adjust these values based on the desired color range you want to extract
    #hue: màu 0-360     saturation: độ tinh khiết (0-đục, 255- đầy đủ)    lightness: độ sáng tổi (0-đen, 255-trắng)
    # '''Hue (°)   | Color
    # --------------------
    # 0-30      | Red
    # 30-90     | Yellow
    # 90-150    | Green
    # 150-210   | Cyan
    # 210-270   | Blue
    # 270-330   | Magenta
    # 330-360   | Red (again) '''
    # Define the lower and upper threshold bounds for HSV channels
    # Saturation also represents the color purity, but in HSL, a 50% saturation value corresponds to a medium grey, 
    # while 0% is achromatic (black or white) and 100% is the purest color.

    lower_bounds = np.array([80, 0, 0])  # Hue, Saturation, Lightness (adjust as needed)
    upper_bounds = np.array([180, 255, 255])  # Hue, Saturation, Lightness (adjust as needed)
    
    # Apply thresholding using inRange faunction
    mask = cv2.inRange(hsv_image, lower_bounds, upper_bounds)

    # Apply the mask to the original image to extract the desired color region
    result = cv2.bitwise_and(brightened_image,brightened_image, mask=mask)


    # Show the original image, mask, and thresholded image
    cv2.imshow("Mask", mask)
    cv2.imshow("HSV Thresholded Image",result)
    cv2.waitKey(0)


# CODE e Khoa :))))
    
    # Đọc ảnh và chuyển đổi sang ảnh xám
    #image = cv2.imread('hsv.png')
    gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
    cv2.imshow('Grayscale Image', gray)
    cv2.waitKey(0)

    # Áp dụng Gaussian Blur để làm mờ ảnh, giúp loại bỏ nhiễu
    blur = cv2.GaussianBlur(gray, (9, 9), 0)
    cv2.imshow('Blurred Image', blur)
    cv2.waitKey(0)

    # Phát hiện cạnh bằng phương pháp Canny
    edges = cv2.Canny(blur, 10, 150)
    cv2.imshow('Edge Image', edges)
    cv2.waitKey(0)


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
            cv2.circle(input_img, center, radius, (0, 255, 0), 2)
            cv2.circle(input_img, center, 2, (0, 0, 255), 3)

            x, y, radius = i[0], i[1], i[2]
            print(f"Center: ({x}, {y}), Radius: {radius}")

            circle_file = open('toa_do_duong_tron.txt', 'a')
            
            for i in range(len(circles)):
                circle_file.write('Center: (' + str(x) + ', ' + str(y) +  '), ' + 'Radius: ' + str(radius) + '\n')
            circle_file.close()
    print ("Done!")
# Hiển thị ảnh với các hình tròn đã phát hiện
    
    cv2.imshow('KET QUA',input_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # if edges is not None:
        
    #     print ("Detecting Hough Circles Started!")
    #     circle_img, circles = find_hough_circles(input_img, edges, r_min, r_max, delta_r, num_thetas, bin_threshold)
        
    #     cv2.imshow('Detected Circles', circle_img)
    #     cv2.waitKey(0)
        
    #     circle_file = open('circles_list.txt', 'w')
    #     circle_file.write('x ,\t y,\t Radius,\t Threshold \n')
    #     for i in range(len(circles)):
    #         circle_file.write(str(circles[i][0]) + ' , ' + str(circles[i][1]) + ' , ' + str(circles[i][2]) + ' , ' + str(circles[i][3]) + '\n')
    #     circle_file.close()
        
    #     if circle_img is not None:
    #         cv2.imwrite("circles_img.png", circle_img)
    # else:
    #     print ("Error in input image!")
            
    # print ("Detecting Hough Circles Complete!")



if __name__ == "__main__":
    main()