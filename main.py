from __future__ import print_function
import cv2 as cv
import numpy as np

max_value = 255
max_value_H = 360 // 2
low_H = 0
low_S = 0
low_V = 0
high_H = max_value_H
high_S = max_value
high_V = max_value
window_detection_name = 'Object Detection'
min_canny_val = 0
max_canny_val = 100
window_Trackbar_name = 'Trackbar '
low_H_name = 'Low H'
low_S_name = 'Low S'
low_V_name = 'Low V'
high_H_name = 'High H'
high_S_name = 'High S'
high_V_name = 'High V'
kernel = np.ones((5, 5), np.uint8)


def on_low_H_thresh_trackbar(val):
    global low_H
    global high_H
    low_H = val
    low_H = min(high_H - 1, low_H)
    cv.setTrackbarPos(low_H_name, window_detection_name, low_H)


def on_high_H_thresh_trackbar(val):
    global low_H
    global high_H
    high_H = val
    high_H = max(high_H, low_H + 1)
    cv.setTrackbarPos(high_H_name, window_detection_name, high_H)


def on_low_S_thresh_trackbar(val):
    global low_S
    global high_S
    low_S = val
    low_S = min(high_S - 1, low_S)
    cv.setTrackbarPos(low_S_name, window_detection_name, low_S)


def on_high_S_thresh_trackbar(val):
    global low_S
    global high_S
    high_S = val
    high_S = max(high_S, low_S + 1)
    cv.setTrackbarPos(high_S_name, window_detection_name, high_S)


def on_low_V_thresh_trackbar(val):
    global low_V
    global high_V
    low_V = val
    low_V = min(high_V - 1, low_V)
    cv.setTrackbarPos(low_V_name, window_detection_name, low_V)


def on_high_V_thresh_trackbar(val):
    global low_V
    global high_V
    high_V = val
    high_V = max(high_V, low_V + 1)
    cv.setTrackbarPos(high_V_name, window_detection_name, high_V)

def on_min_canny_val_trackbar(val):
    global min_canny_val
    min_canny_val = val
    cv.setTrackbarPos("min_canny_val", window_detection_name, min_canny_val)


def on_max_canny_val_trackbar(val):
    global max_canny_val
    max_canny_val = val
    cv.setTrackbarPos("max_canny_val", window_detection_name, max_canny_val)

def image_threshold_fun(image):
    image_threshold= cv.inRange(image, (0, 0, 165), (255, 255, 255))
    #image_threshold = cv.inRange(image, (low_H, low_S, low_V), (high_H, high_S, high_V))
    image_threshold = cv.morphologyEx(image_threshold, cv.MORPH_OPEN, kernel)
    image_threshold = cv.morphologyEx(image_threshold, cv.MORPH_CLOSE, kernel)
    cv.imshow("image_threshold", image_threshold)
    return image_threshold

cv.namedWindow(window_Trackbar_name)
cv.createTrackbar(low_H_name, window_Trackbar_name, low_H, max_value_H, on_low_H_thresh_trackbar)
cv.createTrackbar(high_H_name, window_Trackbar_name, high_H, max_value_H, on_high_H_thresh_trackbar)
cv.createTrackbar(low_S_name, window_Trackbar_name, low_S, max_value, on_low_S_thresh_trackbar)
cv.createTrackbar(high_S_name, window_Trackbar_name, high_S, max_value, on_high_S_thresh_trackbar)
cv.createTrackbar(low_V_name, window_Trackbar_name, low_V, max_value, on_low_V_thresh_trackbar)
cv.createTrackbar(high_V_name, window_Trackbar_name, high_V, max_value, on_high_V_thresh_trackbar)
cv.createTrackbar("min_canny_val", window_Trackbar_name, min_canny_val, 100, on_min_canny_val_trackbar)
cv.createTrackbar("max_canny_val", window_Trackbar_name, max_canny_val, 100, on_max_canny_val_trackbar)
while True:
    #original = cv.imread('./s1/front.jpg')
    original = cv.imread('./s2/front.jpg')
    #original = cv.imread('./OK/s1/front.jpg')
    #original = cv.imread('./OK/s2/front.jpg')

    #original = cv.resize(original, (750, 600), interpolation=cv.INTER_AREA)
    frame = np.copy(original)

    frame_HSV = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    # frame_threshold = cv.inRange(frame_HSV, (low_H, low_S, low_V), (high_H, high_S, high_V))
    frame_threshold = cv.inRange(frame_HSV, (20, 0, 140), (40, 146, 255))
    frame_threshold = cv.morphologyEx(frame_threshold, cv.MORPH_OPEN, kernel)
    frame_threshold = cv.morphologyEx(frame_threshold, cv.MORPH_CLOSE, kernel)

    # 最大面積最小矩形 #
    contours, hierarchy = cv.findContours(frame_threshold, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    areas = []
    for c in range(len(contours)):
        areas.append(cv.contourArea(contours[c]))
    max_id = areas.index(max(areas))
    max_rect = cv.minAreaRect(contours[max_id])
    max_box = cv.boxPoints(max_rect)
    max_box = np.int0(max_box)
    #cv.drawContours(frame, [max_box], 0, (0, 255, 0), 2)

    #print(max_box)
    min_x = min(max_box[0][0], max_box[1][0], max_box[2][0], max_box[3][0])#min(max_box[0][0], max_box[1][0], max_box[2][0], max_box[3][0])
    max_x = max(max_box[0][0], max_box[1][0], max_box[2][0], max_box[3][0])#max(max_box[0][0], max_box[1][0], max_box[2][0], max_box[3][0])
    min_y = min(max_box[0][1], max_box[1][1], max_box[2][1], max_box[3][1])#min(max_box[0][1], max_box[1][1], max_box[2][1], max_box[3][1])
    max_y = max(max_box[0][1], max_box[1][1], max_box[2][1], max_box[3][1])#max(max_box[0][1], max_box[1][1], max_box[2][1], max_box[3][1])
    top_mid_x = int((max_box[1][0] - max_box[0][0]) / 2 + max_box[0][0])
    top_mid_y = int((max_box[1][1] - max_box[0][1]) / 2 + max_box[0][1])
    dow_mid_x = int((max_box[3][0] - max_box[2][0]) / 2 + max_box[2][0])
    dow_mid_y = int((max_box[3][1] - max_box[2][1]) / 2 + max_box[2][1])
    image = frame[min_y:max_y, min_x:max_x]


    image_threshold=image_threshold_fun(image)
    image_HSV = cv.cvtColor(image, cv.COLOR_BGR2HSV)
    blurred = cv.GaussianBlur(image_HSV, (3, 3), 0)
    canny = cv.Canny(blurred, 20, 40, 3, 3, True)
    bitwise_and_img = cv.bitwise_and(image_threshold, canny)
    #canny = cv.Canny(blurred, min_canny_val, max_canny_val, 3, 3, True)
    #canny = cv.Canny(blurred, 10, 30, 3, 3, True)

    # lines = cv.HoughLinesP(canny, 1.0, np.pi / 60, 20, minLineLength=20, maxLineGap=7)
    lines = cv.HoughLinesP(bitwise_and_img, 1.0, np.pi / 180, 20, minLineLength=100, maxLineGap=10)
    #lines = cv.HoughLines(canny, 1.0, np.pi / 60, 20)
    lines = lines[:, 0, :]

    count = 0
    for x1, y1, x2, y2 in lines:
        #cv.line(image, (x1, y1), (x2, y2), (0, 255, 0), 2)#看全部的線

        if (max_x -min_x- 50 > x1 and max_x -min_x - 50 > x2 ) and (50 < x1 and 50 < x2):
            #cv.line(frame, (min_x, min_y), (max_x, max_y), (255, 255, 0), 2)
            #print("x1",x1,"x2",x2,"y1",y1,"y2",y2)
            #print(min_x, max_x, min_y, max_y)
            if((abs(x1 - x2) == 0 or abs(y2-y1)/abs(x2-x1)>1) and 10 < (max(x1,x2)-min(x1,x2))/2+min(x1,x2) and (max(x1,x2)-min(x1,x2))/2+min(x1,x2) < max_x-min_x-10):#判斷斜率以及區域內是否有線
                cv.line(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                count += 1
                #print(x1, y1, x2, y2)
                if x1 < image.shape[1] / 2 and x2 < image.shape[1] / 2:
                    print("The line on the left")
                if x1 > image.shape[1] / 2 and x2 > image.shape[1] / 2:
                    print("The line on the right")
                if ((x1 > image.shape[1] / 2 and x2 < image.shape[1] / 2) or x1 < image.shape[1] / 2 and x2 > image.shape[1] / 2):
                    print("The line on the middle")
    if count == 0:
        print("complete")

    cv.namedWindow("original_window", 0)
    cv.resizeWindow("original_window", 1008, 756)
    cv.imshow("original_window", original)

    cv.namedWindow("img_Capture", 0)
    cv.resizeWindow("img_Capture", 1008, 756)
    cv.imshow("img_Capture", frame)

    cv.namedWindow(window_detection_name, 0)
    cv.resizeWindow(window_detection_name, 1008, 756)
    cv.imshow(window_detection_name, frame_threshold)

    cv.imshow("img_window", image)
    cv.imshow("img_window", image)
    cv.imshow("blurred_window", blurred)
    cv.imshow("bitwise_and_img", bitwise_and_img)
    key = cv.waitKey(30)
    if key == ord('q') or key == 27:
        break
print(max_box)
