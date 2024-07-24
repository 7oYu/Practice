import cv2
import numpy as np
from perspective_projection import cv_show


def learn_hough_linep():
    src_img = cv2.imread('picture/note.jpg', cv2.IMREAD_COLOR)
    src_img = cv2.resize(src_img, (int(src_img.shape[1]/4), int(src_img.shape[0]/4)))
    cv_show('src img', src_img)
    canny_ret = cv2.Canny(src_img, 40, 80)
    cv_show('src img', canny_ret)
    # rho: 霍夫空间(ρ的精度: 默认1像素即可),  theta: 霍夫空间(θ的精度: 默认1度即可)
    # threshold 霍夫空间中大于多少条线相交于一点, 认为是图像空间的一条直线
    # HoughLinesP是HoughLines的改进版输出线段起点终点
    hough_ret = cv2.HoughLinesP(canny_ret, 1, np.pi / 180, 100)
    if hough_ret is None:
        print('no line in img')
        return
    for single_ret in hough_ret:
        cv2.line(src_img, (single_ret[0][0], single_ret[0][1]), (single_ret[0][2], single_ret[0][3]), (0, 255, 0), 2)
    cv_show('src img', src_img)


def learn_hough_line():
    src_img = cv2.imread('picture/note.jpg', cv2.IMREAD_COLOR)
    src_img = cv2.resize(src_img, (int(src_img.shape[1]/4), int(src_img.shape[0]/4)))
    cv_show('src img', src_img)
    canny_ret = cv2.Canny(src_img, 40, 80)
    cv_show('canny img', canny_ret)
    hough_ret = cv2.HoughLines(canny_ret, 1, np.pi / 180, 100)
    if hough_ret is None:
        print('no line in img')
        return
    for single_ret in hough_ret:
        rho = single_ret[0][0]
        theta = single_ret[0][1]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        line_length = 1000
        x1 = int(x0 + line_length * (-b))
        y1 = int(y0 + line_length * (a))
        x2 = int(x0 - line_length * (-b))
        y2 = int(y0 - line_length * (a))
        cv2.line(src_img, (x1, y1), (x2, y2), (0, 255, 0), 1)
    cv_show('line', src_img)


def learn_hough_circles():
    src_img = cv2.imread('picture/circle.jpg', cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(src_img, cv2.COLOR_BGR2GRAY)
    cv_show('src img', gray)
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 10)
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            center = (i[0], i[1])  # 圆心坐标
            radius = i[2]  # 圆半径
            # 在图像中画出圆心
            cv2.circle(src_img, center, 1, (0, 0, 255), 2)
            # 画出圆
            cv2.circle(src_img, center, radius, (0, 255, 0), 2)
    # 显示结果图像
    cv_show('detected circles', src_img)
