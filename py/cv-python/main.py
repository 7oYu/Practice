# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import cv2
import numpy as np
import matplotlib.pyplot as plt
from perspective_projection import cv_show, learn_perspective
from hough import learn_hough_line, learn_hough_linep
from corner import learn_harris, learn_sift, learn_fast, learn_orb
from image_splicing import image_splicing


# sobel算子 水平垂直边缘检测
def learn_sobel():
    img = cv2.imread('picture/lena.jpg', cv2.IMREAD_GRAYSCALE)
    sobel_x = cv2.Sobel(img, cv2.CV_64FC1, 1, 0, ksize=3)
    sobel_x = cv2.convertScaleAbs(sobel_x)
    cv2.imshow('sobel_x', sobel_x)
    sobel_y = cv2.Sobel(img, cv2.CV_64FC1, 0, 1, ksize=3)
    sobel_y = cv2.convertScaleAbs(sobel_y)
    cv2.imshow('sobel_y', sobel_y)
    sobel_xy = cv2.addWeighted(sobel_x, 0.5, sobel_y, 0.5, 0)
    cv_show('sobel_xy', sobel_xy)


# 边缘检测
def learn_canny():
    img = cv2.imread('picture/lena.jpg', cv2.IMREAD_GRAYSCALE)
    canny_1 = cv2.Canny(img, 120, 150)
    canny_2 = cv2.Canny(img, 40, 80)
    ret = np.hstack((canny_1, canny_2))
    cv_show('ret', ret)


# 上、下采样
def learn_pyr():
    img = cv2.imread('picture/lena.jpg', cv2.IMREAD_COLOR)
    print(f'image size {img.shape}')
    up_ret = cv2.pyrUp(img)
    print(f'up image size {up_ret.shape}')
    cv2.imshow('up_ret', up_ret)
    down_ret = cv2.pyrDown(img)
    print(f'down image size {down_ret.shape}')
    cv_show('down_ret', down_ret)


# 轮廓检测
def learn_contour():
    img = cv2.imread('picture/pic.png', cv2.IMREAD_COLOR)
    changed_contour = img.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)  # THRESH_OTSU 自动2值化
    cv2.imshow('threshold img', gray)
    contours, hierarchy = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    # RETR_EXTERNAL 外轮廓 RETR_TREE 所有轮廓
    cv2.drawContours(img, contours, -1, (0, 0, 0), 1)
    cv2.imshow('ret_img img', img)
    for i in range(len(contours)):
        print(f'contour area {i}: {cv2.contourArea(contours[i])}')  # 面积
        print(f'arcLength {i}: {cv2.arcLength(contours[i], True)}')  # 周长 closed: 是否闭合
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # 轮廓近似
    bound_img = changed_contour.copy()
    epsilon = 0.01 * cv2.arcLength(contours[2], True)  # epsilon 设置的越小轮廓越与原来的轮廓近似
    approx = cv2.approxPolyDP(contours[2], epsilon, True)  # 轮廓近似 epsilon表示原始轮廓到近似轮廓最大的距离
    cv2.drawContours(changed_contour, [approx], -1, (0, 0, 0), 1)
    cv_show('changed_contour img', changed_contour)
    # 外接矩形
    circle_img = bound_img.copy()
    bound_rect = cv2.boundingRect(contours[2])  # bound_rect [x, y, h, w]
    cv2.rectangle(bound_img, (bound_rect[0], bound_rect[1]),
                  (bound_rect[0] + bound_rect[2], bound_rect[1] + bound_rect[3]), (0, 0, 0), 1)
    cv_show('bound_rect img', bound_img)
    # 外接圆
    bound_circle = cv2.minEnclosingCircle(contours[2])  # bound_circle [圆心(x,y), 半径]
    cv_show('circle_img img', cv2.circle(circle_img, (int(bound_circle[0][0]), int(bound_circle[0][1])),
                                         int(bound_circle[1]), (0, 0, 0), 1))


def get_roi(img, x, y, w, h):
    roi = img[y:y + h, x:x + w]
    return roi.copy()


def learn_match_template():
    img = cv2.imread('picture/lena.jpg', cv2.IMREAD_GRAYSCALE)
    temp = get_roi(img, 160, 180, 150, 200)
    cv2.imshow('roi img', temp)
    method = cv2.TM_SQDIFF_NORMED
    ret = cv2.matchTemplate(img, temp, method)  # TM_SQDIFF_NORMED 越接近零越相关 另外两种越接近1越相关 NORMED表示归一化
    print(f'match ret size {ret.shape}')  # 结果为一个矩阵 每个元素都对应一个位置的结果
    min_value, max_value, min_loc, max_loc = cv2.minMaxLoc(ret)
    print(f'min_value, max_value, min_loc, max_loc {min_value, max_value, min_loc, max_loc}')
    if method in [cv2.TM_CCORR_NORMED, cv2.TM_CCOEFF_NORMED, cv2.TM_CCORR, cv2.TM_CCOEFF]:
        loc = max_loc
    else:
        loc = min_loc
    cv2.rectangle(img, loc, (loc[0] + 150, loc[1] + 200), (0, 0, 0), 1)
    cv_show('match ret', img)


def learn_hist():
    img = cv2.imread('picture/lena.jpg', cv2.IMREAD_COLOR)
    # 创建掩码
    # mask = np.zeros(img.shape[:2], np.uint8)
    # mask[100:200, 50:150] = 255
    # mask_ret = cv2.bitwise_and(img, img, mask=mask)
    # plt.imshow(cv2.cvtColor(mask_ret, cv2.COLOR_BGR2RGB))
    # plt.show()
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # mask表示需要统计的区域 不指定就是全统计
    hist_ret = cv2.calcHist([img], [0], None, [256], [0, 256])
    print(f'hist size: {hist_ret.shape}')
    plt.plot(hist_ret)
    plt.show()


# 均衡化
def learn_equalize_hist():
    img = cv2.imread('picture/lena.jpg', cv2.IMREAD_GRAYSCALE)
    cv2.imshow('src img', img)
    cv_show('learn_equalize_hist', cv2.equalizeHist(img))


# 自适应均衡化 相当于对图像分块进行均衡化
def learn_clahe():
    clahe = cv2.createCLAHE()
    img = cv2.imread('picture/lena.jpg', cv2.IMREAD_GRAYSCALE)
    ret = clahe.apply(img)
    cv_show('learn_clahe', ret)


def learn_filter():
    img = cv2.imread('picture/noise.jpg', cv2.IMREAD_COLOR)
    cv2.imshow('src', img)
    ret = cv2.medianBlur(img, 5)
    cv_show('ret', ret)


def learn_dft():
    img_src = cv2.imread('picture/lena.jpg', cv2.IMREAD_GRAYSCALE)
    cv2.imshow('src ret', img_src)
    img = np.float32(img_src)
    ret = cv2.dft(img, flags=cv2.DFT_COMPLEX_OUTPUT)  # flag指示用复数形式输出
    ret = np.fft.fftshift(ret)
    magnitude_spectrum = 20 * np.log(cv2.magnitude(ret[:, :, 0], ret[:, :, 1]) + 1)
    cv2.normalize(magnitude_spectrum, magnitude_spectrum, 0, 255, cv2.NORM_MINMAX)
    cv2.imshow('dft ret1', np.uint8(magnitude_spectrum))

    center_y = int(ret.shape[0] / 2)
    center_x = int(ret.shape[1] / 2)
    mask = np.ones(ret.shape, np.uint8)
    mask_size = 35
    mask[center_y - mask_size: center_y + mask_size, center_x - mask_size: center_x + mask_size] = 0
    ret = ret * mask
    magnitude_spectrum = 20 * np.log(cv2.magnitude(ret[:, :, 0], ret[:, :, 1]) + 1)
    cv2.normalize(magnitude_spectrum, magnitude_spectrum, 0, 255, cv2.NORM_MINMAX)
    cv2.imshow('dft ret2', np.uint8(magnitude_spectrum))

    ret = np.fft.ifftshift(ret)
    ret = cv2.idft(ret)
    ret = cv2.magnitude(ret[:, :, 0], ret[:, :, 1])
    cv2.normalize(ret, ret, 0, 255, cv2.NORM_MINMAX)
    cv_show('dft ret3', np.uint8(ret))


def learn_morph():
    img = cv2.imread('picture/vulkanscene.jpg', cv2.IMREAD_GRAYSCALE)
    threshold_ret, img = cv2.threshold(img, 0, 255, cv2.THRESH_OTSU)
    # 腐蚀 getStructuringElement ksize 越大效果越明显， MORPH_RECT 使腐蚀膨胀结果趋近放型 MORPH_ELLIPSE 圆形 MORPH_CROSS 强调水平和垂直方向
    cv2.imshow('src', img)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    ret = cv2.erode(img.copy(), kernel)
    cv_show('erode ret', ret)
    # 膨胀
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (13, 13))
    ret = cv2.dilate(img.copy(), kernel)
    cv2.imshow('src', img)
    cv_show('dilate ret', ret)
    # 开操作 先腐蚀再膨胀
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    ret = cv2.morphologyEx(img.copy(), cv2.MORPH_OPEN, kernel)
    cv2.imshow('src', img)
    cv_show('MORPH_OPEN ret', ret)
    # 闭操作 先膨胀再腐蚀
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    ret = cv2.morphologyEx(img.copy(), cv2.MORPH_TOPHAT, kernel)
    cv2.imshow('src', img)
    cv_show('MORPH_TOPHAT ret', ret)
    # 礼帽 原图-开操作
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    ret = cv2.morphologyEx(img.copy(), cv2.MORPH_BLACKHAT, kernel)
    cv2.imshow('src', img)
    cv_show('MORPH_BLACKHAT ret', ret)
    # MORPH_GRADIENT 梯度
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    ret = cv2.morphologyEx(img.copy(), cv2.MORPH_GRADIENT, kernel)
    cv2.imshow('src', img)
    cv_show('MORPH_GRADIENT ret', ret)


def learn_in_range():
    src_img = cv2.imread('picture/box.jpg', cv2.IMREAD_COLOR)
    src_img = cv2.resize(src_img, (int(src_img.shape[1] / 4), int(src_img.shape[0] / 4)))
    cv_show('src ret', src_img)
    # 低于low和高于up的设置为0，其余设置为255
    src_img = cv2.inRange(src_img, (0, 150, 150), (255, 255, 255))
    cv_show('inRange ret', src_img)


def learn_filter_region():
    src_img = cv2.imread('picture/box.jpg', cv2.IMREAD_COLOR)
    src_img = cv2.resize(src_img, (int(src_img.shape[1] / 4), int(src_img.shape[0] / 4)))
    mask = np.zeros_like(src_img)
    h, w = src_img.shape[:2]
    region = np.array([[(int(w * 0.1), int(h * 0.1)),
                       (int(w * 0.1), int(h * 0.9)),
                       (int(w * 0.9), int(h * 0.9)),
                       (int(w * 0.9), int(h * 0.1))]], dtype=int)
    print(region)
    cv2.fillPoly(mask, region, (255, 255, 255))
    cv_show('mask', mask)
    mask_ret = cv2.bitwise_and(src_img, mask)
    cv_show('mask ret', mask_ret)


# cv doc
# https://www.opencv.org.cn/opencvdoc/2.3.2/html/genindex.html
if __name__ == '__main__':
    print(f'opencv version {cv2.__version__}')
    # learn_sobel()
    # learn_canny()
    # learn_pyr()
    # learn_contour()
    # learn_match_template()
    # learn_hist()
    # learn_equalize_hist()
    # learn_clahe()
    # learn_filter()
    # learn_dft()
    # learn_morph()
    # learn_perspective()
    # learn_hough_line()
    # learn_hough_linep()
    # learn_harris()
    # learn_sift()
    # learn_fast()
    # learn_orb()
    image_splicing()
    # learn_in_range()
    # learn_filter_region()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
