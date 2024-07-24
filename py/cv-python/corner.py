import cv2
import numpy as np
from perspective_projection import cv_show


# harris 光照、对比度变化不敏感：光照发生变化时，检测结果可能不同
# 不同尺度(分辨率) 检测结果不同
def learn_harris():
    src_img = cv2.imread('picture/box.jpg', cv2.IMREAD_COLOR)
    img = cv2.resize(src_img.copy(), (int(src_img.shape[1] / 4), int(src_img.shape[0] / 4)))
    cv_show('src img', img)
    # blockSize: 相邻像素块的大小 ksize: sobel算子的大小 k: 0.04-0.06越小越敏感
    harris_ret = cv2.cornerHarris(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 2, 3, 0.04)
    black_ret = harris_ret.copy()
    cv2.normalize(harris_ret, black_ret, 0, 1000, cv2.NORM_MINMAX)
    black_ret = cv2.convertScaleAbs(black_ret)
    cv_show('harris_ret img', black_ret)
    img[harris_ret > (0.1 * harris_ret.max())] = (0, 255, 0)
    cv_show('harris_ret img', img)


def learn_sift():
    src_img = cv2.imread('picture/pen_far.jpg', cv2.IMREAD_COLOR)
    img_far = cv2.resize(src_img.copy(), (int(src_img.shape[1] / 4), int(src_img.shape[0] / 4)))
    sift = cv2.SIFT.create()
    sift.setNFeatures(15)  # 保留最大特征点数量
    key_point_far, des_far = sift.detectAndCompute(img_far, None)
    img_far_show = cv2.drawKeypoints(img_far, key_point_far, None)
    cv_show('SIFT', img_far_show)
    src_img = cv2.imread('picture/pen_near.jpg', cv2.IMREAD_COLOR)
    img_rot = cv2.resize(src_img.copy(), (int(src_img.shape[1] / 4), int(src_img.shape[0] / 4)))
    key_point_rot, des_rot = sift.detectAndCompute(img_rot, None)
    img_rot_show = cv2.drawKeypoints(img_rot, key_point_rot, None)
    cv_show('SIFT', img_rot_show)
    bf_matcher = cv2.BFMatcher.create()
    match_ret = bf_matcher.match(des_far, des_rot)
    ret = cv2.drawMatches(img_far, key_point_far, img_rot, key_point_rot, match_ret, None)
    cv_show('SIFT bf match', ret)


def learn_fast():
    src_img = cv2.imread('picture/box.jpg', cv2.IMREAD_COLOR)
    img = cv2.resize(src_img.copy(), (int(src_img.shape[1] / 4), int(src_img.shape[0] / 4)))
    fast = cv2.FastFeatureDetector().create(threshold=100)
    key_point = fast.detect(img, None)
    fast_ret = cv2.drawKeypoints(img, key_point, None)
    cv_show('fast', fast_ret)


def learn_orb():
    src_img = cv2.imread('picture/box.jpg', cv2.IMREAD_COLOR)
    img = cv2.resize(src_img.copy(), (int(src_img.shape[1] / 4), int(src_img.shape[0] / 4)))
    orb = cv2.ORB.create()
    key_point = orb.detect(img, None)
    img = cv2.drawKeypoints(img, key_point, None)
    cv_show('orb', img)

