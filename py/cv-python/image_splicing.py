import cv2
import numpy as np
from perspective_projection import cv_show


def image_splicing():
    book_l = cv2.imread('picture/book_l.jpg', cv2.IMREAD_COLOR)
    book_r = cv2.imread('picture/book_r.jpg', cv2.IMREAD_COLOR)
    book_l = cv2.resize(book_l, (int(book_l.shape[1] / 4), int(book_l.shape[0] / 4)))
    book_r = cv2.resize(book_r, (int(book_r.shape[1] / 4), int(book_r.shape[0] / 4)))
    sift_detector = cv2.SIFT.create()
    sift_detector.setContrastThreshold(0.06)
    kp_l, des_l = sift_detector.detectAndCompute(book_l, None)
    kp_r, des_r = sift_detector.detectAndCompute(book_r, None)
    if des_r is not None and des_l is not None:
        des_l = des_l.astype(np.float32)
        des_r = des_r.astype(np.float32)
    else:
        print('splicing fail des is null')
        return
    flann_matcher = cv2.BFMatcher()
    match_ret = flann_matcher.knnMatch(des_l, des_r, k=2)
    good_ret = []
    for match_1, match_2 in match_ret:
        if match_1.distance < 0.7 * match_2.distance:
            good_ret.append(match_1)
    if len(good_ret) < 4:
        print('no enough good match ret')
        return
    match_image = cv2.drawMatches(book_l, kp_l, book_r, kp_r, good_ret, None)
    cv_show('match_image', match_image)

    if len(good_ret) > 4:
        src_pts = np.float32([kp_l[m.queryIdx].pt for m in good_ret])
        dst_pts = np.float32([kp_r[m.trainIdx].pt for m in good_ret])
        # 计算单应性矩阵
        H, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)
        # 透视变换右图像
        h, w, _ = book_l.shape
        book_r_warped = cv2.warpPerspective(book_r, H, (w * 2, h * 2))
    
        # 在左图中填入右图
        book_r_warped[0:h, 0:w] = book_l
    
        # 显示拼接后的图像
        cv_show('Stitched Image', book_r_warped)
    else:
        print('Not enough matches are found - {}/{}'.format(len(good_ret), 4))
        return

