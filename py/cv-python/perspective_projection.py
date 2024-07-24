import cv2
import numpy as np


def cv_show(window_name, img, scale=1):
    if scale != 1:
        img_show = cv2.resize(img.copy(), (int(img.shape[1] / scale), int(img.shape[0] / scale)))
    else:
        img_show = img
    cv2.imshow(window_name, img_show)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def learn_perspective():
    src_img = cv2.imread('picture/box.jpg', cv2.IMREAD_COLOR)
    cv_show('src img', src_img, 4)
    img_gray = cv2.cvtColor(src_img, cv2.COLOR_BGR2GRAY)
    img_gray = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    cont = cv2.findContours(img_gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    approved_cons = []
    for index, contour in enumerate(cont[0]):
        eps = 0.01
        while 1:
            approx_cont = cv2.approxPolyDP(contour, cv2.arcLength(contour, True) * eps, True)
            if len(approx_cont) <= 4:
                break
            else:
                eps = eps * 2
        approved_cons.append(approx_cont)
    sorted_contours = sorted(approved_cons, key=cv2.contourArea, reverse=True)
    print(f'max area contour {sorted_contours[0]}')
    cv_show('cont', cv2.drawContours(src_img.copy(), sorted_contours, 0, (0, 255, 0), 5), 4)
    l2_x_top_dis = np.linalg.norm(sorted_contours[0][1] - sorted_contours[0][0])
    l2_x_bottom_dis = np.linalg.norm(sorted_contours[0][3] - sorted_contours[0][2])
    width_max = max([int(l2_x_top_dis), int(l2_x_bottom_dis)])
    l2_y_left_dis = np.linalg.norm(sorted_contours[0][3] - sorted_contours[0][0])
    l2_y_right_dis = np.linalg.norm(sorted_contours[0][1] - sorted_contours[0][2])
    height_max = max([int(l2_y_left_dis), int(l2_y_right_dis)])
    perspective_mat = cv2.getPerspectiveTransform(np.array(sorted_contours[0], dtype=np.float32),
                                                  np.array([[0, 0], [0, width_max], [height_max, width_max], [height_max, 0]], dtype=np.float32))
    perspective_ret = cv2.warpPerspective(src_img, perspective_mat, (height_max, width_max))
    rotate_mat = cv2.getRotationMatrix2D((int(perspective_ret.shape[1]/2),int(perspective_ret.shape[0]/2)), 180, 1.0)
    rotate_ret = cv2.warpAffine(perspective_ret, rotate_mat, (height_max, width_max))
    cv_show('perspective ret', rotate_ret, 4)
