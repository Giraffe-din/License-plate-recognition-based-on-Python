import cv2 as cv
import os
import numpy as np
from PIL import Image


def img_read(path):
    return cv.imread(path)


def findContours(img):
    # 使用 cv2.RETR_TREE 常量，可以构建一棵树，该树中的每个节点都表示图像中的一个轮廓，并且父节点表示包含子节点的轮廓
    contours, hierarchy = cv.findContours(img, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    # 去除面积过小的轮廓
    contours = [contour for contour in contours if cv.contourArea(contour) > 2000]
    print("len(contours):", len(contours))

    plate_contours = []
    for contour in contours:
        # 得到该轮廓中的最小外接矩形（即最有可能是车牌的轮廓）
        min_Rect = cv.minAreaRect(contour)
        x, y = min_Rect[1]
        if x < y:
            rate = y/x
        else:
            rate = x/y

        if 2 < rate < 7:
            plate_contours.append(min_Rect)

    return plate_contours


def limit(point):
    if point[0] < 0:
        point[0] = 0
    if point[1] < 0:
        point[1] = 0


def angleTransform(plate_contours, init_img, width, height):
    # 对寻找到的轮廓进行角度的矫正
    plate_imgs = []
    # contour 的 值为[轮廓中心坐标（x,y）, （width, height）, angle]
    for contour in plate_contours:
        if -1 < contour[2] < 1:
            angle = 1
        else:
            angle = contour[2]

        contour = (contour[0], (contour[1][0] + 5, contour[1][1] + 5), angle)
        # contour = (contour[0], contour[1], angle)
        # 获得该矩形轮廓的四个角坐标(顺序为左上，左下，右下，右上)
        points = cv.boxPoints(contour)

        up_point = [0, 0]
        down_point = height
        left_point = width
        right_point = [0, 0]
        for point in points:
            if left_point[0] > point[0]:
                if left_point[0] > point[0]:
                    left_point = point
                if down_point[1] > point[1]:
                    down_point = point
                if up_point[1] < point[1]:
                    up_point = point
                if right_point[0] < point[0]:
                    right_point = point
        # 用坐标来判断角度
        limit(left_point)
        limit(right_point)
        limit(up_point)
        if left_point[1] < right_point[1]:
            new_right_point = [right_point[0], up_point[1]]
            original = np.float32([left_point, up_point, right_point])
            transform = np.float32([left_point, up_point, new_right_point])
            # 仿射变换矩阵
            matrix = cv.getAffineTransform(original, transform)
            # 得到变化后的图像
            transformed_img = cv.warpAffine(init_img, matrix, (width, height))
            limit(new_right_point)
            # 需要把坐标点调整为整数
            plate_img = transformed_img[int(left_point[1]):int(up_point[1]), int(left_point[0]):int(new_right_point[0])]
            plate_imgs.append(plate_img)
        # 角度为负：
        elif left_point[1] > right_point[1]:
            new_left_point = [left_point[0], up_point[1]]
            original = np.float32([left_point, up_point, right_point])
            transform = np.float32([new_left_point, up_point, right_point])
            # 仿射变换矩阵
            matrix = cv.getAffineTransform(original, transform)
            # 得到变化后的图像
            transformed_img = cv.warpAffine(init_img, matrix, (width, height))
            limit(new_left_point)
            # 需要把坐标点调整为整数
            plate_img = transformed_img[int(right_point[1]):int(up_point[1]), int(new_left_point[0]):int(right_point[0])]
        plate_imgs.append(plate_img)

    return plate_imgs


def split_plate(img, waves):




















