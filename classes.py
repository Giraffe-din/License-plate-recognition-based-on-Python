import cv2 as cv
import os
import numpy as np
import functions as fc
import recognization as rc

SZ = 20  # 训练图片长宽
MAX_WIDTH = 1000  # 原始图片最大宽度
Min_Area = 2000  # 车牌区域允许最大面积
PROVINCE_START = 1000


class DataModel(object):
    def load(self, fn):
        self.model = self.model.load(fn)

    def save(self, fn):
        self.model.save(fn)


class SVM(DataModel):
    def __init__(self, C=1, gamma=0.5):
        self.model = cv.ml.SVM_create()
        self.model.setGamma(gamma)
        self.model.setC(C)
        self.model.setKernel(cv.ml.SVM_RBF)
        self.model.setType(cv.ml.SVM_C_SVC)

    # 字符识别
    def predict(self, samples):
        result = self.model.predict(samples)
        return result[1].ravel()


class Predictor:
    def __init__(self):
        pass

    def train_svm(self):
        # 识别英文字母和数字
        self.model = SVM(C=1, gamma=0.5)
        # 识别中文
        self.modelchinese = SVM(C=1, gamma=0.5)
        if os.path.exists("svm.dat"):
            self.model.load("svm.dat")
        if os.path.exists("svmchinese.dat"):
            self.modelchinese.load("svmchinese.dat")

    def preprocess_img(self, path):
        """
        :param car_pic_file: 图像文件
        :return:预处理的图像文件（经过灰度化，二值处理，形态学处理） 原图像文件
        """
        init_img = fc.img_read(path)
        height, width = init_img.shape[:2]  # 取彩色图片的高、宽
        # 最大图片宽度定义为1000
        if width > 1000:
            resize_rate = 1000 / width
            # cv.INTER_AREA - 局部像素重采样，适合缩小图片。
            init_img = cv.resize(init_img, (1000, int(height * resize_rate)), interpolation=cv.INTER_AREA)
        # 使用高斯滤波对图像进行平滑处理
        init_img = cv.GaussianBlur(init_img, (5, 5), 0)
        # 转化成灰度图像
        gray_img = cv.cvtColor(init_img, cv.COLOR_BGR2GRAY)
        # 创建一个值全为1的n维数组
        Matrix = np.ones((20, 20), np.uint8)
        # 对图像进行开运算去除冗余噪声
        opened_img = cv.morphologyEx(gray_img, cv.MORPH_OPEN, Matrix)
        # 图片叠加与融合
        # g (x) = (1 − α)f0 (x) + αf1 (x)   a→（0，1）不同的a值可以实现不同的效果
        opened_img = cv.addWeighted(gray_img, 1, opened_img, -1, 0)
        # 将图像进行二值化处理
        ret, thresh_img = cv.threshold(opened_img, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
        # 边缘检测中较大的阈值用于检测图像中明显的边缘，较小的阈值用于将这些间断的边缘连接起来
        edge_img = cv.Canny(thresh_img, 120, 250)
        cv.imwrite("tmp/img_edge.jpg", edge_img)

        Matrix = np.ones((4, 19), np.uint8)
        # 闭运算:先膨胀再腐蚀
        closed_img = cv.morphologyEx(edge_img, cv.MORPH_CLOSE, Matrix)
        # 最后对齐进行形态学处理
        closed_img = cv.morphologyEx(closed_img, cv.MORPH_OPEN, Matrix)
        return closed_img, init_img

    def img_only_color(self, filename, init_img, img_contours):
        """
        :param filename: 图像文件
        :param init_img: 原图像文件
        :return: 识别到的字符、定位的车牌图像、车牌颜色
        """
        height, width = img_contours.shape[:2]  # #取彩色图片的高、宽

        lower_blue = np.array([100, 110, 110])
        upper_blue = np.array([130, 255, 255])
        lower_yellow = np.array([15, 55, 55])
        upper_yellow = np.array([50, 255, 255])
        lower_green = np.array([50, 50, 50])
        upper_green = np.array([100, 255, 255])

        # 将图像模型转为HSL模型便于操作和提取信息
        hsv = cv.cvtColor(filename, cv.COLOR_BGR2HSV)
        # 利用cv.inRange函数设阈值，去除背景部分
        # 参数1：原图
        # 参数2：图像中低于值，图像值变为0
        # 参数3：图像中高于值，图像值变为0
        mask_blue = cv.inRange(hsv, lower_blue, upper_blue)
        mask_yellow = cv.inRange(hsv, lower_yellow, upper_yellow)
        mask_green = cv.inRange(hsv, lower_yellow, upper_green)

        # 图像算术运算  按位运算 按位操作有： AND， OR， NOT， XOR 等
        output = cv.bitwise_and(hsv, hsv, mask=mask_blue + mask_yellow + mask_green)
        # 根据阈值找到对应颜色

        colors = cv.cvtColor(output, cv.COLOR_BGR2GRAY)
        Matrix = np.ones((20, 20), np.uint8)
        closed_img = cv.morphologyEx(colors, cv.MORPH_CLOSE, Matrix)  # 闭运算
        opened_img = cv.morphologyEx(closed_img, cv.MORPH_OPEN, Matrix)  # 开运算

        # cv.imshow("img", img_edge2)
        # cv.waitKey(0)
        plate_contours = fc.img_findContours(opened_img)
        plate_imgs = fc.img_Transform(plate_contours, init_img, width, height)
        colors, car_imgs = fc.img_color(plate_imgs)

        predict_result = []
        predict_plate = ""
        plate_img = None
        plate_color = None

        for i, color in enumerate(colors):

            if color in ("blue", "yello", "green"):
                plate_img = plate_imgs[i]
                try:
                    gray_img = cv.cvtColor(plate_img, cv.COLOR_BGR2GRAY)
                except:
                    print("gray转换失败")

                # 黄、绿车牌字符比背景暗、与蓝车牌刚好相反，所以黄、绿车牌需要反向
                if color == "green" or color == "yello":
                    gray_img = cv.bitwise_not(gray_img)
                ret, gray_img = cv.threshold(gray_img, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
                x_histogram = np.sum(gray_img, axis=1)

                x_min = np.min(x_histogram)
                x_average = np.sum(x_histogram) / x_histogram.shape[0]
                x_threshold = (x_min + x_average) / 2
                wave_peaks = fc.find_waves(x_threshold, x_histogram)

                if len(wave_peaks) == 0:
                    # print("peak less 0:")
                    continue
                # 认为水平方向，最大的波峰为车牌区域
                wave = max(wave_peaks, key=lambda x: x[1] - x[0])

                gray_img = gray_img[wave[0]:wave[1]]
                # 查找垂直直方图波峰
                row_num, col_num = gray_img.shape[:2]
                # 去掉车牌上下边缘1个像素，避免白边影响阈值判断
                gray_img = gray_img[1:row_num - 1]
                y_histogram = np.sum(gray_img, axis=0)
                y_min = np.min(y_histogram)
                y_average = np.sum(y_histogram) / y_histogram.shape[0]
                y_threshold = (y_min + y_average) / 5  # U和0要求阈值偏小，否则U和0会被分成两半
                wave_peaks = fc.find_waves(y_threshold, y_histogram)
                if len(wave_peaks) < 6:
                    print("peak less 1:", len(wave_peaks))
                    continue

                wave = max(wave_peaks, key=lambda x: x[1] - x[0])
                max_wave_dis = wave[1] - wave[0]
                # 判断是否是左侧车牌边缘
                if wave_peaks[0][1] - wave_peaks[0][0] < max_wave_dis / 3 and wave_peaks[0][0] == 0:
                    wave_peaks.pop(0)

                # 组合分离汉字
                cur_dis = 0
                for i, wave in enumerate(wave_peaks):
                    if wave[1] - wave[0] + cur_dis > max_wave_dis * 0.6:
                        break
                    else:
                        cur_dis += wave[1] - wave[0]
                if i > 0:
                    wave = (wave_peaks[0][0], wave_peaks[i][1])
                    wave_peaks = wave_peaks[i + 1:]
                    wave_peaks.insert(0, wave)

                point = wave_peaks[2]
                point_img = gray_img[:, point[0]:point[1]]
                if np.mean(point_img) < 255 / 5:
                    wave_peaks.pop(2)

                if len(wave_peaks) <= 6:
                    # print("peak less 2:", len(wave_peaks))
                    continue
                # print(wave_peaks)

                # wave_peaks  车牌字符 类型列表 包含7个（开始的横坐标，结束的横坐标）

                part_plates = fc.seperate_plate(gray_img, wave_peaks)

                for i, part_plate in enumerate(part_plates):
                    # 可能是固定车牌的铆钉

                    if np.mean(part_plate) < 255 / 5:
                        # print("a point")
                        continue
                    part_plate_old = part_plate

                    w = abs(part_plate.shape[1] - SZ) // 2

                    part_plate = cv.copyMakeBorder(part_plate, 0, 0, w, w, cv.BORDER_CONSTANT, value=[0, 0, 0])
                    part_plate = cv.resize(part_plate, (SZ, SZ), interpolation=cv.INTER_AREA)

                    part_plate = rc.preprocess_hog([part_plate])
                    if i == 0:
                        resp = self.modelchinese.predict(part_plate)
                        charactor = rc.provinces[int(resp[0]) - PROVINCE_START]
                    else:
                        resp = self.model.predict(part_plate)
                        charactor = chr(resp[0])
                    # 判断最后一个数是否是车牌边缘，假设车牌边缘被认为是1
                    if charactor == "1" and i == len(part_plates) - 1:
                        if part_plate_old.shape[0] / part_plate_old.shape[1] >= 7:  # 1太细，认为是边缘
                            continue
                    predict_result.append(charactor)
                    predict_str = "".join(predict_result)

                roi = plate_img
                plate_color = color
                break
        cv.imwrite("tmp/img_caijian.jpg", roi)
        return predict_str, roi, plate_color  # 识别到的字符、定位的车牌图像、车牌颜色

