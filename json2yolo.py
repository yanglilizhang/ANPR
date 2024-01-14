import json
import os
import numpy as np
from copy import deepcopy
import cv2

# 自己的数据集可以通过lablme 软件,create polygons标注车牌四个点即可，然后通过json2yolo.py 将数据集转为yolo格式，即可训练

# 收集指定目录下所有文件的路径，并将这些文件路径存储在名为 allFileList 的列表中
def allFilePath(rootPath, allFIleList):
    # listdir 获取当前目录及子目录的所有文件
    fileList = os.listdir(rootPath) #获得文件下所有文件
    for temp in fileList:
        # os.path.join(rootPath, temp) 获取文件的完整路径 isfile 该路径是否是一个文件
        if os.path.isfile(os.path.join(rootPath, temp)):
            # 将该文件的完整路径添加list中
            allFIleList.append(os.path.join(rootPath, temp))  #如果是图片 则添加到allFIleList
        else:
            # 是子目录-递归调用
            allFilePath(os.path.join(rootPath, temp), allFIleList)  #如果是文件夹 接着调用allFilePath


def xywh2yolo(rect, landmarks_sort, img):
    h, w, c = img.shape
    rect[0] = max(0, rect[0])
    rect[1] = max(0, rect[1])
    rect[2] = min(w - 1, rect[2] - rect[0])
    rect[3] = min(h - 1, rect[3] - rect[1])
    annotation = np.zeros((1, 12))
    annotation[0, 0] = (rect[0] + rect[2] / 2) / w  # cx
    annotation[0, 1] = (rect[1] + rect[3] / 2) / h  # cy
    annotation[0, 2] = rect[2] / w  # w
    annotation[0, 3] = rect[3] / h  # h

    annotation[0, 4] = landmarks_sort[0][0] / w  # l0_x
    annotation[0, 5] = landmarks_sort[0][1] / h  # l0_y
    annotation[0, 6] = landmarks_sort[1][0] / w  # l1_x
    annotation[0, 7] = landmarks_sort[1][1] / h  # l1_y
    annotation[0, 8] = landmarks_sort[2][0] / w  # l2_x
    annotation[0, 9] = landmarks_sort[2][1] / h  # l2_y
    annotation[0, 10] = landmarks_sort[3][0] / w  # l3_x
    annotation[0, 11] = landmarks_sort[3][1] / h  # l3_y
    # annotation[0, 12] = (landmarks_sort[0][0]+landmarks_sort[1][0])/2 / w  # l4_x
    # annotation[0, 13] = (landmarks_sort[0][1]+landmarks_sort[1][1])/2 / h  # l4_y
    return annotation


def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    # return the ordered coordinates
    return rect


def four_point_transform(image, pts):
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    # return the warped image
    return warped

# 将特定目录下的图片文件和相应的标注信息（存储在对应的 JSON 文件中）进行处理，并将标注信息保存为符合 YOLO 格式的文本文件
if __name__ == "__main__":
    # 用于存储图片文件的路径
    pic_file_list = []
    # 指定图片文件的路径
    pic_file = r"/mnt/Gpan/Mydata/pytorchPorject/datasets/ccpd/train_bisai/train_bisai"
    # 指定保存小图片的目录路径
    save_small_path = "small"
    # 定义一个标签列表，与实际标注中的标签对应
    label_file = ['0', '1']
    # 将指定目录下所有图片文件的路径收集到 pic_file_list 列表中
    allFilePath(pic_file, pic_file_list)
    count = 0
    index = 0
    for pic_ in pic_file_list:
        if not pic_.endswith(".jpg"):
            continue
        count += 1
        img = cv2.imread(pic_)
        img_name = os.path.basename(pic_)
        txt_name = img_name.replace(".jpg", ".txt")
        # 读取当前图片文件，提取图片名并生成对应的文本文件名，以及替换图片名中的 .jpg 为 .txt，然后构建文本文件的完整路径
        txt_path = os.path.join(pic_file, txt_name)
        # 根据图片路径生成对应的 JSON 文件路径
        json_file_ = pic_.replace(".jpg", ".json")
        # 打开 JSON 文件，读取其中的数据，并使用内部的信息来处理标注数据。
        if not os.path.exists(json_file_):
            continue
        with open(json_file_, 'r', encoding='utf-8') as a:
            data_dict = json.load(a)
            # print(data_dict['shapes'])
            with open(txt_path, "w") as f:
                for data_message in data_dict['shapes']:
                    index += 1
                    label = data_message['label']
                    points = data_message['points']
                    pts = np.array(points)
                    # pts=order_points(pts)
                    # new_img = four_point_transform(img,pts)
                    roi_img_name = label + "_" + str(index) + ".jpg"
                    save_path = os.path.join(save_small_path, roi_img_name)
                    # cv2.imwrite(save_path,new_img)
                    x_max, y_max = np.max(pts, axis=0)
                    x_min, y_min = np.min(pts, axis=0)
                    rect = [x_min, y_min, x_max, y_max]
                    rect1 = deepcopy(rect)
                    annotation = xywh2yolo(rect1, pts, img)
                    # 打印当前正在处理的 JSON 数据
                    print(data_message)
                    # 提取当前数据的标签
                    label = data_message['label']
                    # 使用标签列表找到标签对应的索引
                    str_label = label_file.index(label)
                    # str_label = "0 "
                    str_label = str(str_label) + " "
                    for i in range(len(annotation[0])):
                        str_label = str_label + " " + str(annotation[0][i])
                    str_label = str_label.replace('[', '').replace(']', '')
                    str_label = str_label.replace(',', '') + '\n'

                    f.write(str_label)
            print(count, img_name)
            # point=data_message[points]