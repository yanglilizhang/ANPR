from plate_recognition.plateNet import myNet_ocr, myNet_ocr_color
import torch
import torch.nn as nn
import cv2
import numpy as np
import os
import time
import sys


def cv_imread(path):  # 可以读取中文路径的图片
    img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), -1)
    return img


def allFilePath(rootPath, allFIleList):
    fileList = os.listdir(rootPath)
    for temp in fileList:
        if os.path.isfile(os.path.join(rootPath, temp)):
            if temp.endswith('.jpg') or temp.endswith('.png') or temp.endswith('.JPG'):
                allFIleList.append(os.path.join(rootPath, temp))
        else:
            allFilePath(os.path.join(rootPath, temp), allFIleList)


device = torch.device('cuda') if torch.cuda.is_available() else torch.device("cpu")
color = ['黑色', '蓝色', '绿色', '白色', '黄色']
plateName = r"#京沪津渝冀晋蒙辽吉黑苏浙皖闽赣鲁豫鄂湘粤桂琼川贵云藏陕甘青宁新学警港澳挂使领民航危0123456789ABCDEFGHJKLMNPQRSTUVWXYZ险品"
mean_value, std_value = (0.588, 0.193)


def decodePlate(preds):
    pre = 0
    newPreds = []
    index = []
    for i in range(len(preds)):
        if preds[i] != 0 and preds[i] != pre:
            newPreds.append(preds[i])
            index.append(i)
        pre = preds[i]
    return newPreds, index


def image_processing(img, device):
    """
    对输入图像进行处理并转换为张量

    参数：
    img：输入图像（numpy数组）
    device：指定设备（torch.device）

    返回值：
    img：处理后的图像张量
    """
    img = cv2.resize(img, (168, 48))
    img = np.reshape(img, (48, 168, 3))

    # normalize
    img = img.astype(np.float32)
    img = (img / 255. - mean_value) / std_value
    img = img.transpose([2, 0, 1])
    img = torch.from_numpy(img)

    img = img.to(device)
    img = img.view(1, *img.size())
    return img


def get_plate_result(img, device, model, is_color=False):
    """
    识别车牌的结果，包括车牌号、每个字符的概率，以及可选的车牌颜色和颜色概率。

    参数:
    img: 输入的图像，可以是图像路径或图像数据。
    device: 指定运行模型的设备，如"cpu"或"cuda:0"。
    model: 用于识别车牌的模型。
    is_color: 是否识别车牌颜色。默认为False，表示不识别颜色。

    返回值:
    如果is_color为True，则返回一个元组，包含车牌号字符串、每个字符的概率、车牌颜色和颜色的概率。
    如果is_color为False，则返回一个元组，包含车牌号字符串和每个字符的概率。
    """
    # 图像预处理
    input = image_processing(img, device)
    if is_color:  # 是否识别颜色
        # 预测字符和颜色
        preds, color_preds = model(input)
        # 对颜色预测应用softmax
        color_preds = torch.softmax(color_preds, dim=-1)
        # 获取颜色的最高置信度和索引
        color_conf, color_index = torch.max(color_preds, dim=-1)
        color_conf = color_conf.item()
    else:
        # 只预测字符
        preds = model(input)
    # 对字符预测应用softmax
    preds = torch.softmax(preds, dim=-1)
    # 获取最高概率的字符和其索引
    prob, index = preds.max(dim=-1)
    # 将tensor转换为numpy数组
    index = index.view(-1).detach().cpu().numpy()
    prob = prob.view(-1).detach().cpu().numpy()
    # 打印prob 21个值
    #  prob:[    0.71047   1     0.80623    1    1     1    1      1   1     0.99995    1     0.94048  1   1
    #  0.99903     0.99994   1  1      1     0.67647     1]
    # print(f"prob:{prob}")

    # 解码字符索引为字符预测
    newPreds, new_index = decodePlate(index)
    # 重新调整概率值，对应于解码后的字符
    prob = prob[new_index]
    # newPreds:[52, 57, 53, 51, 43, 51, 45],new_index:[3, 6, 9, 11, 14, 15, 18]
    # prob:[          1           1     0.99995     0.94048     0.99903     0.99994           1]
    # print(f"newPreds:{newPreds},new_index:{new_index},prob:{prob}")
    plate = ""
    for i in newPreds:
        plate += plateName[i]

    if is_color:
        # 如果识别颜色，返回车牌号、每个字符的概率、车牌颜色和颜色的概率
        return plate, prob, color[color_index], color_conf
    else:
        # 如果不识别颜色，仅返回车牌号和每个字符的概率
        return plate, prob



def init_model(device, model_path, is_color=False):
    """
     初始化识别车牌号及颜色的模型

     参数：
     device：CPU/GPU
     model_path：模型文件的路径
     """
    # print( print(sys.path))
    # model_path ="plate_recognition/model/checkpoint_61_acc_0.9715.pth"
    check_point = torch.load(model_path, map_location=device)
    model_state = check_point['state_dict']
    cfg = check_point['cfg']
    color_classes = 0
    if is_color:
        color_classes = 5  # 颜色类别数

    # 创建模型对象
    model = myNet_ocr_color(num_classes=len(plateName), export=True, cfg=cfg, color_num=color_classes)

    # 载入模型权重
    model.load_state_dict(model_state, strict=False)
    # 将模型移动到指定设备上
    model.to(device)
    # 设置模型为评估模式
    model.eval()
    return model


# model = init_model(device)
if __name__ == '__main__':
    model_path = r"weights/plate_rec_color.pth"
    image_path = "images/tmp2424.png"
    testPath = r"/mnt/Gpan/Mydata/pytorchPorject/CRNN/crnn_plate_recognition/images"
    fileList = []
    allFilePath(testPath, fileList)
    #    result = get_plate_result(image_path,device)
    #    print(result)
    is_color = False
    model = init_model(device, model_path, is_color=is_color)
    right = 0
    begin = time.time()

    for imge_path in fileList:
        img = cv2.imread(imge_path)
        if is_color:
            plate, _, plate_color, _ = get_plate_result(img, device, model, is_color=is_color)
            print(plate)
        else:
            plate, _ = get_plate_result(img, device, model, is_color=is_color)
            print(plate, imge_path)
