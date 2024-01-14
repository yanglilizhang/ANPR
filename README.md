# **Automatic Number Plate Detection system**

github:[https://github.com/201918010332Thomas/ANPR](https://github.com/201918010332Thomas/ANPR)

**Software environment requirements: python >=3.6  pytorch >=1.7**

## **GUI program:**

```
python anpr.py
```

## License plate detection training

1. **Dataset**

   This project uses open source datasets CCPD and CRPD.

   The dataset label format is YOLO format：车牌识别的数据集格式 label x y w h 就跟目标检测一样 右侧的8个点分别是车牌的左上tl、右上tr、左下bl、右下br

   ```
   label x y w h  pt1x pt1y pt2x pt2y pt3x pt3y pt4x pt4y
   ```

   The key points are in order (top left, top right, bottom right, bottom left).

   The coordinates are all normalized, where x and y are the center point divided by the width and height of the image, w and h are the width and height of the box divided by the width and height of the image, ptx and pty are the key point coordinates divided by the width and height.
   关键点依次是（左上，右上，右下，左下） 坐标都是经过归一化，x,y是中心点除以图片宽高，w,h是框的宽高除以图片宽高，ptx，pty是关键点坐标除以宽高
   自己的数据集可以通过lablme 软件,create polygons标注车牌四个点即可，然后通过json2yolo.py 将数据集转为yolo格式，即可训练
   
2. **Modify the data/widerface.yaml file**

   ```
   train: /your/train/path #This is the training dataset, modify to your path.
   val: /your/val/path     #This is the evaluation dataset, modify to your path.
   # number of classes
   nc: 2                   #Here we use 2 categories, 0 single layer license plate 1 double layer license plate.

   # class names
   names: [ 'single','double']

   ```
3. **车牌检测训练-Train**

   ```
   python train.py --data data/widerface.yaml --cfg models/yolov5n-0.5.yaml --weights weights/plate_detect.pt --epoch 250
   python train.py --data data/widerface.yaml --cfg models/yolov5n-0.5.yaml --weights weights/plate_detect-copy.pt --epoch 2
   python train.py --data data/widerface.yaml --cfg models/yolov5n-0.5.yaml --weights weights/plate_detect-copy.pt --batch-size 5 --workers 0 --epoch 2

   ```

   The result exists in the run folder.
4. **Detection model onnx export**
   To export the detection model to onnx, onnx sim needs to be installed. **[onnx-simplifier](https://github.com/daquexian/onnx-simplifier)**

   ```
   1. python export.py --weights ./weights/plate_detect.pt --img 640 --batch 1
   2. onnxsim weights/plate_detect.onnx weights/plate_detect.onnx
   ```

   **Using trained models for detection**

   ```
   python detect_demo.py  --detect_model weights/plate_detect.pt
   ```

## License plate recognition training

The training link for license plate recognition is as follows:

[License plate recognition training](https://github.com/201918010332Thomas/CRNN_LPR)

#### **The results of license plate recognition are as follows:**

[//]: # (![Image]&#40;image/README/test_12.jpg&#41;)

## Arrange

1.**onnx demo**

The onnx model can be found in [onnx model](https://pan.baidu.com/s/1UmWN2kpRP96h2cM6Pi-now), with extraction code: ixyr

python onnx_infer.py --detect_model weights/plate_detect.onnx  --rec_model weights/plate_rec.onnx  --image_path imgs --output result_onnx

2.**tensorrt**

Deployment can be found in [tensorrt_plate](https://github.com/we0091234/chinese_plate_tensorrt)

3.**openvino demo**

Version 2022.2

```
 python openvino_infer.py --detect_model weights/plate_detect.onnx --rec_model weights/plate_rec.onnx --image_path imgs --output result_openvino
```

## References

* [https://github.com/deepcam-cn/yolov5-face](https://github.com/deepcam-cn/yolov5-face)
* [https://github.com/meijieru/crnn.pytorch](https://github.com/meijieru/crnn.pytorch)

  File "/Users/anaconda3/envs/py37/lib/python3.7/site-packages/torch/autograd/__init__.py", line 199, in backward
    allow_unreachable=True, accumulate_grad=True)
Calls into the C++ engine to run the backward pass
RuntimeError: one of the variables needed for gradient computation has been modified by an inplace operation:
* [torch.FloatTensor [38, 2]], which is output 0 of AsStridedBackward0, is at version 4; expected version 0 instead.
* Hint: the backtrace further above shows the operation that failed to compute its gradient. 
* The variable in question was changed in there or anywhere later. Good luck!

Variable._execution_engine.run_backward( # Calls into the C++ engine to run the backward pass
RuntimeError: one of the variables needed for gradient computation has been modified by an inplace operation:
[torch.FloatTensor [204, 2]], which is output 0 of AsStridedBackward0, is at version 4; expected version 0 instead. 
Hint: the backtrace further above shows the operation that failed to compute its gradient.
The variable in question was changed in there or anywhere later. Good luck!

pip install onnx
pip install onnxruntime
安装onnxsim:
pip install onnx-simplifier
安装失败 https://cloud.tencent.com/developer/ask/sof/106982617
Collecting onnx
  Using cached onnx-1.11.0.tar.gz (9.9 MB)
  Preparing metadata (setup.py) ... error
  error: subprocess-exited-with-error

  × python setup.py egg_info did not run successfully.
  │ exit code: 1
  ╰─> [7 lines of output]
      fatal: not a git repository (or any of the parent directories): .git
      Traceback (most recent call last):
        File "<string>", line 2, in <module>
        File "<pip-setuptools-caller>", line 34, in <module>
        File "C:\Users\Red007Master\AppData\Local\Temp\pip-download-coisn9j3\onnx_f49974f8ac4344abaca0eecae41c15e4\setup.py", line 86, in <module>
          assert CMAKE, 'Could not find "cmake" executable!'
      AssertionError: Could not find "cmake" executable!
      [end of output]

  note: This error originates from a subprocess, and is likely not a problem with pip.
error: metadata-generation-failed

需安装cmake：sudo apt install cmake 或 brew install cmake

解决方法：电脑安装cmake
没安装brew，先安装brew 
/bin/zsh -c "$(curl -fsSL https://gitee.com/cunkai/HomebrewCN/raw/master/Homebrew.sh)"
安好brew后输入：
brew install cmake
brew install protobuf
最后 pip install onnx

# [可选] 使用阿里源加速
pip install -i http://mirrors.aliyun.com/pypi/simple onnx-simplifier
# 导出onnx模型 python export.py --weights ./weights/plate_detect-copy.pt --img 640 --batch 1
# 简化onnx模型 onnxsim weights/plate_detect.onnx weights/plate_detect.onnx
