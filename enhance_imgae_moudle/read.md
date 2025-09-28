如果你想在**训练前的数据增强**中使用 `enhance_imgae_moudle` 这个模块（即对你的车牌原始图片进行预处理/增强），建议如下：

---

## 【推荐用法】

### 1. 核心类选择

- **`PlateEnhancer`**（主推，适合大部分场景）
  - 用于常规的车牌图像增强，包括尺寸归一化、对比度、去噪、锐化、自适应阈值化、透视校正等。
  - 支持单张和批量处理，适合数据集批处理。
  - 支持自定义增强配置和多级别增强（light/medium/strong）。

- **`NoiseReducer`**
  - 专用于去噪声，比如高噪声环境下单独做去噪。
  - 可配合 PlateEnhancer 灵活使用。

- **`ImageProcessor`**
  - 用于基础的图像操作，如尺寸调整、对比度增强、边缘增强、直方图均衡化等。
  - 可以手动组合使用。

- **`AdvancedPlateEnhancer`**
  - 针对特殊需求，比如 OCR 优化、字符分割、光照校正、超分辨率等，可用于构建更复杂的增强流水线。

### 2. 推荐典型用法

#### 批量增强训练图片（建议用于训练集制作阶段）

```python
from plate_enhancer import PlateEnhancer
import cv2
import glob

enhancer = PlateEnhancer()  # 可传自定义config
image_paths = glob.glob('train_imgs/*.jpg')
for path in image_paths:
    img = cv2.imread(path)
    enhanced = enhancer.enhance_plate_image(img, 'medium')  # 'light'/'medium'/'strong'可选
    cv2.imwrite('train_imgs_enhanced/' + path.split('/')[-1], enhanced)
```

#### 直接用批量接口
如果有很多图片建议用批量接口：

```python
images = [cv2.imread(p) for p in image_paths]
enhanced_images = enhancer.batch_enhance(images, 'medium')
# 保存 enhanced_images
```

#### 只做去噪
```python
from noise_reducer import NoiseReducer
reducer = NoiseReducer()
denoised = reducer.reduce_noise(img, method='bilateral', level='medium')
```

---

## 【总结表格】

| 增强需求         | 推荐类               | 说明/备注                    |
|------------------|----------------------|------------------------------|
| 常规车牌增强     | PlateEnhancer        | 支持批量、增强等级切换       |
| 只做去噪         | NoiseReducer         | 支持不同去噪算法             |
| 图像基础增强     | ImageProcessor       | 尺寸、对比度、边缘、均衡化等 |
| OCR/特殊增强     | AdvancedPlateEnhancer| 字符分割、光照、超分辨率等   |
| 配置增强参数     | EnhancementConfig    | 可选配置预设或自定义         |

---

## 【典型推荐】
> **优先用 `PlateEnhancer`，大多数场景够用，支持自定义配置和批量增强。特殊需求可以配合其它类。**

如需代码模板或想了解如何集成到你的训练数据 pipeline，可告知你的具体需求！