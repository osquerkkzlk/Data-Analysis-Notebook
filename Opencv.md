# Opencv

---

### 1. 基本概念

- **作用**：提供图像/视频读取、处理、转换、特征提取等功能，适合深度学习中的数据准备和结果可视化。
- **依赖**：需要安装 `opencv-python`（`pip install opencv-python`）。
- **核心模块**：图像读取/写入、颜色空间转换、几何变换、特征检测、视频处理等。

---

### 2. 常用功能与用法

#### （1）图像读取与保存

- **读取图像**：

  ```python
  import cv2
  img = cv2.imread('image.jpg', cv2.IMREAD_COLOR)  # 彩色图像（BGR）
  # 其他模式：cv2.IMREAD_GRAYSCALE（灰度），cv2.IMREAD_UNCHANGED（包含透明通道）
  ```

  - 返回：NumPy 数组，形状 `(height, width, channels)`（彩色为 3，灰度为 1）。
  - 注意：OpenCV 使用 BGR 颜色顺序（非 RGB）。

- **保存图像**：

  ```python
  cv2.imwrite('output.jpg', img)
  ```

  - 支持格式：`.jpg`、`.png`、`.bmp` 等。

#### （2）颜色空间转换

- **BGR ↔ RGB**（深度学习常用 RGB）：

  ```python
  img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
  img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
  ```

- **BGR ↔ 灰度**：

  ```python
  img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  ```

- **其他**：`COLOR_BGR2HSV`、`COLOR_BGR2LAB` 等。

#### （3）图像缩放与裁剪

- **缩放**：

  ```python
  img_resized = cv2.resize(img, (width, height), interpolation=cv2.INTER_LINEAR)
  # 常见插值：INTER_NEAREST（最近邻）、INTER_LINEAR（双线性）、INTER_CUBIC（双三次）
  ```

  - 深度学习中常用于统一输入尺寸（如 224x224）。

- **裁剪**：

  ```python
  img_cropped = img[y:y+h, x:x+w]  # NumPy 切片，[y, x, channels]
  ```

#### （4）图像预处理

- **归一化**（深度学习输入通常 [0, 1] 或 [-1, 1]）：

  ```python
  img_normalized = img.astype(float) / 255.0
  ```

- **标准化**（减均值、除标准差）：

  ```python
  img_standardized = (img - mean) / std
  ```

- **数据增强**：

  - 旋转：

    ```python
    (h, w) = img.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle=45, scale=1.0)
    img_rotated = cv2.warpAffine(img, M, (w, h))
    ```

  - 翻转：

    ```python
    img_flipped = cv2.flip(img, 1)  # 1: 水平翻转，0: 垂直翻转，-1: 两者
    ```

  - 随机裁剪、亮度/对比度调整：

    ```python
    img_adjusted = cv2.convertScaleAbs(img, alpha=1.2, beta=10)  # alpha: 对比度，beta: 亮度
    ```

#### （5）绘制与标注

- **绘制矩形**（用于可视化边界框）：

  ```python
  cv2.rectangle(img, (x1, y1), (x2, y2), color=(0, 255, 0), thickness=2)
  ```

- **绘制文本**（标注类别或分数）：

  ```python
  cv2.putText(img, text="Cat", org=(x, y), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
              fontScale=1.0, color=(0, 0, 255), thickness=2)
  ```

- **绘制线条/圆形**：

  ```python
  cv2.line(img, (x1, y1), (x2, y2), color=(255, 0, 0), thickness=2)
  cv2.circle(img, (x, y), radius=5, color=(0, 255, 0), thickness=-1)
  ```

#### （6）特征检测与匹配

- **边缘检测**（Canny）：

  ```python
  edges = cv2.Canny(img_gray, threshold1=100, threshold2=200)
  ```

- **角点检测**（Harris）：

  ```python
  corners = cv2.cornerHarris(img_gray, blockSize=2, ksize=3, k=0.04)
  ```

- **SIFT/ORB 特征**（用于传统视觉任务）：

  ```python
  orb = cv2.ORB_create()
  keypoints, descriptors = orb.detectAndCompute(img_gray, None)
  ```

#### （7）视频处理

- **读取视频**：

  ```python
  cap = cv2.VideoCapture('video.mp4')  # 或 0 表示摄像头
  while cap.isOpened():
      ret, frame = cap.read()
      if not ret:
          break
      # 处理 frame
  cap.release()
  ```

- **保存视频**：

  ```python
  out = cv2.VideoWriter('output.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps=30, frameSize=(width, height))
  out.write(frame)
  out.release()
  ```

#### （8）显示图像

- **显示窗口**：

  ```python
  cv2.imshow('Image', img)
  cv2.waitKey(0)  # 等待按键（0 表示无限等待）
  cv2.destroyAllWindows()
  ```

  - 注意：Jupyter 环境可能不支持 `imshow`，建议用 `matplotlib` 显示：

    ```python
    import matplotlib.pyplot as plt
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.show()
    ```

---

### 3. 深度学习中的常见应用

1. **数据预处理**：
   - 读取图像、调整大小、归一化、颜色转换（如 BGR 到 RGB）。
   - 数据增强（旋转、翻转、随机裁剪）以提高模型泛化能力。
2. **可视化**：
   - 绘制预测边界框、分割掩码、关键点。
   - 显示训练过程中的图像或中间特征图。
3. **视频处理**：
   - 实时目标检测、跟踪或动作识别。
   - 提取视频帧作为训练数据。
4. **传统特征提取**：
   - 使用 SIFT/ORB 提取特征，结合深度学习模型（如特征融合）。

---

### 4. 高级功能

1. **图像滤波**：

   - **高斯模糊**：

     ```python
     img_blur = cv2.GaussianBlur(img, (5, 5), sigmaX=0)
     ```

   - **中值滤波**（去噪）：

     ```python
     img_median = cv2.medianBlur(img, ksize=5)
     ```

2. **阈值处理**：

   - **全局阈值**：

     ```python
     _, img_thresh = cv2.threshold(img_gray, thresh=127, maxval=255, type=cv2.THRESH_BINARY)
     ```

   - **自适应阈值**：

     ```python
     img_adaptive = cv2.adaptiveThreshold(img_gray, maxValue=255, adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                         thresholdType=cv2.THRESH_BINARY, blockSize=11, C=2)
     ```

3. **轮廓检测**：

   ```python
   contours, _ = cv2.findContours(edges, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)
   cv2.drawContours(img, contours, contourIdx=-1, color=(0, 255, 0), thickness=2)
   ```

4. **模板匹配**：

   ```python
   template = cv2.imread('template.jpg', cv2.IMREAD_GRAYSCALE)
   result = cv2.matchTemplate(img_gray, template, method=cv2.TM_CCOEFF_NORMED)
   min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
   ```

---

### 5. 最佳实践

1. **颜色空间**：

   - 始终确认颜色顺序（OpenCV 用 BGR，深度学习框架如 PyTorch/TensorFlow 用 RGB）。
   - 转换后检查：`img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)`。

2. **内存管理**：

   - 大量图像处理时，释放资源（如 `cap.release()`、`cv2.destroyAllWindows()`）。
   - 使用生成器加载图像，减少内存占用。

3. **性能优化**：

   - 批量处理图像时，使用 NumPy 矢量化操作而非循环。
   - 对于实时应用，优先选择快速插值（如 `INTER_NEAREST`）。

4. **错误处理**：

   - 检查图像是否成功加载：

     ```python
     if img is None:
         raise ValueError("Failed to load image")
     ```

5. **与深度学习框架结合**：

   - 转换为张量：

     ```python
     import torch
     img_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).float() / 255.0
     ```

   - 配合 `PIL` 或 `torchvision` 进行增强。

---

### 6. 常见问题

1. **图像加载失败**：
   - 检查路径是否正确（支持相对/绝对路径）。
   - 确保文件格式支持（`.jpg`、`.png` 等）。
2. **颜色错误**：
   - 显示或输入模型时，确认 BGR/RGB 转换。
3. **性能瓶颈**：
   - 使用多线程或 `multiprocessing` 加速图像加载/处理。
4. **窗口无法显示**：
   - Jupyter 中用 `matplotlib`，或确保 `cv2.waitKey()` 正确调用。

---

### 7. 总结

`cv2` 是深度学习中图像处理的强大工具，常用功能包括：

- **图像读写**：`imread`、`imwrite`。
- **预处理**：缩放、归一化、颜色转换、数据增强。
- **可视化**：绘制矩形、文本、边界框。
- **视频处理**：帧读取、保存。
- **特征提取**：边缘、角点、SIFT/ORB。
  通过熟练掌握这些功能，`cv2` 可无缝集成到深度学习工作流中，提升数据处理效率和结果呈现效果。