import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
# 设置中文字体，这里使用微软雅黑字体作为例子
# 你可以根据需要选择系统内其他支持中文的字体，如 SimHei（黑体）
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体显示中文
plt.rcParams['axes.unicode_minus'] = False  # 解决负号 '-' 显示成方块的问题


# 均值滤波缺点
def denoise_image(image):
    """
    自动检测图像类型并去噪
    :param image: 输入的图像，可以是灰度或彩色图像
    :return: 去噪后的图像
    """
    # 判断图像是灰度图还是彩色图
    if len(image.shape) == 2:
        # 灰度图
        print("Detected: Grayscale Image")
        denoised_image = cv2.medianBlur(image, 5)  # 对灰度图像应用中值滤波
    elif len(image.shape) == 3 and image.shape[2] == 3:
        # 彩色图
        print("Detected: Color Image")
        denoised_image = cv2.medianBlur(image, 5)  # 对彩色图像应用中值滤波
    else:
        raise ValueError("Unknown image format!")

    return denoised_image

# 读取图像
image = cv2.imread('noisy_image.jpg')

# 自动去噪
denoised_image = denoise_image(image)

# 如果是彩色图像，转换为 RGB 格式以便 Matplotlib 显示（OpenCV 默认是 BGR 格式）
if len(image.shape) == 3 and image.shape[2] == 3:
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    denoised_image_rgb = cv2.cvtColor(denoised_image, cv2.COLOR_BGR2RGB)
else:
    image_rgb = image  # 灰度图无需转换
    denoised_image_rgb = denoised_image

# 显示原始图像和去噪后的图像
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title('噪声图')
plt.imshow(image_rgb, cmap='gray' if len(image.shape) == 2 else None)
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title('去噪图')
plt.imshow(denoised_image_rgb, cmap='gray' if len(image.shape) == 2 else None)
plt.axis('off')

plt.show()

# 保存去噪后的图像
cv2.imwrite('denoised_image.jpg', denoised_image)
