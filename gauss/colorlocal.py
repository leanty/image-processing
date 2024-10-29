import cv2
import numpy as np
import matplotlib.pyplot as plt

def local_mean_filter(image, kernel_size=3):
    """
    对图像应用局部均值滤波
    :param image: 输入图像，可以是灰度或彩色图像
    :param kernel_size: 滤波器窗口的大小
    :return: 去噪后的图像
    """
    # 创建一个与输入图像相同大小的空白图像
    denoised_image = np.zeros_like(image)

    # 计算边界
    pad_size = kernel_size // 2
    padded_image = cv2.copyMakeBorder(image, pad_size, pad_size, pad_size, pad_size, cv2.BORDER_REFLECT)

    # 迭代处理每个像素
    for i in range(pad_size, padded_image.shape[0] - pad_size):
        for j in range(pad_size, padded_image.shape[1] - pad_size):
            # 提取邻域
            kernel = padded_image[i - pad_size:i + pad_size + 1, j - pad_size:j + pad_size + 1]
            # 计算均值
            denoised_image[i - pad_size, j - pad_size] = np.mean(kernel, axis=(0, 1))

    return denoised_image.astype(np.uint8)

# 读取彩色图像
image = cv2.imread('bird_noisy.jpg')  # 读取彩色图像

if image is None:
    print("Error: Could not read the image.")
    exit()

# 应用局部均值滤波去噪
denoised_image = local_mean_filter(image, kernel_size=3)

# 显示原始图像和去噪后的图像
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title('Noisy Image')
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))  # 转换为RGB格式进行显示
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title('Denoised Image (Local Mean)')
plt.imshow(cv2.cvtColor(denoised_image, cv2.COLOR_BGR2RGB))  # 转换为RGB格式进行显示
plt.axis('off')

plt.show()

# 保存去噪后的图像
cv2.imwrite('bird_local_mean.jpg', denoised_image)
print("Denoised image saved as 'bird_local_mean.jpg'")
