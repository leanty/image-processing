import numpy as np
import matplotlib.pyplot as plt

def median_filter(image, kernel_size=3):
    """
    使用中值滤波去噪
    :param image: 输入的含噪声图像
    :param kernel_size: 窗口大小，必须为奇数
    :return: 去噪后的图像
    """
    # 确保 kernel_size 为奇数
    if kernel_size % 2 == 0:
        raise ValueError("Kernel size must be odd.")

    # 获取图像的维度
    m, n = image.shape
    pad_size = kernel_size // 2

    # 使用零填充来处理边界
    padded_image = np.pad(image, pad_size, mode='constant', constant_values=0)

    # 创建输出图像
    denoised_image = np.zeros((m, n), dtype=np.uint8)

    # 对每个像素应用中值滤波
    for i in range(m):
        for j in range(n):
            # 取出邻域窗口
            window = padded_image[i:i + kernel_size, j:j + kernel_size]
            # 计算中值
            denoised_image[i, j] = np.median(window)

    return denoised_image

# 读取图像
image = plt.imread('lenasp.jpg')

# 应用自定义中值滤波
denoised_image = median_filter(image, kernel_size=5)

# 显示结果
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.title('Noisy Image')
plt.imshow(image, cmap='gray')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title('Denoised Image (Custom Median Filter)')
plt.imshow(denoised_image, cmap='gray')
plt.axis('off')

plt.tight_layout()
plt.show()

# 保存去噪后的图像
plt.imsave('lena_custom_median.jpg', denoised_image, cmap='gray')

