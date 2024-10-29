import cv2
import numpy as np
import matplotlib.pyplot as plt

def add_gaussian_noise(image, mean=0, sigma=25):
    """
    给灰度图像添加高斯噪声
    :param image: 输入灰度图像
    :param mean: 高斯噪声的均值
    :param sigma: 高斯噪声的标准差
    :return: 含高斯噪声的图像
    """
    noisy_image = image.copy()
    
    # 生成与图像大小相同的高斯噪声
    gaussian_noise = np.random.normal(mean, sigma, image.shape)
    
    # 将噪声添加到图像
    noisy_image = noisy_image + gaussian_noise

    # 将结果裁剪到有效范围 [0, 255]
    noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)

    return noisy_image

# 读取灰度图像
image = cv2.imread('Lena.bmp', cv2.IMREAD_GRAYSCALE)

# 添加高斯噪声
noisy_image = add_gaussian_noise(image, mean=0, sigma=25)

# 保存含噪声的图像
cv2.imwrite('noisy_image_output.jpg', noisy_image)

# 显示原图像和添加噪声后的图像
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.title('Original Image')
plt.imshow(image, cmap='gray')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title('Gaussian Noise Image')
plt.imshow(noisy_image, cmap='gray')
plt.axis('off')

plt.show()
