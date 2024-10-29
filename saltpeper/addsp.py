import numpy as np
import cv2
import matplotlib.pyplot as plt

def add_salt_and_pepper_noise(image, salt_prob, pepper_prob):
    """
    添加椒盐噪声到图像
    :param image: 输入图像
    :param salt_prob: 盐噪声的概率
    :param pepper_prob: 胡椒噪声的概率
    :return: 含有椒盐噪声的图像
    """
    noisy_image = np.copy(image)

    # 随机选择要添加盐噪声的像素
    num_salt = np.ceil(salt_prob * image.size)
    coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image.shape]
    noisy_image[coords[0], coords[1]] = 255 # 盐噪声 (255 for white)

    # 随机选择要添加胡椒噪声的像素
    num_pepper = np.ceil(pepper_prob * image.size)
    coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape]
    noisy_image[coords[0], coords[1]] = 0  # 胡椒噪声 (0 for black)

    return noisy_image

# 读取图像
image = cv2.imread('lena.bmp', cv2.IMREAD_GRAYSCALE)

# 添加椒盐噪声
salt_prob = 0.1# 盐噪声概率
pepper_prob = 0.1  # 胡椒噪声概率
noisy_image = add_salt_and_pepper_noise(image, salt_prob, pepper_prob)

# 显示原始图像和带有噪声的图像
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title('Original Image')
plt.imshow(image, cmap='gray')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title('Image with Salt and Pepper Noise')
plt.imshow(noisy_image, cmap='gray')
plt.axis('off')

plt.show()

# 保存含噪声的图像
cv2.imwrite('lenasp.jpg', noisy_image)
