import cv2
import numpy as np
import matplotlib.pyplot as plt
import time

# 读取图像
image = cv2.imread('lena_gauss.jpg', cv2.IMREAD_GRAYSCALE)

# 设置双边滤波的参数
d = 8# 邻域直径
sigma_color = 25 # 颜色空间的滤波器sigma值
sigma_space = 35  # 坐标空间的滤波器sigma值

# 记录开始时间
start_time = time.time()

# 应用双边滤波进行去噪
denoised_image = cv2.bilateralFilter(image, d, sigma_color, sigma_space)

# 记录结束时间
end_time = time.time()

# 计算并打印运行时间
execution_time = end_time - start_time
print(f"Execution Time: {execution_time:.4f} seconds")

# 显示原始图像和去噪后的图像
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title('Noisy Image')
plt.imshow(image, cmap='gray')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title('Denoised Image (Bilateral Filter)')
plt.imshow(denoised_image, cmap='gray')
plt.axis('off')

plt.show()

# 保存去噪后的图像
cv2.imwrite('lena_bilateral.jpg', denoised_image)
