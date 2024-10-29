import cv2
import numpy as np
import matplotlib.pyplot as plt

def adaptive_median_filter(image, max_window_size=9):
    # 确保输入图像是灰度图像
    if len(image.shape) > 2:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    h, w = image.shape
    out_image = np.zeros_like(image)
    
    for y in range(h):
        for x in range(w):
            window_size = 3
            while True:
                half_win = window_size // 2
                ymin, ymax = max(0, y - half_win), min(h, y + half_win + 1)
                xmin, xmax = max(0, x - half_win), min(w, x + half_win + 1)
                
                # 获取子区域
                region = image[ymin:ymax, xmin:xmax]
                
                med = np.median(region)
                
                # 检查噪声
                if (window_size == 3 and (image[y, x] - med) < 0.1 * med) or \
                   (window_size > 3 and abs(image[y, x] - med) <= 0.1 * med):
                    out_image[y, x] = med
                    break
                
                # 增加窗口大小
                window_size += 2
                if window_size > max_window_size:
                    out_image[y, x] = med
                    break
    
    return out_image

# 读取图像
img = cv2.imread('lenasp.jpg')

# 使用自适应中值滤波器去噪
denoised_img = adaptive_median_filter(img)

# 显示去噪前后对比图像
plt.figure(figsize=(10, 5))

# 显示原图像
plt.subplot(1, 2, 1)
plt.title('Noisy Image')
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cmap='gray')
plt.axis('off')

# 显示去噪后的图像
plt.subplot(1, 2, 2)
plt.title('Denoised Image (Adaptive Median Filter)')
plt.imshow(denoised_img, cmap='gray')
plt.axis('off')

# 调整布局并显示
plt.tight_layout()
plt.show()

# 保存去噪后的图像
plt.imsave('lena_adaptive_median.jpg', denoised_img, cmap='gray')








