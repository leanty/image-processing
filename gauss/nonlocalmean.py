import numpy as np
import cv2
import matplotlib.pyplot as plt
import time

def nl_means(I, ds, Ds, h):
    """
    非局部均值去噪
    :param I: 含噪声图像
    :param ds: 邻域窗口半径
    :param Ds: 搜索窗口半径
    :param h: 高斯函数平滑参数
    :return: 去噪图像
    """
    I = np.double(I)  # 转换为浮点型
    m, n = I.shape
    denoised_img = np.zeros((m, n), dtype=np.float32)  # 创建去噪图像

    # 使用对称填充来处理边界
    padded_img = np.pad(I, ((ds, ds), (ds, ds)), mode='symmetric')
    
    # 创建均匀核
    kernel = np.ones((2 * ds + 1, 2 * ds + 1), dtype=np.float32)
    kernel /= (2 * ds + 1) ** 2
    
    h2 = h * h 
    
    for i in range(m):
        for j in range(n):
            i1 = i + ds
            j1 = j + ds
            
            W1 = padded_img[i1 - ds:i1 + ds + 1, j1 - ds:j1 + ds + 1]  # 邻域窗口 1
            wmax = 0
            average = 0
            sweight = 0
            
            # 定义搜索窗口
            rmin = max(i1 - Ds, ds)
            rmax = min(i1 + Ds, m + ds - 1)
            smin = max(j1 - Ds, ds)
            smax = min(j1 + Ds, n + ds - 1)
            
            for r in range(rmin, rmax + 1):
                for s in range(smin, smax + 1):
                    if r == i1 and s == j1:
                        continue
                    
                    W2 = padded_img[r - ds:r + ds + 1, s - ds:s + ds + 1]  # 邻域窗口 2
                    Dist2 = np.sum(kernel * (W1 - W2) ** 2)  # 邻域间距离
                    w = np.exp(-Dist2 / h2)
                    
                    if w > wmax:
                        wmax = w
                    
                    sweight += w
                    average += w * padded_img[r, s]
            
            # 自身取最大权值
            average += wmax * padded_img[i1, j1]
            sweight += wmax
            
            # 计算去噪后的像素值
            denoised_img[i, j] = average / sweight if sweight > 0 else I[i, j]

    return np.clip(denoised_img, 0, 255).astype(np.uint8)

# 读取图像
image = cv2.imread('lena_gauss.jpg', cv2.IMREAD_GRAYSCALE)

# 设置参数
ds = 3  # 邻域窗口半径
Ds = 7  # 搜索窗口半径
h = 5  # 高斯函数平滑参数

# 记录开始时间
start_time = time.time()

# 应用非局部均值滤波去噪
denoised_image = nl_means(image, ds, Ds, h)

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
plt.title('Denoised Image (Non-Local Means)')
plt.imshow(denoised_image, cmap='gray')
plt.axis('off')

plt.show()

# 保存去噪后的图像
cv2.imwrite('lena_non_local_means.jpg', denoised_image)
