import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim

def calculate_metrics(reference_image, test_image):
    # 检查图像的形状是否一致
    if reference_image.shape != test_image.shape:
        raise ValueError("Reference and test images must have the same dimensions.")

    # 计算均方误差 MSE
    mse = np.mean((reference_image - test_image) ** 2)

    # 计算 PSNR
    if mse == 0:
        psnr = float('inf')  # 如果没有噪声，PSNR 是无穷大
    else:
        psnr = 20 * np.log10(255.0 / np.sqrt(mse))

    # 计算 SSIM
    ssim_value = ssim(reference_image, test_image, data_range=test_image.max() - test_image.min())

    return mse, psnr, ssim_value

# 读取图像
reference_image = cv2.imread('lena.bmp', cv2.IMREAD_GRAYSCALE)

# 确保参考图像成功读取
if reference_image is None:
    print("Error: Could not read reference image 'Lena.bmp'.")
else:
    # 要比较的图像
    # images_to_compare = ['lena_local_mean.jpg', 'lena_non_local_means.jpg', 'lena_bilateral.jpg']
    images_to_compare = ['lena_custom_median.jpg','lena_adaptive_median.jpg']
    for img_path in images_to_compare:
        test_image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        # 确保图像成功读取
        if test_image is None:
            print(f"Error: Could not read image {img_path}.")
            continue
        
        # 计算指标
        try:
            mse, psnr, ssim_value = calculate_metrics(reference_image, test_image)
            print(f"Metrics for {img_path}:")
            print(f"MSE: {mse:.2f}, PSNR: {psnr:.2f}, SSIM: {ssim_value:.4f}")
            print()
        except ValueError as e:
            print(f"Error comparing {img_path}: {e}")