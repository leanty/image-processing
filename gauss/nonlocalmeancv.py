import cv2
import numpy as np
import matplotlib.pyplot as plt

def opencv_nl_means(image, h, templateWindowSize, searchWindowSize):
    return cv2.fastNlMeansDenoising(image, None, h, templateWindowSize, searchWindowSize)

def process_and_display(image, h, templateWindowSize, searchWindowSize):
    denoised = opencv_nl_means(image, h, templateWindowSize, searchWindowSize)
    
    plt.figure(figsize=(12, 6))
    plt.subplot(121)
    plt.imshow(image, cmap='gray')
    plt.title('Original')
    plt.axis('off')
    
    plt.subplot(122)
    plt.imshow(denoised, cmap='gray')
    plt.title('Denoised Image (non_local_means)')
    plt.axis('off')
    
    plt.suptitle(f'h={h}, template={templateWindowSize}, search={searchWindowSize}')
    plt.tight_layout()
    plt.show()
    
    return denoised

# 读取图像
image = cv2.imread('lena_gauss.jpg', cv2.IMREAD_GRAYSCALE)

if image is None:
    print("Error: Could not read the image.")
    exit()

# 尝试不同的参数组合
params = [
    (10, 7, 21),   # 默认参数
    (20, 7, 21),   # 增加 h
    (30, 7, 21),   # 进一步增加 h
    (20, 5, 21),   # 减小模板窗口
    (20, 7, 35),   # 增加搜索窗口
]

best_denoised = None
best_params = None

for h, templateWindowSize, searchWindowSize in params:
    print(f"\nTrying h={h}, template={templateWindowSize}, search={searchWindowSize}")
    denoised = process_and_display(image, h, templateWindowSize, searchWindowSize)
    
    if best_denoised is None:
        best_denoised = denoised
        best_params = (h, templateWindowSize, searchWindowSize)
    else:
        user_input = input("Is this result better than the previous best? (y/n): ").lower()
        if user_input == 'y':
            best_denoised = denoised
            best_params = (h, templateWindowSize, searchWindowSize)

print(f"\nBest parameters: h={best_params[0]}, template={best_params[1]}, search={best_params[2]}")
cv2.imwrite('lena_non_local_means.jpg', best_denoised)
print("Best denoised image saved as 'lena_non_local.jpg'")