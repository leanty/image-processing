import cv2
import numpy as np
import matplotlib.pyplot as plt

def opencv_nl_means_colored(image, h, templateWindowSize, searchWindowSize):
    return cv2.fastNlMeansDenoisingColored(image, None, h, h, templateWindowSize, searchWindowSize)

def process_and_display_colored(image, h, templateWindowSize, searchWindowSize):
    denoised = opencv_nl_means_colored(image, h, templateWindowSize, searchWindowSize)
    
    plt.figure(figsize=(12, 6))
    plt.subplot(121)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))  # 转换为RGB格式进行显示
    plt.title('Original')
    plt.axis('off')
    
    plt.subplot(122)
    plt.imshow(cv2.cvtColor(denoised, cv2.COLOR_BGR2RGB))  # 转换为RGB格式进行显示
    plt.title('Denoised Image (non_local_means)')
    plt.axis('off')
    
    plt.suptitle(f'h={h}, template={templateWindowSize}, search={searchWindowSize}')
    plt.tight_layout()
    plt.show()
    
    return denoised

# 读取彩色图像
image = cv2.imread('bird_noisy.jpg')  # 使用彩色读取

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
    denoised = process_and_display_colored(image, h, templateWindowSize, searchWindowSize)
    
    if best_denoised is None:
        best_denoised = denoised
        best_params = (h, templateWindowSize, searchWindowSize)
    else:
        user_input = input("Is this result better than the previous best? (y/n): ").lower()
        if user_input == 'y':
            best_denoised = denoised
            best_params = (h, templateWindowSize, searchWindowSize)

print(f"\nBest parameters: h={best_params[0]}, template={best_params[1]}, search={best_params[2]}")
cv2.imwrite('bird_color_local_means.jpg', best_denoised)
print("Best denoised image saved as 'bird_color_local_means.jpg'")
