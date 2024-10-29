import cv2
import numpy as np
import matplotlib.pyplot as plt

# 读取左右视图图像
left_image = cv2.imread('tsukuba_l.png', cv2.IMREAD_GRAYSCALE)
right_image = cv2.imread('tsukuba_r.png', cv2.IMREAD_GRAYSCALE)

# 创建块匹配对象BM
block_matcher = cv2.StereoBM_create(numDisparities=16, blockSize=15)

# 使用块匹配计算视差
disparity_bm = block_matcher.compute(left_image, right_image)

# 创建SGBM对象
min_disparity = 0
num_disparities = 16 * 5  # 必须是16的倍数
block_size = 7
stereo_sgbm = cv2.StereoSGBM_create(minDisparity=min_disparity,
                                    numDisparities=num_disparities,
                                    blockSize=block_size,
                                    P1=8 * 3 * block_size ** 2,
                                    P2=32 * 3 * block_size ** 2,
                                    disp12MaxDiff=1,
                                    uniquenessRatio=15,
                                    speckleWindowSize=100,
                                    speckleRange=32)

# 使用SGBM算法计算视差
disparity_sgbm = stereo_sgbm.compute(left_image, right_image)

# 可视化视差图
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.title('Block Matching')
plt.imshow(disparity_bm, cmap='gray')
plt.colorbar()

plt.subplot(1, 2, 2)
plt.title('SGBM')
plt.imshow(disparity_sgbm, cmap='gray')
plt.colorbar()

plt.show()
