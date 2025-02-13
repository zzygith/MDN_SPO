import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

np.random.seed(42)
n_samples=300
noise_num = 200
# 混合分布的权重
weights = [0.5, 0.5]

# 定义第一个高斯分布的参数：均值和协方差矩阵
mean1 = [-1, 0]
cov1 = [[1, 0.8],
        [0.8, 1]]
# 定义第二个高斯分布的参数
mean2 = [1, 3]
cov2 = [[1, -0.7],
        [-0.7, 1]]

# 根据权重确定每个分布生成的样本数
n_samples1 = int(n_samples * weights[0])
n_samples2 = n_samples - n_samples1

# 分别从两个高斯分布中生成样本
samples1 = np.random.multivariate_normal(mean1, cov1, n_samples1)
samples2 = np.random.multivariate_normal(mean2, cov2, n_samples2)

# 合并样本数据
samples = np.vstack((samples1, samples2))

##################### noise method 1 ##################################### noise method 1 ################
noise_x = np.random.uniform(-4, 4, noise_num)
noise_y = np.random.uniform(-4, 8, noise_num)
noise_samples = np.column_stack((noise_x, noise_y))
# 将噪声点与混合分布样本合并
all_samples = np.vstack((samples, noise_samples))
##################### noise method 1 ##################################### noise method 1 ################

##################### noise method 2 ##################################### noise method 2 ################
# # 混合分布的权重
# weights_noise = [0.5, 0.5]
# # 定义第一个高斯分布的参数：均值和协方差矩阵
# mean1_noise = [-1, 0]
# cov1_noise = [[1, -0.8],
#         [-0.8, 1]]
# # 定义第二个高斯分布的参数
# mean2_noise = [1, 3]
# cov2_noise = [[1, 0.7],
#         [0.7, 1]]
# # 根据权重确定每个分布生成的样本数
# n_samples1_noise = int(noise_num * weights_noise[0])
# n_samples2_noise = noise_num - n_samples1_noise
# # 分别从两个高斯分布中生成样本
# samples1_noise = np.random.multivariate_normal(mean1_noise, cov1_noise, n_samples1_noise)
# samples2_noise = np.random.multivariate_normal(mean2_noise, cov2_noise, n_samples2_noise)
# # 合并样本数据
# noise_samples = np.vstack((samples1_noise, samples2_noise))
# # 将噪声点与混合分布样本合并
# all_samples = np.vstack((samples, noise_samples))
##################### noise method 2 ##################################### noise method 2 ################

##################### noise method 3 ##################################### noise method 3 ################
# noise = np.random.normal(loc=0.0, scale=1.0, size=samples.shape[0])
# # 将该噪声仅加到 y 维度上（samples[:,1]）
# samples[:, 1] += noise
# all_samples=samples
##################### noise method 3 ##################################### noise method 3 ################

np.savetxt('samplesZandU.txt', all_samples)

#########################################################################
#########################################################################
#########################################################################
# 混合分布的权重
weights = [0.5, 0.5]
# 定义第一个高斯分布的参数：均值和协方差矩阵
mean1 = [-1, 0]
cov1 = [[1, 0.8],
        [0.8, 1]]
# 定义第二个高斯分布的参数
mean2 = [1, 3]
cov2 = [[1, -0.7],
        [-0.7, 1]]

samples = np.loadtxt('samplesZandU.txt')

# 绘制样本散点图
plt.figure(figsize=(8, 6))
plt.scatter(samples[:, 0], samples[:, 1], s=10, color='gray', alpha=0.6, label='Samples')

# 构建网格，用于计算混合分布的概率密度
x = np.linspace(-4, 6, 300)
y = np.linspace(-4, 8, 300)
X, Y = np.meshgrid(x, y)
pos = np.dstack((X, Y))

# 分别定义两个多元正态分布对象
rv1 = multivariate_normal(mean1, cov1)
rv2 = multivariate_normal(mean2, cov2)

# 计算混合分布的概率密度：各分布的概率密度按权重相加
Z = weights[0] * rv1.pdf(pos) + weights[1] * rv2.pdf(pos)

# 绘制概率密度的等高线图
plt.contour(X, Y, Z, levels=10, cmap='jet')

# plt.title('二维高斯混合分布')
plt.xlabel('z')
plt.ylabel('u')
plt.legend()
plt.show()
