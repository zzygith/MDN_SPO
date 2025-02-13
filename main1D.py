import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

from MDNModel1D import MDN
from MDNModel1D import mdn_loss

# # 1. 准备数据
# def generate_data(n_samples=2000):
#     x = torch.linspace(-1, 1, n_samples)
#     mu1 = 2.0 * torch.sin(2.0 * x)
#     mu2 = 0.5 * torch.cos(5.0 * x)
#     sigma1 = 0.1
#     sigma2 = 0.1
#     pis = torch.bernoulli(torch.full((n_samples,), 0.5))  # 随机 0/1
#     y = torch.where(
#         pis==0,
#         torch.normal(mean=mu1, std=sigma1),
#         torch.normal(mean=mu2, std=sigma2)
#     )
#     return x.unsqueeze(-1), y.unsqueeze(-1)

# # # 1. 准备数据, x,y符合二元高斯混合分布
# def generate_data(n_samples=2000):
#     # 设置随机数种子，保证结果可重复
#     np.random.seed(42)

#     # 混合分布的权重
#     weights = [0.5, 0.5]

# #     # 定义第一个高斯分布的参数：均值和协方差矩阵
# #     mean1 = [0, 0]
# #     cov1 = [[1, 0.5],
# #             [0.5, 1]]
# #     # 定义第二个高斯分布的参数
# #     mean2 = [2, 4]
# #     cov2 = [[1, -0.6],
# #             [-0.6, 1]]
    
#     # 定义第一个高斯分布的参数：均值和协方差矩阵
#     mean1 = [0, 2]
#     cov1 = [[1, 0.5],
#             [0.5, 1]]
#     # 定义第二个高斯分布的参数
#     mean2 = [1, 6]
#     cov2 = [[1, -0.7],
#             [-0.7, 1]]

#     # 根据权重确定每个分布生成的样本数
#     n_samples1 = int(n_samples * weights[0])
#     n_samples2 = n_samples - n_samples1

#     # 分别从两个高斯分布中生成样本
#     samples1 = np.random.multivariate_normal(mean1, cov1, n_samples1)
#     samples2 = np.random.multivariate_normal(mean2, cov2, n_samples2)

#     # 合并样本数据
#     samples = np.vstack((samples1, samples2))

#     # 转换为 torch tensor，并转换为 float32
#     x = torch.from_numpy(samples[:, 0]).float()
#     y = torch.from_numpy(samples[:, 1]).float()

#     return x.unsqueeze(-1), y.unsqueeze(-1)


# # 1. 准备数据, 读取z,u
def generate_data(sampleNumber):
    AllSamples = np.loadtxt('samplesZandU.txt')
    indices = np.random.choice(AllSamples.shape[0], sampleNumber, replace=False)
    samples = AllSamples[indices]
    # 转换为 torch tensor，并转换为 float32
    x = torch.from_numpy(samples[:, 0]).float()
    y = torch.from_numpy(samples[:, 1]).float()
    return x.unsqueeze(-1), y.unsqueeze(-1)


# modelNum=10
for modelNum in range (1,11,1):

    # 2. 定义 MDN 模型
    # 3. 定义 MDN loss
    # 4. 训练循环
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    torch.manual_seed(0)

    x_data, y_data = generate_data(sampleNumber=450)

    x_data, y_data = x_data.to(device), y_data.to(device)

    model = MDN(in_dim=1, hidden_dim=64, n_components=2).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    num_epochs = 2000
    batch_size = 64

    for epoch in range(num_epochs):
        perm = torch.randperm(x_data.size(0))
        batch_indices = perm[:batch_size]
        x_batch = x_data[batch_indices]
        y_batch = y_data[batch_indices]

        optimizer.zero_grad()
        pi, mu, sigma = model(x_batch)
        loss = mdn_loss(pi, mu, sigma, y_batch)
        loss.backward()
        optimizer.step()

        if (epoch+1) % 200 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

    # 5. 可视化结果
    model.eval()
    torch.save(model.state_dict(), './savedModel/mdn_model'+str(modelNum)+'.pth')

    # x_plot = torch.linspace(-1, 1, 3).unsqueeze(-1).to(device)
    # pi_plot, mu_plot, sigma_plot = model(x_plot)
    # print(x_plot)
    # print(pi_plot)
    # print(mu_plot)
    # print(sigma_plot)
    # plt.scatter(x_data.cpu().numpy(), y_data.cpu().numpy(), s=2, alpha=0.3, label='Training Data')
    # plt.show()

