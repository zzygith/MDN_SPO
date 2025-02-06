import torch.nn as nn
import torch.nn.functional as F
import torch

class MDN(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_components):
        super(MDN, self).__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_pi = nn.Linear(hidden_dim, n_components)
        self.fc_mu = nn.Linear(hidden_dim, n_components)
        self.fc_sigma = nn.Linear(hidden_dim, n_components)
        self.relu = nn.ReLU()

    def forward(self, x):
        h = self.relu(self.fc1(x))
        h = self.relu(self.fc2(h))
        
        # 输出混合系数
        pi = F.softmax(self.fc_pi(h), dim=-1)
        # 输出均值
        mu = self.fc_mu(h)
        # 使用 F.softplus 而不是 torch.softplus
        sigma = F.softplus(self.fc_sigma(h)) + 1e-5

        return pi, mu, sigma
    

def mdn_loss(pi, mu, sigma, y):
    y = y.expand_as(mu)
    normal_log_component = -0.5 * ((y - mu)**2 / sigma**2) \
                           - torch.log(sigma) \
                           - 0.5 * torch.log(torch.tensor(2.0 * 3.1415926))
    log_pi = torch.log(pi + 1e-8)
    log_prob = normal_log_component + log_pi
    max_log_prob, _ = torch.max(log_prob, dim=1, keepdim=True)
    log_sum_exp = max_log_prob + torch.log(torch.sum(torch.exp(log_prob - max_log_prob), dim=1, keepdim=True) + 1e-8)
    nll = - log_sum_exp.mean()
    return nll