import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import linprog

from MDNModel1D import MDN
from MDNModel1D import mdn_loss
from solveDRO import solveDROFunc

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

#inputValue=1.0

samples = np.loadtxt('samplesZandU.txt')
allZlist=samples[:, 0]
allUlist=samples[:, 1]
optimalX1=[]
optimalX2=[]

for inputValue in allZlist:
    GMMobj={}
    for modelNum in range(1,11,1):
        #print(modelNum)
        model = MDN(in_dim=1, hidden_dim=64, n_components=2).to(device)
        model.load_state_dict(torch.load('./savedModel/mdn_model'+str(modelNum)+'.pth'))
        model.eval()
        x_input = torch.tensor([[inputValue]], device=device, dtype=torch.float32)
        with torch.no_grad():
        # 模型返回混合系数pi、均值mu和标准差sigma
            pi, mu, sigma = model(x_input)

        # 输出混合分布参数-1D
        piCPU=pi.cpu().numpy().flatten()
        muCPU=mu.cpu().numpy().flatten()
        sigmaCPU=sigma.cpu().numpy().flatten()
        # print("Mixing coefficients (pi):", piCPU)
        # print("Component means (mu):", muCPU)
        # print("Component std deviations (sigma):", sigmaCPU)
        GMMobj[modelNum]={'pi':piCPU,'mu':muCPU,'sigma':sigmaCPU}

    def WDdistance(GMMobj,modelNum,calculateModelNum):
        w1=GMMobj[modelNum]['pi'][0]
        w2=1-w1  #因为小数位的原因,直接读取的话可能出现w1+w2=1.000321这种情况
        mu1,mu2=GMMobj[modelNum]['mu']
        sigma1,sigma2=GMMobj[modelNum]['sigma']

        w1Prime=GMMobj[calculateModelNum]['pi'][0]
        w2Prime=1-w1Prime
        mu1Prime,mu2Prime=GMMobj[calculateModelNum]['mu']
        sigma1Prime,sigma2Prime=GMMobj[calculateModelNum]['sigma']

        c11P=(mu1-mu1Prime)**2+(sigma1**2+sigma1Prime**2-2*sigma1*sigma1Prime)
        c12P=(mu1-mu2Prime)**2+(sigma1**2+sigma2Prime**2-2*sigma1*sigma2Prime)
        c21P=(mu2-mu1Prime)**2+(sigma2**2+sigma1Prime**2-2*sigma2*sigma1Prime)
        c22P=(mu2-mu2Prime)**2+(sigma2**2+sigma2Prime**2-2*sigma2*sigma2Prime)
        c=[c11P,c12P,c21P,c22P]
        A_eq=[  [1,1,0,0],
                [0,0,1,1],
                [1,0,1,0],
                [0,1,0,1]  ]
        b_eq=[w1,w2,w1Prime,w2Prime]
        bounds = [(0, None), (0, None),(0, None),(0, None)]
        res = linprog(c, A_eq=A_eq, b_eq=b_eq,
                bounds=bounds, method='highs')
        optimal_value = res.fun
        return optimal_value**0.5

    WDlist={}
    for modelNum in range(1,11,1):
        WDlistNum=[]
        for calculateModelNum in range(1,11,1):
            if calculateModelNum==modelNum:
                continue
            else:
                s=WDdistance(GMMobj,modelNum,calculateModelNum)
                WDlistNum.append(s)
        WDlistNum=np.array(WDlistNum)
        WDlist[modelNum]=WDlistNum

    print(WDlist)
    # 计算每个对象的数组和
    sums = {key: np.sum(arr) for key, arr in WDlist.items()}

    # 找出数组和最小的对象的 key
    min_key = min(sums, key=sums.get)
    print("和最小的对象的 key 为:", min_key)
    print("该对象的数组和为:", sums[min_key])

    # 对该最小对象的数组求最大值
    max_value = np.max(WDlist[min_key])
    print("该对象数组的最大值为:", max_value)

    print('#############################')

    print('input:',inputValue)
    print('nominal distribution:')
    print('weight:',GMMobj[min_key]['pi'][0],1-GMMobj[min_key]['pi'][0])
    print('mu:',GMMobj[min_key]['mu'])
    print('sigma:',GMMobj[min_key]['sigma'])
    print('radius:',max_value)
    print('#############################')
    w1_0P=GMMobj[min_key]['pi'][0]
    w2_0P=1-GMMobj[min_key]['pi'][0]
    mu1_0P=GMMobj[min_key]['mu'][0]
    mu2_0P=GMMobj[min_key]['mu'][1]
    sigma1_0P=GMMobj[min_key]['sigma'][0]
    sigma2_0P=GMMobj[min_key]['sigma'][1]
    epsilonP=max_value
    x1Optimal, x2Optimal=solveDROFunc(w1_0P,w2_0P,mu1_0P,mu2_0P,sigma1_0P,sigma2_0P,epsilonP)
    print("最优解:x =", x1Optimal, x2Optimal)
    optimalX1.append(x1Optimal)
    optimalX2.append(x2Optimal)

optimalX1=np.array(optimalX1)
optimalX2=np.array(optimalX2)

# 拼接成 (n, 4) 的二维数组
combined = np.column_stack((allZlist, allUlist, optimalX1, optimalX2))
# 保存为 CSV 文件
# fmt 可以指定小数位，比如保留 3 位小数： "%.3f"
np.savetxt("samplesZandUandXopt.csv", combined, delimiter=",", fmt="%.3f", 
           header="z,u,x1,x2", comments="")