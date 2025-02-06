from scipy.optimize import linprog

w1_0=0.66 
w2_0=0.34
mu1_0=6.06
mu2_0=2.40
sigma1_0=0.69
sigma2_0=0.80
epsilon=1.02

def solveDROFunc(w1_0,w2_0,mu1_0,mu2_0,sigma1_0,sigma2_0,epsilon):

    c11=0
    c12=(mu1_0-mu2_0)**2+(sigma1_0**2+sigma2_0**2-2*sigma1_0*sigma2_0)
    c21=(mu1_0-mu2_0)**2+(sigma1_0**2+sigma2_0**2-2*sigma1_0*sigma2_0)
    c22=0

    # obj=w1_0*y1+w2_0*y2+epsilon**2*n+0*x1+0*x2
    # 目标函数系数：minimize 8x+2y
    c = [w1_0, w2_0, epsilon**2, 0, 0]

    # 将约束 5x+2y>=80 和 6x+8y>=200 转换为 <= 形式：
    # -5x - 2y <= -80
    # # -6x - 8y <= -200
    # A_ub = [
    #     [-1, 0, -c11,  -mu1_0, -2],
    #     [-1, 0, -c12,  -mu2_0, -2],
    #     [0, -1, -c21,  -mu1_0, -2],
    #     [0, -1, -c22,  -mu2_0, -2],
    #     [0, 0,   0,   5,     2],
    #     [0, 0,   0,   6,     8]
    # ]
    A_ub = [
        [-1, 0, -c11,  -mu1_0, -1],
        [-1, 0, -c12,  -mu2_0, -1],
        [0, -1, -c21,  -mu1_0, -1],
        [0, -1, -c22,  -mu2_0, -1],
        [0, 0,   0,   5,     2],
        [0, 0,   0,   6,     8]
    ]
    b_ub = [0, 0, 0, 0,80,200]


    # 定义变量的取值范围：x >= 0, y >= 0
    bounds = [(-1000, 1000), (-1000, 1000),(0, None),(0, None),(0, None)]

    # 求解线性规划问题
    result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
    #print("最优解:x =", result.x[-2],result.x[1])
    return result.x[-2],result.x[-1]

    # # 输出结果
    # print("最优解:x =", result.x)
    # print("最小目标函数值 =", result.fun)
