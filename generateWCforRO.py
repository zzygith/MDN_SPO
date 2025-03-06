
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import linprog
from sklearn.neighbors import NearestNeighbors
import pandas
import cplex


def uncertainty_function(x, y, Q, data, SV, alphas):
    """
    计算不确定性函数值 F(x,y) = sum_{j in SV} alpha_j * (|Q_row1*( (x,y)-data[j] )| + |Q_row2*( (x,y)-data[j] )|)
    输入:
      x, y    : 网格数组（对应二维区域的 x 和 y 坐标）
      Q       : 2*2 的白化矩阵
      data    : 原始数据数组，形状为 (n, 2)
      SV      : 支持向量的索引列表
      alphas  : 全部的 alpha 值（我们仅使用 SV 中的）
    输出:
      F : 与 x, y 同尺寸的函数值数组
    """
    F = np.zeros_like(x)
    for j in SV:
        sv = data[j]  # 取出支持向量坐标 [x1, x2]
        # Q 的第一行： Q[0,0]*(x - sv[0]) + Q[0,1]*(y - sv[1])
        term1 = Q[0,0]*(x - sv[0]) + Q[0,1]*(y - sv[1])
        # Q 的第二行： Q[1,0]*(x - sv[0]) + Q[1,1]*(y - sv[1])
        term2 = Q[1,0]*(x - sv[0]) + Q[1,1]*(y - sv[1])
        F += alphas[j] * (np.abs(term1) + np.abs(term2))
    return F


#fileNum=1

for fileNum in range(1,11):
    DRO_SPO_path='./samplesZandUandXopt500_0/'
    if fileNum==1:
        filepath=DRO_SPO_path+'samplesZandUandXopt.csv'
    else:
        filepath=DRO_SPO_path+'samplesZandUandXopt'+str(fileNum)+'.csv'
    
    ########################### 寻找neighbor ###########################
    argv_1=filepath
    argv_2=0.80
    neighborNum=100

    dataAll = np.array(pandas.read_csv(argv_1, header=0))[:,0:2]
    nbrs = NearestNeighbors(n_neighbors=neighborNum, algorithm='ball_tree').fit(dataAll)
    distances, indices = nbrs.kneighbors(dataAll)
    allZlist=[]
    allU_WClist=[]
    #################################################################################

    for dataIndice in range(0,len(dataAll)):
        #dataIndice=218

        data_zValue=dataAll[dataIndice][0]
        data_uValue=dataAll[dataIndice][1]
        allZlist.append(data_zValue)
        dataSelected=dataAll[indices[dataIndice],:]

        data = np.transpose(dataSelected)

        n=len(data)
        N=len(data[0])

        print ("n=",n)
        print ("N=",N)


        print("Calculating Eigenvalues.") 


        A = np.cov(data)

        eigVals, eigVecs = np.linalg.eig(A)

        eigVals = eigVals**(-1/2)
        eigVals = np.diag(eigVals)
        Q = eigVecs.dot(eigVals).dot(eigVecs.T)


        print("Creating kernel matrix.") 

        l = np.zeros(n)
        for i in range(n):
            l[i] = np.amax(Q[i].dot(data)) - np.amin(Q[i].dot(data)) + 0.01
        lsum = np.sum(l)


        #####prepare K values

        v = []
        for i in range(N):
            v.append(Q.dot(np.transpose(data)[i]))

        K = [[lsum - np.linalg.norm(v[i]-v[j], ord=1) for j in range(N)] for i in range(N)]


        nu = 1-float(argv_2)

        print(nu)





        ####solve qp

        print("Solving QP.") 

        p = cplex.Cplex()

        ub = (1/(N*nu))*np.ones(N)
        obj = np.zeros(N)
        for i in range(N):
            obj[i] = -K[i][i]

        varnames = ["alpha"+str(j) for j in range(N)]

        p.variables.add(obj=obj, ub=ub, names = varnames)

        p.linear_constraints.add(lin_expr = [cplex.SparsePair(ind = range(N), val = [1.0]*N)], senses = ["E"], rhs = [1])


        p.objective.set_sense(p.objective.sense.minimize)

        ind = np.array([[x for x in range(N)] for x in range(N)])

        qmat = []
        # for i in range(N):
        #     qmat.append([np.arange(N), K[i]])

        for i in range(N):
            qmat.append([list(range(N)), K[i]])

        p.objective.set_quadratic(qmat)


        p.solve()


        ####calculate output

        print("Calculating polyhedron.") 
        eps=0.00001
        alphas = [p.solution.get_values(i) for i in range(N)]
        SV = [i for i in range(N) if alphas[i] > eps]
        BSV = [i for i in SV if alphas[i] < 1/(N*nu)]

        val = []
        for i in BSV:
            vec = [alphas[j]*np.linalg.norm(Q.dot(np.transpose(data)[i]-np.transpose(data)[j]), ord=1) for j in SV]
            val.append(np.sum(vec))
        theta = np.min(val)



        ###print results

        # print("SVBEGIN")
        # print(SV)
        # print("SVEND")

        # print("QBEGIN")
        # print(Q)
        # print("QEND")

        # print("THETABEGIN")
        # print(theta)
        # print("THETAEND")

        # print("ALPHASBEGIN")
        # print(alphas)
        # print("ALPHASEND")


        # print("thetaval", theta)

        data=dataSelected



        # 设置绘图区域：根据数据范围适当扩展
        margin = 2.0
        x_min = data[:,0].min() - margin
        x_max = data[:,0].max() + margin
        y_min = data[:,1].min() - margin
        y_max = data[:,1].max() + margin

        # 生成网格数据
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 400),
                            np.linspace(y_min, y_max, 400))

        # 计算网格上不确定性函数的值
        F_val = uncertainty_function(xx, yy, Q, data, SV, alphas)

        # # 绘制不确定性集合区域
        # plt.figure(figsize=(8,6))
        # # 填充 F(x,y) <= theta 的区域
        # plt.contourf(xx, yy, F_val, levels=[F_val.min(), theta], colors=['lightblue'], alpha=0.5)
        # # 绘制 F(x,y)=theta 的边界
        # contour = plt.contour(xx, yy, F_val, levels=[theta], colors='blue', linewidths=2)
        # plt.clabel(contour, fmt="F = theta", inline=True)

        # # 绘制所有数据点（黑色圆点）
        # plt.scatter(dataAll[:,0], dataAll[:,1], color='k', s=5, label='Data Points All')
        # plt.scatter(data[:,0], data[:,1], color='g', s=5, label='Data Points')
        # # 绘制支持向量（红色圆点）
        # SV_data = data[SV]
        # plt.scatter(SV_data[:,0], SV_data[:,1], color='red', s=5, label='Support Vectors')


        # plt.xlabel("x1")
        # plt.ylabel("x2")
        # plt.title("Uncertainty Set Region defined by SVC-based Linear Inequalities")
        # plt.legend(loc='best')
        # plt.show()


        # 固定 x1=200, 求对应 x2 的范围，使得 (200,x2) 落在不确定性集合内
        x_fixed = data_zValue

        # 设置一个足够大的 x2 搜索范围，根据之前的 y_min, y_max 适当扩展
        # 注意：这里的 y_min 和 y_max 已经在你的绘图代码中定义
        y_search = np.linspace(y_min - 10, y_max + 10, 1000)  # 可根据需要调整

        # 生成对应的 x 数组，全为 x_fixed
        x_array = np.full_like(y_search, x_fixed)

        # 计算在 x1=200 时每个 x2 点的 F 值
        F_at_x_fixed = uncertainty_function(x_array, y_search, Q, data, SV, alphas)

        # 找出满足 F(x_fixed, x2) <= theta 的 x2 值
        indices_result = np.where(F_at_x_fixed <= theta)[0]

        if len(indices_result) > 0:
            x2_lower = y_search[indices_result[0]]
            x2_upper = y_search[indices_result[-1]]
            print(f"当 x1 = {x_fixed} 时, x2 的范围为：[{x2_lower:.4f}, {x2_upper:.4f}]")
            data_robustUValue=x2_lower
        else:
            print(f"在搜索范围内,x1 = {x_fixed} 对应的不确定性集合中没有 x2 满足 F(x,y) <= theta. 保存原u值{data_uValue}")
            data_robustUValue=data_uValue

        allU_WClist.append(data_robustUValue)


    allZlist=np.array(allZlist)
    allU_WClist=np.array(allU_WClist)

    # 拼接维数组
    combined = np.column_stack((allZlist, allU_WClist))
    np.savetxt(DRO_SPO_path+"samplesZandU_WC"+str(fileNum)+".csv", combined, delimiter=",", fmt="%.3f", 
            header="z,u_wc", comments="")



# import matplotlib.pyplot as plt
# dataAll = np.array(pandas.read_csv(DRO_SPO_path+"samplesZandU_WC.csv", header=0))[:,0:2]
# plt.scatter(dataAll[:,0:1],dataAll[:,1:2])
# plt.show()