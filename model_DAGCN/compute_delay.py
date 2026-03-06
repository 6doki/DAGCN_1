import os
import pandas as pd
import numpy as np
import pickle
import torch
import time
from torch.autograd import Variable

from Data.dataset.multi_step_dataset import MultiStepDataset


def gen_data(data, ntr, N):
    '''
    if flag:
        data=pd.read_csv(fname)
    else:
        data=pd.read_csv(fname,header=None)
    data:输入的数据，一个包含时间序列的数组或者矩阵
    ntr:处理后返回的数据的时间步长
    N：节点数量
    '''
    # data=data.as_matrix()
    '''
    将data数据整形为三维矩阵:
    (1) 形状 (天数,每天的时间步数,节点数)
    (2) -1表示自动推断天数
    (3) 288表示一天的时间步
    (4) 取前ntr天的数据
    '''
    data = np.reshape(data, [-1, 288, N])  # （91,288,358）
    return data[0:ntr]  # （54,288,358）


# def normalize(a):
#     '''
#     np.mean函数计算输入数据a在每一行上的平均值，axis=1表示沿着每一行计算
#     keepdims=True表示保持原始数据的维度，返回的mu是一个与输入a形状相同的二维数组
#     '''
#     mu = np.mean(a, axis=1, keepdims=True)
#     '''
#     np.std函数计算输入数据a在每一行上的标准差，axis=1表示沿着每一行计算
#     keepdims=True表示保持原始数据的维度，返回的std是一个与输入a形状相同的二维数组
#     '''
#     std = np.std(a, axis=1, keepdims=True)
#     '''
#     返回标准化后的数据，即将输入数据a的每个元素减去对应行的平均值mu，再除以对应行的标准差std
#     返回的数据形状与输入a相同，即为一个二维数组
#     '''
#     return (a - mu) / std

'''对输入的数据进行标准化处理'''


# def normalize(a):
#     '''
#     修改后的鲁棒标准化函数
#     '''
#     # 计算均值
#     mu = np.mean(a, axis=1, keepdims=True)
#     # 计算标准差
#     std = np.std(a, axis=1, keepdims=True)
#
#     # [核心修改] 在分母 std 后面加上一个极小值 1e-5
#     # 这样即使 std 为 0，分母也不会是 0
#     return (a - mu) / (std + 1e-5)

def normalize(a):
    """
    修改后的鲁棒标准化函数，兼容 1D 和 2D 数组
    """
    # 如果是 1D 数组 (例如 flatten 后的单条时间序列)
    if a.ndim == 1:
        mu = np.mean(a)
        std = np.std(a)
        return (a - mu) / (std + 1e-5)

    # 如果是 2D 数组 (例如原始处理逻辑中的批量数据)
    else:
        mu = np.mean(a, axis=1, keepdims=True)
        std = np.std(a, axis=1, keepdims=True)
        return (a - mu) / (std + 1e-5)


# 新增辅助函数：计算两个序列的最佳滞后步数 (Lag)
def compute_delay(a, b, max_lag=3):
    """
    计算序列 a 和 b 之间的最佳时间延迟。
    返回 lag值，表示 b 比 a 滞后多少个时间步。
    """
    # 标准化
    a = (a - np.mean(a)) / (np.std(a) + 1e-5)
    b = (b - np.mean(b)) / (np.std(b) + 1e-5)

    # 计算互相关
    correlation = np.correlate(a, b, mode='full')
    lags = np.arange(-len(a) + 1, len(a))

    # 找到相关性最大的位置
    best_index = np.argmax(correlation)
    best_lag = lags[best_index]

    # 限制 lag 范围，比如限制在 [0, max_lag] 之间
    # 我们只关心 b 滞后于 a 的情况 (传播)
    if best_lag < 0: best_lag = 0
    if best_lag > max_lag: best_lag = max_lag

    return int(best_lag)


def compute_dtw(a, b, order=1, Ts=12, normal=True):
    if normal:
        a = normalize(a)  # （54,288）
        b = normalize(b)

     # [核心修改] 如果输入是一维数组，强行升维成 (1, Length) 的二维数组
    if a.ndim == 1:
         a = a.reshape(1, -1)
    if b.ndim == 1:
         b = b.reshape(1, -1)

    T0 = a.shape[1]  # 288
    d = np.reshape(a, [-1, 1, T0]) - np.reshape(b, [-1, T0, 1])  # dist matrix: (54, 288, 288)
    d = np.linalg.norm(d, axis=0, ord=order)  # 范式1max(sum(abs(x), axis=0))（绝对值和的最大值）: (288,288)
    D = np.zeros([T0, T0])  # dtw matrix: (288,288)
    for i in range(T0):
        for j in range(max(0, i - Ts), min(T0, i + Ts + 1)):
            if (i == 0) and (j == 0):
                D[i, j] = d[i, j] ** order
                continue
            if (i == 0):
                D[i, j] = d[i, j] ** order + D[i, j - 1]
                continue
            if (j == 0):
                D[i, j] = d[i, j] ** order + D[i - 1, j]
                continue
            if (j == i - Ts):
                D[i, j] = d[i, j] ** order + min(D[i - 1, j - 1], D[i - 1, j])
                continue
            if (j == i + Ts):
                D[i, j] = d[i, j] ** order + min(D[i - 1, j - 1], D[i, j - 1])
                continue
            D[i, j] = d[i, j] ** order + min(D[i - 1, j - 1], D[i - 1, j], D[i, j - 1])
    return D[-1, -1] ** (1.0 / order)  # value


def construct_adj_fusion(A, A_dtw, steps, Lag_mx):
    N = len(A)
    adj = np.zeros([N * steps] * 2)  # 创建一个新的零矩阵[N * steps,N * steps]

    # 1,填充空间图
    for i in range(steps):
        if (i == 1) or (i == 2):
            adj[i * N: (i + 1) * N, i * N: (i + 1) * N] = A     # (N:2N,N:2N)  (2N:3N, 2N:3N)
        else:
            adj[i * N: (i + 1) * N, i * N: (i + 1) * N] = A_dtw     # (0:N, 0:N)   (3N:4N, 3N:4N)
    print('第一步：空间图填充完成！！')
    # 2，填充时间图
    for i in range(N):
        for k in range(steps - 1):
            '''
            k * N + i                            表示时间步k中的节点i
            (k + 1) * N + i                      表示时间步k+1中相同的节点i
            adj[k * N + i, (k + 1) * N + i] = 1  表示节点i在相邻时间步有连接
            '''
            adj[k * N + i, (k + 1) * N + i] = 1
            # 保持矩阵对称性质
            adj[(k + 1) * N + i, k * N + i] = 1
    print('第二步：时间图填充完成！！！')

    # 3. [创新点] 填充延迟感知语义图 (Delay-Aware Semantic Graph)
    # 遍历所有节点对 (i, j)
    # 如果 A_dtw[i, j] == 1 (说明它们语义相似)
    # 我们根据 Lag_mx[i, j] 决定把边连在哪个时间块之间

    rows, cols = np.nonzero(A_dtw)  # 获取所有有连接的节点对
    for idx in range(len(rows)):
        i = rows[idx]
        j = cols[idx]
        lag = int(Lag_mx[i, j])  # 获取 i 到 j 的延迟步数

        # 遍历当前的时间窗 steps
        for t in range(steps):
            target_t = t + lag

            # 如果目标时间还在窗口内，建立跨时间步的连接
            if target_t < steps:
                # 建立从 (时刻t, 节点i) 到 (时刻t+lag, 节点j) 的有向边
                # 这意味着：节点i在t时刻的状态，会影响节点j在t+lag时刻的状态
                adj[t * N + i, target_t * N + j] = 1

                # 如果做无向图，通常需要对称，但在延迟场景下，传播是有方向的。
                # 这种有向性是比 CMGCN 更高级的地方。
    print('第三步：延迟感知语义图填充完成！！！')
    for i in range(len(adj)):  # 为每个节点添加自身连接
        adj[i, i] = 1
    print('第四步：自环填充完成！！！')
    return adj  # (4N,4N)


class COMPUTE_delay(MultiStepDataset):
    '''
    空间图：继承于父类MultiStepDataset
    时间图：DTW生成
    可以直接使用MultiStepDataset生成的data：train_loader/valid_loader/test_loader/scaler/num_batches
    '''
    def __init__(self, config):
        super().__init__(config)
        self.strides = self.config.get("strides", 4)  # 步长
        self.order = self.config.get("order", 1)  # 计算DTW距离的时候的选取哪种距离（1：曼哈顿距离 2：欧式距离）
        self.lag = self.config.get("lag", 12)  # 模型滞后期（表示时间上的延迟） -----------估计用不上？？？！！！！
        self.period = self.config.get("period", 288)  # 一天288个时间步 -----------好像没用上？？？！！！
        self.sparsity = self.config.get("sparsity", 0.01)  # 稀疏度，邻接矩阵中非零元素占比 -----------好像没用上？？？！！！
        self.train_rate = self.config.get("train_rate", 0.6)  # 训练数据比例
        self.adj_percent= self.config.get("adj_percent", 0.01)  # 邻接矩阵稀疏度，节点数*稀疏度=top=k （与该节点最相似的前top个节点）
        self.adj_mx = torch.FloatTensor(self._construct_adj())  # SG和TG融合后的graph：(4N,4N)
        # self.adj_mx = torch.FloatTensor(self._construct_adj())
        # self.adj_mx = torch.randn((1432, 1432))

    # 构建节点之间的邻接矩阵（节点i与其最相似的top个节点）:(N,N) ---------> 时间相似性图
    # def _construct_dtw(self):
    #     print("------------------进入COMPUTE_delay类_construct_dtw方法：-------------------")
    #     # 定义 lag_mx 的保存路径
    #     lag_file = self.dtw_file.replace('.npy', '_lag.npy')
    #
    #     # 情况1: 如果文件都存在，直接加载
    #     if os.path.exists(self.dtw_file) and os.path.exists(lag_file):
    #         print("加载已有的 DTW 和 Lag 矩阵...")
    #         w_adj = np.load(self.dtw_file)
    #         lag_mx = np.load(lag_file)
    #
    #     # 情况2: 重新计算
    #     else:
    #         data = self.rawdat[:, :, 0]
    #         total_day = data.shape[0] // 288
    #         train_day = int(total_day * 0.6)
    #         n_route = data.shape[1]
    #         xtr = gen_data(data, train_day, n_route)
    #
    #         N = n_route
    #         d = np.zeros([N, N])
    #         lag_mx = np.zeros([N, N], dtype=int)  # 初始化延迟矩阵
    #
    #         for i in range(N):
    #             for j in range(i + 1, N):
    #                 # # 计算 DTW 距离
    #                 # d[i, j] = compute_dtw(xtr[:, :, i], xtr[:, :, j])
    #                 seq_i = xtr[:, :, i].flatten()
    #                 seq_j = xtr[:, :, j].flatten()
    #
    #                 # [新增] 检查并处理 NaN (防止之前的漏网之鱼)
    #                 if np.isnan(seq_i).any():
    #                     seq_i = np.nan_to_num(seq_i)
    #                     print(f"节点 {i} 和 {j} 之间的序列有 NaN 值，已处理。")
    #                 if np.isnan(seq_j).any():
    #                     seq_j = np.nan_to_num(seq_j)
    #                     print(f"节点 {i} 和 {j} 之间的序列有 NaN 值，已处理。")
    #
    #                 d[i, j] = compute_dtw(seq_i, seq_j)  # 注意这里不需要再传 xtr[:,:,i] 这种切片了，直接传处理好的 seq_i
    #
    #                 # 计算延迟
    #                 lag_mx[i, j] = compute_delay(seq_i, seq_j, max_lag=self.strides - 1)
    #                 lag_mx[j, i] = compute_delay(seq_j, seq_i, max_lag=self.strides - 1)
    #
    #         dtw = d + d.T
    #         print("节点之间的相似性矩阵（时间图）已经生成！")
    #
    #         # 生成稀疏的 w_adj
    #         n = dtw.shape[0]
    #         w_adj = np.zeros([n, n])
    #         adj_percent = self.adj_percent
    #         top = int(n * adj_percent)
    #
    #         for i in range(dtw.shape[0]):
    #             a = dtw[i, :].argsort()[0:top]
    #             for j in range(top):
    #                 w_adj[i, a[j]] = 1
    #
    #         # 对称化 w_adj
    #         for i in range(n):
    #             for j in range(n):
    #                 if (w_adj[i][j] != w_adj[j][i] and w_adj[i][j] == 0):
    #                     w_adj[i][j] = 1
    #                 if (i == j):
    #                     w_adj[i][j] = 1
    #
    #         # [关键] 过滤 lag_mx，只保留有语义连接的节点的延迟信息
    #         lag_mx = lag_mx * w_adj
    #
    #         print("保存 DTW 和 Lag 矩阵...")
    #         np.save(self.dtw_file, w_adj)
    #         np.save(lag_file, lag_mx)  # 保存 lag_mx
    #
    #     # [关键] 赋值给类属性，供后续使用
    #     self.dtw = w_adj
    #     self.lag_mx = lag_mx

    def _construct_dtw(self):
        print("------------------进入COMPUTE_delay类_construct_dtw方法：-------------------")
        lag_file = self.dtw_file.replace('.npy', '_lag.npy')

        if os.path.exists(self.dtw_file) and os.path.exists(lag_file):
            print("加载已有的 DTW 和 Lag 矩阵...")
            w_adj = np.load(self.dtw_file)
            lag_mx = np.load(lag_file)
        else:
            data = self.rawdat[:, :, 0]
            total_day = data.shape[0] // 288
            train_day = int(total_day * 0.6)
            n_route = data.shape[1]
            xtr = gen_data(data, train_day, n_route)  # (Days, 288, Nodes)

            N = n_route
            d = np.zeros([N, N])
            lag_mx = np.zeros([N, N], dtype=int)

            print(f"开始计算DTW和延迟矩阵，共 {N * (N - 1) // 2} 对节点...")

            # [核心优化] 预先计算所有节点的“日均流量模式”
            # 形状从 (Days, 288, N) 变为 (288, N)
            # 这样序列长度从 15000+ 降到了 288，计算速度提升 2500 倍
            mean_traffic = np.mean(xtr, axis=0)

            count = 0
            total_pairs = N * (N - 1) // 2

            for i in range(N):
                for j in range(i + 1, N):
                    # 显示进度，防止以为卡死
                    if count % 1000 == 0:
                        print(f"正在处理第 {count}/{total_pairs} 对节点...", flush=True)
                    count += 1

                    # [修改] 使用日均流量模式进行计算
                    # seq_i = xtr[:, :, i].flatten()  <-- 之前这里导致了卡死
                    seq_i = mean_traffic[:, i]  # 长度 288
                    seq_j = mean_traffic[:, j]  # 长度 288

                    # 确保处理 NaN
                    if np.isnan(seq_i).any(): seq_i = np.nan_to_num(seq_i)
                    if np.isnan(seq_j).any(): seq_j = np.nan_to_num(seq_j)

                    # 计算 DTW (现在非常快了)
                    d[i, j] = compute_dtw(seq_i, seq_j)

                    # 计算延迟 (Lag)
                    # Lag 计算使用平均模式也是合理的，或者你可以单独为 Lag 使用长序列(np.correlate很快)
                    # 这里为了统一和速度，建议也使用平均模式
                    lag_mx[i, j] = compute_delay(seq_i, seq_j, max_lag=self.strides - 1)
                    lag_mx[j, i] = compute_delay(seq_j, seq_i, max_lag=self.strides - 1)

            dtw = d + d.T
            print("节点之间的相似性矩阵（时间图）已经生成！")

            # ... (后续生成 w_adj 的代码保持不变) ...
            n = dtw.shape[0]
            w_adj = np.zeros([n, n])
            adj_percent = self.adj_percent
            top = int(n * adj_percent)

            for i in range(dtw.shape[0]):
                a = dtw[i, :].argsort()[0:top]
                for j in range(top):
                    w_adj[i, a[j]] = 1

            for i in range(n):
                for j in range(n):
                    if (w_adj[i][j] != w_adj[j][i] and w_adj[i][j] == 0):
                        w_adj[i][j] = 1
                    if (i == j):
                        w_adj[i][j] = 1

            lag_mx = lag_mx * w_adj

            print("保存 DTW 和 Lag 矩阵...")
            np.save(self.dtw_file, w_adj)
            np.save(lag_file, lag_mx)

        self.dtw = w_adj
        self.lag_mx = lag_mx

    # 构建融合后的graph： （4N,4N)
    def _construct_adj(self):
        print("------------------进入COMPUTE_delay类_construct_adj方法(开始构造空间图和时间图)：-------------------")
        self._construct_dtw()  # 构建self.dtw: (N,N)  时间相似性图
        print("第一步：调用COMPUTE_delay类_construct_dtw方法 生成时间相似图")
        print("self.adj_mx.shape（空间图）",self.adj_mx.shape)
        print("self.dtw.shape（时间图）", self.dtw.shape)
        adj_mx = construct_adj_fusion(self.adj_mx, self.dtw, self.strides, self.lag_mx)  # (4N,4N)
        print("The shape of localized adjacency matrix: {}".format(adj_mx.shape), flush=True)
        print("融合之后的大矩阵：",adj_mx)
        print("------------------退出COMPUTE_delay类_construct_adj方法：-------------------")
        return adj_mx


    def get_data(self):
        """
        返回数据的DataLoader，包括训练数据、测试数据、验证数据
        这些DataLoader是用来迭代访问数据集的工具
        Returns:
            tuple: tuple contains:
                train_dataloader:
                eval_dataloader:
                test_dataloader:
        """
        # 加载数据集

        return self.data["train_loader"], self.data["valid_loader"], self.data["test_loader"]

    def get_data_feature(self):

        pattern_file = 'model_DAGCN/data/PEMS04/patterns.npy'  # 确保路径对应
        if os.path.exists(pattern_file):
            patterns = np.load(pattern_file)
            patterns = torch.FloatTensor(patterns)  # 转为 Tensor
        else:
            print("警告: 未找到 patterns.npy，使用随机初始化代替测试！")
            patterns = torch.randn(16, 12)  # 仅用于防止报错

        feature = {
            "scaler": self.data["scaler"], # 数据缩放器 对原始数据进行标准化和归一化操作的工具。
            "adj_mx": self.adj_mx, # 融合图
            "num_batches": self.data['num_batches'], # 训练过程一次模型训练需要多少个批次的数据
            "patterns": patterns
        }

        return feature
