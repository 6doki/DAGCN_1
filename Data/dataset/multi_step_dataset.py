import os
import pandas as pd
import numpy as np
import pickle
import csv
import torch
from torch.autograd import Variable

from Data.utils import DataLoader, load_pickle, DataLoaderM_new
from utils.normalization import StandardScaler, NormalScaler, NoneScaler, \
    MinMax01Scaler, MinMax11Scaler, LogScaler


def get_adjacency_matrix(distance_df, sensor_ids, normalized_k=0.01):
    print("-----------进入get_adjacency_matrix方法：使用传感器间的距离数据构建邻接矩阵，从而得到空间图-----------")
    """
    功能：
      用于从传感器间的距离数据构建归一化的邻接矩阵 从而得到从数据到空间图的转变
    distance_df: 三列 [from , to , distance] 传感器from到传感器to的距离
    sensor_ids: 传感器ID列表
    normalized_k: 归一化的阈值，低于这个阈值的条目被设置为0，以保持稀疏性
    """
    num_sensors = len(sensor_ids) # 传感器的数量
    dist_mx = np.zeros((num_sensors, num_sensors), dtype=np.float32) # 初始化矩阵 其元素值表示每两个传感器之间的距离
    dist_mx[:] = np.inf  # 所有值设置为无穷大

    '''建立传感器到索引的映射，从而通过传感器的ID快速找到它在dist_mx矩阵中的位置'''
    sensor_id_to_ind = {}  # 字典
    # i是传感器ID在ID列表中的索引，sensor_id是ID列表中的具体传感器ID，enumerate返回一个(index,value)元组
    for i, sensor_id in enumerate(sensor_ids):
        sensor_id_to_ind[sensor_id] = i # 键值对（key:sensor_id,value:i） 这样就可以通过传感器ID找到列表中的索引
        # print('字典中第'+sensor_id+'个元素的')

    '''填充矩阵'''
    for row in distance_df.values:  # 遍历每一行 每一行包含一对传感器以及距离
        if row[0] not in sensor_id_to_ind or row[1] not in sensor_id_to_ind:  # 如果起点或者终点传感器ID不在映射字典中，skip
            continue
        '''
        sensor_id_to_ind[row[0]]:传感器ID在ID列表中的索引 其中row[0]表示所遍历的某一行的起点传感器 row[1]表示所遍历的某一行的终点传感器
        '''
        dist_mx[sensor_id_to_ind[row[0]], sensor_id_to_ind[row[1]]] = row[2] # 将传感器对的距离填充到对应的矩阵dist_mx矩阵的位置

    print('dist_mx的shape：',dist_mx.shape)
    '''计算距离的标准差'''
    distances = dist_mx[~np.isinf(dist_mx)].flatten() # 将矩阵中所有不是无穷大的值提取出来（即有效的距离数据）转成一个一维数组
    std = distances.std() # 计算这些有效距离的标准差
    print("交通流空间三元组的距离标准差为：",std)
    adj_mx = np.exp(-np.square(dist_mx / std)) # 将距离转换为邻接矩阵的权重，平方后再取负指数，使得距离近的传感器之间的连接权重较大，距离远的权重小

    # Make the adjacent matrix symmetric by taking the max.
    # adj_mx = np.maximum.reduce([adj_mx, adj_mx.T])

    # Sets entries that lower than a threshold, i.e., k, to zero for sparsity.
    # 译文：对于稀疏度，将矩阵中值低于阈值(即k)的条目设置为零。
    adj_mx[adj_mx < normalized_k] = 0
    '''
    返回结果：sensor_ids:传感器的ID列表
            sensor_id_to_ind:传感器ID到索引的映射字典
            adj_mx：计算得到的邻接矩阵，表示传感器之间的空间关系
    '''
    print("-----------退出get_adjacency_matrix方法（返回：传感器ID列表、映射字典、邻接矩阵空间图）-----------")
    return sensor_ids, sensor_id_to_ind, adj_mx


class MultiStepDataset(object):

    def __init__(self, config):

        self.config = config
        self.file_name = self.config.get("filename", " ")  # 原始数据
        self.adj_filename = self.config.get("adj_filename", "")  # 空间数据
        self.dtw_file = self.config.get("dtw_file", None)  # 时间数据
        self.graph_sensor_ids = self.config.get("graph_sensor_ids", "")
        self.distances_file = self.config.get("distances_file", "")
        self.adj_type = self.config.get("adj_type", None)

        self.train_rate = self.config.get("train_rate", 0.6)
        self.valid_rate = self.config.get("eval_rate", 0.2)
        self.cuda = self.config.get("cuda", True)

        self.horizon = self.config.get("horizon", 24)
        self.window = self.config.get("window", 12)

        self.normalize = self.config.get("normalize", 2)
        self.batch_size = self.config.get("batch_size", 64)
        self.adj_mx = None
        self.add_time_in_day = self.config.get("add_time_in_day", False)
        self.add_day_in_week = self.config.get("add_day_in_week", False)
        self.input_dim = self.config.get("input_dim", 1)
        self.output_dim = self.config.get("output_dim", 1)
        self.ensure_adj_mat()
        self._load_origin_data(self.file_name, self.adj_filename) # adj_filename = adj_name

        self.data = self._gene_dataset()

    def ensure_adj_mat(self):
        '''
        如果邻接矩阵文件不存在，代码会根据传感器之间的距离数据（存储在distances_file中）使用get_adjacency_matrix函数来生成空间邻接矩阵
        :return:邻接矩阵（被保存为pkl文件格式）
        '''
        if os.path.exists(self.adj_filename):
            print(f"空间邻接矩阵已经存在：{self.adj_filename}")
            return
        else:
            print("——————————————————开始生成空间图的邻接矩阵！！！！！——————————————————")
            with open(self.graph_sensor_ids) as f:
                sensor_ids = f.read().strip().split(',')
            distance_df = pd.read_csv(self.distances_file, dtype={'from': 'str', 'to': 'str'})
            '''调用函数通过距离数据构建邻接矩阵'''
            _, sensor_id_to_ind, adj_mx = get_adjacency_matrix(distance_df, sensor_ids, normalized_k=0.01)
            with open(self.adj_filename, 'wb') as f:
                pickle.dump([sensor_ids, sensor_id_to_ind, adj_mx], f, protocol=2)

            # with open(self.graph_sensor_ids) as f:
            #     sensor_ids = f.read().strip().split(',')
            # distance_df = pd.read_csv(self.adj_filename, dtype={'from': 'str', 'to': 'str'})
            # '''调用函数通过距离数据构建邻接矩阵'''
            # _, sensor_id_to_ind, adj_mx = get_adjacency_matrix(distance_df, sensor_ids, normalized_k=0.01)
            # with open(self.adj_filename, 'wb') as f:
            #     pickle.dump([sensor_ids, sensor_id_to_ind, adj_mx], f, protocol=2)
            print("——————————————————已生成空间图的邻接矩阵！！！！！——————————————————")
            return

    def _load_origin_data(self, file_name, adj_name):
        '''
        功能：
          加载数据文件和空间图（邻接矩阵）
        :param file_name: 包含原始数据的文件名
        :param adj_name: 包含邻接矩阵数据的文件名
        :return:
        '''
        # 根据文件扩展名加载不同类型的数据文件
        print("---------------------进入_load_origin_data方法---------------------")
        if file_name[-3:] == "txt":
            fin = open(file_name)
            self.rawdat = np.loadtxt(fin, delimiter=',')
            print("检测到file_name文件扩展名是：txt ，加载数据文件！")
        elif file_name[-3:] == "csv":
            self.rawdat = pd.read_csv(file_name).values
            print("rawdat样子："+self.rawdat)
            print("检测到file_name文件扩展名是：csv ，加载数据文件！")
        elif file_name[-2:] == "h5":
            self.rawdat = pd.read_hdf(file_name)
            print("检测到file_name文件扩展名是：csv ，加载数据文件！")
        elif file_name[-3:] == "npz":
            mid_dat = np.load(file_name)
            self.rawdat = mid_dat[mid_dat.files[0]]
            print("读取npz文件中的第一个数组生成了self.rawdat,其形状是：",self.rawdat.shape)  # PEMS04:(16992,307,3)
            if len(self.rawdat.shape) == 2:
                self.rawdat = np.expand_dims(self.rawdat, axis=-1)
            print("检测到file_name文件扩展名是：npz ，加载数据文件！")
        else:
            raise ValueError('检测到的file_name文件类型错误!')
        # 载入空间图
        if adj_name == "":
            self.adj_mx = None
            print("检测到的adj_name文件为空！")
        elif adj_name[-3:] == "pkl":  # 如果邻接矩阵文件是pkl格式
            sensor_ids, sensor_id_to_ind, adj = load_pickle(adj_name)  # adj: 表示距离（N,N)
            if self.adj_type == "distance":
                self.adj_mx = adj
            else:  # "adj_type": "connectivity" 大于0的设置为1，其他值设置为0
                row, col = adj.shape
                for i in range(row):
                    for j in range(i, col):
                        if adj[i][j] > 0:
                            adj[i][j] = 1
                            adj[j][i] = 1
                        else:
                            adj[i][j] = 0
                            adj[j][i] = 0
                self.adj_mx = adj
            print("检测到adj_name文件格式是:pkl，加载邻接矩阵文件！")
        elif adj_name[-3:] == "csv":  # 如果邻接矩阵文件是csv格式
            num_of_vertices = self.rawdat.shape[1]
            adj = np.zeros((num_of_vertices, num_of_vertices), dtype=np.float32)  # 初始化一个全零的邻接矩阵adj
            with open(adj_name, 'r') as f:
                f.readline()
                reader = csv.reader(f)
                for row in reader:
                    if len(row) != 3:
                        continue
                    i, j, d = int(row[0]), int(row[1]), float(row[2])
                    if self.adj_type == "distance":
                        adj[i][j] = 1 / d
                        adj[j][i] = 1 / d
                    else:
                        adj[i][j] = 1
                        adj[j][i] = 1
            self.adj_mx = adj
            print("检测到adj_name文件格式是:csv，加载邻接矩阵文件！")
        else:
            raise ValueError('检测到的adj_name文件类型错误!')
        print("---------------------退出_load_origin_data方法---------------------")

    def _get_scalar(self, x_train, y_train):
        """
        根据全局参数`scaler_type`选择数据归一化方法

        Args:
            x_train: 训练数据X
            y_train: 训练数据y

        Returns:
            Scaler: 归一化对象
        """
        if self.normalize == 2:
            scaler = NormalScaler(maxx=max(x_train.max(), y_train.max()))
            print('NormalScaler max: ' + str(scaler.max))
        elif self.normalize == 1:  # 1
            scaler = StandardScaler(mean=x_train.mean(), std=x_train.std())
            print('StandardScaler mean: ' + str(scaler.mean) + ', std: ' + str(scaler.std))
        elif self.normalize == 3:
            scaler = MinMax01Scaler(
                maxx=max(x_train.max(), y_train.max()), minn=min(x_train.min(), y_train.min()))
            print('MinMax01Scaler max: ' + str(scaler.max) + ', min: ' + str(scaler.min))
        elif self.normalize == 4:
            scaler = MinMax11Scaler(
                maxx=max(x_train.max(), y_train.max()), minn=min(x_train.min(), y_train.min()))
            print('MinMax11Scaler max: ' + str(scaler.max) + ', min: ' + str(scaler.min))
        elif self.normalize == 5:
            scaler = LogScaler()
            print('LogScaler')
        elif self.normalize == 0:
            scaler = NoneScaler()
            print('NoneScaler')
        else:
            raise ValueError('Scaler type error!')
        return scaler

    def _generate_graph_seq2seq_io_data(
            self, df, x_offsets, y_offsets, add_time_in_day=False, add_day_in_week=False, scaler=None
    ):
        """
        功能：
            生成seq2seq样本数据
        参数：
            data: np数据 [B, N, D] 其中D特征维度为3
            df:输入数据，通常是pandas.DataFrame，每行代表一个时间步，每列代表一个节点
            x_offsets：输入序列的时间步偏移列表
            y_offsets：输出序列的时间步偏移列表
            add_time_in_day：是否添加一天中的时间特征
            add_day_in_week：是否添加一周中的时间特征
            scaler:缩放器，用于数据预处理
        """
        num_samples, num_nodes = df.shape[0], df.shape[1]  # 样本数 节点数
        if not isinstance(df, np.ndarray):  # 如果不是numpy类型
            data = np.expand_dims(df.values, axis=-1)  # 转换并且扩展最内维度
            data_list = [data]
        else:
            data_list = [df]
        if add_time_in_day:  # 添加一天中的时间特征
            time_ind = (df.index.values - df.index.values.astype("datetime64[D]")) / np.timedelta64(1, "D")
            time_in_day = np.tile(time_ind, [1, num_nodes, 1]).transpose((2, 1, 0))
            data_list.append(time_in_day)
        if add_day_in_week:  # 添加一周中的时间特征
            day_in_week = np.zeros(shape=(num_samples, num_nodes, 7))
            day_in_week[np.arange(num_samples), :, df.index.dayofweek] = 1
            data_list.append(day_in_week)

        data = np.concatenate(data_list, axis=-1)

        x, y = [], []
        # t is the index of the last observation.
        min_t = abs(min(x_offsets))  # 最小的输入时间步
        max_t = abs(num_samples - abs(max(y_offsets)))  # 最大的输出时间步
        for t in range(min_t, max_t):
            x_t = data[t + x_offsets, ...]
            y_t = data[t + y_offsets, ...]
            x.append(x_t)
            y.append(y_t)
        x = np.stack(x, axis=0)  # (num_samples,T_x,num_nodes,D) 即PEMS04输入样本对X:(16957,12,307,3)
        y = np.stack(y, axis=0)  # (num_samples,T_y,num_nodes,D) 即PEMS04输出样本对Y:(16957,24,307,3)

        return x, y

    def _generate_train_val_test(self):
        print("开始将连续的数据集序列变成一个个独立样本..........")
        seq_length_x, seq_length_y = self.window, self.horizon  # 第一个是输入长度（窗口大小） 第二个是输出长度（预测视窗）
        x_offsets = np.arange(-(seq_length_x - 1), 1, 1)
        y_offsets = np.arange(1, (seq_length_y + 1), 1)
        x, y = self._generate_graph_seq2seq_io_data(self.rawdat, x_offsets, y_offsets, self.add_time_in_day, self.add_day_in_week)
        print("原始数据self.rawdat shape: ", self.rawdat.shape)
        print("原始数据序列生成的输入样本对x.shape", x.shape, ", 原始数据序列生成的输出样本对y shape: ", y.shape)
        num_samples = x.shape[0]  # 样本总数 PEMS04:16957
        num_val = round(num_samples * self.valid_rate)  # 验证集样本数
        print("验证集样本数:", num_val)
        num_train = round(num_samples * self.train_rate)  # 训练集样本数
        print("训练集样本数:", num_train)
        num_test = num_samples - num_train - num_val  # 测试集样本数
        print("测试集样本数:", num_test)

        # 输出训练集、验证集、测试集的形状
        print("*************************************************")
        print("训练集 x shape:", x[:num_train].shape, ", 训练集 y shape:", y[:num_train].shape)
        print("验证集 x shape:", x[num_train:num_train + num_val].shape, ", 验证集 y shape:",
              y[num_train:num_train + num_val].shape)
        print("测试集 x shape:", x[num_train + num_val:].shape, ", 测试集 y shape:", y[num_train + num_val:].shape)
        print("*************************************************")
        '''
        self.train:     [x[:num_train], y[:num_train]]
        self.train[0]:  x[:num_train] 输入序列
        self.train[1]:  y[:num_train] 输出序列
        '''
        return [x[:num_train], y[:num_train]], \
               [x[num_train:num_train + num_val], y[num_train:num_train + num_val]], \
               [x[num_train + num_val:], y[num_train + num_val:]]

    def _gene_dataset(self):
        data = {}
        self.train, self.valid, self.test = self._generate_train_val_test()  # train:[x,y];    x or y: narray(15711,12,358,1)
        x_train, y_train = self.train[0], self.train[1]  # 训练集的输入序列和输出序列
        x_valid, y_valid = self.valid[0], self.valid[1]  # 验证集的输入序列和输出序列
        x_test, y_test = self.test[0], self.test[1]      # 测试集的输入序列和输出序列
        self.scaler = self._get_scalar(x_train[..., :self.output_dim],
                                       y_train[..., :self.output_dim])  # std标准化（Z-score）
        x_train[..., :self.output_dim] = self.scaler.transform(x_train[..., :self.output_dim])
        y_train[..., :self.output_dim] = self.scaler.transform(y_train[..., :self.output_dim])
        x_valid[..., :self.output_dim] = self.scaler.transform(x_valid[..., :self.output_dim])
        y_valid[..., :self.output_dim] = self.scaler.transform(y_valid[..., :self.output_dim])
        x_test[..., :self.output_dim] = self.scaler.transform(x_test[..., :self.output_dim])
        y_test[..., :self.output_dim] = self.scaler.transform(y_test[..., :self.output_dim])

        '''数据集三个特征（流量、占有率、速度）只选取流量一个特征'''
        data['train_loader'] = DataLoader(x_train[..., :self.input_dim], y_train[..., :self.output_dim],
                                          self.batch_size)
        data['valid_loader'] = DataLoader(x_valid[..., :self.input_dim], y_valid[..., :self.output_dim],
                                          self.batch_size)
        data['test_loader'] = DataLoader(x_test[..., :self.input_dim], y_test[..., :self.output_dim], self.batch_size)
        data['scaler'] = self.scaler
        data['num_batches'] = x_train.shape[0] / self.batch_size
        return data

    def get_data(self):
        """
        返回数据的DataLoader，包括训练数据、测试数据、验证数据

        Returns:
            tuple: tuple contains:
                train_dataloader:
                eval_dataloader:
                test_dataloader:
        """
        # 加载数据集

        return self.data["train_loader"], self.data["valid_loader"], self.data["test_loader"]

    def get_data_feature(self):
        """
        返回数据集特征，子类必须实现这个函数，返回必要的特征

        Returns:
            dict: 包含数据集的相关特征的字典
        """
        feature = {
            "scaler": self.data["scaler"],
            "adj_mx": self.adj_mx,
            "num_batches": self.data['num_batches']
        }

        return feature


