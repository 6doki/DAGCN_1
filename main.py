import os
import sys

sys.path.append(os.getcwd())
# from torch.utils.data import DataLoader, TensorDataset
import json

from Data.dataset.stfgnn_dataset import STFGNNDataset
from model_DAGCN.compute_delay import COMPUTE_delay
from model.STFGNN import STFGNN
from model_DAGCN.DAGCN import DAGCN
from executor.multi_step_executor import MultiStepExecutor as STFGNNExecutor

if __name__ == '__main__':
    config = {}  # 空的字典，用来存储配置文件中的配置信息
    for filename in ["config/PEMS04.json", "config/DAGCN.json"]:
        with open(filename, "r") as f: # 打开配置文件之后将其文件对象赋值给f
            _config = json.load(f)     # 将f中的JSON内容加载到_config字典中，该字典存储了当前文件中的所有配置项
            for key in _config:        # 遍历字典中的每一个键
                if key not in config:
                    config[key] = _config[key]
    # print('config字典输出：',config)

    for key,value in config.items():
        print(f"{key}:{value}")       # 打印config字典中的键值对  key:配置项名称  value:配置项值


    '''
    构建STFGNNDataset类的一个实例，dataset包含以下内容：
    1，config中的配置信息
    2，空间邻接矩阵和时间图以及融合的图
    3，数据加载器：get_data()返回的训练、验证和测试数据加载器
    4，数据特征：get_data_feature()返回的数据集相关特征，包括数据的缩放器、邻接矩阵等
    '''
    # dataset = STFGNNDataset(config)
    dataset = COMPUTE_delay(config)
    # print('dataset对象中所有属性以及其值',dataset.__dict__)

    # # 查看dataset的属性
    print("Strides:", dataset.strides)
    # print("Order:", dataset.order)
    # print("Lag:", dataset.lag)
    # print("Period:", dataset.period)
    # print("Sparsity:", dataset.sparsity)
    # print("Train Rate:", dataset.train_rate)
    # print("Adjacency Percent:", dataset.adj_percent)
    # print("Adjacency Matrix:", dataset.adj_mx)

    # 调用方法查看数据加载器和特征
    train_data, valid_data, test_data = dataset.get_data()  # DataLoader

    #  查看所有属性及其值
    # print("Dataset Attributes:", dataset.__dict__)

    data_feature = dataset.get_data_feature()   # 字典，key包括：scaler, adj_mx, num_batches, patterns
    # print("data Feature Size:", data_feature)

    # model_cache_file = 'cache/model_cache/PEMS0410_STFGNN.m'
    # model_cache_file = 'cache/model_cache/PEMS0430_STFGNN.m'
    # model_cache_file = 'cache/model_cache/PEMS08_STFGNN.m'
    # model_cache_file = 'cache/model_cache/PEMS0810_STFGNN.m'
    # model_cache_file = 'cache/model_cache/PEMS0810_STFGNN_UPDATE_0122.m'
    # model_cache_file = 'cache/model_cache/PEMS04_DAGCN_0126_epoch20.m'
    # model_cache_file = 'cache/model_cache/PEMS04_DAGCN_0126_epoch200.m'
    # model_cache_file = 'cache/model_cache/PEMS04_DAGCN_0127_epoch50.m'
    # model_cache_file = 'cache/model_cache/PEMS04_DAGCN_0128_epoch200.m'
    # model_cache_file = 'cache/model_cache/PEMS04_DAGCN_0205_epoch50.m'
    # model_cache_file = 'cache/model_cache/PEMS04_DAGCN_0302_epoch50.m'
    # model_cache_file = 'cache/model_cache/PEMS04_DAGCN_0303_epoch50.m'
    model_cache_file = 'cache/model_cache/PEMS04_DAGCN_0306_epoch50_evaluator_png.m'

    # model = STFGNN(config, data_feature)
    model = DAGCN(config, data_feature)

    executor = STFGNNExecutor(config, model)   # 使用自己定义的评估模型
    train = False  # 标识是否需要重新训练

    if train or not os.path.exists(model_cache_file):
        executor.train(train_data, valid_data)
        executor.save_model(model_cache_file)
    else:
        executor.load_model(model_cache_file)
    # 评估，评估结果将会放在 cache/evaluate_cache 下
    executor.evaluate(test_data) # 调用的是multi_step_executor.py文件中MultiStepExecutor类的evaluate方法
    # executor.plot_losses()
