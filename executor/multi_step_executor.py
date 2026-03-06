import os
import time
import numpy as np
import torch
import math
import time
import torch.nn as nn
from torch.autograd import Variable
from logging import getLogger
import logging
import tqdm
from torch.utils.tensorboard import SummaryWriter
from executor.utils import get_train_loss
from utils.Optim import Optim
from evaluator.evaluator import Evaluator
from utils.utils import ensure_dir

from model import loss
from functools import partial

import matplotlib.pyplot as plt

'''
创建一个配置好的日志记录器logger
'''
def get_logger(root, name=None, debug=True):
    print("进入multi_step_executor.py中的get_logger方法：")
    '''
    :param root: 日志文件存储的根目录
    :param name: 日志记录器的名字
    :param debug: 布尔值，是否在控制台显示debug级别的日志
    :return:
    '''
    # when debug is true, show DEBUG and INFO in screen
    # when debug is false, show DEBUG in file and info in both screen&file
    # INFO will always be in screen
    # create a logger
    logger = logging.getLogger(name)
    # critical > error > warning > info > debug > notset
    logger.setLevel(logging.DEBUG) # 处理所有级别的日志（debug,info,warning,error,critical）

    '''
    日志格式：
    %(asctime)s 表示日志记录的时间戳，会被格式化为指定的时间格式（由第二个参数定义）
    %(message)s 表示实际的日志消息内容
    "%Y-%m-%d %H:%M" 时间格式"YYYY-MM-DD HH"
    '''
    formatter = logging.Formatter('%(asctime)s: %(message)s', "%Y-%m-%d %H:%M")
    # create another handler for output log to console
    console_handler = logging.StreamHandler() # 创建一个控制台处理器（StreamHandler），用于将日志输出到控制台
    if debug:
        console_handler.setLevel(logging.DEBUG)
    else:
        console_handler.setLevel(logging.INFO)
    # create a handler for write log to file
    logfile = os.path.join(root, name)  # 创建日志文件的完整路径
    print('Creat Log File in(创建的日志文件路径在): ', logfile)
    # 创建一个文件处理器（FileHandler），用于将日志输出到指定的文件。模式为 'w'，表示每次打开文件时清空之前的内容
    file_handler = logging.FileHandler(logfile, mode='w')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    console_handler.setFormatter(formatter)
    # 将控制台处理器和文件处理器添加到日志记录器，使得日志信息可以同时输出到控制台和文件
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    return logger


class MultiStepExecutor(object):
    def __init__(self, config, model):
        self.config = config
        self.evaluator = Evaluator(config)  # 调用evaluator.py中的Evaluator类创建一个评估器对象
        _device = self.config.get('device', torch.device('cpu'))  # 从config配置中获取device，并且将模型移动到指定的设备上
        self.device = torch.device(_device)
        self.model = model.to(self.device)
        self.cache_dir = os.path.join('cache/model_cache')
        self.evaluate_res_dir = 'cache/evaluate_cache'
        self.summary_writer_dir = 'log/runs'
        # ensure_dir 确保目录存在 不存在则创建
        ensure_dir(self.cache_dir)
        ensure_dir(self.evaluate_res_dir)
        ensure_dir(self.summary_writer_dir)
        self._writer = SummaryWriter(self.summary_writer_dir)   # PyTorch框架下的可视化工具
        self._logger = get_logger(self.cache_dir,time.strftime("%Y%m%d-%H:%M", time.localtime()) + ".log")
        self._logger.info(self.model)
        for name, param in self.model.named_parameters():   # 参数统计
            self._logger.debug(str(name) + '\t' + str(param.shape) + '\t' +
                              str(param.device) + '\t' + str(param.requires_grad))
        # 统计模型参数数量
        total_num = sum([param.nelement() for param in self.model.parameters()])
        self._logger.debug('Total parameter numbers: {}'.format(total_num))
        # 训练配置
        self.train_loss = self.config.get("train_loss", "masked_mae")   # mae
        self.criterion = get_train_loss(self.train_loss) 

        self.cuda = self.config.get("cuda", True)   # true
        self.best_val = 10000000
        self.optim = Optim(
            model.parameters(), self.config
        )
        self.epochs = self.config.get("epochs", 100)    # 100
        self.scaler = self.model.scaler     # norm=1, 即Z-score
        self.num_batches = self.model.num_batches   # n_train/batch_size64
        self.num_nodes = self.config.get("num_nodes", 0)    # N
        self.batch_size = self.config.get("batch_size", 64)  # 64
        self.patience = self.config.get("patience", 20)     # 20
        self.lr_decay = self.config.get("lr_decay", False)  # False
        self.mask = self.config.get("mask", True)   # False

        # 补充
        self.train_losses_per_epoch = []  # 存储每个epoch的训练损失
        self.valid_losses_per_epoch = []  # 存储每个epoch的验证损失


    def train(self, train_data, valid_data):
        print("进入multi_step_executor.py中的train方法：")
        print("begin training.......")
        wait = 0  # 用于早停策略的计数
        batches_seen = self.num_batches * 0  # 用于跟踪已处理的批次数量
        

        for epoch in tqdm.tqdm(range(1, self.epochs + 1)): # 使用 tqdm 进行进度条显示，循环训练轮次从 1 到 self.epochs
            epoch_start_time = time.time() # 记录每个 epoch 的开始时间
            train_loss = []
            train_data.shuffle() # 在每个 epoch 开始时打乱训练数据的顺序，以提高模型的泛化能力

            for iter, (x,y) in enumerate(train_data.get_iterator()):
                '''遍历训练数据集的迭代器，(x, y) 分别是输入和目标（标签）数据'''
                self.model.train()  # 将模型设置为训练模式，启用 dropout 和 batch normalization
                self.model.zero_grad()  # 清空模型的梯度，以便开始新的反向传播
                '''将输入 x 和目标 y 转换为 PyTorch 张量，并移动到指定的设备'''
                trainx = torch.Tensor(x).to(self.device)  # [batch_size, window, num_nodes, dim]: (64, 12, N, 1)
                trainy = torch.Tensor(y).to(self.device)  # [batch_size, horizon, num_nodes, dim]
                output = self.model(trainx)
                '''将模型输出output与目标y反归一化之后，计算其损失'''
                loss = self.criterion(self.scaler.inverse_transform(output), 
                    self.scaler.inverse_transform(trainy))
                
                loss.backward() # 计算梯度
                self.optim.step() # 梯度更新
                train_loss.append(loss.item()) # 当前批次的损失值添加到 train_loss 列表中
                
            
            if self.lr_decay: # 学习率更新
                self.optim.lr_scheduler.step()

            valid_loss = []
            valid_mape = []
            valid_rmse = []
            valid_pcc = []
            for iter, (x, y) in enumerate(valid_data.get_iterator()):
                '''遍历验证数据集的迭代器'''
                self.model.eval() # 将模型设置为评估模式，以禁用dropout和batch normalization
                valx = torch.Tensor(x).to(self.device)
                valy = torch.Tensor(y).to(self.device)
                with torch.no_grad(): # 禁用梯度计算，避免不必要的内存消耗
                    output = self.model(valx) # 获取模型在验证数据上的输出
                '''调用评估器的 evaluate 方法，计算验证损失和其他性能指标'''
                score = self.evaluator.evaluate(self.scaler.inverse_transform(output), \
                    self.scaler.inverse_transform(valy))
                if self.mask:
                    vloss = score["masked_MAE"]["all"]
                else:
                    vloss = score["MAE"]["all"]
                    
                valid_loss.append(vloss)
            
            # 平均
            mtrain_loss = np.mean(train_loss)
            mvalid_loss = np.mean(valid_loss)

            self.train_losses_per_epoch.append(mtrain_loss)
            self.valid_losses_per_epoch.append(mvalid_loss)

            print(
                '| end of epoch {:3d} | time: {:5.2f}s | train_loss平均训练损失 {:5.4f} | valid mae平均验证损失 {:5.4f}'.format(
                    epoch, (time.time() - epoch_start_time), mtrain_loss, \
                        mvalid_loss))

            '''若当前验证损失小于之前的最佳值，则更新最佳损失和最佳模型，并重置等待计数'''
            if mvalid_loss < self.best_val:
                self.best_val = mvalid_loss
                wait = 0
                self.best_val = mvalid_loss
                self.best_model = self.model
            else:
                wait += 1
            '''等待计数达到耐心阈值，则提前停止训练，并打印停止的 epoch 信息'''
            if wait >= self.patience:
                print('early stop at epoch早停: {:04d}'.format(epoch))
                break
        
        self.model = self.best_model

    "补充"
    def plot_losses(self):
        """
        绘制训练和验证损失随epoch变化的曲线
        """
        epochs = range(1, len(self.train_losses_per_epoch) + 1)
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, self.train_losses_per_epoch, label='Training Loss')
        plt.plot(epochs, self.valid_losses_per_epoch, label='Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Training and Validation Losses over Epochs')
        plt.legend()
        plt.grid(True)

        # 确保保存图像的目录存在
        save_dir = 'png'
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, 'loss_0810.png')

        # 保存图像
        plt.savefig(save_path)
        plt.close()  # 关闭图形以释放内存

        self._logger.info(f'Loss plot saved to {save_path}')

    def plot_prediction_curve(self, preds, realy, node_idx=0, start_step=0, length=288, horizon_idx=0):
        """
        绘制具体节点的真实值与预测值对比曲线
        preds / realy 形状: (n_samples, horizon, num_nodes, 1)
        length=288 刚好是一天的步数 (5分钟一次)
        """
        import matplotlib.pyplot as plt
        import os

        # 提取真实值和预测值的一维序列
        truth_seq = realy[start_step: start_step + length, horizon_idx, node_idx, 0]
        pred_seq = preds[start_step: start_step + length, horizon_idx, node_idx, 0]

        plt.figure(figsize=(12, 4))
        plt.plot(truth_seq, label='Ground Truth', color='blue', linewidth=1.5)
        plt.plot(pred_seq, label='Prediction (DAGCN)', color='red', linestyle='--', linewidth=1.5)
        plt.title(f'Traffic Flow Prediction at Node {node_idx} (Horizon {horizon_idx + 1})')
        plt.xlabel('Time Steps (5 min/step)')
        plt.ylabel('Traffic Flow')
        plt.legend()
        plt.grid(True, linestyle=':', alpha=0.6)

        save_dir = 'png'
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f'pred_curve_node{node_idx}.png')
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
        self._logger.info(f'Prediction curve saved to {save_path}')

    def evaluate(self, test_data):
        """
        use model to test data
        test_data：测试数据加载器
        Args:
            test_dataloader(torch.Dataloader): Dataloader
        """
        self._logger.debug('Start evaluating（MultiStepExecutor类的evaluate方法） ...')
        outputs = [] # 存储模型输出
        realy = [] # 存储真实的目标值
        seq_len = test_data.seq_len  #test_data["y_test"] 获取测试数据的序列长度，这通常是预测结果的长度
        self.model.eval() # 将模型设置为评估模式（self.model.eval()），以禁用 dropout 和 batch normalization

        for iter, (x, y) in enumerate(test_data.get_iterator()):
            testx = torch.Tensor(x).to(self.device)
            testy = torch.Tensor(y).to(self.device)
            with torch.no_grad():
                # self.evaluator.clear()
                pred = self.model(testx)
                '''将预测结果 pred 和真实目标 testy 分别添加到 outputs 和 realy 列表中。'''
                outputs.append(pred) 
                realy.append(testy)
        # 拼接
        realy = torch.cat(realy, dim=0)
        yhat = torch.cat(outputs, dim=0)
        # 将 realy 和 yhat 切片到 seq_len，确保预测结果和真实目标的长度一致。（取前seq_len个）
        realy = realy[:seq_len, ...]
        yhat = yhat[:seq_len, ...]

        '''反归一化'''
        realy = self.scaler.inverse_transform(realy)
        preds = self.scaler.inverse_transform(yhat)

        '''调用评估器的 evaluate 方法，将反归一化后的预测结果 preds 和真实目标 realy 作为参数传入，计算评估指标'''
        res_scores = self.evaluator.evaluate(preds, realy)
        for _index in res_scores.keys(): # 遍历评估结果 res_scores 的每个指标
            self._logger.debug( "{} :".format(_index))
            step_dict = res_scores[_index]
            for j, k in step_dict.items():
                self._logger.debug("{} : {}".format(j, k.item()))

        # 新增评估
        self.plot_prediction_curve(preds, realy, node_idx=10, horizon_idx=0)
        self.plot_prediction_curve(preds, realy, node_idx=10, horizon_idx=11)

        # 【新增：计算每个节点的平均 MAE 并绘制直方图】
        # preds/realy shape: (samples, horizon=12, nodes, 1)
        # 我们对所有样本 (axis=0) 和所有时间步 (axis=1) 求平均绝对误差
        abs_error = np.abs(preds[..., 0] - realy[..., 0])  # shape: (samples, horizon, nodes)
        node_mae = np.mean(abs_error, axis=(0, 1))  # shape: (nodes,)

        plt.figure(figsize=(8, 5))
        plt.hist(node_mae, bins=30, color='skyblue', edgecolor='black', alpha=0.7)
        plt.axvline(np.mean(node_mae), color='red', linestyle='dashed', linewidth=2,
                    label=f'Mean MAE: {np.mean(node_mae):.2f}')
        plt.title('Distribution of MAE Across All Nodes')
        plt.xlabel('Node Mean Absolute Error (MAE)')
        plt.ylabel('Number of Nodes')
        plt.legend()
        plt.grid(True, linestyle=':', alpha=0.6)

        dist_path = os.path.join('png', 'node_mae_distribution.png')
        plt.savefig(dist_path, bbox_inches='tight')
        plt.close()
        self._logger.info(f'Node MAE distribution saved to {dist_path}')

        # 可选：打印出最难预测的5个节点（MAE最大）
        worst_nodes = np.argsort(node_mae)[-5:]
        self._logger.info(f'Top 5 hardest nodes to predict (Highest MAE): {worst_nodes}')
        
        

    def save_model(self, cache_name):
        """
        将当前的模型保存到文件

        Args:
            cache_name(str): 保存的文件名 通常包含文件路径和扩展名
        """
        ensure_dir(self.cache_dir)
        self._logger.info("Saved model at " + cache_name)
        '''
        使用 torch.save 函数保存模型的状态字典（state_dict），这包括模型的所有参数和缓冲区
        state_dict 是 PyTorch 中用于序列化模型的标准方法，它只保存模型的参数，不包括模型结构信息。通常，重建模型时需要知道模型的结构
        '''
        torch.save(self.model.state_dict(), cache_name)

    def load_model(self, cache_name):
        """
        加载对应模型的 cache

        Args:
            cache_name(str): 保存的文件名 通常包含文件路径和扩展名
        """
        self._logger.info("Loaded model at " + cache_name)
        model_state = torch.load(cache_name) # 使用 torch.load 函数从指定文件中加载模型状态字典
        self.model.load_state_dict(model_state) # 使用 load_state_dict 方法将加载的状态字典应用到当前模型中，以恢复模型的参数和状态。
