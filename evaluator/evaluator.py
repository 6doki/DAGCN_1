import pickle
import numpy as np
import os
import scipy.sparse as sp
import torch
from scipy.sparse import linalg
from torch.autograd import Variable
from collections import defaultdict
import matplotlib.pyplot as plt
'''
计算相对标准误差RSE：RSE越小，预测值和真实值之间相对误差越小，模型预测性能越好
'''


def rse_np(preds, labels):  # 参数preds:预测值 labels:真实值
    if not isinstance(preds, np.ndarray):
        preds = preds.cpu().numpy()
        labels = labels.cpu().numpy()
    mse = np.sum(np.square(np.subtract(preds, labels)).astype('float32'))  # 分子部分
    means = np.mean(labels)  # 分母部分
    labels_mse = np.sum(np.square(np.subtract(labels, means)).astype('float32'))
    return np.sqrt(mse / labels_mse)


'''
计算平均绝对误差MAE：MAE越小，预测值和真实值之间平均绝对误差越小，模型预测性能越好
'''


def mae_np(preds, labels):
    if isinstance(preds, np.ndarray):
        mae = np.abs(np.subtract(preds, labels)).astype('float32')
    else:
        mae = np.abs(np.subtract(preds.cpu().numpy(), labels.cpu().numpy())).astype('float32')
    return np.mean(mae)


'''
计算均方根误差RMSE：RMSE越小，预测值和真实值之间均方根误差越小，模型预测性能越好
'''


def rmse_np(preds, labels):
    mse = mse_np(preds, labels)  # 调用下面的函数
    return np.sqrt(mse)


def mse_np(preds, labels):
    if isinstance(preds, np.ndarray):
        return np.mean(np.square(np.subtract(preds, labels)).astype('float32'))
    else:
        return np.mean(np.square(np.subtract(preds.cpu().numpy(), labels.cpu().numpy())).astype('float32'))


'''计算预测值与真实值之间的平均绝对百分比误差MAPE（没对异常值进行处理）'''


def mape_np(preds, labels):
    if isinstance(preds, np.ndarray):
        mape = np.abs(np.divide(np.subtract(preds, labels).astype('float32'), labels))
    else:
        mape = np.abs(
            np.divide(np.subtract(preds.cpu().numpy(), labels.cpu().numpy()).astype('float32'), labels.cpu().numpy()))
    return np.mean(mape)


'''计算预测值与真实值之间的相对绝对误差（RAE）'''


def rae_np(preds, labels):
    mse = np.sum(np.abs(np.subtract(preds, labels)).astype('float32'))
    means = np.mean(labels)
    labels_mse = np.sum(np.abs(np.subtract(labels, means)).astype('float32'))
    return mse / labels_mse


'''计算皮尔逊相关系数'''


def pcc_np(x, y):
    if not isinstance(x, np.ndarray):
        x, y = x.cpu().numpy(), y.cpu().numpy()
    x, y = x.reshape(-1), y.reshape(-1)
    return np.corrcoef(x, y)[0][1]


'''计算节点级别的皮尔逊相关系数，即两个矩阵中每个节点对应位置的向量的相关性'''


def node_pcc_np(x, y):
    if not isinstance(x, np.ndarray):
        x, y = x.cpu().numpy(), y.cpu().numpy()
    sigma_x = x.std(axis=0)
    sigma_y = y.std(axis=0)
    mean_x = x.mean(axis=0)
    mean_y = y.mean(axis=0)
    cor = ((x - mean_x) * (y - mean_y)).mean(0) / (sigma_x * sigma_y + 0.000000000001)
    return cor.mean()


'''
计算预测值与标签之间的相关性系数。
具体来说，它计算了每个特征（列）上的皮尔逊相关系数，
并对非零标准差的特征取平均值，从而得到整体的相关性系数。
'''


def corr_np(preds, labels):
    sigma_p = (preds).std(axis=0)
    sigma_g = (labels).std(axis=0)
    mean_p = preds.mean(axis=0)
    mean_g = labels.mean(axis=0)
    index = (sigma_g != 0)
    correlation = ((preds - mean_p) * (labels - mean_g)).mean(axis=0) / (sigma_p * sigma_g)
    correlation = (correlation[index]).mean()
    return correlation


'''计算平均绝对百分比误差MAPE(对异常值进行了处理)'''


def stemgnn_mape(preds, labels, axis=None):
    '''
    Mean absolute percentage error.
    :param labels: np.ndarray or int, ground truth.
    :param preds: np.ndarray or int, prediction.
    :param axis: axis to do calculation.
    :return: int, MAPE averages on all elements of input.
    '''
    if not isinstance(preds, np.ndarray):
        preds = preds.cpu().numpy()
        labels = labels.cpu().numpy()
    mape = (np.abs(preds - labels) / (np.abs(labels) + 1e-5)).astype(np.float64)
    mape = np.where(mape > 5, 5, mape)  # 将计算得到的 MAPE 中大于 5 的值替换为 5，用于限制异常值对评估指标的影响。
    return np.mean(mape, axis)


'''
计算带有遮罩值的均方根误差
遮罩值通常用于表示数据中的缺失值或者无效值
'''


def masked_rmse_np(preds, labels, null_val=np.nan):
    return np.sqrt(masked_mse_np(preds=preds, labels=labels, null_val=null_val))


def masked_mse_np(preds, labels, null_val=np.nan):
    with np.errstate(divide='ignore', invalid='ignore'):
        if np.isnan(null_val):
            mask = ~np.isnan(labels)
        else:
            mask = np.not_equal(labels, null_val)
        mask = mask.astype('float32')
        mask /= np.mean(mask)
        mse = np.square(np.subtract(preds, labels)).astype('float32')
        mse = np.nan_to_num(mse * mask)
        return np.mean(mse)


def masked_mae_np(preds, labels, null_val=np.nan):
    with np.errstate(divide='ignore', invalid='ignore'):
        if np.isnan(null_val):
            mask = ~np.isnan(labels)
        else:
            mask = np.not_equal(labels, null_val)
        mask = mask.astype('float32')
        mask /= np.mean(mask)
        mae = np.abs(np.subtract(preds, labels)).astype('float32')
        mae = np.nan_to_num(mae * mask)
        return np.mean(mae)


# def masked_mape_np(preds, labels, null_val=np.nan):
#     if not isinstance(preds, np.ndarray):
#         preds = preds.cpu().numpy()
#         labels = labels.cpu().numpy()
#     with np.errstate(divide='ignore', invalid='ignore'):
#         if np.isnan(null_val):
#             mask = ~np.isnan(labels)
#         else:
#             # mask = np.not_equal(labels, null_val)
#             mask = np.where(labels > null_val, True, False)
#         # mask = mask.astype('float32')
#         # mask /= np.mean(mask)
#         # mape = np.abs(np.divide(np.subtract(preds, labels).astype('float32'), labels))
#         # mape = np.nan_to_num(mask * mape)
#         # return np.mean(mape)
#         preds = preds[mask]
#         labels = labels[mask]
#         return np.mean(np.absolute(np.divide((labels - preds), labels)))

def masked_mape_np(preds, labels, null_val=np.nan):
    # 强制将可能漏网的 PyTorch tensor 转换为 numpy
    if torch.is_tensor(preds):
        preds = preds.detach().cpu().numpy()
    if torch.is_tensor(labels):
        labels = labels.detach().cpu().numpy()

    with np.errstate(divide='ignore', invalid='ignore'):
        if np.isnan(null_val):
            mask = ~np.isnan(labels)
        else:
            mask = np.where(labels > null_val, True, False)
        preds = preds[mask]
        labels = labels[mask]
        return np.mean(np.absolute(np.divide((labels - preds), labels)))

class Evaluator(object):
    '''
    初始化、评估方法_evaluate和主评估入口evaluate
    '''
    def __init__(self, config):
        self.config = config
        self.mask = self.config.get("mask", False)  # False
        self.out_catagory = "multi" # 输出类别

    def _evaluate(self, output: np.ndarray, groud_truth: np.ndarray, mask: int, out_catagory: str):
        """
        评估模型性能
        : multi 多维度
        :param output: [n_samples, 12, n_nodes, n_features]      预测值
        :param groud_truth: [n_samples, 12, n_nodes, n_features] 真实值
        : single 单维度
        """
        # print('进入evaluator.py中的_evaluate方法：')
        if out_catagory == 'multi':  # 输出类别
            if bool(mask):
                '''
                启用mask，计算指标时候会忽略数据中的空值
                先确认预测值和真实值维度是否一致，如果不一致时将groud_truth的最后一个维度扩展进行匹配output
                '''
                print("--------------启用了mask！！！！------------------")
                if output.shape != groud_truth.shape:
                    groud_truth = np.expand_dims(groud_truth[..., 0], axis=-1)
                assert output.shape == groud_truth.shape, f'{output.shape}, {groud_truth.shape}' # 这行代码主要就是起到一个检查的作用
                # batch, steps, scores, node = output.shape[0], output.shape[1], defaultdict(dict), output.shape[2]
                '''
                scores: 字典 用于存储各个评估指标的结果
                batch: 样本数量 output.shape[0]=n_samples
                steps: 时间步数 output.shape[1]=12
                node: 节点数量 output.shape[2]=n_nodes
                '''
                scores, batch, steps, node = defaultdict(dict), output.shape[0], output.shape[1], output.shape[2]
                for step in range(steps): # 遍历所有时间步
                    '''
                    代码解析：
                    output形状:[n_samples, 12, n_nodes, n_features]
                    output[:, step]:切片 提取第step个时间步的数据 形状变为[n_samples, n_nodes, n_features]
                    (batch, -1): batch=output.shape[0]为样本数量，-1表示自动计算这一维的大小，使得重塑之后的数组元素个数与原始保持一致
                    '''
                    y_pred = np.reshape(output[:, step], (batch, -1))  # y_pred [n_samples, n_nodes * n_features]
                    y_true = np.reshape(groud_truth[:, step], (batch, -1))  #y_true [n_samples, n_nodes * n_features]

                    '''单个时间步'''
                    scores['masked_MAE'][f'horizon-{step}'] = masked_mae_np(y_pred, y_true, null_val=0.0)
                    scores['masked_RMSE'][f'horizon-{step}'] = masked_rmse_np(y_pred, y_true, null_val=0.0)
                    scores['masked_MAPE'][f'horizon-{step}'] = masked_mape_np(y_pred, y_true, null_val=0.0) * 100.0
                    # scores['node_wise_PCC'][f'horizon-{step}'] = node_pcc_np(y_pred.swapaxes(1, -1).reshape((-1, node)),y_true.swapaxes(1, -1).reshape((-1, node)))
                    scores['PCC'][f'horizon-{step}'] = pcc_np(y_pred, y_true)
                '''所有时间步'''
                scores['masked_MAE']['all'] = masked_mae_np(output, groud_truth, null_val=0.0)
                scores['masked_RMSE']['all'] = masked_rmse_np(output, groud_truth, null_val=0.0)
                scores['masked_MAPE']['all'] = masked_mape_np(output, groud_truth, null_val=0.0) * 100.0
                scores['PCC']['all'] = pcc_np(output, groud_truth)
                # scores["node_pcc"]['all'] = node_pcc_np(output, groud_truth)
            else:
                '''
                未启用mask，计算指标时候不会忽略数据中的空值
                '''
                print("--------------没有启用mask！！！！------------------")
                if output.shape != groud_truth.shape:
                    groud_truth = np.expand_dims(groud_truth[..., 0], axis=-1)
                assert output.shape == groud_truth.shape, f'{output.shape}, {groud_truth.shape}'
                batch, steps, scores, node = output.shape[0], output.shape[1], defaultdict(dict), output.shape[2]
                for step in range(steps):
                    y_pred = output[:, step]
                    y_true = groud_truth[:, step]
                    scores['MAE'][f'horizon-{step}'] = mae_np(y_pred, y_true)
                    scores['RMSE'][f'horizon-{step}'] = rmse_np(y_pred, y_true)
                    # # scores['MAPE'][f'horizon-{step}'] = mape_np(y_pred,y_true) * 100.0
                    # scores['masked_MAE'][f'horizon-{step}'] = masked_mae_np(y_pred, y_true)
                    # scores['masked_RMSE'][f'horizon-{step}'] = masked_rmse_np(y_pred, y_true)
                    # scores['MAPE'][f'horizon-{step}'] = mape_np(y_pred, y_true, null_val=0.01) * 100.0
                    scores['masked_MAPE'][f'horizon-{step}'] = masked_mape_np(y_pred, y_true, null_val=0.01) * 100.0
                    # scores['StemGNN_MAPE'][f'horizon-{step}'] = stemgnn_mape(y_pred, y_true) * 100.0
                    scores['PCC'][f'horizon-{step}'] = pcc_np(y_pred, y_true)
                    # scores['node_wise_PCC'][f'horizon-{step}'] = node_pcc_np(y_pred.swapaxes(1, -1).reshape((-1, node)),y_true.swapaxes(1, -1).reshape((-1, node)))
                scores['MAE']['all'] = mae_np(output, groud_truth)
                scores['RMSE']['all'] = rmse_np(output, groud_truth)
                # scores['MAPE']['all'] = mape_np(output, groud_truth,null_val=0.01) * 100.0
                # # scores['MAPE']['all'] = mape_np(output, groud_truth) * 100.0
                # scores['masked_MAE']['all'] = masked_mae_np(y_pred, y_true)
                # scores['masked_RMSE']['all'] = masked_rmse_np(y_pred, y_true)
                # scores['MAPE']['all'] = masked_mape_np(output, groud_truth, null_val=0.01) * 100.0
                scores['masked_MAPE']['all'] = masked_mape_np(output, groud_truth, null_val=0.01) * 100.0
                # scores['StemGNN_MAPE']['all'] = stemgnn_mape(output, groud_truth) * 100.0
                scores['PCC']['all'] = pcc_np(output, groud_truth)
                # scores['node_wise_PCC']['all'] = node_pcc_np(output.swapaxes(2, -1).reshape((-1, node)),
                #                                              groud_truth.swapaxes(2, -1).reshape((-1, node)))

        else:  # 如果输出是single
            output = output.squeeze()  # 预测值去掉多余的维度
            groud_truth = groud_truth.squeeze() # 真实值去掉多余的维度
            assert output.shape == groud_truth.shape, f'{output.shape}, {groud_truth.shape}'
            scores = defaultdict(dict)

            scores['RMSE']['all'] = rmse_np(output, groud_truth)
            scores['masked_MAPE']['all'] = masked_mape_np(output, groud_truth, null_val=0.0) * 100.0
            scores['PCC']['all'] = node_pcc_np(output, groud_truth)
            scores['rse']['all'] = rse_np(output, groud_truth)
            scores['rae']['all'] = rae_np(output, groud_truth)
            scores['MAPE']['all'] = stemgnn_mape(output, groud_truth) * 100.0
            scores['MAE']['all'] = mae_np(output, groud_truth)
            scores["node_pcc"]['all'] = node_pcc_np(output, groud_truth)
            scores['CORR']['all'] = corr_np(output, groud_truth)
        return scores

    def evaluate(self, output, groud_truth):  # 确保输入的张量为numpy数组，再调用_evaluate方法进行评估
        # 增加 detach()，确保即使有梯度也能安全转换
        if torch.is_tensor(output):
            output = output.detach().cpu().numpy()
        if torch.is_tensor(groud_truth):
            groud_truth = groud_truth.detach().cpu().numpy()
        # if not isinstance(output, np.ndarray):
        #     output = output.cpu().numpy()
        # if not isinstance(groud_truth, np.ndarray):
        #     groud_truth = groud_truth.cpu().numpy()
        scores=self._evaluate(output, groud_truth, self.mask, self.out_catagory)
        self.plot_metrics(scores)
        # return self._evaluate(output, groud_truth, self.mask, self.out_catagory)
        return scores

    def plot_metrics(self,scores):
        # metrics=['MAE','RMSE','masked_MAPE', 'StemGNN_MAPE', 'PCC', 'node_wise_PCC']
        metrics_three=['masked_MAE','masked_RMSE','masked_MAPE']
        # steps = [f'horizon-{i}' for i in range(len(scores['MAE']) - 1)]
        steps = [f'horizon-{i}' for i in range(len(scores['masked_MAE']) - 1)]

        plt.figure(figsize=(15, 5.3))
        for i, metric in enumerate(metrics_three):
            plt.subplot(1, 3, i + 1)
            values = [scores[metric][step] for step in steps]
            plt.plot(steps, values, marker='o')
            plt.title(f'{metric} over time steps')
            plt.xlabel('Time steps')
            plt.ylabel(metric)
            plt.xticks(rotation=45)

        plt.tight_layout()

        # 确保保存图像的目录存在
        save_dir = 'png'
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, 'every_loss_0830_three_mask.png')

        # 保存图像
        plt.savefig(save_path)
        plt.close()  # 关闭图形以释放内存