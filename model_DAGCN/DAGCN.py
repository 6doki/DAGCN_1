# from _typeshed import Self
import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh


def compute_laplacian_pe(adj_mx, max_freqs=10):
    """
    计算拉普拉斯位置编码 (Laplacian PE)
    :param adj_mx: 邻接矩阵 (N, N)
    :param max_freqs: 保留前k个特征向量 (用于降维)
    :return: torch.Tensor (N, max_freqs)
    """
    # 1. 确保是 numpy 矩阵
    if isinstance(adj_mx, torch.Tensor):
        adj_mx = adj_mx.cpu().numpy()

    N = adj_mx.shape[0]

    # 2. 计算归一化拉普拉斯矩阵 L = I - D^-1/2 * A * D^-1/2
    # 添加自环
    adj_mx = adj_mx + np.eye(N)
    # 度矩阵
    d = np.sum(adj_mx, axis=1)
    d_inv_sqrt = np.power(d, -0.5)
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = np.diag(d_inv_sqrt)

    # 对称归一化邻接矩阵
    sym_adj = adj_mx.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt)

    # 拉普拉斯矩阵
    L = np.eye(N) - sym_adj

    # 3. 特征分解 (Eigendecomposition)
    # 使用 scipy 计算最小的 k 个特征值和特征向量，这些特征向量不仅包含了图的频率信息，还天然地构成了节点在图谱空间中的坐标
    # k = max_freqs
    try:
        # eigsh 用于稀疏/实对称矩阵，'SM' 表示 Smallest Magnitude (最小特征值)
        eigenvals, eigenvecs = eigsh(L, k=max_freqs + 1, which='SM')
        # 第一个特征向量通常是常数向量 (对应特征值0)，一般丢弃或保留均可
        # 这里取后 max_freqs 个
        pe = eigenvecs[:, 1:]
        print("特征向量生成！")
    except:
        # 如果特征分解失败 (极少情况)，用随机初始化兜底
        print("Warning: Laplacian decomposition failed, using random PE.")
        pe = np.random.randn(N, max_freqs)

    return torch.FloatTensor(pe)


class AdvancedDataEmbedding(nn.Module):
    """
    创新点三：增强型数据嵌入
    包含：Input Projection + Laplacian PE + Learnable Temporal/Spatial Embedding
    """

    def __init__(self, input_dim, embed_dim, adj_mx, num_nodes, pe_dim=16,dropout=0.1):
        super(AdvancedDataEmbedding, self).__init__()

        # 1. 原始输入投影 (1 -> 64)
        self.input_proj = nn.Linear(input_dim, embed_dim)

        # 2. 拉普拉斯位置编码 (Spatial Structure)
        # 预先计算好 PE
        self.pe_dim = pe_dim  # 取前8个特征向量 超参数 后续调整
        pe = compute_laplacian_pe(adj_mx, max_freqs=self.pe_dim)
        # 注册为 buffer，不参与梯度更新，但随模型移动设备
        self.register_buffer('laplacian_pe', pe)
        # 将 PE 映射到 embed_dim
        self.pe_proj = nn.Linear(self.pe_dim, embed_dim)

        # 3. 可学习的时间/空间嵌入 (辅助)
        # 既然没有具体的 time_of_day 数据，我们用可学习的参数来增强
        self.node_emb = nn.Parameter(torch.randn(num_nodes, embed_dim))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        x: (B, T, N, Cin)
        """
        # A. 基础特征映射 val_emb
        val_emb = self.input_proj(x)  # (B, T, N, D)

        # B. 注入拉普拉斯空间信息 pe_feat
        # laplacian_pe: (N, 8) -> (N, D)
        pe_feat = self.pe_proj(self.laplacian_pe)
        # 扩展维度以相加: (1, 1, N, D)
        pe_feat = pe_feat.unsqueeze(0).unsqueeze(0)

        # C. 注入可学习节点嵌入 node_feat
        node_feat = self.node_emb.unsqueeze(0).unsqueeze(0)  # (1, 1, N, D)

        # D. 融合
        # 原始数值 + 结构信息 + 节点身份
        out = val_emb + pe_feat + node_feat

        return self.dropout(out)


class GlobalTemporalTransformer(nn.Module):
    """
    创新点二：全局时间 Transformer
    用于捕捉长距离的时间依赖，弥补 CMGCN 滑动窗口的"近视"问题。
    """

    def __init__(self, in_dim, num_heads=2, dropout=0.1):
        super(GlobalTemporalTransformer, self).__init__()
        self.num_heads = num_heads
        self.in_dim = in_dim

        # 多头自注意力层 (Temporal Self-Attention)
        # batch_first=True 表示输入格式为 (Batch, Seq_Len, Feature)
        self.attention = nn.MultiheadAttention(embed_dim=in_dim, num_heads=num_heads, dropout=dropout, batch_first=True)

        # 前馈网络 (Feed Forward Network)
        self.ffn = nn.Sequential(
            nn.Linear(in_dim, in_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(in_dim * 2, in_dim)
        )

        # 层归一化 (LayerNorm)
        self.norm1 = nn.LayerNorm(in_dim)
        self.norm2 = nn.LayerNorm(in_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        :param x: (Batch, N, T, C) 注意这里的 T 是全局时间步
        """
        B, N, T, C = x.shape

        # 1. 维度变换：合并 Batch 和 Node，把 Time 独立出来做 Attention
        # (B, N, T, C) -> (B*N, T, C)
        x_reshaped = x.reshape(B * N, T, C)

        # 2. Self-Attention 计算
        # attn_output: (B*N, T, C)
        attn_output, _ = self.attention(x_reshaped, x_reshaped, x_reshaped)

        # 3. 残差连接 + Norm
        x_res = self.norm1(x_reshaped + self.dropout(attn_output))

        # 4. FFN + 残差连接 + Norm
        ffn_output = self.ffn(x_res)
        x_out = self.norm2(x_res + self.dropout(ffn_output))

        # 5. 还原维度
        # (B*N, T, C) -> (B, N, T, C)
        x_out = x_out.reshape(B, N, T, C)

        return x_out

class DelayAware_gcn_operation(nn.Module):
    def __init__(self, adj, in_dim, out_dim, num_vertices, patterns, activation='GLU'):
        """
        图卷积模块
        :param adj: 邻接图
        :param in_dim: 输入维度
        :param out_dim: 输出维度
        :param num_vertices: 节点数量
        :param activation: 激活方式 {'relu', 'GLU'}
        """
        super(DelayAware_gcn_operation, self).__init__()
        self.adj = adj          # (4N,4N) 4倍节点数
        self.in_dim = in_dim    # 64
        self.out_dim = out_dim  # 64
        self.num_vertices = num_vertices    # N
        self.activation = activation    # 'GLU'

        # 新增部分
        self.patterns = nn.Parameter(patterns, requires_grad=False)
        self.num_patterns = patterns.shape[0]
        self.pattern_len = patterns.shape[1]

        # 模式特征提取层：将匹配到的Pattern映射回输入特征维度
        # self.pattern_conv = nn.Linear(self.pattern_len, in_dim)
        self.pattern_embed = nn.Linear(self.pattern_len, in_dim)

        # 融合门控机制：决定保留多少原特征，注入多少模式特征
        self.fusion_gate = nn.Linear(in_dim * 2, out_dim)
        # 初始化偏置，使 sigmoid(fusion_gate) 初始值很小，优先保留原始特征
        nn.init.constant_(self.fusion_gate.bias, -2.0)

        assert self.activation in {'GLU', 'relu'}
        if self.activation == 'GLU':
            self.FC = nn.Linear(self.in_dim, 2 * self.out_dim, bias=True)   # (in=64, out= 64*2)
        else:
            self.FC = nn.Linear(self.in_dim, self.out_dim, bias=True)

    # def forward(self, x, pattern_weight, mask=None):
    #
    #
    #     # 模式匹配
    #     x_tmp = x.permute(1, 0, 2)  # 调整维度以便计算: (N, B, C) -> (B, N, C)
    #
    #     # 将输入特征投影为 Query: (B, N, Pattern_Len)
    #     query = self.query_proj(x_tmp)
    #
    #     # 计算与所有 Patterns 的相似度 (点积)
    #     # patterns: (K, Pattern_Len) -> 转置 (Pattern_Len, K)
    #     # score: (B, N, K) 表示每个节点当前时刻与 K 个模式的匹配程度
    #     score = torch.matmul(query, self.patterns.t())
    #     attn = torch.softmax(score, dim=-1)  # 归一化权重
    #
    #     # 加权组合最匹配的 Patterns
    #     # (B, N, K) * (K, Pattern_Len) -> (B, N, Pattern_Len)
    #     matched_pattern_feat = torch.matmul(attn, self.patterns)
    #
    #     # 映射回特征维度: (B, N, In_Dim)
    #     delay_feat = self.pattern_conv(matched_pattern_feat)
    #
    #     # --- 过程 2: 特征融合 ---
    #     # 拼接原特征和延迟特征
    #     combined = torch.cat([x_tmp, delay_feat], dim=-1)
    #     # 计算门控值 (0~1)
    #     z = torch.sigmoid(self.fusion_gate(combined))
    #     # 融合：原特征 * z + 延迟特征 * (1-z)
    #     x_enhanced = x_tmp * z + delay_feat * (1 - z)
    #
    #     # 转回原维度 (N, B, C) 以便进行后续的 GCN 操作
    #     x = x_enhanced.permute(1, 0, 2)
    #
    #     adj = self.adj
    #     '''如果提供了mask，将邻接矩阵adj移动到与mask相同的设备上，并乘以mask，将无效节点对应的邻接关系置0，从而忽略掉那些不应考虑的边'''
    #     if mask is not None:
    #         adj = adj.to(mask.device) * mask
    #
    #     x = torch.einsum('nm, mbc->nbc', adj.to(x.device), x)  # 4*N, B, Cin  对邻接矩阵adj(维度nm)和特征矩阵x(维度mbc)进行乘法操作
    #
    #     if self.activation == 'GLU':
    #         lhs_rhs = self.FC(x)  # 全连接层输出：(4*N, B, 2*Cout)
    #         lhs, rhs = torch.split(lhs_rhs, self.out_dim, dim=-1)  # 拆分之后的lhs和rhs:(4*N, B, Cout)
    #         out = lhs * torch.sigmoid(rhs)
    #         del lhs, rhs, lhs_rhs
    #         return out
    #     elif self.activation == 'relu':
    #         return torch.relu(self.FC(x))  # 3*N, B, Cout

    def forward(self, x, pattern_weights, mask=None):
        """
        :param x: (4*N, B, Cin)
        :param pattern_weights: (B, N, K) 从最顶层传下来的权重
        """
        # 1. 准备 Pattern 特征
        # patterns: (K, L) -> pattern_embed -> (K, Cin)
        # 这一步将固定的 Pattern 投影到当前的特征空间
        projected_patterns = self.pattern_embed(self.patterns)

        # 2. 根据权重聚合特征
        # pattern_weights: (B, N, K)
        # projected_patterns: (K, Cin)
        # matmul -> (B, N, Cin)

        # 3. 维度对齐
        total_gcn_nodes = x.shape[0]  # 获取当前的节点总数 (例如 1228)
        original_nodes = pattern_weights.shape[1]  # 获取原始节点数 (例如 307)

        # 自动计算倍数 (1228 // 307 = 4)
        if original_nodes > 0:
            repeat_factor = total_gcn_nodes // original_nodes
        else:
            repeat_factor = 1  # 防止除零，虽然理论上不会发生
        delay_feat = torch.matmul(pattern_weights, projected_patterns)
        # 动态复制
        delay_feat = delay_feat.repeat(1, repeat_factor, 1)

        # 调整为 (4N, B, Cin) 以匹配 x
        delay_feat = delay_feat.permute(1, 0, 2)

        # 4. 融合
        # 拼接
        combined = torch.cat([x, delay_feat], dim=-1)
        # 计算门控 (B, 4N, Cin)
        z = torch.sigmoid(self.fusion_gate(combined))

        # 残差融合：原特征 + 门控 * 延迟特征
        # 这样如果 z 接近 0，就退化为原始 GCN，保证下限
        x_enhanced = x + z * delay_feat

        # 5. GCN 操作 (不变)
        adj = self.adj
        if mask is not None:
            adj = adj.to(mask.device) * mask
        x_out = torch.einsum('nm, mbc->nbc', adj.to(x_enhanced.device), x_enhanced)

        if self.activation == 'GLU':
            lhs_rhs = self.FC(x_out)
            lhs, rhs = torch.split(lhs_rhs, self.out_dim, dim=-1)
            return lhs * torch.sigmoid(rhs)
        elif self.activation == 'relu':
            return torch.relu(self.FC(x_out))


class STSGCM(nn.Module):
    def __init__(self, adj, in_dim, out_dims, num_of_vertices, patterns, activation='GLU'):
        """
        :param adj: 邻接矩阵
        :param in_dim: 输入维度
        :param out_dims: list 各个图卷积的输出维度
        :param num_of_vertices: 节点数量
        :param activation: 激活方式 {'relu', 'GLU'}
        """
        super(STSGCM, self).__init__()
        self.adj = adj  # (4N,4N)
        self.in_dim = in_dim    # 64
        self.out_dims = out_dims    # out_dims：[64,64,64]
        self.num_of_vertices = num_of_vertices  # N
        self.activation = activation
        self.patterns = patterns
        self.gcn_operations = nn.ModuleList()  # 先初始化一个模块列表，用于存储多个gcn_operation

        '''第一个gcn_operation'''
        self.gcn_operations.append(   # 第0个
            DelayAware_gcn_operation(
                adj=self.adj,
                in_dim=self.in_dim,
                out_dim=self.out_dims[0],  # out_dims[0]：64
                num_vertices=self.num_of_vertices,
                activation=self.activation,
                patterns=self.patterns
            )
        )
        '''剩下的gcn_operations'''
        for i in range(1, len(self.out_dims)):  # 第1-2个
            self.gcn_operations.append(
                DelayAware_gcn_operation(
                    adj=self.adj,
                    in_dim=self.out_dims[i-1],  # 输入：前一个图卷积操作的输出维度
                    out_dim=self.out_dims[i],   # 输出：64
                    num_vertices=self.num_of_vertices,
                    activation=self.activation,
                    patterns=self.patterns
                )
            )

    def forward(self, x, pattern_weights, mask=None):
        """
        :param x: (3N, B, Cin)
        :param mask: (3N, 3N)
        :return: (N, B, Cout)
        """
        need_concat = []  # 空列表，用于存储每个gcn_operation的输出

        for i in range(len(self.out_dims)):
            x = self.gcn_operations[i](x, pattern_weights, mask)     # 4N, B, Cin
            need_concat.append(x)

        # shape of each element is (1, N, B, Cout)
        '''计算每个gcn_operation的输出，取出第num_of_vertices到2*num_of_vertices个节点的输出，并在第一个维度上增加一个维度，结果维度(1,N,B,Cout)'''
        need_concat = [
            torch.unsqueeze(
                h[self.num_of_vertices: 2 * self.num_of_vertices], dim=0
            ) for h in need_concat
        ]

        '''将上面的输出在第一个维度上拼接'''
        out = torch.max(torch.cat(need_concat, dim=0), dim=0).values  # (N, B, Cout)
        del need_concat
        return out


class STSGCL(nn.Module):
    def __init__(self,
                 adj,
                 history,
                 num_of_vertices,
                 in_dim,
                 out_dims,
                 patterns,
                 strides=4,
                 activation='GLU',
                 # temporal_emb=False,
                 temporal_emb=True,
                 # spatial_emb=False,
                 spatial_emb=True
                 ):
        """
        :param adj: 邻接矩阵
        :param history: 输入时间步长
        :param in_dim: 输入维度
        :param out_dims: list 各个图卷积的输出维度
        :param strides: 滑动窗口步长，local时空图使用几个时间步构建的，默认为4
        :param num_of_vertices: 节点数量
        :param activation: 激活方式 {'relu', 'GLU'}
        :param temporal_emb: 加入时间位置嵌入向量
        :param spatial_emb: 加入空间位置嵌入向量
        """
        super(STSGCL, self).__init__()
        print("-------------------进入STSGCL---------------------")
        self.patterns = patterns
        self.adj = adj  # (4N,4N)
        self.strides = strides  # 4
        self.history = history  # 12
        self.in_dim = in_dim    # 64
        self.out_dims = out_dims    # [64,64,64]
        self.num_of_vertices = num_of_vertices  # N
        self.activation = activation    # 'GLU'
        self.temporal_emb = temporal_emb    # True
        self.spatial_emb = spatial_emb  # True
        self.conv1 = nn.Conv2d(self.in_dim, self.out_dims[-1], kernel_size=(1, 2), stride=(1, 1), dilation=(1, 3))  # Conv1d(64, 64, kernel_size=(1, 2), stride=(1, 1), dilation=(1, 3))
        self.conv2 = nn.Conv2d(self.in_dim, self.out_dims[-1], kernel_size=(1, 2), stride=(1, 1), dilation=(1, 3))  # Conv1d(64, 64, kernel_size=(1, 2), stride=(1, 1), dilation=(1, 3))
        # self.conv1 = nn.Conv2d(self.in_dim, self.out_dims[-1], kernel_size=(1, 2), stride=(1, 1), dilation=(1, 1))
        # self.conv2 = nn.Conv2d(self.in_dim, self.out_dims[-1], kernel_size=(1, 2), stride=(1, 1), dilation=(1, 1))


        "--- [创新点二新增] 全局 Transformer 分支 ---"
        # 它的输入维度是 in_dim，输出维度也是 in_dim (为了方便残差融合)
        self.global_transformer = GlobalTemporalTransformer(in_dim=in_dim, num_heads=2)

        # 降维/对齐层：
        # 因为 STSGCM (局部GCN) 会改变通道数 (例如 64->64)，
        # 如果 GCN 的输出维度 out_dims[-1] 和输入 in_dim 不一样，我们需要一个线性层来对齐
        if in_dim != out_dims[-1]:
            self.align_proj = nn.Linear(in_dim, out_dims[-1])
        else:
            self.align_proj = None

        self.STSGCMS = nn.ModuleList()
        for i in range(self.history - self.strides + 1):    # (0,...,8)
            self.STSGCMS.append(
                STSGCM(
                    adj=self.adj,
                    in_dim=self.in_dim,
                    out_dims=self.out_dims,
                    patterns = self.patterns,
                    num_of_vertices=self.num_of_vertices,
                    activation=self.activation
                )
            )


        if self.temporal_emb:
            self.temporal_embedding = nn.Parameter(torch.FloatTensor(1, self.history, 1, self.in_dim))  # (1,12,1,64)
            # 1, T=12, 1, Cin=64

        if self.spatial_emb:
            self.spatial_embedding = nn.Parameter(torch.FloatTensor(1, 1, self.num_of_vertices, self.in_dim))   # (1,1,N,64)
            # 1, 1, N, Cin=64

        self.reset()  # 初始化嵌入向量的权重

    def reset(self):
        print("--------------进入STSGCL中的reset()-----------------")
        if self.temporal_emb:
            nn.init.xavier_normal_(self.temporal_embedding, gain=0.0003)

        if self.spatial_emb:
            nn.init.xavier_normal_(self.spatial_embedding, gain=0.0003)

    def forward(self, x, pattern_weights, mask=None):
        """
        x: (B, T, N, C)
        """
        # ... (原有代码: Embedding 叠加，保持不变) ...
        if self.temporal_emb:
            x = x + self.temporal_embedding
        if self.spatial_emb:
            x = x + self.spatial_embedding

        # ==================== 分支 1: 局部滑动窗口 GCN (原 CMGCN 逻辑) ====================
        need_concat = []
        batch_size = x.shape[0]

        # 这是一个滑动窗口循环，比如 T=12, strides=4, 循环会执行 T-4+1 次
        for i in range(self.history - self.strides + 1):
            t = x[:, i: i + self.strides, :, :]  # 切片: 取局部 4 个时间步 t形状(B,4,N,64)

            # 变形为 (B, 4N, 64) 送入 GCN
            t = torch.reshape(t, shape=[batch_size, self.strides * self.num_of_vertices, self.in_dim])

            # GCN 运算:变形为(4N,B,64)
            t = self.STSGCMS[i](t.permute(1, 0, 2), pattern_weights, mask)

            # 还原形状 (N, B, Cout) -> (B, N, Cout)
            t = t.permute(1, 0, 2)
            t = torch.unsqueeze(t, dim=1)  # (B, 1, N, Cout)
            need_concat.append(t)

        # local_out 形状: (B, T_new, N, Cout)
        # T_new = history - strides + 1
        local_out = torch.cat(need_concat, dim=1)

        # ==================== 分支 2: [创新点二] 全局 Transformer ====================
        # 输入 x: (B, T_old, N, Cin)

        # 1. 调整维度适应 Transformer: (B, N, T, C)
        x_trans_in = x.permute(0, 2, 1, 3)

        # 2. 全局 Attention 计算
        # global_out: (B, N, T_old, Cin)
        global_out = self.global_transformer(x_trans_in)

        # 3. 时间维度对齐 (关键步骤)
        # GCN 把时间 T 缩短了 (例如 12 -> 9)，Transformer 输出了 12。
        # 我们只取 Transformer 输出的“后半部分”来和 GCN 对齐
        # 或者取对应的切片。这里我们取与 GCN 对应的部分。
        # local_out 的长度是 self.history - self.strides + 1
        target_len = local_out.shape[1]

        # 简单策略：取最后 target_len 个时间步 (假设我们要预测未来，最近的信息最重要)
        # (B, N, T_target, Cin)
        global_out = global_out[:, :, -target_len:, :]

        # 4. 还原维度: (B, T_target, N, Cin)
        global_out = global_out.permute(0, 2, 1, 3)

        # 5. 通道对齐
        if self.align_proj is not None:
            global_out = self.align_proj(global_out)

        # ==================== 双流融合 ====================
        # Local (GCN) + Global (Transformer)
        # 这是一个类似 ResNet 的加法融合
        final_out = local_out + global_out

        return final_out

    # def forward(self, x, pattern_weights, mask=None):  # 原有方案
    #     """
    #     :param x: B, T, N, Cin    (x数据：表示有B个批次，每个批次有T个时间步，每个时间步有N个节点，输入特征维度是Cin)
    #     :param mask: (N, N)
    #     :return: B, T-3, N, Cout  (输出out数据:表示有B个批次，每个批次有T-3个时间步，每个时间步有N个节点，输出特征维度是Cout)
    #     """
    #     # 消融
    #     if self.temporal_emb:   # x: B=64, T=12, N, Cin=64      temporal_embedding:1, history=12, 1, Cin=64
    #         x = x + self.temporal_embedding     # x: B=64, T=12, N, Cin=64
    #
    #     if self.spatial_emb:    # 1, 1, N, Cin=64   spatial_embedding: 1, 1, N , Cin=64
    #         x = x + self.spatial_embedding      # x: B=64, T=12, N, Cin=64
    #
    #
    #     '''
    #     下面代码对应论文中：两个二维的扩张卷积神经网络（捕获全局的）
    #     '''
    #     data_temp = x.permute(0, 3, 2, 1)                 # 交换位置x(B=64, Cin=64, N, T=12)
    #     data_left = torch.sigmoid(self.conv1(data_temp))  # (64, 64, 358, 9)
    #     data_right = torch.tanh(self.conv2(data_temp))    # (64, 64, 358, 9)
    #     data_time_axis = data_left * data_right           # (64, 64, 358, 9)
    #     data_res = data_time_axis.permute(0, 3, 2, 1)     # (64,9,358,64) 再次交换位置:(B,T-3,N,Cin)
    #     # shape is (B, T-3, N, C)
    #     #############################################
    #
    #     need_concat = []
    #     batch_size = x.shape[0]  # 64
    #
    #     for i in range(self.history - self.strides + 1):
    #         t = x[:, i: i+self.strides, :, :]  # 从x中提取一个滑动窗口:t(B, self.stride, N, Cin),这里stride=4
    #         t = torch.reshape(t, shape=[batch_size, self.strides * self.num_of_vertices, self.in_dim])   #将t重塑为(B, self.stride*N, Cin)
    #         # (B, 4*N, Cin)
    #         t = self.STSGCMS[i](t.permute(1, 0, 2), pattern_weights, mask)  # (4*N, B, Cin) -> (N, B, Cout)
    #         t = torch.unsqueeze(t.permute(1, 0, 2), dim=1)  # (N, B, Cout) -> (B, N, Cout) ->(B, 1, N, Cout)
    #         need_concat.append(t)
    #
    #     mid_out = torch.cat(need_concat, dim=1)  # (B, T-3, N, Cout)
    #     out = mid_out + data_res
    #     del need_concat, batch_size
    #     return out


class output_layer(nn.Module):
    def __init__(self, num_of_vertices, history, in_dim, out_dim, hidden_dim=128, horizon=24):
        """
        预测层，注意在作者的实验中是对每一个预测时间step做处理的，也即他会令horizon=1
        :param num_of_vertices:节点数量
        :param history:输入时间步长
        :param in_dim: 输入维度
        :param hidden_dim:中间层维度
        :param horizon:输出(预测)时间步长
        """
        super(output_layer, self).__init__()
        self.num_of_vertices = num_of_vertices
        self.history = history  # 3
        self.in_dim = in_dim    # 64
        self.out_dim = out_dim  #1
        self.hidden_dim = hidden_dim    # 128
        self.horizon = horizon  # 1

        '''
        FC1:第一个全连接层，将输入维度转换为隐藏维度
        FC2:第二个全连接层，将隐藏维度转换为输出维度
        '''
        self.FC1 = nn.Linear(self.in_dim * self.history, self.hidden_dim, bias=True)    # Linear(in_features=192, out_features=128, bias=True)
        #self.FC2 = nn.Linear(self.hidden_dim, self.horizon , bias=True)
        self.FC2 = nn.Linear(self.hidden_dim, self.horizon * self.out_dim, bias=True)   # 128, 1

    def forward(self, x):
        """
        :param x: (B, Tin, N, Cin)：（批次大小，输入时间步长，节点数，输入特征维度）
        :return: (B, Tout, N)：（批次大小，预测时间步长，节点数量）
        """
        batch_size = x.shape[0]  # B
        x = x.permute(0, 2, 1, 3)  # (B, Tin, N, Cin)-->(B, N, Tin, Cin)
        out1 = torch.relu(self.FC1(x.reshape(batch_size, self.num_of_vertices, -1)))  # (B, N, Tin, Cin) -> (B, N, Tin * Cin) -> (B, N, hidden128)
        out2 = self.FC2(out1)  # (B, N, hidden) -> (B, N, horizon*1) 这里的1是Cout也就是out_dim
        out2 = out2.reshape(batch_size, self.num_of_vertices, self.horizon, self.out_dim)  # (B, N, horizon, Cout) : (B, N, 1, 1)
        del out1, batch_size
        return out2.permute(0, 2, 1, 3)  # B, horizon, N, Cout
        # return out2.permute(0, 2, 1)  # B, horizon, N


class DAGCN(nn.Module):
    print("-------------------------进入DAGCN类----------------------------")
    def __init__(self, config, data_feature):  # config:模型配置参数  data_feature:数据的特征信息
        """

        :param adj: local时空间矩阵
        :param history: 输入时间步长
        :param num_of_vertices: 节点数量
        :param in_dim:输入维度
        :param hidden_dims: lists, 中间各STSGCL层的卷积操作维度
        :param first_layer_embedding_size: 第一层输入层的维度
        :param out_layer_dim: 输出模块中间层维度
        :param activation: 激活函数 {relu, GlU}
        :param use_mask: 是否使用mask矩阵对adj进行优化
        :param temporal_emb:是否使用时间嵌入向量
        :param spatial_emb:是否使用空间嵌入向量
        :param horizon:预测时间步长
        :param strides:滑动窗口步长，local时空图使用几个时间步构建的，默认为4
        """
        super(DAGCN, self).__init__()
        self.config = config
        self.data_feature = data_feature
        self.scaler = data_feature["scaler"]    # std: Z-score 标准化
        self.num_batches = data_feature["num_batches"]  # 批次数量

        # self在”=“右侧:表示正在访问类实例的属性或者方法，获取其值或调用其功能
        adj = self.data_feature["adj_mx"]  # 融合的邻接矩阵
        history = self.config.get("window", 12)     # 12 历史时间步长 决定模型每次输入多少历史数据进行预测
        num_of_vertices = self.config.get("num_nodes", None)  # 节点数量
        in_dim = self.config.get("input_dim", 1)    # 1
        out_dim = self.config.get("output_dim", 1)  # 1
        hidden_dims = self.config.get("hidden_dims", None)  # [[64, 64, 64], [64, 64, 64], [64, 64, 64]] 隐藏层
        first_layer_embedding_size = self.config.get("first_layer_embedding_size", None)    # 第一层全连接层的输出维度：64
        out_layer_dim = self.config.get("out_layer_dim", None)  # 128 输出层中间层维度
        activation = self.config.get("activation", "GLU")
        use_mask = self.config.get("mask")  # False
        temporal_emb = self.config.get("temporal_emb", True)    # True 使用时间嵌入
        spatial_emb = self.config.get("spatial_emb", True)  # True 使用空间嵌入
        horizon = self.config.get("horizon", 24)    # 12 输出的预测时间步长
        strides = self.config.get("strides", 4)  # 4 滑动窗口步长 用于控制每次处理的时间步数量
        # pe_dim = self.config.get("pe_dim", 8)
        pe_dim = self.config.get("pe_dim", 16)
        # pattern = self.data_feature["pattern"]

        # 将局部变量存储为实例属性
        self.adj = adj  # (4N，4N）
        self.num_of_vertices = num_of_vertices  # N
        self.hidden_dims = hidden_dims  # [[64, 64, 64], [64, 64, 64], [64, 64, 64]]
        self.out_layer_dim = out_layer_dim  # 128
        self.activation = activation    # "GLU"
        self.use_mask = use_mask    # false

        self.temporal_emb = temporal_emb    # true
        self.spatial_emb = spatial_emb  # true
        self.horizon = horizon  # 12
        self.strides = 4
        # self.strides = 3

        self.input_proj = nn.Linear(1, 12)

        # [新增] 从 data_feature 获取 patterns
        # 注意：要送到 GPU 上 (假设 device 是 self.adj.device 的位置，后续 forward 会自动处理，但最好这里转一下)
        if "patterns" in data_feature:
            # [核心修改] 使用 nn.Parameter 包装，并设置为不可训练
            # 这样 model.cuda() 时，它会自动跟着去 GPU
            self.patterns = nn.Parameter(data_feature["patterns"], requires_grad=False)
        else:
            raise ValueError("Data feature missing 'patterns'!")

        # self.First_FC = nn.Linear(in_dim, first_layer_embedding_size, bias=True)

        # ----------------------------------------
        # [创新点三修改] 替换原来的 self.First_FC
        # 原代码: self.First_FC = nn.Linear(in_dim, first_layer_embedding_size, bias=True)

        if isinstance(self.adj, torch.Tensor):
            spatial_adj = self.adj[:self.num_of_vertices, :self.num_of_vertices].cpu().numpy()
        else:
            spatial_adj = self.adj[:self.num_of_vertices, :self.num_of_vertices]

        self.embedding_layer = AdvancedDataEmbedding(
            input_dim=in_dim,
            embed_dim=first_layer_embedding_size,
            adj_mx=spatial_adj,  # [修改] 传入切片后的 307x307 矩阵
            pe_dim=pe_dim,
            num_nodes=self.num_of_vertices,
            dropout=0.1
        )

        self.STSGCLS = nn.ModuleList()
        self.STSGCLS.append(
            STSGCL(
                adj=self.adj, # 融合的图
                history=history,
                num_of_vertices=self.num_of_vertices,
                in_dim=first_layer_embedding_size,
                out_dims=self.hidden_dims[0], # 64
                strides=self.strides,
                activation=self.activation,
                temporal_emb=self.temporal_emb,
                spatial_emb=self.spatial_emb,
                patterns=self.patterns
            )
        )

        in_dim = self.hidden_dims[0][-1]    # 64 设置为第一层隐藏维度的最后一个值
        history -= (self.strides - 1)   # 12-3=9 更新history
        print("***********************")
        print("经过第一个STSGCLS后的历史时间步长history更新为：", history)

        for idx, hidden_list in enumerate(self.hidden_dims):    # hidden_dims: [[64, 64, 64], [64, 64, 64], [64, 64, 64]]
            print("idx值：", idx)
            if idx == 0:
                continue
            self.STSGCLS.append(    # 2个STSGCL
                STSGCL(
                    adj=self.adj,
                    history=history,    # 9,6
                    num_of_vertices=self.num_of_vertices,
                    in_dim=in_dim,  # 64
                    out_dims=hidden_list,
                    strides=self.strides,
                    activation=self.activation,
                    temporal_emb=self.temporal_emb,
                    spatial_emb=self.spatial_emb,
                    patterns=self.patterns
                )
            )
            history -= (self.strides - 1)   # 9->6, 6->3 更新history
            print("***********************")
            print("经过第%s个STSGCL层后的历史时间步长history更新为%s：" %(idx+1,history))
            in_dim = hidden_list[-1]    # 64

        # predictLayer包含多个output_layer，每个output_layer负责在每个时间步长上生成预测。预测层是最后的输出层，用来将时序数据映射为具体的预测值。
        self.predictLayer = nn.ModuleList()
        for t in range(self.horizon):   #12
            self.predictLayer.append(
                output_layer(     # 多个output_layer，用于每个时间步长的预测
                    num_of_vertices=self.num_of_vertices,
                    history=history,    # 3
                    in_dim=in_dim,  # 64
                    out_dim=out_dim,  # 1
                    hidden_dim=out_layer_dim,   # 128
                    horizon=1
                )
            )

        if self.use_mask:
            mask = torch.zeros_like(self.adj)
            mask[self.adj != 0] = self.adj[self.adj != 0]
            self.mask = nn.Parameter(mask)
        else:   # False
            self.mask = None

    def forward(self, x):
        # --- [新增] 第一步：计算 Pattern Weights ---
        # x: (B, 12, N, 1) -> permute -> (B, N, 12, 1)
        # 我们取输入的特征，或者简单的 reshape
        # 这里假设输入序列长度就是 12 (history)，pattern 长度也是 12

        # 简单归一化输入，以便和 Z-score 的 patterns 匹配
        x_norm = (x - x.mean(dim=1, keepdim=True)) / (x.std(dim=1, keepdim=True) + 1e-5)
        # 取第0个特征 (B, T, N) -> (B, N, T)
        x_in = x_norm[..., 0].permute(0, 2, 1)

        # 计算相似度: (B, N, T) @ (K, T).T -> (B, N, K)
        # self.patterns 是 (K, 12)
        # 注意：self.patterns 需要在该类里访问到，确保它在 __init__ 里被保存了
        score = torch.matmul(x_in, self.patterns.t())
        pattern_weights = torch.softmax(score, dim=-1)  # (B, N, K)

        # --- 步骤 2: 数据嵌入 (创新点三修改) ---
        # [修改] 不再使用 self.First_FC(x)
        # 而是使用增强型 embedding_layer
        x = self.embedding_layer(x)
        # 注意: AdvancedDataEmbedding 内部已经包含了 Input Projection 和 Feature Fusion
        # 所以 x 现在的维度已经是 (B, T, N, 64) 了

        # x = torch.relu(self.First_FC(x))  # B=64, Tin=12, N, Cin=1 -> (64,12,N, Cout=64) 这一行可以去掉，因为 Embedding 类里通常不需要 ReLU，或者已经在内部处理了
        # print("经过定义的First_FC层之后x的数据形状：",x.size())  # pems08_30:([64,12,170,64]) pems08_10:([64,12,170,64])
        for model in self.STSGCLS:
            x = model(x, pattern_weights ,self.mask)
        need_concat = []
        for i in range(self.horizon):   # 12 对每一个时间步长
            out_step = self.predictLayer[i](x)  # (B, 1, N, 1) 每个时间步的预测
            need_concat.append(out_step)
        out = torch.cat(need_concat, dim=1)  # B, Tout, N, 1 合并所有时间步的预测进行输出
        del need_concat
        return out



