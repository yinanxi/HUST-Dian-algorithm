import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()

        self.d_model = d_model  # 模型输入特征维度
        self.num_heads = num_heads  # 注意力机制的头数
        self.d_k = d_model // num_heads  # 每个头的维度

        # 定义查询(Q)、键(K)、值(V)的线性变换层
        self.linear_Q = nn.Linear(d_model, d_model)
        self.linear_K = nn.Linear(d_model, d_model)
        self.linear_V = nn.Linear(d_model, d_model)

        # 最终输出的线性变换层
        self.linear_out = nn.Linear(d_model, d_model)

    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        """
        缩放点积注意力的具体计算过程
        """
        # 计算Q和K之间的点积，并进行缩放以防止梯度消失
        attn_weights = torch.matmul(Q, K.transpose(-2, -1)) / (self.d_k ** 0.5)

        # 若有mask则应用mask（通常用于掩盖未来的信息）
        if mask is not None:
            attn_weights = attn_weights.masked_fill(mask == 0, float('-inf'))

        # 通过softmax计算注意力权重
        attn_weights = F.softmax(attn_weights, dim=-1)

        # 将注意力权重应用于值V上，得到输出
        output = torch.matmul(attn_weights, V)
        return output, attn_weights

    def forward(self, query, key, value, mask=None, kv_cache=None):
        batch_size = query.size(0)

        # 线性变换并将维度划分成多头：(batch_size, num_heads, seq_length, d_k)
        Q = self.linear_Q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        # KV缓存处理，若缓存存在则复用，否则计算新的K和V
        if kv_cache is not None:
            K, V = kv_cache
        else:
            K = self.linear_K(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
            V = self.linear_V(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        # 进行注意力机制计算
        attn_output, attn_weights = self.scaled_dot_product_attention(Q, K, V, mask=mask)

        # 将多头注意力结果合并：(batch_size, seq_length, d_model)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)

        # 最终输出的线性变换
        output = self.linear_out(attn_output)

        # 返回最终结果、注意力权重、KV缓存
        return output, attn_weights, (K, V)

# 测试多头注意力机制并输出随机矩阵和注意力权重
def test_multi_head_attention():
    torch.manual_seed(42)  # 为复现结果设定随机种子

    # 定义示例参数
    batch_size = 2
    seq_length = 4
    d_model = 32
    num_heads = 4

    # 创建随机矩阵作为模拟输入
    query = torch.rand(batch_size, seq_length, d_model)
    key = torch.rand(batch_size, seq_length, d_model)
    value = torch.rand(batch_size, seq_length, d_model)

    print("随机生成的Query矩阵:\n", query)
    print("随机生成的Key矩阵:\n", key)
    print("随机生成的Value矩阵:\n", value)

    # 实例化多头注意力模型
    mha = MultiHeadAttention(d_model, num_heads)

    # 首次调用，无缓存
    output, attn_weights, kv_cache = mha(query, key, value)

    print("\n初次调用的注意力权重形状:", attn_weights.shape)
    print("初次调用注意力权重示例:\n", attn_weights[0, 0])

    # 模拟后续调用，复用KV缓存
    query_new = torch.rand(batch_size, seq_length, d_model)  # 新的query输入
    print("\n新的Query矩阵（复用KV缓存）:\n", query_new)

    output_cached, attn_weights_cached, _ = mha(query_new, key, value, kv_cache=kv_cache)

    print("\n复用KV缓存后的注意力权重形状:", attn_weights_cached.shape)
    print("复用KV缓存后的注意力权重示例:\n", attn_weights_cached[0, 0])

if __name__ == '__main__':
    test_multi_head_attention()
