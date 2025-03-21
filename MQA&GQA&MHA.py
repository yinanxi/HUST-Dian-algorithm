import torch
import torch.nn as nn
import torch.nn.functional as F

# Scaled Dot-Product Attention(basic)
def scaled_dot_product_attention(Q, K, V):
    attn_weights = torch.matmul(Q, K.transpose(-2, -1)) / (Q.size(-1) ** 0.5)
    attn_weights = F.softmax(attn_weights, dim=-1)
    output = torch.matmul(attn_weights, V)
    return output, attn_weights

# 标准MHA实现
class MHA(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.Q_proj = nn.Linear(d_model, d_model)
        self.K_proj = nn.Linear(d_model, d_model)
        self.V_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, query, key, value):
        B = query.size(0)

        Q = self.Q_proj(query).view(B, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.K_proj(key).view(B, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.V_proj(value).view(B, -1, self.num_heads, self.d_k).transpose(1, 2)

        output, attn = scaled_dot_product_attention(Q, K, V)

        output = output.transpose(1, 2).contiguous().view(B, -1, self.num_heads * self.d_k)
        output = self.out_proj(output)
        return output, attn

# MQA实现 (共享单一KV头)
class MQA(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.Q_proj = nn.Linear(d_model, d_model)
        self.K_proj = nn.Linear(d_model, self.d_k)
        self.V_proj = nn.Linear(d_model, self.d_k)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, query, key, value):
        B = query.size(0)

        Q = self.Q_proj(query).view(B, -1, self.num_heads, self.d_k).transpose(1, 2)
        # K、V只有一个头，被所有头共享
        K = self.K_proj(key).unsqueeze(1)
        V = self.V_proj(value).unsqueeze(1)

        output, attn = scaled_dot_product_attention(Q, K, V)

        output = output.transpose(1, 2).contiguous().view(B, -1, self.num_heads * self.d_k)
        output = self.out_proj(output)
        return output, attn

# GQA实现 (分组共享KV头)
class GQA(nn.Module):
    def __init__(self, d_model, num_heads, num_groups):
        super().__init__()
        self.num_heads = num_heads
        self.num_groups = num_groups
        self.heads_per_group = num_heads // num_groups
        self.d_k = d_model // num_heads

        self.Q_proj = nn.Linear(d_model, d_model)
        self.K_proj = nn.Linear(d_model, num_groups * self.d_k)
        self.V_proj = nn.Linear(d_model, num_groups * self.d_k)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, query, key, value):
        B = query.size(0)

        Q = self.Q_proj(query).view(B, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.K_proj(key).view(B, -1, self.num_groups, self.d_k).transpose(1, 2)
        V = self.V_proj(value).view(B, -1, self.num_groups, self.d_k).transpose(1, 2)

        # 扩展K、V以匹配head数量
        K = K.unsqueeze(2).repeat(1, 1, self.heads_per_group, 1, 1).view(B, self.num_heads, -1, self.d_k)
        V = V.unsqueeze(2).repeat(1, 1, self.heads_per_group, 1, 1).view(B, self.num_heads, -1, self.d_k)

        output, attn = scaled_dot_product_attention(Q, K, V)

        output = output.transpose(1, 2).contiguous().view(B, -1, self.num_heads * self.d_k)
        output = self.out_proj(output)
        return output, attn

# 测试三种注意力机制
def test_attention_methods():
    torch.manual_seed(42)
    B, seq_len, d_model, heads, groups = 2, 4, 32, 4, 2

    query = torch.rand(B, seq_len, d_model)
    key = torch.rand(B, seq_len, d_model)
    value = torch.rand(B, seq_len, d_model)

    mha = MHA(d_model, heads)
    mqa = MQA(d_model, heads)
    gqa = GQA(d_model, heads, groups)

    output_mha, attn_mha = mha(query, key, value)
    output_mqa, attn_mqa = mqa(query, key, value)
    output_gqa, attn_gqa = gqa(query, key, value)

    print("MHA注意力权重形状:", attn_mha.shape)
    print("MHA注意力权重示例:\n", attn_mha[0, 0])

    print("\nMQA注意力权重形状:", attn_mqa.shape)
    print("MQA注意力权重示例:\n", attn_mqa[0, 0])

    print("\nGQA注意力权重形状:", attn_gqa.shape)
    print("GQA注意力权重示例:\n", attn_gqa[0, 0])

if __name__ == '__main__':
    test_attention_methods()
