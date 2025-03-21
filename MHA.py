import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"

        # 线性变换层
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        batch_size, seq_len, _ = x.size()

        # 生成Q,K,V并分头
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # 计算注意力权重
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn_weights = F.softmax(scores, dim=-1)

        # 注意力加权求和
        output = torch.matmul(attn_weights, v)
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)
        output = self.out_proj(output)

        return attn_weights, output


# 测试
embed_dim = 8
num_heads = 2
x = torch.randn(2, 5, embed_dim)  # 随机输入
mha = MultiHeadAttention(embed_dim, num_heads)
attn_weights, _ = mha(x)
print("多头注意力权重形状:", attn_weights.shape)  # [2, 2, 5, 5]

#具体测试随机矩阵的注意力权重

# 在原有代码基础上增加数值验证部分

# 设置随机种子保证可重复性
torch.manual_seed(42)

# 使用更小的维度便于观察
embed_dim = 6
num_heads = 2

# 构造确定性输入 (batch_size=1, seq_len=3)
x = torch.tensor([[
    [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],  # 序列位置1
    [0.7, 0.8, 0.9, 1.0, 1.1, 1.2],  # 序列位置2
    [1.3, 1.4, 1.5, 1.6, 1.7, 1.8]   # 序列位置3
]], dtype=torch.float32)

# 初始化模型
mha = MultiHeadAttention(embed_dim, num_heads)

# 手动设置投影参数（便于观察）
def init_weights(m):
    if isinstance(m, nn.Linear):
        # 使用简单递增数值初始化权重
        with torch.no_grad():
            for i in range(m.weight.size(0)):
                m.weight[i] = i * 0.1
            m.bias.zero_()

mha.apply(init_weights)

# 前向传播计算
attn_weights, output = mha(x)

# 打印详细计算过程
print("输入 x:\n", x)
print("\nQ 投影权重:\n", mha.q_proj.weight)
print("\nK 投影权重:\n", mha.k_proj.weight)
print("\n计算得到的注意力权重:")
print(attn_weights.detach().numpy().round(3))

# 手动验证第一个头的计算
print("\n--- 手动验证第一个头的计算 ---")

# 获取第一个头的参数
q_head1 = mha.q_proj.weight[:3] @ x[0].T  # 前3行权重对应第一个头
k_head1 = mha.k_proj.weight[:3] @ x[0].T  # 前3行权重对应第一个头

# 计算第一个头的QK^T
manual_scores = q_head1.T @ k_head1 / (3**0.5)  # head_dim=3
manual_weights = F.softmax(manual_scores, dim=-1)

print("\n手动计算的第一个头注意力权重:")
print(manual_weights.detach().numpy().round(3))

# 验证与模型输出的差异
model_head1 = attn_weights[0, 0].detach()
diff = torch.abs(manual_weights - model_head1).sum()
print(f"\n手动计算与模型输出差异: {diff.item():.6f} (应接近0)")