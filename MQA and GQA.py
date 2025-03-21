import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiQueryAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"

        # Q的投影与MHA相同，K和V的投影输出为单个头
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, self.head_dim)
        self.v_proj = nn.Linear(embed_dim, self.head_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        batch_size, seq_len, _ = x.size()

        # 生成Q并分头
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        # 生成K和V并分头（单个头）
        k = self.k_proj(x).view(batch_size, seq_len, 1, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, 1, self.head_dim).transpose(1, 2)

        # 计算注意力权重
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn_weights = F.softmax(scores, dim=-1)

        # 加权求和
        output = torch.matmul(attn_weights, v)
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)
        output = self.out_proj(output)

        return attn_weights, output


class GroupedQueryAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, group_num):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.group_num = group_num
        self.head_dim = embed_dim // num_heads
        self.num_heads_per_group = num_heads // group_num

        assert num_heads % group_num == 0, "num_heads must be divisible by group_num"
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"

        # Q投影与MHA相同，K和V投影为组数×头维度
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, group_num * self.head_dim)
        self.v_proj = nn.Linear(embed_dim, group_num * self.head_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        batch_size, seq_len, _ = x.size()

        # 生成Q并分头
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        # 生成K和V并分头为组数
        k = self.k_proj(x).view(batch_size, seq_len, self.group_num, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.group_num, self.head_dim).transpose(1, 2)

        # 将Q分组并计算注意力
        q = q.view(batch_size, self.group_num, self.num_heads_per_group, seq_len, self.head_dim)
        k = k.unsqueeze(2)  # 扩展维度以匹配组内头数
        v = v.unsqueeze(2)

        # 计算注意力得分
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn_weights = F.softmax(scores, dim=-1)

        # 加权求和并恢复形状
        output = torch.matmul(attn_weights, v)
        output = output.view(batch_size, self.num_heads, seq_len, self.head_dim)
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)
        output = self.out_proj(output)

        # 调整注意力权重形状
        attn_weights = attn_weights.view(batch_size, self.num_heads, seq_len, seq_len)

        return attn_weights, output

# 测试参数
#embed_dim = 8
#num_heads = 4
#group_num = 2
#x = torch.randn(2, 5, embed_dim)  # 随机输入


# 测试MQA
#mqa = MultiQueryAttention(embed_dim, num_heads)
#attn_mqa, _ = mqa(x)
#print("MQA注意力权重形状:", attn_mqa.shape)  # [2, 4, 5, 5]

# 测试GQA
#gqa = GroupedQueryAttention(embed_dim, num_heads, group_num)
#attn_gqa, _ = gqa(x)
#print("GQA注意力权重形状:", attn_gqa.shape)  # [2, 4, 5, 5]

# 3-2 注意力机制变体实现与验证
# 修改自3-1的验证代码，保留相同的输入和参数初始化方式
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


# 使用与3-1完全相同的输入和初始化参数
def init_weights(m):
    if isinstance(m, nn.Linear):
        with torch.no_grad():
            for i in range(m.weight.size(0)):
                m.weight[i] = i * 0.1
            m.bias.zero_()

# 测试MQA --------------------------------------------------
print("\n" + "="*50 + "\nMulti-Query Attention验证\n" + "="*50)
mqa = MultiQueryAttention(embed_dim=6, num_heads=2)
mqa.apply(init_weights)  # 应用相同的初始化方式

# 前向传播
attn_mqa, output_mqa = mqa(x)

# 打印参数
print("\nMQA的K投影层权重:")
print(mqa.k_proj.weight)  # 应只有3行（head_dim=3）
print("\nMQA的V投影层权重:")
print(mqa.v_proj.weight)  # 同样只有3行

# 手动验证第一个头的计算
print("\n--- 手动验证MQA第一个头的计算 ---")
# 第一个头的Q投影（前3行权重）
q_head1 = mqa.q_proj.weight[:3] @ x[0].T  # 与MHA相同
# K投影（所有头共享，只有3个参数）
k_shared = mqa.k_proj.weight @ x[0].T  # [3,3]
# 计算QK^T
manual_scores_mqa = q_head1.T @ k_shared / (3**0.5)
manual_weights_mqa = F.softmax(manual_scores_mqa, dim=-1)

print("\n手动计算的MQA第一个头注意力权重:")
print(manual_weights_mqa.detach().numpy().round(3))
print("模型计算的MQA第一个头注意力权重:")
print(attn_mqa[0, 0].detach().numpy().round(3))
diff_mqa = torch.abs(manual_weights_mqa - attn_mqa[0, 0]).sum()
print(f"差异值: {diff_mqa.item():.6f} (应接近0)")

# 测试GQA --------------------------------------------------
print("\n" + "="*50 + "\nGrouped Query Attention验证 (group_num=2)\n" + "="*50)
gqa = GroupedQueryAttention(embed_dim=6, num_heads=2, group_num=2)
gqa.apply(init_weights)

# 前向传播
attn_gqa, output_gqa = gqa(x)

# 打印参数
print("\nGQA的K投影层权重形状:", gqa.k_proj.weight.shape)  # [2 * 3,6] = [6,6]

# 手动验证第一个组的计算
print("\n--- 手动验证GQA第一个组的计算 ---")
# 第一个组的Q（对应第一个头）
q_group1 = gqa.q_proj.weight[:3] @ x[0].T  # 前3行权重
# 第一个组的K投影（前3行权重）
k_group1 = gqa.k_proj.weight[:3] @ x[0].T
# 计算QK^T
manual_scores_gqa = q_group1.T @ k_group1 / (3**0.5)
manual_weights_gqa = F.softmax(manual_scores_gqa, dim=-1)

print("\n手动计算的GQA第一组注意力权重:")
print(manual_weights_gqa.detach().numpy().round(3))
print("模型计算的GQA第一个头注意力权重:")
print(attn_gqa[0, 0].detach().numpy().round(3))
diff_gqa = torch.abs(manual_weights_gqa - attn_gqa[0, 0]).sum()
print(f"差异值: {diff_gqa.item():.6f} (应接近0)")

# 对比注意力权重 --------------------------------------------------
print("\n" + "="*50 + "\n注意力权重对比\n" + "="*50)

print("MQA第一个头权重:\n", attn_mqa[0,0].detach().numpy().round(3))
print("GQA第一个头权重:\n", attn_gqa[0,0].detach().numpy().round(3))

# KV缓存大小对比
print("\nKV缓存大小对比 (batch_size=1, seq_len=3, embed_dim=6)")

print(f"MQA的KV缓存: 1 head * 3 dim = 3 parameters per position → Total: {1 * 3 * 3 * 2} floats (减少50%)")
print(f"GQA的KV缓存: 2 groups * 3 dim = 6 parameters per position → Total: {2 * 3 * 3 * 2} floats (与MHA相同但结构不同)")