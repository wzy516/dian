import json
import pandas as pd
import re
from tqdm import tqdm  # 进度条工具，可选

# ----------------------------------
# 步骤1：逐行读取JSON文件
# ----------------------------------
data = []
with open('test.jsonl', 'r', encoding='utf-8') as f:
    for line in tqdm(f):  # 使用tqdm显示读取进度
        line = line.strip()
        if line:
            try:
                item = json.loads(line)
                # 统一字段名称（将point改为label）
                item['label'] = item.pop('point')
                data.append(item)
            except (json.JSONDecodeError, KeyError) as e:
                print(f"解析失败的行: {line}\n错误信息: {e}")
df = pd.DataFrame(data)

# ----------------------------------
# 步骤3：数据清洗
# ----------------------------------
def clean_text(text):
    # 处理换行符（你的数据中有包含\n的评论）
    text = re.sub(r'\n+', ' ', text)
    # 去除HTML标签
    text = re.sub(r'<[^>]+>', '', text)
    # 去除特殊符号（保留中文、英文、数字和常用标点）
    text = re.sub(r'[^\w\u4e00-\u9fa5，。！？、：；“”‘’（）《》【】]', '', text)
    # 合并多余空格
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# 应用清洗
df['text'] = df['text'].apply(clean_text)

# 过滤无效数据
df = df.dropna(subset=['text', 'label'])
df = df.drop_duplicates(subset=['text'])
df = df[df['text'].str.len() > 5]  # 保留长度大于5的评论

# 处理分数（假设你的评分是1-10分制）
df['label'] = pd.to_numeric(df['label'], errors='coerce')  # 转换为数字
df = df.dropna(subset=['label'])
df['label'] = df['label'].astype(int)
df = df[(df['label'] >= 1) & (df['label'] <= 10)]  # 过滤异常分数

# ----------------------------------
# 步骤4：保存清洗结果
# ----------------------------------
df.to_csv('cleaned_test.csv', index=False)
print(f"清洗完成，有效数据量: {len(df)}")
#print(df.sample(3))  # 随机查看3条数据