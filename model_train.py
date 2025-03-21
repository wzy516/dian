import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    TrainingArguments,
    Trainer
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, accuracy_score
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# ----------------------
# 数据准备阶段
# ----------------------
# 加载清洗后的数据（假设CSV包含text和label两列）
train_csv = "cleaned_comments.csv"
df = pd.read_csv(train_csv)
df['label'] = df['label'].astype(int) - 1  # 将标签转换为0-9

# 文本编码 模型配置
model_path = "./bert-base-chinese"  # 假设模型文件夹在项目根目录下
tokenizer = BertTokenizer.from_pretrained(model_path)
model = BertForSequenceClassification.from_pretrained(
    model_path,
    num_labels=10,
    problem_type="single_label_classification"
)
max_length = 128

def encode_texts(texts):
    return tokenizer(
        texts.tolist(),
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    )

# 数据集划分
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
train_encodings = encode_texts(train_df['text'])
val_encodings = encode_texts(val_df['text'])

# ----------------------
# 自定义数据集类
# ----------------------
class CommentDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {
            key: val[idx]
            for key, val in self.encodings.items()
        }
        item['labels'] = torch.tensor(self.labels.iloc[idx])
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = CommentDataset(train_encodings, train_df['label'])
val_dataset = CommentDataset(val_encodings, val_df['label'])

# ----------------------


# ----------------------
# 训练参数配置
# ----------------------
training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",          # 更新参数名
    save_strategy="epoch",          # 新增保存策略
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir="./logs",
    load_best_model_at_end=True,
    metric_for_best_model="mae",
    greater_is_better=False
)

# ----------------------
# 评估指标计算
# ----------------------
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    mae = mean_absolute_error(labels, preds)
    acc = accuracy_score(labels, preds)
    return {"mae": mae, "accuracy": acc}

# ----------------------
# 训练流程
# ----------------------
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics
)

# 执行训练
trainer.train()

# 保存最佳模型
model.save_pretrained("./best_model")
tokenizer.save_pretrained("./best_model")

# ----------------------
# 预测示例
# ----------------------
# ----------------------
# 预测示例（修正版本）
# ----------------------
def predict(text):
    # 生成输入并移至模型设备
    inputs = tokenizer(
        text,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    ).to(model.device)

    # 推理计算
    model.eval()  # 进入评估模式
    with torch.no_grad():
        outputs = model(**inputs)

    # 结果解析
    pred = torch.argmax(outputs.logits).item() + 1
    return pred

# 测试预测
test_text = "这部作品的作画质量堪称业界标杆，但剧情发展有些拖沓"
print(f"预测评分：{predict(test_text)}")


# 验证集预测与评估
# ----------------------

plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置全局字体
plt.rcParams['axes.unicode_minus'] = False    # 解决负号显示异常问题
def evaluate_on_validation_set(trainer, val_dataset, val_df):
    # 执行预测
    predictions = trainer.predict(val_dataset)

    # 获取预测结果
    pred_logits = predictions.predictions
    pred_labels = np.argmax(pred_logits, axis=1)
    true_labels = predictions.label_ids

    # 转换为DataFrame
    results_df = val_df.copy()
    results_df['pred_label'] = pred_labels + 1  # 恢复1-10评分

    # 保存预测结果
    results_df[['text', 'label', 'pred_label']].to_csv("validation_predictions.csv", index=False)

    # 扩展评估指标
    from sklearn.metrics import classification_report, confusion_matrix
    import seaborn as sns
    import matplotlib.pyplot as plt

    # 打印详细分类报告
    print(classification_report(
        true_labels,
        pred_labels,
        labels=np.arange(10),  # 显式指定所有可能的标签
        target_names=[str(i + 1) for i in range(10)],
        digits=4
    ))

    # 对应的混淆矩阵修正
    cm = confusion_matrix(true_labels, pred_labels, labels=np.arange(10))
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d',
                xticklabels=range(1, 11),
                yticklabels=range(1, 11),
                cmap='Blues')
    plt.xlabel('预测标签')
    plt.ylabel('真实标签')
    plt.title('混淆矩阵')
    plt.savefig('confusion_matrix.png')
    plt.close()

    # 返回关键指标
    return {
        'mae': mean_absolute_error(true_labels, pred_labels),
        'accuracy': accuracy_score(true_labels, pred_labels)
    }


# 执行验证集评估
val_metrics = evaluate_on_validation_set(trainer, val_dataset, val_df)
print(f"\n最终验证集指标 - MAE: {val_metrics['mae']:.4f}, 准确率: {val_metrics['accuracy']:.4f}")
