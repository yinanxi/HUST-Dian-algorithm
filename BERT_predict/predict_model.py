import torch
import json
from sklearn.model_selection import train_test_split
from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    Trainer,
    TrainingArguments
)
from datasets import Dataset
from evaluate import load
import numpy as np

# 定义数据文件路径
data_path = r'C:\Users\DELL\Desktop\BERT\catch-main\comments_and_ratings_final_cleaned.jsonl'

# 加载数据
texts, labels = [], []
with open(data_path, 'r', encoding='utf-8') as f:
    for line in f:
        item = json.loads(line)
        texts.append(item['text'])
        labels.append(float(item['rating']))  # 转换分数为浮点数

# 将数据集划分为训练集和验证集（90%训练，10%验证）
train_texts, val_texts, train_labels, val_labels = train_test_split(
    texts, labels, test_size=0.1, random_state=42
)

# 初始化中文BERT tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")

# 定义tokenize函数，用于将文本编码为模型输入
def tokenize_fn(examples):
    return tokenizer(examples['text'], padding="max_length", truncation=True, max_length=128)

# 将训练数据转为Dataset格式，并执行tokenize操作
train_dataset = Dataset.from_dict({"text": train_texts, "label": train_labels}).map(tokenize_fn, batched=True)
# 将验证数据转为Dataset格式，并执行tokenize操作
val_dataset = Dataset.from_dict({"text": val_texts, "label": val_labels}).map(tokenize_fn, batched=True)

# 设置数据格式为PyTorch tensor
train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
val_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])

# 初始化BERT模型，设定任务为回归问题
model = BertForSequenceClassification.from_pretrained(
    "bert-base-chinese",
    num_labels=1,                 # 回归任务输出维度为1
    problem_type="regression"     # 明确指定任务类型为回归
)

# 加载评估指标（均方误差）
metric = load("mse")

# 定义计算指标函数（计算MSE和RMSE）
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.squeeze(predictions)
    mse = metric.compute(predictions=predictions, references=labels)
    rmse = np.sqrt(mse['mse'])
    return {"mse": mse['mse'], "rmse": rmse}

# 设置训练参数
training_args = TrainingArguments(
    output_dir='./results_regression',    # 模型和日志输出目录
    eval_strategy="epoch",               # 每个epoch后执行一次评估
    save_strategy="epoch",               # 每个epoch保存一次模型
    learning_rate=1e-5,                   # 学习率
    per_device_train_batch_size=8,        # 训练batch大小
    per_device_eval_batch_size=8,         # 验证batch大小
    num_train_epochs=10,                   # 训练总共进行10个epoch
    weight_decay=0.01,                    # 权重衰减防止过拟合
    logging_steps=50,                     # 每50步记录一次训练日志
    save_total_limit=2,                   # 最多保存2个模型checkpoint
    load_best_model_at_end=True,          # 训练结束时加载最佳模型
    metric_for_best_model='rmse',         # 使用RMSE作为挑选最佳模型的指标
)

# 初始化Trainer对象，负责训练和评估
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
)

# 开始模型训练
trainer.train()

# 保存训练后的最佳模型到指定路径
trainer.save_model('./bangumi_rating_regression_model')
tokenizer.save_pretrained('./bangumi_rating_regression_model')

