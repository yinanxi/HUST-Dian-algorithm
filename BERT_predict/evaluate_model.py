import torch
import json
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.metrics import mean_squared_error
import numpy as np

# 加载模型
model_path = "./bangumi_rating_regression_model"
tokenizer = BertTokenizer.from_pretrained(model_path)
model = BertForSequenceClassification.from_pretrained(model_path)
model.eval()

# 加载验证数据
test_path = r'C:\Users\DELL\Desktop\BERT\catch-main\test.jsonl'
texts, labels = [], []
with open(test_path, 'r', encoding='utf-8') as f:
    for line in f:
        item = json.loads(line)
        texts.append(item['text'])
        labels.append(float(item['point']))  # 注意修改这里的字段名

# 模型预测
predictions = []
with torch.no_grad():
    for text in texts:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
        outputs = model(**inputs)
        pred = outputs.logits.item()
        predictions.append(pred)

# 计算指标
rmse = mean_squared_error(labels, predictions, squared=False)
mse = mean_squared_error(labels, predictions)
print(f"模型在验证集上的表现：RMSE={rmse:.4f}, MSE={mse:.4f}")
