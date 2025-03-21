import json

# 输入数据路径（原始数据）
input_path = r'C:\Users\DELL\Desktop\BERT\catch-main\comments_and_ratings_cleaned.jsonl'
# 输出数据路径（清理后的数据）
output_path = r'C:\Users\DELL\Desktop\BERT\catch-main\comments_and_ratings_final_cleaned.jsonl'

# 评分合理范围（自行根据数据调整）
MIN_RATING, MAX_RATING = 1.0, 10.0

cleaned_data = []
seen_texts = set()

with open(input_path, 'r', encoding='utf-8') as infile:
    for line in infile:
        item = json.loads(line)
        text = item.get('text', '').strip()
        rating = item.get('point', None)

        # 清洗条件:
        # 1. 文本不能为空
        # 2. 评分必须存在且在合理范围内
        # 3. 数据项不能重复
        if (
            text and
            rating is not None and
            MIN_RATING <= float(rating) <= MAX_RATING and
            (text, rating) not in seen_texts
        ):
            cleaned_data.append({'text': text, 'rating': float(rating)})
            seen_texts.add((text, rating))

# 保存清洗后的数据
with open(output_path, 'w', encoding='utf-8') as outfile:
    for item in cleaned_data:
        json.dump(item, outfile, ensure_ascii=False)
        outfile.write('\n')

print(f"数据清洗完成！共保留 {len(cleaned_data)} 条有效数据。")
