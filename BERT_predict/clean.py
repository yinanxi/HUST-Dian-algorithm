import json
import re

def clean_comments_and_ratings(input_file, output_file):
    """
    数据清洗函数：
    :param input_file: 原始数据文件路径
    :param output_file: 清洗后数据文件路径
    """
    cleaned_data = []

    # 读取原始数据
    with open(input_file, 'r', encoding='utf-8') as f:
        seen = set()
        for line in f:
            record = json.loads(line.strip())

            # 去除重复数据
            record_tuple = (record['text'], record['point'])
            if record_tuple in seen:
                continue
            seen.add(record_tuple)

            # 去除缺失数据
            if not record['text'].strip() or not (1 <= record['point'] <= 10):
                continue

            # 去除评论过短或无意义评论
            if len(record['text'].strip()) < 3 or re.match(r'^[\W_]+$', record['text'].strip()):
                continue

            cleaned_data.append(record)

    # 保存清洗后的数据
    with open(output_file, 'w', encoding='utf-8') as f:
        for record in cleaned_data:
            json.dump(record, f, ensure_ascii=False)
            f.write('\n')

    print(f"原始数据条数：{len(seen)}，清洗后数据条数：{len(cleaned_data)}")


if __name__ == '__main__':
    # 数据清洗
    clean_comments_and_ratings('comments_and_ratings.jsonl', 'comments_and_ratings_cleaned.jsonl')
