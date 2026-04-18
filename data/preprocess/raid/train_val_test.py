import os
import json
import random
from pathlib import Path

def split_and_save_titles(domain, input_file, output_root, val_size=50, test_size=100):
    """
    处理单个领域的标题分割与保存
    """
    # 读取所有标题
    titles = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data = json.loads(line.strip())
                if 'title' in data:
                    titles.append(data['title'])
            except json.JSONDecodeError:
                print(f"警告：无法解析 {input_file} 中的一行数据")
                continue
    
    # 确保有足够的数据
    total_needed = val_size + test_size
    if len(titles) < total_needed:
        print(f"警告：{domain} 领域的数据不足 {total_needed} 条，实际有 {len(titles)} 条")
        return
    
    # 随机打乱数据
    random.seed(42)  # 设置随机种子，确保结果可复现
    random.shuffle(titles)
    
    # 分割数据集
    test_titles = titles[:test_size]
    val_titles = titles[test_size:test_size+val_size]
    train_titles = titles[test_size+val_size:]
    
    # 创建领域文件夹
    domain_dir = os.path.join(output_root, domain)
    Path(domain_dir).mkdir(parents=True, exist_ok=True)
    
    # 保存各数据集
    def save_json(data, filename):
        with open(os.path.join(domain_dir, filename), 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    save_json(train_titles, 'train_titles.json')
    save_json(val_titles, 'val_titles.json')
    save_json(test_titles, 'test_titles.json')
    
    print(f"已处理 {domain}：训练集 {len(train_titles)} 条，验证集 {len(val_titles)} 条，测试集 {len(test_titles)} 条")

def main():
    # 配置参数
    domains = ['abstracts', 'books', 'news', 'poetry', 'recipes', 'reddit', 'reviews', 'wiki']
    input_root = '/home/cxsj25f/raid/train_data_raw/none/gpt2'
    output_root = './title_datasets'  # 输出根目录，可根据需要修改
    val_size = 50
    test_size = 100
    
    # 创建输出根目录
    Path(output_root).mkdir(parents=True, exist_ok=True)
    
    # 处理每个领域
    for domain in domains:
        input_file = os.path.join(
            input_root, 
            domain, 
            'greedy', 
            'no', 
            f'none_gpt2_{domain}_greedy_no.jsonl'
        )
        
        if not os.path.exists(input_file):
            print(f"警告：文件 {input_file} 不存在，跳过该领域")
            continue
        
        split_and_save_titles(domain, input_file, output_root, val_size, test_size)
    
    print("所有领域处理完成")

if __name__ == "__main__":
    main()