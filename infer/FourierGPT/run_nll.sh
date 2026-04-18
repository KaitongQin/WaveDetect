#!/usr/bin/env bash

# ================== 配置区域 ==================
INPUT_DIR='/home/cxsj25f/baseline/FourierGPT/data/raid-all/model'  # 第一个参数：输入文件夹
MODEL_PATH="/data1/model/gpt2-small"
SCRIPT="run_nll.py"
# =============================================

for input_file in "$INPUT_DIR"/*.txt; do
  # 如果目录下没有 txt 文件
  [ -e "$input_file" ] || continue

  base_name=$(basename "$input_file" .txt)
  output_file="${INPUT_DIR}/${base_name}.nll.txt"

  echo "Processing: $input_file"
  echo "Output:     $output_file"

  python "$SCRIPT" \
    --input "$input_file" \
    --output "$output_file" \
    --model_path "$MODEL_PATH"
done

echo "All files processed."
