import pandas as pd
import os
import json
from pathlib import Path

def organize_raid_data():
    """
    Creates a nested directory structure:
    attack -> model -> domain -> decoding -> repetition_penalty
    """
    # Load data (Modify the path as needed for your environment)
    root_dir = Path('./RAID/')
    base_dir = root_dir / 'data'
    train_csv_path = root_dir / 'train.csv'
    print(f"Target CSV path: {train_csv_path}")

    print("Loading train.csv...")
    df = pd.read_csv(train_csv_path)
    
    # Handle missing values
    df['decoding'] = df['decoding'].fillna('none')
    df['repetition_penalty'] = df['repetition_penalty'].fillna('none')

    # Required columns to extract
    required_columns = ['id', 'adv_source_id', 'source_id', 'title', 'prompt', 'generation']

    # Group data by the specified hierarchical parameters
    grouped = df.groupby(['attack', 'model', 'domain', 'decoding', 'repetition_penalty'])
    total_groups = len(grouped)
    print(f"Total combinations found: {total_groups}")

    processed = 0
    for (attack, model, domain, decoding, repetition_penalty), group in grouped:
        processed += 1
        print(f"Processing: {processed}/{total_groups} - {attack}/{model}/{domain}/{decoding}/{repetition_penalty}")

        # Create the nested directory path
        folder_path = base_dir / str(attack) / str(model) / str(domain) / str(decoding) / str(repetition_penalty)
        folder_path.mkdir(parents=True, exist_ok=True)

        # Construct the target filename
        filename = f'{attack}_{model}_{domain}_{decoding}_{repetition_penalty}.jsonl'
        file_path = folder_path / filename

        # Prepare the records for writing
        records = []
        for _, row in group.iterrows():
            record = {}
            for col in required_columns:
                record[col] = row[col] if pd.notna(row[col]) else ""
            records.append(record)

        # Write the formatted records to the JSONL file
        with open(file_path, 'w', encoding='utf-8') as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False) + '\n')

        print(f"  Saved {len(records)} samples to {file_path}")

    print("Completed data organization!")

if __name__ == "__main__":
    organize_raid_data()