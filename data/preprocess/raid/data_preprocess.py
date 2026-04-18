import os
import argparse
import json
import random
from typing import List, Dict, Tuple
from collections import defaultdict

# Define all possible parameter options
CONFIG = {
    'attack': ['alternative_spelling', 'article_deletion', 'homoglyph', 'insert_paragraphs', 
               'none', 'number', 'paraphrase', 'perlexity_misspelling', 'synonym', 
               'upper_lower', 'whitespace', 'zero_width_space'],
    'model': ['chatgpt', 'cohere', 'cohere-chat', 'gpt2', 'gpt3', 'gpt4', 'human', 
              'llama-chat', 'mistral', 'mistral-chat', 'mpt', 'mpt-chat'],
    'domain': ['abstracts', 'books', 'news', 'poetry', 'recipes', 'reddit', 'reviews', 'wiki'],
    'decoding': ['greedy', 'sampling', 'none'],
    'repetition_penalty': ['yes', 'no', 'none']
}

def parse_filename(filename: str) -> dict:
    """Parse filename to extract parameter values"""
    if not filename.endswith('.jsonl'):
        return None
    
    # Remove .jsonl suffix and split by underscore
    parts = filename[:-6].split('_')
    if len(parts) < 5:
        return None  # Doesn't match naming format
    
    # Extract each parameter from filename parts
    repetition_penalty = parts[-1]
    decoding = parts[-2]
    domain = parts[-3]
    model = parts[-4]
    # Attack may consist of multiple parts joined by underscores
    attack = '_'.join(parts[:-4])
    
    return {
        'attack': attack,
        'model': model,
        'domain': domain,
        'decoding': decoding,
        'repetition_penalty': repetition_penalty,
        'filename': filename
    }

def filter_files(directory: str, criteria: dict) -> List[str]:
    """Recursively filter all .jsonl files that meet the criteria in the directory and subdirectories"""
    total_files = 0
    matched_files = []
    for root, dirs, files in os.walk(directory):
        for filename in files:
            if not filename.endswith('.jsonl'):
                continue
            total_files += 1
            parsed = parse_filename(filename)
            if not parsed:
                continue 
            
            match = True
            for key, values in criteria.items():
                # Check if the current parameter is in the allowed list
                if parsed[key] not in values:
                    match = False
                    break
            
            if match:
                matched_files.append(os.path.join(root, filename))  # Store full path for reading
    print(f'total_files: {total_files}')
    return matched_files

def split_train_test(data: List[Dict], test_size: float = 0.2) -> Tuple[List[Dict], List[Dict]]:
    """Split data into training and testing sets"""
    if test_size <= 0 or test_size >= 1:
        raise ValueError("test_size must be between 0 and 1")
    
    # Calculate split index
    split_idx = int(len(data) * (1 - test_size))
    
    # Split data
    train_data = data[:split_idx]
    test_data = data[split_idx:]
    
    return train_data, test_data

def save_data(data: List[Dict], file_path: str):
    """Save data to JSONL file"""
    with open(file_path, 'w', encoding='utf-8') as f:
        for sample in data:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    print(f"Saved {len(data)} samples to {file_path}")

def count_domain_samples(samples: List[Dict]) -> Dict[str, int]:
    """Count the number of samples for each domain in the sample list"""
    domain_counts = defaultdict(int)
    for sample in samples:
        domain = sample.get('domain', 'unknown')
        domain_counts[domain] += 1
    return domain_counts

def print_domain_counts(split_name: str, counts: Dict[str, int]):
    """Print domain count statistics for a specific dataset split"""
    print(f"\n{split_name} set domain distribution:")
    total = sum(counts.values())
    for domain, count in sorted(counts.items()):
        percentage = (count / total) * 100 if total > 0 else 0
        print(f"  {domain}: {count} samples ({percentage:.1f}%)")
    print(f"  Total: {total} samples")

def process_and_save_data(matched_files: List[str], criteria: Dict[str, List[str]], 
                         base_dir: str = '../data/binary/sample', title_root: str = './title_datasets'):
    """Process matched files, split into train/val/test sets based on title datasets, and save"""
    
    # Preload title mappings for all domains into datasets
    domain_title_maps = {}  # Structure: {domain: {'train': set(titles), 'val': set(titles), 'test': set(titles)}}
    
    # Collect all possible domains involved
    domains = set()
    for file_path in matched_files:
        filename = os.path.basename(file_path)
        parsed = parse_filename(filename)
        if parsed and parsed['domain']:
            domains.add(parsed['domain'])
    
    # Load title mappings for each domain
    for domain in domains:
        domain_title_maps[domain] = {}
        for split in ['train', 'val', 'test']:
            title_file = os.path.join(title_root, domain, f'{split}_titles.json')
            if not os.path.exists(title_file):
                print(f"Warning: Title file {title_file} does not exist, the {split} set for this domain will be empty.")
                domain_title_maps[domain][split] = set()
                continue
            
            # Read and store the title set (using a set for faster lookup)
            with open(title_file, 'r', encoding='utf-8') as f:
                titles = json.load(f)
                domain_title_maps[domain][split] = set(titles)
    
    # Initialize the three datasets
    train_samples = []
    val_samples = []
    test_samples = []
    unclassified_samples = []  # Samples that cannot be matched to any title set
    
    # Process samples in each file
    for file_path in matched_files:
        filename = os.path.basename(file_path)
        parsed = parse_filename(filename)
        if not parsed:
            continue
        
        domain = parsed['domain']
        # Determine the title mapping for the current domain (use an empty mapping if the domain does not exist)
        title_map = domain_title_maps.get(domain, {'train': set(), 'val': set(), 'test': set()})
        
        # Determine label (0: human, 1: other models)
        label = 0 if parsed['model'] == 'human' else 1
        
        # Read file content and classify samples
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line.strip())
                    # Check required fields
                    if 'prompt' not in data or 'generation' not in data or 'title' not in data:
                        continue
                    
                    # Construct sample text
                    text = f"{data['title']} {data['generation']}".strip()
                    sample = {
                        'text': text,
                        'label': label,
                        'attack': parsed['attack'],
                        'model': parsed['model'],
                        'domain': domain,
                        'decoding': parsed['decoding'],
                        'repetition_penalty': parsed['repetition_penalty']
                    }
                    
                    # Match dataset based on title
                    title = data['title']
                    if title in title_map['train']:
                        train_samples.append(sample)
                    elif title in title_map['val']:
                        val_samples.append(sample)
                    elif title in title_map['test']:
                        test_samples.append(sample)
                    else:
                        unclassified_samples.append(sample)
                
                except json.JSONDecodeError:
                    print(f"Warning: Unable to parse line in {file_path}")
                    continue
    
    # Create output directory
    os.makedirs(base_dir, exist_ok=True)
    
    # Generate output filename (connect multiple parameters with commas, replace all values with 'mixed')
    output_params = []
    for k, v in criteria.items():
        if len(v) == len(CONFIG[k]):
            output_params.append('mixed')
        else:
            output_params.append(','.join(v))
    
    base_filename = f"{output_params[0]}_{output_params[1]}_{output_params[2]}_{output_params[3]}_{output_params[4]}"
    
    # Save the three datasets
    def save_split(split_data, split_name):
        if split_data:
            random.shuffle(split_data)
            file_path = os.path.join(base_dir, f"{split_name}.jsonl")
            with open(file_path, 'w', encoding='utf-8') as f:
                for sample in split_data:
                    f.write(json.dumps(sample, ensure_ascii=False) + '\n')
            print(f"Saved {len(split_data)} samples to the {split_name} set: {file_path}")
        else:
            print(f"No samples to save for the {split_name} set.")
    
    save_split(train_samples, 'train')
    save_split(val_samples, 'val')
    save_split(test_samples, 'test')
    
    # Count and print domain distributions for the validation and test sets
    val_domain_counts = count_domain_samples(val_samples)
    test_domain_counts = count_domain_samples(test_samples)
    
    print_domain_counts("Validation", val_domain_counts)
    print_domain_counts("Test", test_domain_counts)
    
    # Print warning for unclassified samples
    if unclassified_samples:
        print(f"\nWarning: {len(unclassified_samples)} samples had titles that did not match any dataset and were ignored.")
    
    # Print overall statistics
    total = len(train_samples) + len(val_samples) + len(test_samples)
    print(f"\nTotal valid samples: {total} (Train: {len(train_samples)}, Val: {len(val_samples)}, Test: {len(test_samples)})")

def main():
    parser = argparse.ArgumentParser(description='Filter jsonl files and prepare classification training data with train/test split')
    parser.add_argument('--directory', type=str, default='./train_data_raw', help='Directory path to search in')
    parser.add_argument('--title_dir', type=str, default='./title_datasets', help='Directory path to title datasets')
    parser.add_argument('--attack', type=str, nargs='+', default=CONFIG['attack'], 
                       help=f'Attack types (space-separated), possible values: {CONFIG["attack"]}')
    parser.add_argument('--model', type=str, nargs='+', default=CONFIG['model'], 
                       help=f'Model names (space-separated), possible values: {CONFIG["model"]}')
    parser.add_argument('--domain', type=str, nargs='+', default=CONFIG['domain'], 
                       help=f'Domain types (space-separated), possible values: {CONFIG["domain"]}')
    parser.add_argument('--decoding', type=str, nargs='+', default=CONFIG['decoding'], 
                       help=f'Decoding methods (space-separated), possible values: {CONFIG["decoding"]}')
    parser.add_argument('--repetition_penalty', type=str, nargs='+', default=CONFIG['repetition_penalty'], 
                       help=f'Repetition penalties (space-separated), possible values: {CONFIG["repetition_penalty"]}')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for shuffling data')
    parser.add_argument('--base_dir', type=str, default='./binary/attack', help='Directory to store data')
    args = parser.parse_args()
    
    # Set random seed for reproducibility
    random.seed(args.seed)
    
    # Validate parameter values
    for key in ['attack', 'model', 'domain', 'decoding', 'repetition_penalty']:
        values = getattr(args, key)
        for value in values:
            if value not in CONFIG[key]:
                raise ValueError(f"Invalid {key} value: {value}, possible values: {CONFIG[key]}")
    
    # Build filter criteria - each parameter corresponds to a list of allowed values
    criteria = {
        'attack': args.attack,
        'model': args.model,
        'domain': args.domain,
        'decoding': args.decoding,
        'repetition_penalty': args.repetition_penalty
    }
    
    # Filter files
    matched_files = filter_files(args.directory, criteria)
    print(f"Found {len(matched_files)} matching files to process")
    
    # Process and save data if there are matched files
    if matched_files:
        process_and_save_data(matched_files, criteria, base_dir=args.base_dir, title_root=args.title_dir)
    else:
        print("No matching files found, skipping data processing.")

if __name__ == "__main__":
    main()