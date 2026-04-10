"""
Prepares the steelbar_dataset for APGCC training.
- Converts JSON labels ({"points": [[x,y], ...]}) to .txt files (one "x y" per line)
- Splits into train/test (80/20 by default)
- Generates train.list and test.list
"""
import os
import json
import random
import argparse

def prepare(dataset_root, train_ratio=0.8, seed=42):
    images_dir = os.path.join(dataset_root, 'images')
    labels_dir = os.path.join(dataset_root, 'labels')
    txt_labels_dir = os.path.join(dataset_root, 'labels_txt')
    os.makedirs(txt_labels_dir, exist_ok=True)

    # Collect all images that have a matching JSON label
    image_files = sorted([f for f in os.listdir(images_dir) if f.lower().endswith('.jpg')])
    valid_pairs = []
    for img_file in image_files:
        stem = os.path.splitext(img_file)[0]
        json_path = os.path.join(labels_dir, stem + '.json')
        if os.path.exists(json_path):
            valid_pairs.append((img_file, stem))
        else:
            print(f"  [WARN] No label found for {img_file}, skipping.")

    print(f"Found {len(valid_pairs)} image-label pairs.")

    # Convert JSON -> TXT
    for img_file, stem in valid_pairs:
        json_path = os.path.join(labels_dir, stem + '.json')
        txt_path = os.path.join(txt_labels_dir, stem + '.txt')
        with open(json_path) as f:
            data = json.load(f)
        points = data.get('points', [])
        with open(txt_path, 'w') as f:
            for p in points:
                f.write(f'{p[0]} {p[1]}\n')

    print(f"Converted {len(valid_pairs)} JSON labels to TXT in '{txt_labels_dir}'")

    # Train/test split
    random.seed(seed)
    indices = list(range(len(valid_pairs)))
    random.shuffle(indices)
    split = int(len(indices) * train_ratio)
    train_indices = sorted(indices[:split])
    test_indices  = sorted(indices[split:])

    def write_list(indices, list_path):
        with open(list_path, 'w') as f:
            for i in indices:
                img_file, stem = valid_pairs[i]
                img_rel  = os.path.join('images', img_file).replace('\\', '/')
                txt_rel  = os.path.join('labels_txt', stem + '.txt').replace('\\', '/')
                f.write(f'{img_rel} {txt_rel}\n')

    train_list = os.path.join(dataset_root, 'train.list')
    test_list  = os.path.join(dataset_root, 'test.list')
    write_list(train_indices, train_list)
    write_list(test_indices,  test_list)

    print(f"Train samples : {len(train_indices)}  -> {train_list}")
    print(f"Test  samples : {len(test_indices)}   -> {test_list}")
    print("Done.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_root', default=r'C:\Users\User\Downloads\APGCC\steelbar_dataset')
    parser.add_argument('--train_ratio', type=float, default=0.8)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    prepare(args.dataset_root, args.train_ratio, args.seed)
