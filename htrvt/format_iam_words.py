import os
import shutil
import random
from pathlib import Path

def format_iam_words():
    source_words_dir = Path("/Users/abhinandan/Documents/Krishna/HTR-VT/archive (3)/iam_words/words")
    words_txt = "/Users/abhinandan/Documents/Krishna/HTR-VT/archive (3)/words_new.txt"
    out_dir = Path("/Users/abhinandan/Documents/Krishna/HTR-VT/data/iam/words_formatted")
    
    out_dir.mkdir(parents=True, exist_ok=True)
    
    valid_samples = []
    
    with open(words_txt, "r") as f:
        lines = f.readlines()
        
    for line in lines:
        if line.startswith("#") or not line.strip():
            continue
            
        parts = line.split()
        if len(parts) >= 9 and parts[1] == "ok":
            word_id = parts[0]
            label = " ".join(parts[8:])
            
            # Extract folder structure
            # e.g., a01-000u-00-00 -> a01/a01-000u
            id_parts = word_id.split('-')
            dir1 = id_parts[0]
            dir2 = f"{id_parts[0]}-{id_parts[1]}"
            
            img_path = source_words_dir / dir1 / dir2 / f"{word_id}.png"
            
            if img_path.exists():
                # We can symlink to save space, but let's just create the txt file and symlink the image inside out_dir
                target_img = out_dir / f"{word_id}.png"
                target_txt = out_dir / f"{word_id}.txt"
                
                if not target_img.exists():
                    os.symlink(img_path, target_img)
                    
                with open(target_txt, "w") as tf:
                    tf.write(label)
                    
                valid_samples.append(f"{word_id}.png")

    print(f"Found {len(valid_samples)} valid word samples.")
    
    # Shuffle and split
    random.seed(42)
    random.shuffle(valid_samples)
    
    num_samples = len(valid_samples)
    train_end = int(num_samples * 0.8)
    val_end = int(num_samples * 0.9)
    
    train_split = valid_samples[:train_end]
    val_split = valid_samples[train_end:val_end]
    test_split = valid_samples[val_end:]
    
    data_dir = Path("/Users/abhinandan/Documents/Krishna/HTR-VT/data/iam")
    
    with open(data_dir / "train.ln", "w") as f:
        f.write("\n".join(train_split))
    with open(data_dir / "val.ln", "w") as f:
        f.write("\n".join(val_split))
    with open(data_dir / "test.ln", "w") as f:
        f.write("\n".join(test_split))
        
    print(f"Splits created: Train({len(train_split)}), Val({len(val_split)}), Test({len(test_split)})")

if __name__ == "__main__":
    format_iam_words()
