import os
import sys
import shutil

raw_dirs_path = "/home/enzo/Cyano_Detection/data/raw"

all_directories = os.listdir(raw_dirs_path)

for d in all_directories:
    
    d_full_path = os.path.join(raw_dirs_path, d)
    
    for d2 in os.listdir(d_full_path):
        d2_full_path = os.path.join(d_full_path, d2)

        if "_LST_" in d2_full_path:
            shutil.rmtree(d2_full_path)
            print(f"deleting {d2_full_path}")
            continue

        if d2_full_path.endswith(".zip"):
            os.remove(d2_full_path)
            print(f"deleting {d2_full_path}")
            continue
        files_in_dir = len(os.listdir(d2_full_path))
        if files_in_dir == 0:
            print(f"deleting {d2_full_path}")
            os.rmdir(d2_full_path)
