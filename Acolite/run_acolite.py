import os, sys
sys.path.append("../")
# Get paths of products to run acolite on
import settings

data_path = settings.data_path

days_to_process = []
for day_directory in os.listdir(data_path):
    day_directory_path = os.path.join(data_path, day_directory)
    day_diretory_files = os.listdir(day_directory_path)
    if any(".SAFE" in file for file in day_diretory_files):
        
