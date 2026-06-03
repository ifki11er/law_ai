import os
import glob
from config.settings import AI_HUB_DATA_PATH

def main():
    base_dir = os.path.join(AI_HUB_DATA_PATH, "01.원천데이터")
    print(f"Base Dir: {base_dir}")
    print(f"Exists: {os.path.exists(base_dir)}")
    
    if os.path.exists(base_dir):
        for d in os.listdir(base_dir):
            dpath = os.path.join(base_dir, d)
            if os.path.isdir(dpath):
                csv_files = glob.glob(os.path.join(dpath, "*.csv"))
                print(f"Subdir: {d} | CSV count: {len(csv_files)}")

if __name__ == "__main__":
    main()
