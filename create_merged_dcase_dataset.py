import os
import shutil

def merge_folders(source1, source2, source1name, source2name, destination):
    # Create the destination directory if it doesn't exist
    if not os.path.exists(destination):
        os.makedirs(destination)
    
    # Merge folders from source1
    for root, dirs, files in os.walk(source1):
        relative_path = os.path.relpath(root, source1)
        dest_path = os.path.join(destination, relative_path)
        for file in files:
            renamed_file = source1name + '_' + file
            src_file = os.path.join(root, file)
            dest_file = os.path.join(dest_path, renamed_file)
            if not os.path.exists(dest_path):
                os.makedirs(dest_path)
            shutil.copy(src_file, dest_file)
    
    # Merge folders from source2
    for root, dirs, files in os.walk(source2):
        relative_path = os.path.relpath(root, source2)
        dest_path = os.path.join(destination, relative_path)
        for file in files:
            renamed_file = source2name + '_' + file
            src_file = os.path.join(root, file)
            dest_file = os.path.join(dest_path, renamed_file)
            # Create the destination directory if it doesn't exist
            if not os.path.exists(dest_path):
                os.makedirs(dest_path)
            # If file already exists, skip
            if not os.path.exists(dest_file):
                shutil.copy(src_file, dest_file)

if __name__ == "__main__":
    source1 = "DCASE_2023_Challenge_Task_7_Dataset/dev"
    source2 = "DCASE_2023_Challenge_Task_7_Dataset/eval"
    source1name = 'dev'
    source2name = 'eval'
    destination = "DCASE_2023_Challenge_Task_7_Dataset/merged"
    merge_folders(source1, source2, source1name, source2name, destination)
    print("Folders merged successfully.")
