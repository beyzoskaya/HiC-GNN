import os

def rename_files_in_directory(root_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for subdir, dirs, files in os.walk(root_dir):
        for file in files:
            input_path = os.path.join(subdir, file)
            output_file_name = file + '.txt'
            output_path = os.path.join(output_dir, output_file_name)
            os.rename(input_path, output_path)
            print(f'Renamed {input_path} to {output_path}')

if __name__ == "__main__":
    root_dir = 'Datasets'
    output_dir = 'Dataset_CH12-LX_txt'

    rename_files_in_directory(root_dir, output_dir)

    print("Renaming completed.")
