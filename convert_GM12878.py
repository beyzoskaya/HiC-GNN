import os
import shutil

def convert_to_txt(input_file_path, output_file_path):
  
    shutil.copy(input_file_path, output_file_path)

def process_gm12878_dataset(base_dir, output_base_dir):
    
    resolutions = ['1mb', '5kb', '10kb', '25kb', '50kb', '100kb', '250kb', '500kb']
    
    for resolution in resolutions:
        resolution_pattern = f"{resolution}_resolution_intrachromosomal"
        resolution_dir = os.path.join(base_dir, resolution_pattern)
        
        if not os.path.exists(resolution_dir):
            print(f"Resolution folder does not exist: {resolution_dir}")
            continue
        
        output_resolution_dir = os.path.join(output_base_dir, resolution)
        if not os.path.exists(output_resolution_dir):
            os.makedirs(output_resolution_dir)
        
        for chr_folder in os.listdir(resolution_dir):
            chr_folder_path = os.path.join(resolution_dir, chr_folder)
            if not os.path.isdir(chr_folder_path):
                print(f"Skipping non-directory: {chr_folder_path}")
                continue
            
            for subfolder in os.listdir(chr_folder_path):
                subfolder_path = os.path.join(chr_folder_path, subfolder)
                if not os.path.isdir(subfolder_path):
                    print(f"Skipping non-directory: {subfolder_path}")
                    continue
                
                for file_name in os.listdir(subfolder_path):
                    if 'RAWobserved' in file_name:  # Filter files based on name containing 'RAWobserved'
                        input_file_path = os.path.join(subfolder_path, file_name)
                        output_file_name = file_name + '.txt'  # Append .txt extension
                        output_file_path = os.path.join(output_resolution_dir, output_file_name)
                        
                        if not os.path.exists(output_file_path):
                            convert_to_txt(input_file_path, output_file_path)
                            print(f'Converted and saved: {output_file_path}')
                        else:
                            print(f'File already exists: {output_file_path}')
if __name__ == "__main__":
    base_directory = '/Users/beyzakaya/Desktop/process/GM12878_primary'
    output_directory = '/Users/beyzakaya/Desktop/process/Hi-C_dataset'
    
    process_gm12878_dataset(base_directory, output_directory)
