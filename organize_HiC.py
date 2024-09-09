import os
import shutil

source_dir = 'Dataset_CH12-LX_txt'  
dest_base_dir = 'Hi-C_dataset/CH12-LX' 

resolutions = ['1mb', '5kb', '10kb', '25kb', '50kb', '100kb', '250kb', '500kb']

for filename in os.listdir(source_dir):
    if filename.endswith(".txt") and 'RAWobserved' in filename: 
        print(f'Processing file: {filename}') 

        base_name, ext = os.path.splitext(filename) 
        parts = base_name.split('_') 
        
        if len(parts) == 2:
            chr_info = parts[0] 
            resolution_with_extension = parts[1] 
            
           
            resolution_parts = resolution_with_extension.split('.')
            if len(resolution_parts) == 2:
                resolution = resolution_parts[0]  
                file_extension = resolution_parts[1] 
                
                if file_extension == 'RAWobserved':
                    if resolution in resolutions:
                        dest_folder = os.path.join(dest_base_dir, resolution)
                        
                        if os.path.exists(dest_folder):
                            source_path = os.path.join(source_dir, filename)
                            dest_path = os.path.join(dest_folder, filename)
                            shutil.copy(source_path, dest_path)
                            print(f'Copied {filename} to {dest_folder}')
                        else:
                            print(f'Folder {dest_folder} does not exist for resolution {resolution}.')
                    else:
                        print(f'Unknown resolution for file: {filename}')
                else:
                    print(f'File extension is not RAWobserved for file: {filename}')
            else:
                print(f'Unable to extract resolution from: {filename}')
        else:
            print(f'Invalid filename format: {filename}')