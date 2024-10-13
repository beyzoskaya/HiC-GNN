import os

def extract_base_name(filename, suffix):
    if filename.endswith(suffix):
        return filename[:-len(suffix)]
    return None

def read_training_log(file_path):
    values = {}
    with open(file_path, 'r') as f:
        for line in f:
            if 'Optimal dSCC' in line:
                try:
                    values['dSCC'] = float(line.split(': ')[1].strip())
                except ValueError:
                    values['dSCC'] = None
            elif 'Final MSE loss' in line:
                try:
                    values['MSE'] = float(line.split(': ')[1].strip())
                except ValueError:
                    values['MSE'] = None
            elif 'dRMSD' in line:
                try:
                    values['dRMSD'] = float(line.split(': ')[1].strip())
                except ValueError:
                    values['dRMSD'] = None
    return values

def read_evaluation_log(file_path):
    values = {}
    with open(file_path, 'r') as f:
        for line in f:
            if 'Optimal dSCC' in line:
                try:
                    values['dSCC'] = float(line.split(': ')[1].strip())
                except ValueError:
                    values['dSCC'] = None
            elif 'MSE' in line and 'Final' not in line:
                try:
                    values['MSE'] = float(line.split(': ')[1].strip())
                except ValueError:
                    values['MSE'] = None
            elif 'dRMSD' in line:
                try:
                    values['dRMSD'] = float(line.split(': ')[1].strip())
                except ValueError:
                    values['dRMSD'] = None
    return values

def main():
    # Define your training and evaluation directories
    training_dir = 'Outputs/GATNetHeadsChanged4Layers_lr_0.0.0009_dropout_0.3_threshold_1e-8_node2vecParamsChanged_GM12878'
    evaluation_dir = 'Outputs/GATNetHeadsChanged4Layers_lr_0.0.0009_dropout_0.3_threshold_1e-8_node2vecParamsChanged_GM12878_evaluation'
    
    training_suffix = '_node2vec_log_GAT.txt'
    evaluation_suffix = '_evaluation_log.txt'
    
    if not os.path.isdir(training_dir):
        print(f"Training directory '{training_dir}' does not exist.")
        return
    if not os.path.isdir(evaluation_dir):
        print(f"Evaluation directory '{evaluation_dir}' does not exist.")
        return
    
    training_files = [f for f in os.listdir(training_dir) if f.endswith(training_suffix)]
    
    if not training_files:
        print("No training log files found.")
        return
    
    for train_file in training_files:
        base_name = extract_base_name(train_file, training_suffix)
        if not base_name:
            continue  
        
        eval_file = f'{base_name}_evaluation_log.txt'
        train_path = os.path.join(training_dir, train_file)
        eval_path = os.path.join(evaluation_dir, eval_file)
        
        if not os.path.exists(eval_path):
            print(f'Evaluation log not found for {base_name}. Skipping.')
            continue
        
        training_metrics = read_training_log(train_path)
        evaluation_metrics = read_evaluation_log(eval_path)
        
        metrics_same = {}
        for metric in ['dSCC', 'MSE', 'dRMSD']:
            train_val = training_metrics.get(metric, None)
            eval_val = evaluation_metrics.get(metric, None)
            if train_val is not None and eval_val is not None:
                same = abs(train_val - eval_val) < 1e-10
                metrics_same[metric] = same
            else:
                metrics_same[metric] = False  
        
        print(f'Comparison for Chromosome: {base_name}')
        print('-' * 80)
        print(f"{'Metric':<10} {'Training Value':<25} {'Evaluation Value':<25} {'Same?':<10}")
        print('-' * 80)
        for metric in ['dSCC', 'MSE', 'dRMSD']:
            train_val = training_metrics.get(metric, 'N/A')
            eval_val = evaluation_metrics.get(metric, 'N/A')
            same = 'Yes' if metrics_same.get(metric, False) else 'No'
            if train_val == 'N/A' or eval_val == 'N/A':
                same = 'N/A'
            print(f"{metric:<10} {train_val:<25} {eval_val:<25} {same:<10}")
        print('-' * 80 + '\n')

if __name__ == "__main__":
    main()
