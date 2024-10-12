import os

base_output_dir = 'Outputs'

model_folders = [
    'HiC_GNN_LINE_evaluation',
    'node2vec_GAT_original_GM12878_evaluation',
    'GATNetConvLayerChanged_GM12878_evaluation',
    'GATNetConvLayerChanged_lr_0.0001_GM12878_evaluation',
    'GATNetHeadsChangedLeakyReLU_lr_0.0001_epoch_500_GM12878_evaluation',
    'node2vec_GAT_moreReduced_overfitted_GM12878_evaluation',
    'node2vec_GAT_reduced_overfitted_GM12878_evaluation',
    'GATNetHeadsChanged4LayerEmbedding256_lr_0.0.001_dropout_0.3_threshold_1e-8_AdamW_GM12878_evaluation',
    'GATNetHeadsChanged4LayerEmbedding256Dense_lr_0.0.0009_dropout_0.3_threshold_1e-8_GM12878_evaluation',
    'GATNetHeadsChanged4LayerEmbedding512Dense_lr_0.0.0009_dropout_0.3_threshold_1e-7_GM12878_evaluation'
    
]

comparison_results = {}

for model_folder in model_folders:
    model_path = os.path.join(base_output_dir, model_folder)
    
    for file_name in os.listdir(model_path):
        if file_name.endswith('_evaluation_log.txt'):
            chromosome_name = file_name.split('_evaluation_log.txt')[0]

            dSCC = None
            MSE = None
            with open(os.path.join(model_path, file_name), 'r') as file:
                lines = file.readlines()
                if len(lines) >= 2:
                    try:
                        dSCC = float(lines[0].split(': ')[1].strip()) 
                        MSE = float(lines[1].split(': ')[1].strip()) 
                    except (IndexError, ValueError):
                        dSCC = None
                        MSE = None

                if chromosome_name not in comparison_results:
                    comparison_results[chromosome_name] = []

                comparison_results[chromosome_name].append({
                    'model': model_folder,
                    'dSCC': dSCC,
                    'MSE': MSE
                })

for chromosome, results in comparison_results.items():
    valid_results = [r for r in results if r['dSCC'] is not None and r['MSE'] is not None]
    
    if valid_results:
        best_model_dSCC = max(valid_results, key=lambda x: x['dSCC'])

        best_model_MSE = min(valid_results, key=lambda x: x['MSE'])

        print(f"For {chromosome}:")
        print(f"  Best model by dSCC: {best_model_dSCC['model']} with dSCC: {best_model_dSCC['dSCC']}")
        print(f"  Best model by MSE: {best_model_MSE['model']} with MSE: {best_model_MSE['MSE']}")
    else:
        print(f"For {chromosome}: No valid dSCC or MSE found.")

    print("-" * 50)