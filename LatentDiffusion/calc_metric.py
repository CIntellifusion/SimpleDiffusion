import torch_fidelity
import os 
ground_truth_dir = "ground-truth-celeba"
path="logs/trigflow/2025-11-03_03-34/samples/fid-eval/"
epochs = sorted(os.listdir(path))
for epoch_dir in epochs:
    output_dirs = os.listdir(os.path.join(path,epoch_dir))
    for output_dir in sorted(output_dirs):
        output_dir = os.path.join(path,epoch_dir,output_dir)
        metrics_dict = torch_fidelity.calculate_metrics(
            input1=output_dir,
            input2=ground_truth_dir,
            fid=True
        )
        print(f"{output_dir} : {metrics_dict}")
        with open("fid_results.txt","a") as f:
            f.write(f"{output_dir} : {metrics_dict}\n")