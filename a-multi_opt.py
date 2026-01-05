from Gen_Model.sample_VAE2stg import generate_synthetic_data
from lego_torch.SO3 import SO3
from lego_torch.batch_sym import Symmetry
import time
import numpy as np
from lego_torch.analyze_results_to_db import analyze_crystal_results
import os
from lego_torch.pre_relax_GPU import process_all_batches as process_all_batches_only_rep
from lego_torch.dual_opt import run_latent_optimization
import pandas as pd
import shutil
from lego_torch.write_metric import write_metrics_file

def save_invalid_crystals(valid_xtal_index, input_path='results/dis.csv'):
    """
    Remove valid crystal indices from the input CSV and save the invalid ones.
    
    Args:
        valid_xtal_index: List or array of indices of valid crystals to remove
        input_path: Path to the input CSV file
        output_path: Path to save the invalid crystals CSV (optional)
    """
    df = pd.read_csv(input_path)
    df_invalid = df.drop(valid_xtal_index)
    
    df_invalid.to_csv(input_path, index=False)

    print(f"Saved {len(df_invalid)} invalid crystals to {input_path}")
    print(f"Removed {len(valid_xtal_index)} valid crystals")
    return df_invalid


dis_path = 'VAE2stg-v4-synthetic-sorted-dis.csv'
#dis_path = 'test.csv'
model = 'VAE_2nd_stage_DiffGMM-DT-NEWDATA_e128_hd1024_b500_KLF1_CLF2_NLF0.1_e250'
output_path = f'results/{model}'
# Create results directory if it doesn't exist
os.makedirs(output_path, exist_ok=True)

model_path = f'data/models/{model}.pt'

# Copy input file to results directory
if os.path.exists(dis_path):
    shutil.copy(dis_path, f'{output_path}/dis.csv')
    print(f"Copied {dis_path} to {output_path}/dis.csv")
else:
    print(f"Warning: Input file {dis_path} not found!")

# Generate synthetic data
synthetic_data = generate_synthetic_data(
        f'{output_path}/dis.csv',
        model_path,
        latent_dim=128)
print(f"Generated {len(synthetic_data)} synthetic samples")

WP = Symmetry(csv_file="wyckoff_list.csv")
f0 = SO3(lmax=4, nmax=2, alpha=1.5, rcut=2.1, max_N=100)
batch_size = 1000
batch_data = synthetic_data #[:100]
steps = 1000

# Track overall metrics
overall_start_time = time.time()
metrics_data = {
    'input_file': dis_path,
    'model_path': model_path,
    'batch_size': batch_size,
    'steps': steps,
    'latent_dim': 128,
    'total_runs': 3,
    'run_results': [],
    'initial_samples': len(synthetic_data)
}

# Initial representation optimization (not VAE)
print("Running initial representation optimization...")
combined_results = process_all_batches_only_rep(
    batch_size=batch_size,
    batch_data=batch_data,
    steps=steps,
    WP_obj=WP,
    f0_obj=f0,
    p_ref0_obj=None
)
initial_results, valid_xtal_index = analyze_crystal_results(
    combined_results, 
    loss_threshold=1e6, 
    min_loss=-1, 
    output_dir=output_path, 
    verbose=True
)
# Copy database AFTER analyzing results
shutil.copy(f'{output_path}/mof-0.db', f'{output_path}/run-0.db')
print(f"Copied {output_path}/mof-0.db to {output_path}/run-0.db")

# Store initial representation optimization results
metrics_data['initial_rep_results'] = initial_results

#remove valid_xtal_index rows from dis.csv 
save_invalid_crystals(valid_xtal_index, input_path=f'{output_path}/dis.csv')

total_runs = 3
for run in range(1, total_runs + 1):
    run_start_time = time.time()
    print(f"\n=== Run {run}/{total_runs} ===")
    print(f"Running latent optimization with batch size {batch_size} and {steps} steps")
    
    # Count samples remaining before this run

    df_current = pd.read_csv(f'{output_path}/dis.csv')
    samples_remaining = len(df_current)
    print(f"Samples remaining before run {run}: {samples_remaining}")


    run_data = {
        'samples_remaining': samples_remaining,
        'valid_structures': 0,
        'validity_rate': 0.0,
        'runtime': 'N/A',
        'error': None
    }
    
    combined_results = run_latent_optimization(
        csv_path=f'{output_path}/dis.csv',
        model_path_template=model_path,
        batch_size=batch_size,
        steps=500,
        latent_dim=128,
        run=run
    )
    
    print(f"Analyzing results from latent optimization run {run}")
    run_results, valid_xtal_index = analyze_crystal_results(
        combined_results, 
        loss_threshold=1e6, 
        min_loss=-1, 
        output_dir=output_path, 
        verbose=True
    )
    
    # Copy database AFTER analyzing results
    shutil.copy(f'{output_path}/mof-0.db', f'{output_path}/run-{run}.db')
    print(f"Copied {output_path}/mof-0.db to {output_path}/run-{run}.db")
    
    # Update the dis.csv file by removing valid structures
    save_invalid_crystals(valid_xtal_index, input_path=f'{output_path}/dis.csv')
    
    # Store run metrics
    run_data['valid_structures'] = len(valid_xtal_index)
    run_data['validity_rate'] = run_results.get('validity_rate', 0.0)
    run_data['runtime'] = f"{time.time() - run_start_time:.2f}s"
    
    print(f"Completed run {run}: found {len(valid_xtal_index)} valid structures")
    

    
    # Add run data to metrics
    metrics_data['run_results'].append(run_data)

# Final metrics calculation
try:
    df_final = pd.read_csv(f'{output_path}/dis.csv')
    final_remaining = len(df_final)
except:
    final_remaining = 0

metrics_data['final_remaining_samples'] = final_remaining
metrics_data['total_runtime'] = f"{time.time() - overall_start_time:.2f}s"

# Write comprehensive metrics file
write_metrics_file(metrics_data, f'{output_path}/metric.txt')

print(f"\nCompleted all {total_runs} runs!")
print(f"Final metrics written to {output_path}/metric.txt")