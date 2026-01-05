import torch
from lego_torch.SO3 import SO3
from lego_torch.batch_sym import Symmetry
import time
import numpy as np
import pickle
import argparse
import os
from lego_torch.pre_relax_GPU import process_all_batches
import os
import numpy as np
import random


def _fmt_time(seconds: float) -> str:
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    if h:
        return f"{int(h)}h {int(m)}m {s:.1f}s"
    if m:
        return f"{int(m)}m {s:.1f}s"
    return f"{s:.1f}s"

def compute_ref_p(f, sym):
    ref_row=torch.tensor([[194, 2.46, 2.46, 6.70, 1.5708, 1.5708, 2.0944,
                           9, 1/3, 2/3, 1/4, 10, 0, 0, 1/4]],
                           dtype=torch.float64)
    spg, wps, rep = sym.get_batch_from_rows(ref_row,
                                            normalize_in=False,
                                            normalize_out=False, 
                                            tol=1)

    res = sym.get_tuple_from_batch(spg, wps, rep, normalize=False)
    p_ref = f.compute_p(*res[:4])[0, 0].view(1,1,-1)
    #print(f"{res[0]} {res[1]} {p_ref}")#; import sys; sys.exit(0)
    return p_ref

if __name__ == "__main__":

    torch.manual_seed(0)
    #cuda manual seed
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)
    #set all seeds
    random.seed(0)
    np.random.seed(0)

    parser = argparse.ArgumentParser(description="Optimize VAE representation parameters for crystal structure generation.")
    parser.add_argument('--start', type=int, default=0, help='Start index for data selection')
    parser.add_argument('--num', type=int, default=100000, help='Total number of samples to process')
    parser.add_argument('--batch_size', type=int, default=1000, help='Number of samples per batch')
    parser.add_argument('--per_sample_clip', type=float, default=10.0,
                        help='Per-sample gradient clip norm; set <0 to disable')
    parser.add_argument('--steps', type=int, default=1000, help='Number of optimization steps per sample')
    parser.add_argument('--data_file', type=str, default='data/sample/TVAE_v4-cont_hd_512512_e_500.csv', help='Input CSV file with initial data')

    args = parser.parse_args()

    import pandas as pd
    
    # Initialize global variables
    WP = Symmetry(csv_file="lego_torch/wyckoff_list.csv")
    f0 = SO3(lmax=4, nmax=2, alpha=1.5, rcut=2.1, max_N=100)
    p_ref0 = compute_ref_p(f0, WP)
    start = args.start
    num = args.num
    batch_size = args.batch_size
    end = start + num

    overall_t0 = time.time()

    # Load the entire CSV data once
    csv_path = args.data_file
    filename = csv_path.split('/')[-1].replace('.csv','')
    dirname = f"results/{filename}"
    os.makedirs(dirname, exist_ok=True)
    #sort by column 'spg

    df = pd.read_csv(csv_path).iloc[start:end]
    df = df.sort_values(by=['spg'])
    batch_data = df.values    
    print(f"Loaded data from {csv_path}, shape: {batch_data.shape}")
    print(f"Start index: {start}, End index: {end}, Total samples: {num}, Batch size: {batch_size}")
    print(f"Total batches: {(num + batch_size - 1) // batch_size}")

    # Process data in batches
    all_results = []
    combined_results = process_all_batches(
        batch_size, batch_data,args.steps, WP, f0, p_ref0
    )

    overall_elapsed = time.time() - overall_t0
    print(f"Total runtime: {_fmt_time(overall_elapsed)} for {num} samples (batch size {batch_size})")
    # Save combined results
    with open(f'{dirname}/combined_results.pkl', 'wb') as f:
        pickle.dump(combined_results, f)
    #print(f"Saved combined results to {dirname}/combined_results.pkl")
    # Analyze and save final results
    #results, valid_xtal_index = analyze_crystal_results(combined_results, output_dir=dirname, verbose=True)
    #print(f"Analysis results saved to {dirname}")


    metric_path = f'{dirname}/metric.txt'
    mode = 'a' if os.path.exists(metric_path) else 'w'
    with open(metric_path, mode) as f:
        f.write(f"filename: {dirname}\n")
        f.write(f"\n=== Analysis run: {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n")
        f.write(f"Total Optimization time: {_fmt_time(overall_elapsed)}\n")
        #f.write(f"Total structures: {results['total_structures']}\n")
        #f.write(f"Valid structures: {results['valid_structures']}\n")
