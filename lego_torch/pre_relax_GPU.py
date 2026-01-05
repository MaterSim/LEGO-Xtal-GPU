import torch
from lego_torch.SO3 import SO3
from lego_torch.batch_sym import Symmetry
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import time
from pyxtal.lego.builder import builder
import numpy as np
import pickle
from lego_torch.analyze_results_to_db import analyze_crystal_results
from typing import Optional
from pyxtal.lego.builder import builder
import os
# Simple time formatter
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

def compute_loss(rep_batch, spg_batch, generators, g_map, xyz_map,
                 weights, p_ref, f, WP):
    generators = generators.clone().detach()
    res = WP.get_tuple_from_batch_opt(spg_batch, rep_batch, generators, g_map, xyz_map)
    plist = f.compute_p(*res[:4])                         # [B, N, L]
    p_ref_expanded = p_ref.expand_as(plist)               # [B, N, L]
    loss_batch = torch.sum((plist - p_ref_expanded) ** 2, dim=2)  # [B, N]
    valid_mask = (res[3] != -1).to(loss_batch.device)                           # bool [B, N]
    w = weights.detach().to(dtype=loss_batch.dtype, device=loss_batch.device)  # [B, N]
    w_valid = w * valid_mask.to(loss_batch.dtype)                               # [B, N]
    numerator = torch.sum(loss_batch * w_valid, dim=1)     # [B]
    den = torch.sum(w_valid, dim=1)                        # [B]
    den = den.clamp_min(torch.finfo(loss_batch.dtype).eps) # avoid div-by-zero
    final_loss = numerator / den                           # [B]
    return final_loss

def optimize_loss(
    spg_batch,
    rep_batch,
    generators,
    g_map,
    xyz_map,
    weights,
    opt_type: str = 'Adam',
    lr: float = 1e-2,
    num_steps: int = 2000,
    verbose: bool = True,
    per_sample_clip: Optional[float] = 10.0,
    # Per-sample LR scaling controls
    enable_per_sample_lr: bool = True,
    ps_patience: int = 10,      # Number of steps to wait before reducing scale if no improvement
    ps_factor: float = 0.5,     # Factor by which to reduce the scale (e.g., learning rate) when triggered
    ps_min_scale: float = 0.1,  # Minimum allowed scale value (prevents scale from becoming too small)
    ps_max_scale: float = 2.0,  # Maximum allowed scale value (prevents scale from becoming too large)
    ps_eps: float = 1e-4,       # Small epsilon value to avoid division by zero or for convergence checks
):

    global p_ref0, f0, WP

    """
    Optimize a batch of 'rep' parameters subject to bounds,
    minimizing loss computed by `compute_loss_from_reps_batch`.
    """
    torch.autograd.set_detect_anomaly(True)

    def apply_bounds(tensor):
        """Clamps tensor values between 0 and 1."""
        with torch.no_grad():
            tensor.clamp_(0.0, 1.0)

    # Clone and enable gradients for `reps`
    rep_batch = rep_batch.clone().detach().requires_grad_(True)
    generators = generators.clone().detach()

    # Choose optimizer
    if opt_type == 'SGD':
        optimizer = optim.SGD([rep_batch], lr=lr)
    elif opt_type == 'Adam':
        optimizer = torch.optim.AdamW([rep_batch], lr=lr)
    else:
        raise ValueError(f"Unsupported optimizer type: {opt_type}")

    # Dynamic learning rate tracking
    loss_history = []
    
    # Initialize per-sample LR state (row-wise on rep_batch)
    B = rep_batch.shape[0]
    _device = rep_batch.device
    print(f"Using device: {_device}")
    if enable_per_sample_lr:
        ps_best_loss = torch.full((B,), float('inf'), dtype=rep_batch.dtype, device=_device)
        ps_pat = torch.zeros((B,), dtype=torch.long, device=_device)
        ps_scale = torch.ones((B,), dtype=rep_batch.dtype, device=_device)
        ps_min_scale = torch.tensor(ps_min_scale, dtype=rep_batch.dtype, device=_device)
        ps_max_scale = torch.tensor(ps_max_scale, dtype=rep_batch.dtype, device=_device)

    # === Optimization loop ===
    # For each optimization step, compute the loss, backpropagate, and update parameters.
    for step in range(1, num_steps + 1):
        optimizer.zero_grad()

        # Compute per-sample losses (shape: [batch_size])
        losses = compute_loss(rep_batch, spg_batch, generators, g_map,
                              xyz_map, weights, p_ref0, f0, WP)
        # Aggregate all sample losses to a scalar for optimizer step
        loss_scalar = losses.sum()

        metric = loss_scalar.detach().item()
        #loss_scalar.backward()
        # Backpropagate using per-sample losses (vector-Jacobian product)
        losses.backward(torch.ones_like(losses))
         
        # --- Row-wise gradient manipulation: per-sample LR scaling and clipping ---
        # This block enables adaptive learning rates and gradient clipping per sample in the batch.
        if rep_batch.grad is not None:
            g = rep_batch.grad.view(B, -1)
            #print(f"[DEBUG] Step {step}: Gradient shape: {g.shape}")
            # Build row-wise scale from per-sample LR scheduler
            row_scale = None
            if enable_per_sample_lr:
                # Update per-sample scheduler state from current losses
                with torch.no_grad():
                    improved = losses < (ps_best_loss - ps_eps)
                    #print(f"[DEBUG] Step {step}: Improved mask: {improved.cpu().numpy()}")
                    ps_best_loss = torch.where(improved, losses, ps_best_loss)
                    ps_pat = torch.where(improved, torch.zeros_like(ps_pat), ps_pat + 1)

                    # On plateau, reduce effective LR (scale)
                    plateau = ps_pat > ps_patience
                    #print(f"[DEBUG] Step {step}: Plateau mask: {plateau.cpu().numpy()}")
                    if plateau.any():
                        new_scale = (ps_scale * ps_factor).clamp(ps_min_scale, ps_max_scale)
                        #print(f"[DEBUG] Step {step}: Reducing scale for plateaued samples. New scales: {new_scale.cpu().numpy()}")
                        ps_scale = torch.where(plateau.to(ps_scale.dtype).bool(), new_scale, ps_scale)
                        ps_pat = torch.where(plateau, torch.zeros_like(ps_pat), ps_pat)

                row_scale = ps_scale.view(B, 1).to(g.device)
                #print(f"[DEBUG] Step {step}: Per-sample LR scale: {ps_scale.cpu().numpy()}")
            else:
                row_scale = torch.ones((B, 1), dtype=g.dtype, device=g.device)

            # Optional per-sample gradient clipping (after LR scaling)
            if per_sample_clip is not None:
                norms = g.norm(dim=1, keepdim=True).clamp_min(1e-12)
                #print(f"[DEBUG] Step {step}: Per-sample grad norms: {norms.cpu().numpy().flatten()}")
                clip_scale = (float(per_sample_clip) / norms).clamp(max=1.0)
                #print(f"[DEBUG] Step {step}: Per-sample clip scale: {clip_scale.cpu().numpy().flatten()}")
                row_scale = row_scale * clip_scale
            # Apply row-wise scaling and clipping to gradients
            #print(f"grad before step {step}: {g[0].cpu().numpy() if B > 0 else 'N/A'}")
            g.mul_(row_scale)
            
            #print(f"[DEBUG] Step {step}: Scaled gradients (first row): {g[0].cpu().numpy() if B > 0 else 'N/A'}")
            rep_batch.grad.copy_(g.view_as(rep_batch))
            #import sys; sys.exit(0)
        
        optimizer.step()
        #apply_bounds(rep_batch)  # Ensure `reps` stay in range [0,1]
        
        # Dynamic learning rate adjustment
        loss_history.append(metric)
            
        # Early stopping if loss becomes very small
        if metric < 1e-6:
            if verbose:
                print(f"Step {step}: Early stopping - loss converged to {metric:.2e}")
            break
        
        # Logging
        if verbose and step % 50 == 0:
            print(f"Step {step}, loss(sum)={metric:.6f}")
            
        if step + 1 == num_steps:
            if verbose:
                print(f"🛑 stopping at last iteration")
                
    return rep_batch.detach(), losses.detach()


def prepare_wyckoff_list(wps_b,batch_size):
    """Convert Wyckoff position tensor to list format for pyxtal."""
    max_idx = batch_size
    wps_list = [[] for _ in range(max_idx)]
    
    for val, idx in wps_b.cpu().numpy().tolist():
        wps_list[idx].append(val)
    
    return wps_list



def process_batch(
    batch_start,
    batch_end,
    batch_data,
    per_sample_clip: Optional[float] = 10.0,
    enable_per_sample_lr: bool = True,
    ps_patience: int = 10,
    ps_factor: float = 0.5,
    ps_min_scale: float = 0.1,
    ps_max_scale: float = 2.0,
    steps: int = 700,
):
    """Process a single batch of samples from the loaded CSV data."""
    b_t0 = time.time()
    # Select batch data from the loaded CSV
    batch_data_slice = batch_data[batch_start:batch_end]
    batch_data_slice = torch.tensor(batch_data_slice, dtype=torch.float64,device='cuda' if torch.cuda.is_available() else 'cpu')
    spg_b, wps_b, rep_b = WP.get_batch_from_rows(batch_data_slice, radian=True,
                                                 normalize_in=False,
                                                 normalize_out=True,
                                                 tol=1)

    results = WP.get_tuple_from_batch(spg_b, wps_b, rep_b, normalize=True)
    _, _, _, _, weights, generators, g_map, xyz_map = results
    
    # Compute initial losses
    #pre_losses = compute_loss(rep_b, spg_b, generators, g_map,
    #                      xyz_map, weights, p_ref0, f0, WP) 
    #print(f"Initial losses for batch {batch_start}-{batch_end}: {pre_losses}")
    

    # Optimize
    t1 = time.time()
    rep_opt, loss_opt = optimize_loss(
        spg_b, rep_b, generators, g_map, xyz_map, weights,
        opt_type='Adam',
        lr=2e-3,
        num_steps=steps,
        verbose=True,
        per_sample_clip=per_sample_clip,
        enable_per_sample_lr=enable_per_sample_lr,
        ps_patience=ps_patience,
        ps_factor=ps_factor,
        ps_min_scale=ps_min_scale,
        ps_max_scale=ps_max_scale)
    t2 = time.time()
    print(f"Batch {batch_start}-{batch_end} optimization took {_fmt_time(t2 - t1)}")
    batch_size = batch_end - batch_start
    # Prepare results
    wps_list = prepare_wyckoff_list(wps_b, batch_size)
    
    results_dict = {
        "spg": spg_b.cpu().numpy().tolist(),
        "wps": wps_list,
        "rep": rep_opt.cpu().numpy().tolist(),
        "loss": loss_opt.cpu().numpy().tolist()
    }

    b_elapsed = time.time() - b_t0
    print(f"Batch {batch_start}-{batch_end} total time: {_fmt_time(b_elapsed)}")

    return results_dict 


def process_all_batches(batch_size, batch_data, steps, WP_obj=None, f0_obj=None, p_ref0_obj=None):
    """
    Process all batches, save intermediate results, and combine final results.
    
    Args:
        batch_size: Number of samples per batch
        batch_data: The loaded CSV data
        steps: Number of optimization steps per sample
        WP_obj: Symmetry object (will be initialized if None)
        f0_obj: SO3 object (will be initialized if None)
        p_ref0_obj: Reference p tensor (will be computed if None)
    
    Returns:
        combined_results: Dictionary containing combined results from all batches
    """
    # Initialize required objects if not provided
    global WP, f0, p_ref0
    
    if WP_obj is None:
        from lego_torch.batch_sym import Symmetry
        WP_obj = Symmetry(csv_file="wyckoff_list.csv")
    if f0_obj is None:
        from lego_torch.SO3 import SO3
        f0_obj = SO3(lmax=4, nmax=2, alpha=1.5, rcut=2.1, max_N=100)
    if p_ref0_obj is None:
        p_ref0_obj = compute_ref_p(f0_obj, WP_obj)
    
    # Set globals for use in other functions
    WP = WP_obj
    f0 = f0_obj 
    p_ref0 = p_ref0_obj
    num = len(batch_data)

    
    # Process data in batches
    all_results = []
    
    for batch_idx in range(0, num, batch_size):
        batch_start_idx = batch_idx
        batch_end_idx = min(batch_idx + batch_size, num)
        actual_batch_size = batch_end_idx - batch_start_idx
        
        print(f"\n=== Processing Batch {batch_idx//batch_size + 1}/{(num + batch_size - 1) // batch_size} ===")
        print(f"Indices: {batch_start_idx} to {batch_end_idx} ({actual_batch_size} samples)")
        
        per_batch_t0 = time.time()
        # Resolve per-sample clip argument
        per_sample_clip = 10 
        # Process this batch with requested loss reduction and clipping
        batch_results = process_batch(
            batch_start_idx, batch_end_idx, batch_data,
            per_sample_clip=per_sample_clip,
            steps=steps
        )
        per_batch_elapsed = time.time() - per_batch_t0
        print(f"Batch time: {_fmt_time(per_batch_elapsed)} (indices {batch_start_idx}-{batch_end_idx})")
    
        all_results.append(batch_results)

    # Combine all results and create final summary
    print(f"\n=== Final Summary ===")
    print(f"Total samples processed: {num}")

    # Save combined results (all indices, including failed ones)
    combined_results = {
        "spg": [],
        "wps": [],
        "rep": [],
        "loss": []
    }
    
    for batch_results in all_results:
        for key in combined_results.keys():
            combined_results[key].extend(batch_results[key])
    
    return combined_results


if __name__ == "__main__":
    import argparse
    import os
    torch.manual_seed(0)
    #cuda manual seed
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)
    #set all seeds
    import random
    random.seed(0)
    import numpy as np
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
