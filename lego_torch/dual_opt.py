"""
Crystal structure generator with latent space optimization.
This script optimizes VAE latent vectors to generate valid crystal structures
with improved symmetry properties.
"""

import pandas as pd
import torch
import numpy as np
import torch.optim as optim
import time
import os

from Gen_Model.VAE import VAE
from Gen_Model.CVAE import CVAE
from .SO3 import SO3
from .batch_sym import Symmetry
from typing import Optional
# Global constants
import time
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

# Global variables (initialized later)
WP = None
f0 = None
p_ref0 = None

# Simple time formatter
def _fmt_time(seconds: float) -> str:
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    if h:
        return f"{int(h)}h {int(m)}m {s:.1f}s"
    if m:
        return f"{int(m)}m {s:.1f}s"
    return f"{s:.1f}s"

def setup_environment():
    """Initialize global variables and create output directories."""
    global WP, f0, p_ref0

    # Initialize symmetry and SO3 objects
    WP = Symmetry(csv_file=os.path.join(os.path.dirname(__file__), "wyckoff_list.csv"))
    f0 = SO3(lmax=4, nmax=2, alpha=1.5, rcut=2.1, max_N=100)
    p_ref0 = compute_ref_p(f0,WP)
    print(f"p_ref0 : {p_ref0}")

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
            # Build row-wise scale from per-sample LR scheduler
            row_scale = None
            if enable_per_sample_lr:
                # Update per-sample scheduler state from current losses
                with torch.no_grad():
                    improved = losses < (ps_best_loss - ps_eps)
                    ps_best_loss = torch.where(improved, losses, ps_best_loss)
                    ps_pat = torch.where(improved, torch.zeros_like(ps_pat), ps_pat + 1)

                    # On plateau, reduce effective LR (scale)
                    plateau = ps_pat > ps_patience
                    if plateau.any():
                        new_scale = (ps_scale * ps_factor).clamp(ps_min_scale, ps_max_scale)
                        ps_scale = torch.where(plateau.to(ps_scale.dtype).bool(), new_scale, ps_scale)
                        ps_pat = torch.where(plateau, torch.zeros_like(ps_pat), ps_pat)

                row_scale = ps_scale.view(B, 1).to(g.device)
            else:
                row_scale = torch.ones((B, 1), dtype=g.dtype, device=g.device)

            # Optional per-sample gradient clipping (after LR scaling)
            if per_sample_clip is not None:
                norms = g.norm(dim=1, keepdim=True).clamp_min(1e-12)
                clip_scale = (float(per_sample_clip) / norms).clamp(max=1.0)
                row_scale = row_scale * clip_scale
            g.mul_(row_scale)
            rep_batch.grad.copy_(g.view_as(rep_batch))
        
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
        if verbose and step % 10 == 0:
            print(f"Step {step}, loss(sum)={metric:.6f}")
            
        if step + 1 == num_steps:
            if verbose:
                print(f"🛑 stopping at last iteration")
                
    return rep_batch.detach(), losses.detach()

def assemble_data(dis, cont_inv):
    """
    Assemble the final synthetic data from discrete and continuous parts.
    
    Args:
        dis: Discrete data tensor (space group, Wyckoff positions)
        cont_inv: Continuous data tensor (lattice parameters, coordinates)
        
    Returns:
        Combined data tensor in the proper format
    """
    # Ensure both tensors have the same dtype and device for concatenation
    if dis.dtype != cont_inv.dtype:
        dis = dis.to(dtype=cont_inv.dtype)
    if dis.device != cont_inv.device:
        dis = dis.to(device=cont_inv.device)
    
    synthetic_data = torch.cat((dis, cont_inv), dim=1)

    spg     = synthetic_data[:, [0]]            # (N, 1)
    wps     = synthetic_data[:, 1:9]            # (N, 8)
    abc_ang = synthetic_data[:, 9:15]           # (N, 6)
    xyzs    = synthetic_data[:, 15:]            # (N, 24)

    # Stack wp{i} + x{i}y{i}z{i} blocks
    blocks = []
    for i in range(8):
        wp_i   = wps[:, [i]]                    # (N, 1)
        xyz_i  = xyzs[:, 3*i:3*(i+1)]           # (N, 3)
        block  = torch.cat([wp_i, xyz_i], dim=1)
        blocks.append(block)

    # Final reassembly
    synthetic_data_reordered = torch.cat([spg, abc_ang] + blocks, dim=1)
    return synthetic_data_reordered
def process_lattice_parameters(para):
    """
    Process lattice parameters based on space group constraints.
    
    Args:
        para: Tensor of shape (N, 6+) where first column is space group,
              columns 1-6 are lattice parameters (a,b,c,alpha,beta,gamma)
    
    Returns:
        Processed tensor with crystallographic constraints applied
    """
    # Clone to avoid modifying the original tensor
    para = para.clone()
    spg = para[:, 0]

    # Create masks for each condition
    mask_194 = spg > 194  # Cubic
    mask_142 = (spg > 142) & (spg <= 194)  # Hexagonal
    mask_74 = (spg > 74) & (spg <= 142)   # Tetragonal
    mask_15 = (spg > 15) & (spg <= 74)    # Orthorhombic
    mask_2 = (spg > 2) & (spg <= 15)      # Monoclinic

    # Apply transformations using vectorized operations
    # Cubic: a=b=c, α=β=γ=90°
    if mask_194.any():
        a_mean_194 = para[mask_194, 1:4].mean(dim=1, keepdim=True)
        para[mask_194, 1:4] = a_mean_194
        para[mask_194, 4:7] = torch.pi/2

    # Hexagonal: a=b≠c, α=β=90°, γ=120°
    if mask_142.any():
        a_mean_142 = para[mask_142, 1:3].mean(dim=1, keepdim=True)
        para[mask_142, 1:3] = a_mean_142
        para[mask_142, 4:6] = torch.pi/2
        para[mask_142, 6] = torch.pi * 2/3

    # Tetragonal: a=b≠c, α=β=γ=90°
    if mask_74.any():
        a_mean_74 = para[mask_74, 1:3].mean(dim=1, keepdim=True)
        para[mask_74, 1:3] = a_mean_74
        para[mask_74, 4:7] = torch.pi/2

    # Orthorhombic: a≠b≠c, α=β=γ=90°
    if mask_15.any():
        para[mask_15, 4:7] = torch.pi/2

    # Monoclinic: a≠b≠c, α=γ=90°, β≠90°
    if mask_2.any():
        para[mask_2, 4] = torch.pi/2  # α
        para[mask_2, 6] = torch.pi/2  # γ

    return para

def evaluate_crystals(results_dict, criteria=None):
    """
    Evaluate crystal structures and save valid ones to CIF files.
    
    Args:
        results_dict: Dictionary with space groups, Wyckoff positions, representations, and losses
        output_dir: Directory to save valid structures
        criteria: Dictionary of validation criteria
        
    Returns:
        Number of valid crystal structures
    """
    if criteria is None:
        criteria = {
            "CN": {"C": [3]},
            "cutoff": None
        }
    xtal_1d_batch = []
    valid_count = 0
    for i in range(len(results_dict["loss"])):
        try:
            xtal = WP.get_pyxtal_from_spg_wps_rep(
                results_dict["spg"][i],
                results_dict["wps"][i],
                results_dict["rep"][i],
                normalize=True
            )
            xtal_1d = xtal.get_tabular_representation()
            #print(f"xtal_1d: {xtal_1d}")
            # Convert numpy array to list and append loss
            xtal_1d_list = xtal_1d.tolist() if hasattr(xtal_1d, 'tolist') else list(xtal_1d)
            xtal_1d_list.append(results_dict["loss"][i])
            #print(f"xtal_1d_with_loss: {xtal_1d_list}")
            xtal_1d_batch.append(xtal_1d_list)
            if xtal.check_validity(criteria) and results_dict['loss'][i] < 1:
                print(f" ===== index: {i} loss: {results_dict['loss'][i]:.3f} ====")

                valid_count += 1
        except Exception as e:
            print(f"Error processing crystal at index {i}: {e}")
            continue
            
    return valid_count, xtal_1d_batch

def scale_lattice_params(synthetic_data, WP):
    """
    Scale lattice parameters in synthetic data for normalization.
    
    Args:
        synthetic_data: Tensor containing crystal structure data
        WP: Symmetry object containing max_abc and max_angle for scaling
        
    Returns:
        Normalized tensor with scaled lattice parameters
    """
    scale = torch.ones(synthetic_data.shape[1], dtype=synthetic_data.dtype, device=synthetic_data.device)
    scale[1:4] = 1.0 / WP.max_abc       # a, b, c
    scale[4:7] = 1.0 / WP.max_angle     # alpha, beta, gamma
    norm_rows = synthetic_data * scale  # out-of-place; keeps autograd graph
    norm_rows_double = norm_rows.double() 
    return norm_rows_double

def prepare_wyckoff_list(wps_b,batch_size):
    """Convert Wyckoff position tensor to list format for pyxtal."""
    max_idx = batch_size
    wps_list = [[] for _ in range(max_idx)]
    
    for val, idx in wps_b.cpu().numpy().tolist():
        wps_list[idx].append(val)
    
    return wps_list


def process_batch(dis_batch, 
        batch_start, 
        batch_end, 
        CVAE_model, 
        num_steps, 
        ed=128,
        per_sample_clip: Optional[float] = 10.0,
        # Per-sample LR scaling controls
        enable_per_sample_lr: bool = True,
        ps_patience: int = 10,      # Number of steps to wait before reducing scale if no improvement
        ps_factor: float = 0.5,     # Factor by which to reduce the scale (e.g., learning rate) when triggered
        ps_min_scale: float = 0.1,  # Minimum allowed scale value (prevents scale from becoming too small)
        ps_max_scale: float = 2.0,  # Maximum allowed scale value (prevents scale from becoming too large)
        ps_eps: float = 1e-4,       # Small epsilon value to avoid division by zero or for convergence checks
        ):
    """Process a single batch of samples."""
    batch_size = len(dis_batch)
    
    # Set seed for reproducibility BEFORE creating tensors
    #torch.manual_seed(42)
    
    # Initialize latent vectors for this batch
    latent_dim = ed
    z = torch.randn(batch_size, latent_dim, device=CVAE_model._device, requires_grad=True)
    #print(f"Initial z: {z.detach().cpu().numpy()[:2,:5]} ...")
    
    
    # Prepare conditioning input
    dis_tensor = torch.tensor(dis_batch.values, dtype=torch.float32, device=CVAE_model._device)
    
    condition = CVAE_model.transformer_condition.transform(dis_batch)
    cond_tensor = torch.tensor(condition, dtype=torch.float32, device=CVAE_model._device)
    cond_latent = CVAE_model.condition_layer(cond_tensor)
    #initialize and store initial synthetic data
    with torch.no_grad():
        z_initial = z.clone()
        full_latent = torch.cat((z_initial, cond_latent), dim=1)
        decoded, _ = CVAE_model.decoder(full_latent)
        output_orig = CVAE_model.transformer_data.inverse(decoded)
        # Convert dis_tensor to same dtype as output_orig for assembly
        dis_tensor_for_assembly = dis_tensor.to(dtype=output_orig.dtype)
        synthetic_data = assemble_data(dis_tensor_for_assembly, output_orig)

        
    # Convert to double precision for symmetry operations
    synthetic_data_double = synthetic_data.to(dtype=torch.float64)
    synthetic_data_double = process_lattice_parameters(synthetic_data_double)
    
    spg_b, wps_b, rep_ids_batch = WP.get_batch_rep_ids_from_rows(
        synthetic_data_double, 
        radian=True,
        normalize_in=False,
        normalize_out=True,
        tol=1
    )
    safe_ids = rep_ids_batch.clone()
    safe_ids[safe_ids == -1] = 0
    safe_ids = safe_ids.long()
    #print(f"Processed synthetic_data_double: {synthetic_data_double.dtype} {synthetic_data_double.device} {synthetic_data_double.shape} {synthetic_data_double[:2,:20]} ...")
    synthetic_data_double[:,1:4] = synthetic_data_double[:,1:4] / WP.max_abc  
    synthetic_data_double[:,4:7] = synthetic_data_double[:,4:7] / WP.max_angle  
    # Fix: Ensure rep_b is of the same dtype as synthetic_data
    rep_b = torch.gather(synthetic_data_double, dim=1, index=safe_ids)
    rep_b = rep_b.to(dtype=torch.float64)  
    rep_b[rep_ids_batch == -1] = -1.0
    #print(f"spg_b : {spg_b[:10]}, wps_b : {wps_b[:10]}, rep_b : {rep_b[:10]}") ; import sys; sys.exit(0)
    _, _, _, _, weights, generators, g_map, xyz_map = WP.get_tuple_from_batch(
        spg_b, wps_b, rep_b, normalize=True
    )
    # First stage optimization: latent space
    print(f"Starting first stage optimization with latent_dim={latent_dim} for batch {batch_start}-{batch_end} (latent space)...")

    stage1_t0 = time.time()
    
    #optimizer = torch.optim.AdamW([z], lr=lr * 0.1)  
    optimizer = torch.optim.AdamW([z], lr=2e-3)
    loss_history = []
    B = z.shape[0]
    _device = z.device
    if enable_per_sample_lr:
        ps_best_loss = torch.full((B,), float('inf'), dtype=z.dtype, device=_device)
        ps_pat = torch.zeros((B,), dtype=torch.long, device=_device)
        ps_scale = torch.ones((B,), dtype=z.dtype, device=_device)
        ps_min_scale = torch.tensor(ps_min_scale, dtype=z.dtype, device=_device)
        ps_max_scale = torch.tensor(ps_max_scale, dtype=z.dtype, device=_device)

    for step in range(1, num_steps + 1):
        # Ensure z still requires gradients (safety check)
        if not z.requires_grad:
            print(f"WARNING: z.requires_grad became False at step {step}. Fixing...")
            z.requires_grad_(True)
            
        optimizer.zero_grad()
        
        # Generate from latent vectors
        full_latent = torch.cat((z, cond_latent), dim=1)
        decoded, _ = CVAE_model.decoder(full_latent)
        output_orig = CVAE_model.transformer_data.inverse(decoded)
        # Convert dis_tensor to same dtype as output_orig for assembly
        dis_tensor_for_assembly = dis_tensor.to(dtype=output_orig.dtype)
        synthetic_data = assemble_data(dis_tensor_for_assembly, output_orig)
        synthetic_data= process_lattice_parameters(synthetic_data)
        norm_rows_double=scale_lattice_params(synthetic_data, WP)
        rep_b = torch.gather(norm_rows_double, dim=1, index=safe_ids)
        # Use masked_fill instead of in-place assignment to preserve gradients
        mask = (rep_ids_batch == -1)
        rep_b = torch.where(mask, torch.tensor(-1.0, device=rep_b.device, dtype=rep_b.dtype), rep_b)
        
        losses = compute_loss(rep_b, spg_b, generators, g_map,
                            xyz_map, weights, p_ref0, f0, WP)

        #if step ==1:
            #print(f"spg_b : {spg_b[:10]}, wps_b : {wps_b[:10]}, rep_b : {rep_b[:10]}")
            #print(f"generators : {generators[:10]}, g_map : {g_map[:10]}, xyz_map : {xyz_map[:10]}")
            #print(f"Initial losses: {losses.detach().cpu().numpy().tolist()} \n \n")
            #print(f"total initial loss: {losses.detach().sum().item():.6f}")

        loss_scalar = losses.sum()
        metric = loss_scalar.detach().item()
        
        # Optimization step
        losses.backward(torch.ones_like(losses))
        total_loss = losses.detach().sum()
        
        g = z.grad.view(B, -1)
        # Build row-wise scale from per-sample LR scheduler
        row_scale = None
        if enable_per_sample_lr:
            # Update per-sample scheduler state from current losses
            with torch.no_grad():
                improved = losses < (ps_best_loss - ps_eps)
                ps_best_loss = torch.where(improved, losses, ps_best_loss)
                ps_pat = torch.where(improved, torch.zeros_like(ps_pat), ps_pat + 1)

                # On plateau, reduce effective LR (scale)
                plateau = ps_pat > ps_patience
                if plateau.any():
                    new_scale = (ps_scale * ps_factor).clamp(ps_min_scale, ps_max_scale)
                    ps_scale = torch.where(plateau.to(ps_scale.dtype).bool(), new_scale, ps_scale)
                    ps_pat = torch.where(plateau, torch.zeros_like(ps_pat), ps_pat)

            row_scale = ps_scale.view(B, 1).to(g.device)
        else:
            row_scale = torch.ones((B, 1), dtype=g.dtype, device=g.device)

        # Optional per-sample gradient clipping (after LR scaling)
        if per_sample_clip is not None:
            norms = g.norm(dim=1, keepdim=True).clamp_min(1e-12)
            clip_scale = (float(per_sample_clip) / norms).clamp(max=1.0)
            row_scale = row_scale * clip_scale
        g.mul_(row_scale)
        z.grad.copy_(g.view_as(z))
        optimizer.step()        
        loss_history.append(metric)
        # Progress reporting
        if step % 50 == 0:
            print(f"  Batch {batch_start}-{batch_end}, Step {step}: loss = {metric:.6f}")
        

    
    stage1_elapsed = time.time() - stage1_t0
    print(f"First stage time: {_fmt_time(stage1_elapsed)} (batch {batch_start}-{batch_end})")



    # Second stage optimization: direct representation optimization
    print(f"Starting second stage optimization for batch {batch_start}-{batch_end} (representation parameters)...")

    stage2_t0 = time.time()
    
    rep_opt, loss_opt = optimize_loss(
        spg_b, 
        rep_b, 
        generators, 
        g_map, 
        xyz_map, 
        weights,
        opt_type='Adam',
        lr=2e-3,
        num_steps=num_steps,
        verbose=True,
        per_sample_clip=per_sample_clip,
        enable_per_sample_lr=enable_per_sample_lr,
        ps_patience=ps_patience,
        ps_factor=ps_factor,
        ps_min_scale=ps_min_scale,
        ps_max_scale=ps_max_scale
    )
    
    stage2_elapsed = time.time() - stage2_t0
    print(f"Second stage time: {_fmt_time(stage2_elapsed)} (batch {batch_start}-{batch_end})")
    
    # Evaluate batch results
    wps_list = prepare_wyckoff_list(wps_b, batch_size)
    
    results_dict = {
        "spg": spg_b.cpu().numpy().tolist(),
        "wps": wps_list,
        "rep": rep_opt.cpu().numpy().tolist(),
        "loss": loss_opt.cpu().numpy().tolist()
    }



    return results_dict #, valid_count, xtal_1d_batch 


def run_latent_optimization(
    csv_path='filtered_data/VAE2stg-v4-synthetic-sorted-dis-3rd-run.csv',
    model_path_template='../2_stage_VAE_GMM/VAE_2nd_stage_DiffGMM-DT_e{}_hd1024_b500_KLF1_CLF2_NLF0.1_e250.pt',
    batch_size=1000,
    steps=1000,
    latent_dim=128,
    run=1,
    start=0
):
    """
    Run latent space optimization for crystal structure generation.
    
    Args:
        csv_path: Path to the input CSV file with discrete data
        model_path_template: Template path for VAE model (with {} for latent_dim)
        batch_size: Number of samples per batch
        steps: Number of optimization steps
        latent_dim: Latent dimension for VAE
        run: Run number for output file naming
        start: Starting index for data selection
    
    Returns:
        dict: Combined results containing spg, wps, rep, and loss
    """
    #print args
    print(f"Arguments: csv_path={csv_path}, model_path_template={model_path_template}, batch_size={batch_size}, steps={steps}, latent_dim={latent_dim}, run={run}, start={start}")  
    dis = pd.read_csv(csv_path)
    num = dis.shape[0] 
    print(f"Total samples in dataset: {num}")
    print(f"Using latent dimension: {latent_dim} for this run {run} of latent and representation optimization")
    overall_t0 = time.time()
    setup_environment()

    print(f"Total batches: {(num + batch_size - 1) // batch_size}")

    # Load VAE model
    CVAE_model = CVAE()
    
    try:
        model_path = model_path_template.format(latent_dim)
        CVAE_model.load(model_path)
    except FileNotFoundError:
        print(f"Model file not found at {model_path}, trying local path...")

    print(f"VAE model loaded with latent dimension {latent_dim}")
    
    # Prepare model for inference - freeze decoder and use eval mode
    CVAE_model.decoder.eval()  # Use eval mode for deterministic behavior
    CVAE_model.condition_layer.eval()  
    for param in CVAE_model.decoder.parameters():
        param.requires_grad = False
    for param in CVAE_model.condition_layer.parameters():
        param.requires_grad = False
    

    all_results = []
    # Initialize file for storing all batch losses
    for batch_idx in range(0, num, batch_size):
        batch_start_idx = start + batch_idx
        batch_end_idx = min(start + batch_idx + batch_size, num)
        actual_batch_size = batch_end_idx - batch_start_idx
        
        print(f"\n=== Processing Batch {batch_idx//batch_size + 1}/{(num + batch_size - 1) // batch_size} ===")
        print(f"Indices: {batch_start_idx} to {batch_end_idx} ({actual_batch_size} samples)")
        
        # Get batch data
        dis_batch = dis.iloc[batch_idx:batch_idx + actual_batch_size].copy()
        
        b_t0 = time.time()
        # Process this batch
        batch_results = process_batch(
            dis_batch, 
            batch_start_idx, 
            batch_end_idx, 
            CVAE_model,
            steps,
            ed=latent_dim
        )
        b_elapsed = time.time() - b_t0
        print(f"Batch time: {_fmt_time(b_elapsed)} (indices {batch_start_idx}-{batch_end_idx})")
        
        all_results.append(batch_results)

    # Combine all results and create final summary
    # Save combined results
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


def main():
    """Main function to execute the latent space optimization process."""
    import argparse
    argparse.ArgumentParser(description="Optimize VAE latent vectors for crystal structure generation.")
    # take start , num, batch_size, and steps as command line arguments
    parser = argparse.ArgumentParser(description="Optimize VAE latent vectors for crystal structure generation.")
    parser.add_argument('--start', type=int, default=0, help='Start index for data selection')
    parser.add_argument('--batch_size', type=int, default=1000, help='Number of samples per batch')
    parser.add_argument('--steps', type=int, default=1000, help='Number of optimization steps')
    parser.add_argument('--ed', type=int, default=128, help='Latent dimension for VAE')
    parser.add_argument('--run', type=int, default=1, help='Run number for output file naming')
    args = parser.parse_args()
    
    # Call the function with command line arguments
    combined_results = run_latent_optimization(
        batch_size=args.batch_size,
        steps=args.steps,
        latent_dim=args.ed,
        run=args.run,
        start=args.start
    )


if __name__ == "__main__":
    main()
