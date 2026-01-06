import argparse
import pandas as pd
import torch
from Gen_Model.CVAE import CVAE

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
    synthetic_data_reordered_df = pd.DataFrame(
        synthetic_data_reordered.detach().cpu().numpy(),
        columns=['spg', 'a', 'b', 'c', 'alpha', 'beta', 'gamma'] + 
                [col for i in range(8) for col in [f'wp{i}', f'x{i}', f'y{i}', f'z{i}']]
    )
    return synthetic_data_reordered_df

def generate_synthetic_data(
    dis_csv_path: str,
    vae_cont_model_path: str,
    latent_dim: int = 128
):  
    dis = pd.read_csv(dis_csv_path, header=0) #.iloc[:, :1000]
    VAE_cont_model = CVAE()
    VAE_cont_model.load(vae_cont_model_path)
    VAE_cont_model.decoder.train()
    VAE_cont_model.condition_layer.eval()
    for param in VAE_cont_model.decoder.parameters():
        param.requires_grad = False
    for param in VAE_cont_model.condition_layer.parameters():
        param.requires_grad = False

    dis_tensor = torch.tensor(dis.values, dtype=torch.float32, device=VAE_cont_model._device)
    condition = VAE_cont_model.transformer_condition.transform(dis)
    cond_tensor = torch.tensor(condition, dtype=torch.float32, device=VAE_cont_model._device)
    cond_latent = VAE_cont_model.condition_layer(cond_tensor)
    z = torch.randn(len(condition), latent_dim, device=VAE_cont_model._device, requires_grad=True)
    full_latent = torch.cat((z, cond_latent), dim=1)
    decoded, _ = VAE_cont_model.decoder(full_latent)
    output_orig = VAE_cont_model.transformer_data.inverse(decoded)
    dis_tensor_for_assembly = dis_tensor.to(dtype=output_orig.dtype)
    synthetic_data = assemble_data(dis_tensor_for_assembly, output_orig)
    return synthetic_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic data from discrete CSV and trained VAE")
    parser.add_argument(
        "--dis-csv",
        default="TVAE-v4-dis.csv",
        help="Path to discrete-stage CSV input"
    )
    parser.add_argument(
        "--model-path",
        default="data/models/VAE_2nd_stage_DiffGMM-DT_e128_hd1024_b500_KLF1_CLF2_NLF0.1_e250.pt",
        help="Path to trained continuous VAE model"
    )
    parser.add_argument(
        "--latent-dim",
        type=int,
        default=128,
        help="Latent dimension used for sampling"
    )
    parser.add_argument(
        "--output",
        default="CVAE_TVAE_Dis_input-sorted-full.csv",
        help="Output CSV path"
    )
    args = parser.parse_args()

    synthetic_data_df = generate_synthetic_data(
        args.dis_csv,
        args.model_path,
        latent_dim=args.latent_dim
    )
    synthetic_data_df.to_csv(args.output, index=False)
    print(f"Saved {len(synthetic_data_df)} rows to {args.output}")