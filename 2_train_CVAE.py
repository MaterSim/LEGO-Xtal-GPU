
import argparse
import os
import time

import pandas as pd
import torch

from Gen_Model.CVAE import CVAE


def main():
    parser = argparse.ArgumentParser(description="Train CVAE (stage 2) on continuous data conditioned on discrete columns.")
    parser.add_argument("--data", default="./data/train/train-v4.csv", help="Path to training CSV")
    parser.add_argument("--output", default="VAE_2nd_stage_DiffGMM-DT-NEW_DATASET.pt", help="Path to save trained model")
    parser.add_argument("--embedding-dim", type=int, default=512, help="Latent embedding dimension")
    parser.add_argument("--hidden-dim", type=int, default=1024, help="Hidden dimension for encoder/decoder (uses two layers)")
    parser.add_argument("--batch-size", type=int, default=2000, help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=250, help="Number of training epochs")
    parser.add_argument("--kl-factor", type=float, default=1.0, help="KL loss factor")
    parser.add_argument("--ce-factor", type=float, default=2.0, help="Cross-entropy loss factor")
    parser.add_argument("--nll-factor", type=float, default=0.1, help="Negative log-likelihood loss factor")
    args = parser.parse_args()

    start_time = time.time()
    df = pd.read_csv(args.data)

    cat_cols = ["spg"] + [f"wp{i}" for i in range(8)]
    condition = df[cat_cols]
    cont_data = df.drop(columns=cat_cols)

    print(f"Discrete data shape: {condition.shape}")
    print(f"Continuous data shape: {cont_data.shape}")

    synthesizer = CVAE(
        embedding_dim=args.embedding_dim,
        compress_dims=(args.hidden_dim, args.hidden_dim),
        decompress_dims=(args.hidden_dim, args.hidden_dim),
        l2scale=1e-5,
        batch_size=args.batch_size,
        epochs=args.epochs,
        loss_factor_KL=args.kl_factor,
        loss_factor_CE=args.ce_factor,
        loss_factor_NLL=args.nll_factor,
    )

    fit_start = time.time()
    synthesizer.fit(cont_data, condition)
    fit_end = time.time()
    print(f"VAE fit time: {fit_end - fit_start:.2f} seconds")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    synthesizer.save(args.output)
    print(f"Saved trained model to {args.output}")

    end_time = time.time()
    print(f"Total script time: {end_time - start_time:.2f} seconds")


if __name__ == "__main__":
    main()
