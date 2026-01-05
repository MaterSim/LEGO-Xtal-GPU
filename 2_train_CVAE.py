
from lego.VAE_cont_Diff_GMM import VAE_cont

import pandas as pd
import torch
import numpy as np
import time

#np.random.seed(42)
#torch.manual_seed(42)
#if torch.cuda.is_available(): torch.cuda.manual_seed(42)

# Start timing
start_time = time.time()
df=pd.read_csv('./data/train/train-v4.csv') #.iloc[:1000]  
#take only 'spg', 'wp1', 'wp2, ... , 'wp7' named columns
cat_cols = ['spg'] + [f'wp{i}' for i in range(0, 8)]
condition= df[cat_cols]
cont_data = df.drop(columns=cat_cols) #+ ['label'] + ['energy'])

#print(df.head())

print("Dis data shape: ", condition.shape)
print("Cont data shape: ", cont_data.shape)
HD=1024
KLF=1
CLF=2
NLF=0.1
ed=512
batch_size=500
epochs=250
synthesizer = VAE_cont()
synthesizer = VAE_cont(embedding_dim=ed,
        compress_dims=(HD, HD),
        decompress_dims=(HD, HD),
        l2scale=1e-5,
        batch_size=2000,
        epochs=epochs,
        loss_factor_KL=KLF,
        loss_factor_CE=CLF,
        loss_factor_NLL= NLF)

fit_start = time.time()
synthesizer.fit(cont_data, condition)
fit_end = time.time()
print(f"VAE fit time: {fit_end - fit_start:.2f} seconds")

filename = f'TVAE-v4-40-dis_VAE_2stage_DiffGMM-DT-NEW-DATASET_e{ed}_hd{HD}_b{batch_size}_KLF{KLF}_CLF{CLF}_NLF{NLF}_e{epochs}.csv'
#print(f"Saving synthesizer to {filename}")
synthesizer.save(f'VAE_2nd_stage_DiffGMM-DT-NEW_DATASET_e{ed}_hd{HD}_b{batch_size}_KLF{KLF}_CLF{CLF}_NLF{NLF}_e{epochs}.pt')
#synthesizer.load(f'VAE_2nd_stage_DiffGMM-DT_e{ed}_hd{HD}_b{batch_size}_KLF{KLF}_CLF{CLF}_NLF{NLF}_e{epochs}.pt')

#dis = pd.read_csv('dis_synthetic_data_100k.csv')
#dis=pd.read_csv("backup/dis_CTGAN_v6_synthetic_data__hd256_e250.csv")
#TVAE-v4-cont_hd_512512_e_500.csv
# Sampling timing
sample_start = time.time()
data=pd.read_csv("TVAE-v4-cont_hd_512512_e_500.csv")
dis=data[cat_cols]
cont = synthesizer.sample(100000,dis)
sample_end = time.time()
print(f"VAE sample time: {sample_end - sample_start:.2f} seconds")
#synthetic_data.to_csv('vae_synthetic_data_1k.csv', index=False)


dis=torch.tensor(dis.values, dtype=torch.float32, device='cuda' if torch.cuda.is_available() else 'cpu')
#cont=torch.tensor(cont, dtype=torch.float32, device='cuda' if torch.cuda.is_available() else 'cpu')
print(f"device dis: {dis.device}, dtype: {dis.dtype}, shape: {dis.shape}")
print(f"device cont: {cont.device}, dtype: {cont.dtype}, shape: {cont.shape}")

#cont is already tensor and dis is tensor, combine cont and dis
synthetic_data = torch.cat((dis, cont), dim=1)
#[spg,a,b,c,alpha,beta,gamma,wp0,x0,y0,z0,wp1,x1,y1,z1,wp2,x2,y2,z2,wp3,x3,y3,z3,wp4,x4,y4,z4,wp5,x5,y5,z5,wp6,x6,y6,z6,wp7,x7,y7,z7] 
#use this column sequence based on the original data
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
synthetic_data_reordered = pd.DataFrame(
    synthetic_data_reordered.detach().cpu().numpy(),
    columns=['spg', 'a', 'b', 'c', 'alpha', 'beta', 'gamma'] + 
            [col for i in range(8) for col in [f'wp{i}', f'x{i}', f'y{i}', f'z{i}']]
)


print(synthetic_data_reordered[:5])
synthetic_data_reordered.to_csv(filename, index=False)
print("Synthetic data saved to ", filename)
end_time = time.time()
print(f"Total script time: {end_time - start_time:.2f} seconds")
