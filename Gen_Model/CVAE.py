"""
VAE module.
"""
import os
import joblib
import numpy as np
import pandas as pd
import torch
from torch.nn import Linear, Module, Parameter, ReLU, Sequential
from torch.nn.functional import cross_entropy
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from .data_transformer import DataTransformer,ConditionDataset
from .base import BaseSynthesizer, random_state
from torch.utils.tensorboard import SummaryWriter
from .Diff_GMM import GMMDataTransformer

#random.seed(42)
#np.random.seed(42)
#torch.manual_seed(42)

class Encoder(Module):
    """Encoder for the VAE.

    Args:
        data_dim (int): Dimensions of the data.
        compress_dims (tuple or list of ints): Size of each hidden layer.
        embedding_dim (int): Size of the output vector.
    """

    def __init__(self, data_dim, compress_dims, embedding_dim):
        super(Encoder, self).__init__()
        dim = data_dim
        seq = []
        for item in list(compress_dims):
            seq += [Linear(dim, item), ReLU()]
            dim = item

        self.seq = Sequential(*seq)
        self.fc1 = Linear(dim, embedding_dim)
        self.fc2 = Linear(dim, embedding_dim)

    def forward(self, input_):
        """Encode the passed `input_`."""
        feature = self.seq(input_)
        mu = self.fc1(feature)
        logvar = self.fc2(feature)
        std = torch.exp(0.5 * logvar)
        return mu, std, logvar


class Decoder(Module):
    """Decoder for the TVAE.

    Args:
        embedding_dim (int): Size of the input vector.
        decompress_dims (tuple or list of ints): Size of each hidden layer.
        data_dim (int): Dimensions of the data.
    """

    def __init__(self, embedding_dim, decompress_dims, data_dim):
        super(Decoder, self).__init__()
        dim = embedding_dim
        seq = []
        for item in list(decompress_dims):
            seq += [Linear(dim, item), ReLU()]
            dim = item

        seq.append(Linear(dim, data_dim))
        self.seq = Sequential(*seq)
        self.sigma = Parameter(torch.ones(data_dim) * 0.1)

    def forward(self, input_):
        """Decode the passed `input_`."""
        return self.seq(input_), self.sigma
    
class Condition_embedding(Module):
    def __init__(self, input_dim, embedding_dim):
        super(Condition_embedding, self).__init__()
        self.fc = Linear(input_dim, embedding_dim)
        self.act = ReLU()

    def forward(self, input):
        """Encode the passed `input_`."""
        feature = self.fc(input)
        output = self.act(feature)
        return output


class VAE_cont(BaseSynthesizer):
    """
    Variational Autoencoder (VAE) for tabular data.

    Args:
        embedding_dim (int): Size of the output vector.
        compress_dims (tuple or list of ints): Size of each hidden layer.
        decompress_dims (tuple or list of ints): Size of each hidden layer.
        l2scale (float): L2 regularization factor.
        batch_size (int): Batch size for training.
        epochs (int): Number of epochs for training.
        loss_factor (float): Loss factor for the VAE.
        cuda (bool or str): Device to use ('cuda' or 'cpu').
        verbose (bool): Verbosity flag.
        folder (str): Folder to save models and samples.
    """
    def __init__(
        self,
        embedding_dim=128,
        compress_dims=(512, 512),
        decompress_dims=(512, 512),
        l2scale=1e-5,
        batch_size=2000,
        epochs=300,
        loss_factor_KL=2,
        loss_factor_CE=1.0,
        loss_factor_NLL=1.0,
        cuda=True,
        verbose=True,
        type='discrete',
        folder='LEGO-VAE',
    ):
        self.embedding_dim = embedding_dim
        self.compress_dims = compress_dims
        self.decompress_dims = decompress_dims

        self.l2scale = l2scale
        self.batch_size = batch_size
        self.loss_factor_KL = loss_factor_KL
        self.loss_factor_CE = loss_factor_CE 
        self.loss_factor_NLL = loss_factor_NLL
        self.epochs = epochs
        self.loss_values = pd.DataFrame(columns=['Epoch', 'Batch', 'Loss'])
        self.verbose = verbose

        if not cuda or not torch.cuda.is_available():
            device = 'cpu'
        elif isinstance(cuda, str):
            device = cuda
        else:
            device = 'cuda'

        self._device = torch.device(device)
        #store encoded features for last step

        self.type = type
        self.root_folder = folder
        self.type_folder = os.path.join(self.root_folder, self.type)
        self.samples_folder = os.path.join(self.type_folder, 'samples')
        self.model_folder = os.path.join(self.type_folder, 'models')
        os.makedirs(self.root_folder, exist_ok=True)
        os.makedirs(self.type_folder, exist_ok=True)
        os.makedirs(self.samples_folder, exist_ok=True)
        os.makedirs(self.model_folder, exist_ok=True)

    def plot_losses(self, loss_values, filename='average_loss_plot.png'):
        """
        Plot the average loss per epoch.
        """
        import matplotlib.pyplot as plt
        
        # Calculate average loss per epoch
        avg_loss_per_epoch = loss_values.groupby('Epoch')['Loss'].mean()
        
        # Plot the average loss
        plt.figure(figsize=(10, 5))
        plt.plot(avg_loss_per_epoch.index, avg_loss_per_epoch.values, label='Average Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Average Loss')
        plt.title('VAE Training Average Loss per Epoch')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(self.type_folder, filename))
        plt.close()

    def save(self, filepath):
        """
        Save the trained model to a file.
        """
        with open(filepath, 'wb') as f:
            joblib.dump({
                'encoder': self.encoder,
                'decoder': self.decoder,
                'transformer_data': self.transformer_data,
                'transformer_condition': self.transformer_condition,
                'device': self._device,
                'condition_layer': self.condition_layer,
            }, f)

    def load(self, filepath):
        """
        Load a trained model from a file.
        """
        with open(filepath, 'rb') as f:
            state = joblib.load(f)
            self.encoder = state['encoder']
            self.decoder = state['decoder']
            self.transformer_data = state['transformer_data']
            self.transformer_condition = state['transformer_condition']
            self._device = state['device']
            self.condition_layer = state['condition_layer']
            self.encoder.to(self._device)
            self.decoder.to(self._device)
        return self

    def vae_gmm_loss(
        self,
        rep_hat: torch.Tensor,
        rep_true: torch.Tensor,
        sigmas: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor,
        factor_KL: float = 1.0,
        factor_CE: float = 1.0,
        factor_NLL: float = 1.0
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        rep_hat / rep_true : (B, D*(K+1))  – tensors in GMM-representation space
        mu, logvar         : (B, latent_dim) – encoder outputs

        Returns
        -------
        total_loss, ce_gamma, nll_z, kl_latent
        """
        B = rep_true.size(0)
        D = self.transformer_data.D          # number of feature groups
        K = self.transformer_data.K          # one‑hot length (10)
        assert (K + 1) * D == rep_true.size(1), "Dimension mismatch"

        # Clamp σ for numerical stability once
        sigmas = torch.clamp(sigmas, min=1e-3, max=1e2)

        st = 0
        ce_total  = 0.0
        nll_total = 0.0

        for d in range(D):
            # --- categorical CE --------------------------------------
            logits      = rep_hat[:, st : st + K]                 # (B, K)
            target_idx  = rep_true[:, st : st + K].argmax(dim=-1) # (B,)
            ce_total   += cross_entropy(logits, target_idx, reduction='sum')
            # --- residual NLL ----------------------------------------
            res_hat     = torch.tanh(rep_hat[:, st + K])           # (B,)
            res_true    = rep_true[:, st + K]                      # (B,)
            std         = sigmas[st + K]                           # scalar σ_d
            eq          = res_true - res_hat                       # (B,)
            nll_total  += (eq.pow(2) / (2 * std ** 2)).sum()       # Σ (B)
            nll_total  += torch.log(std) * B                       # B·log σ
            st += K + 1

        assert st == rep_true.size(1), "Pointer did not cover all columns"

        # Batch‑average each term
        ce_gamma = ce_total / B
        nll_z    = nll_total / B
        kl_latent = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / B

        total_loss = ce_gamma * factor_CE + \
                     nll_z * factor_NLL + \
                     kl_latent * factor_KL
        

        return total_loss, ce_gamma, nll_z, kl_latent

    @random_state
    def fit(self, train_data, condition, discrete_columns=()):
        """
        Fit the TVAE Synthesizer models to the training data.

        Args:
            train_data (np.ndarray or pd.DataFrame):
                Training Data. It must be a 2-dimensional numpy array or a pandas.DataFrame.
            discrete_columns (list-like):
                List of discrete columns to be used to generate the Conditional
                Vector. If ``train_data`` is a Numpy array, this list should
                contain the integer indices of the columns. Otherwise, if it is
                a ``pandas.DataFrame``, this list should contain the column names.
        """
        train_data=torch.from_numpy(np.array(train_data, dtype=np.float32)).to(self._device)
        train_data_cpu = train_data.detach().cpu()

        self.transformer_data =   GMMDataTransformer(
                                    n_components   = 10,            # K for every feature
                                    n_features     = train_data.shape[1],
                                    gumbel_softmax = True,
                                    tau            = 1.0,
                                ).fit(train_data_cpu)
        self.transformer_data.freeze()
        self.transformer_data = self.transformer_data.to(self._device)

        train_data_T = self.transformer_data(train_data)
        print("Transformed Train Data", np.shape(train_data_T), '\n', train_data_T[:10,:22])

        self.transformer_condition = DataTransformer()
        self.transformer_condition.fit(condition, discrete_columns=["spg", "wp0", "wp1", "wp2", "wp3", "wp4", "wp5", "wp6", "wp7"])
        condition = self.transformer_condition.transform(condition)
        print(condition.shape)
        print(f"Model Hyperparameters: embedding_dim={self.embedding_dim}, compress_dims={self.compress_dims}, decompress_dims={self.decompress_dims}, l2scale={self.l2scale}, batch_size={self.batch_size}, epochs={self.epochs}, loss_factor_KL={self.loss_factor_KL}, loss_factor_CE={self.loss_factor_CE}, loss_factor_NLL={self.loss_factor_NLL}")
        dataset = ConditionDataset(train_data_T, condition)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, drop_last=False) 

        data_dim = train_data_T.shape[1]
        print(f'Input data dimension: {data_dim}')

        condition_dim = self.transformer_condition.output_dimensions
        self.condition_input_dim = condition_dim
        self.condition_output_dim = 128

        self.encoder = Encoder(data_dim, self.compress_dims, self.embedding_dim).to(self._device)
        self.condition_layer = Condition_embedding(self.condition_input_dim, self.condition_output_dim).to(self._device)
        self.decoder = Decoder(self.embedding_dim + self.condition_output_dim, self.decompress_dims, data_dim).to(self._device)
        optimizerAE = Adam(
            list(self.encoder.parameters()) + list(self.condition_layer.parameters()) + list(self.decoder.parameters()), weight_decay=self.l2scale
        )


        self.loss_values = pd.DataFrame(columns=['Epoch', 'Batch', 'Loss'])
        iterator = tqdm(range(self.epochs), disable=(not self.verbose))
        #if self.verbose:
        #    iterator_description = 'Loss: {loss:.3f}'
        #    iterator.set_description(iterator_description.format(loss=0))
        writer = SummaryWriter()
        for i in iterator:
            loss_values = []
            batch = []
            Cross_entropy_loss = 0.0
            kl_loss = 0.0
            nll_total = 0.0
            recon_mse = 0.0
            #reset full encoded features
            for id_, data in enumerate(loader):
                optimizerAE.zero_grad()
                # keep sigma within a reasonable range *before* it is used
                with torch.no_grad():
                    self.decoder.sigma.clamp_(0.01, 1.0)

                first = data[0].to(self._device) #first = cont
                condition_value = data[1].to(self._device).float()   
                mu, std, logvar = self.encoder(first.float())         

                eps = torch.randn_like(std)
                emb = eps * std + mu
                condition_embedding = self.condition_layer(condition_value) # bs, 128
                #print(emb.size(), condition_embedding.size()) # bs, 128
                emb_energy = torch.cat((emb, condition_embedding), 1) #conditional latent
                rec, sigmas = self.decoder(emb_energy)
                loss, CE_loss, nll_val, kld = self.vae_gmm_loss(
                    rec,
                    first,
                    sigmas,
                    mu,
                    logvar,
                    factor_KL=self.loss_factor_KL,
                    factor_CE=self.loss_factor_CE,
                    factor_NLL=self.loss_factor_NLL
                )

                loss.backward()
                '''
                torch.nn.utils.clip_grad_norm_(                           
                    list(self.encoder.parameters()) +
                    list(self.condition_layer.parameters()) +
                    list(self.decoder.parameters()),
                    max_norm=1.0,     
                    )
                ''' 
                optimizerAE.step()
                

                batch.append(id_)
                loss_values.append(loss.detach().cpu().item())
                Cross_entropy_loss += (CE_loss.detach().cpu().item())
                nll_total += nll_val.detach().cpu().item()
                kl_loss += (kld.detach().cpu().item())

            writer.add_scalar('Cross_entropy Loss', Cross_entropy_loss/len(batch), i)
            writer.add_scalar('NLL Loss', nll_total/len(batch), i)
            writer.add_scalar('KL Div', kl_loss/len(batch), i)
            writer.add_scalar('Total Loss', sum(loss_values)/len(batch), i)
            
            print(f'epoch: {i} with Cross_entropy_loss: {Cross_entropy_loss/len(batch)},nll_total: {nll_total/len(batch)}, kld loss: {kl_loss/len(batch)}, and total loss: {sum(loss_values)/len(batch)}')
            # print(f'epoch: {i} with contrastive loss: {contras_loss/len(batch)} and total loss: {sum(loss_values)/len(batch)}')

            epoch_loss_df = pd.DataFrame({
                'Epoch': [i] * len(batch),
                'Batch': batch,
                'Loss': loss_values,
            })
            if not self.loss_values.empty:
                self.loss_values = pd.concat([self.loss_values, epoch_loss_df]).reset_index(
                    drop=True
                )
            else:
                self.loss_values = epoch_loss_df

            #if self.verbose:
            #    iterator.set_description(
            #        iterator_description.format(loss=loss.detach().cpu().item())
            #    )
        writer.close()

    @random_state
    def sample(self, samples, condition_data):
        """
        Sample data similar to the training data.

        Args:
            samples (int): Number of rows to sample.

        Returns:
            torch.Tensor: Sampled data in the original domain.
        """
        # --- sanity -----------------------------------------------------
        if samples > len(condition_data):
            raise ValueError(
                f"Asked to sample {samples} rows but only {len(condition_data)} "
                "conditions were provided."
            )

        # ---------------------------------------------------------------
        # 1. latent prior  ~  N(0, I)
        # ---------------------------------------------------------------
        z = torch.randn(samples, self.embedding_dim, device=self._device)  # (B, Z)

        # ---------------------------------------------------------------
        # 2. condition embedding  (frozen)
        # ---------------------------------------------------------------
        cond_slice = condition_data.iloc[:samples].copy()                     # (B, raw)
        cond_enc   = self.transformer_condition.transform(cond_slice)         # (B, C_cond)
        cond_tensor = torch.from_numpy(cond_enc.astype(np.float32)).to(self._device)
        cond_emb    = self.condition_layer(cond_tensor)                       # (B, 128)

        # ---------------------------------------------------------------
        # 3. decode  (no tanh – keep raw GMM representation)
        # ---------------------------------------------------------------
        latent_input = torch.cat((z, cond_emb), dim=1)                        # (B, Z+128)
        rep_fake, _ = self.decoder(latent_input)              # (B, D*(K+1))

        print(f"rep_fake [10 rows]: {rep_fake[:10,:22]}")
        data = self.transformer_data.inverse(rep_fake)       # (B, D_raw)

        return data


    def set_device(self, device):
        """
        Set the `device` to be used ('cuda' or 'cpu') and move all modules.
        """
        if isinstance(device, str):
            device = torch.device(device)
        self._device = device
        # move all relevant modules
        for module in (getattr(self, 'encoder', None),
                       getattr(self, 'decoder', None),
                       getattr(self, 'condition_layer', None)):
            if module is not None:
                module.to(self._device)
