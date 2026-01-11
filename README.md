# LEGO-Xtal GPU: Local Environment Geometry-Oriented Crystal Generator

LEGO-Xtal (Local Environment Geometry-Oriented Crystal Generator) is designed for the rapid generation of crystal structures for complex materials characterized by well-defined local structural building units (SBUs). By leveraging a small set of known material examples, LEGO-Xtal learns target crystal structures by exploiting crystal symmetry. A generative model navigates the vast combinatorial space of symmetry operations to produce numerous trial structures, which are then optimized to ensure each atom satisfies the desired local atomic environment. This repository is the GPU-accelerated version of LEGO-Xtal, with relaxation and optimization stages running on GPUs for speed. For the original implementation, see https://github.com/MaterSim/LEGO-xtal.



## Workflow

1.  **Training Data Collection and Augmentation:**
    *   **Data Selection:** Curate a dataset of known structures (e.g., 140 sp² carbon structures from the SACADA database).
    *   **Crystal Representation:** Describe each structure using an irreducible crystal representation, encoding space group number, cell parameters, and Wyckoff site information (typically resulting in a ~40-column feature vector).
    *   **Symmetry-Based Data Augmentation:** Generate alternative representations based on group-subgroup relationships to maximize the utility of crystal symmetry. This allows the model to learn from diverse symmetry perspectives and ensures sufficient data for training.

2.  **Generative Model Training:**
    *   **Two-Stage Training Approach:**
        *   **Stage 1 (Discrete):** Train a generative model (GAN or VAE) on discrete columns (space group and Wyckoff positions wp0-wp7) to learn the symmetry distribution.
        *   **Stage 2 (CVAE):** Train a Conditional Variational Autoencoder (CVAE) on continuous parameters (lattice parameters and atomic coordinates), conditioned on the discrete features.
    *   The models use data transformers to normalize and encode features, enabling effective learning of the complex crystal structure space.
    *   After training, sample from the discrete model, then use the CVAE to generate complete synthetic crystal structures.

3.  **Pre-relaxation (GPU):**
    LEGO-Xtal offers two GPU-accelerated optimization modes to refine generated structures:
    
    *   **Single-Pass Optimization (3_relax.py):** 
        *   Fast GPU-accelerated direct optimization of representation parameters using PyTorch.
        *   Minimizes the difference between target local environment (SO3 descriptors) and generated structures.
        *   Ideal for large-scale rapid screening of structures.
    
    *   **Multi-Optimization (a-multi_opt.py):**
        *   Two-stage iterative refinement process:
            1. **Initial Representation Optimization:** Optimize all generated structures in parallel on GPU.
            2. **Latent Space + Representation Optimization:** For remaining invalid structures, iteratively optimize both the CVAE latent vectors and representation parameters.
        *   Removes valid structures after each iteration and focuses computational effort on difficult cases.
        *   Achieves higher success rates through adaptive optimization in both latent and parameter space.

4.  **Energy Ranking and Database (CPU):**
    Post-processing runs on CPU nodes for efficient parallel energy calculations:
    
    *   **Topology Analysis:** Use CrystalNets.jl to identify and classify structural topologies (e.g., dia, pcu, srs).
    *   **Energy Calculations:**
        *   **GULP:** Fast empirical potential energy evaluation for initial screening.
        *   **MACE:** Machine-learned force field energy calculations with three accuracy levels (small, medium, large step sizes).
    *   **Structure Filtering:** Filter structures based on energy thresholds, validity criteria (coordination numbers, overlaps), and uniqueness.
    *   **Database Storage:** Store all results in SQLite databases with ASE format for efficient querying and analysis.
    *   **Low-Energy Selection:** Identify and extract the N lowest-energy unique structures for further analysis or experimental validation. 

By utilizing different local environments as training sources, LEGO-Xtal can rapidly generate high-quality crystal candidates for designing complex materials with target motif. Combining materials science domain knowledge with advanced AI methodologies, LEGO-Xtal aims to advance AI-driven crystal structure generation, paving the way for more efficient materials discovery and design.

---

## Installation

### 1. Install Julia in base
Ensure you have a clean installation of Julia. If you have a broken installation, remove `~/.juliaup` and `~/.julia` first.
```bash
rm -rf ~/.juliaup
rm -rf ~/.julia   
```
To install Juliaup (Still in base):
```bash
curl -fsSL https://install.julialang.org | sh
# Follow the prompts: install for current user -> yes, default location -> yes
```
After installation, reload your shell:
```bash
source ~/.bashrc
julia --version
```

### 2. Set up Conda Environment
Create a new environment and install Python dependencies.
```bash
conda create -n xtal python=3.10.8
conda activate xtal

# Install Python packages
pip install -r requirements.txt
```

### 3. Clean Conflicting Julia Installations
Ensure no conflicting Julia installations exist within the Conda environment.
```bash
conda activate xtal
conda remove julia
rm -rf $CONDA_PREFIX/julia_env
rm -rf $CONDA_PREFIX/share/julia
rm -f  $CONDA_PREFIX/bin/julia
```

### 4. Install Julia Packages
Install `CrystalNets` and `JSON` in the Julia environment.
```bash
julia -e 'import Pkg; Pkg.add(name="CrystalNets", version="0.4.9"); Pkg.add("JSON")'
```

### 5. Verification
To verify that `CrystalNets` is working correctly, first check the Julia script:

```bash
cd util/Julia_Installation/
julia process_topology.jl C.cif
```
*Expected output:* `[{"name":"dia","dim":3,"count":1}]`

Then run the Python test script:

```bash
python test_calculate_topology.py
```
*Expected output:*
```text
Batch output length: 11, expected: 11, elapsed: 13.74s
Row 7: 141 ['32i', '32i', '32i'] dim=3 aaa
Row 8: 155 ['18f', '18f', '18f'] dim=3 aaa
...
```

---

## Usage

LEGO-Xtal provides a unified pipeline controlled via YAML configuration for reproducible workflows.

### 1. Configure Your Pipeline

Edit `config/run_gen_model.yaml` to set your parameters:

```yaml
data:
  train: "data/train/train-v4.csv"

discrete:
  model: "GAN"              # or "VAE"
  epochs: 250
  batch_size: 100
  sample_size: 100

cvae:
  hidden_dims: "1024"       
  batch_size: 500
  epochs: 10
  embed_dim: 128
  output: "models/VAE_stage2.pt"

relax:
  type: "single"            # "single" or "multi"
  num: 10000                # total number of structures to process
  log_dir: "results"

```

### 2. Train Models and Generate Structures

Run the complete training and generation pipeline:

```bash
bash run_gen_model.sh
```

This will:
1. Train the discrete model (Stage 1)
2. Train the CVAE model (Stage 2)
3. Generate synthetic structures (Stage 3)
4. Run GPU pre-relaxation (single or multi-optimization based on config)

### 3. Energy Analysis and Ranking (CPU)

After GPU relaxation, run CPU-based energy analysis and extract low-energy structures:

```bash
bash run_energy.sh
```

This will automatically select the appropriate script based on `relax.type` in your config:
- **Single mode:** Runs `4_energy.py` on relaxed structures
- **Multi mode:** Runs `b-post_process_multi-opt.py` on multi-optimized results

Both compute:
- Topologies with CrystalNets
- GULP energies
- MACE energies at multiple accuracy levels
- Comprehensive metrics
- Extract N lowest-energy structures

### 4. Complete Pipeline with SLURM

For HPC clusters, submit both stages as dependent jobs:

```bash
bash run.sh
```

This submits:
- **GPU job:** Training, generation, and GPU pre-relaxation (`run_gen_model.sh`)
- **CPU job (dependent):** Energy analysis and post-processing on CPU nodes (`run_energy.sh`)

### Individual Scripts

You can also run individual components:

**Train Discrete Model:**
```bash
python 1_train_Discrete.py --data data/train/train-v4.csv --model GAN --epochs 250 --batch 512 --sample 1000
```

**Train CVAE:**
```bash
python 2_train_CVAE.py --dis-csv data/sample/GAN_dis1000.csv --data data/train/train-v4.csv --embed 128 --hidden 1024 --epochs 250 --output models/VAE_stage2.pt
```

**Generate Synthetic Data:**
```bash
python Gen_Model/sample_CVAE.py --dis-csv data/sample/GAN_dis1000.csv --model-path models/VAE_stage2.pt --latent-dim 128 --output data/sample/synthetic_full.csv
```

#### For Single-Pass Mode (relax.type: "single")

**GPU Relaxation:**
```bash
python 3_relax.py --data_file data/sample/test.csv --num 1000 --results_dir results/test
```

**Energy Analysis:**
```bash
python 4_energy.py --name results/test --cpu 32 
```

#### For Multi-Optimization Mode (relax.type: "multi")

**GPU Relaxation (with iterative optimization):**
```bash
python a-multi_opt.py --dis-csv data/sample/test_dis.csv --model-path models/VAE_stage2.pt --output-dir results/test-multi-opt --batch-size 1000 --total-runs 3
```

**CPU Post-Processing and Energy Analysis:**
```bash
python b-post_process_multi-opt.py --name results/test-multi-opt --cpu 32 
```

---

## Output Structure

After running the pipeline, your results directory will contain:

```
results/
└── your_output_dir/
    ├── mof-0.db              # Valid (CN=3) structure database (ASE format)
    ├── metric.txt            # Comprehensive metrics and statistics
    ├── unique_0.db           # Unique Structures database
    └── dump/                 # Intermediate pickle files
```

---

## Citation

If you use LEGO-Xtal-GPU in your research, please cite:

```
@article{ridwan2026crystal,
  title={Crystal Generation using the Fully Differentiable Pipeline and Latent Space Optimization},
  author={Ridwan, Osman Goni and Frapper, Gilles and Xue, Hongfei and Zhu, Qiang},
  journal={arXiv preprint arXiv:2601.04606},
  year={2026}
}
```

---

## License

See [LICENSE](LICENSE) for details.
