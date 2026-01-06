#!/usr/bin/env bash
set -euo pipefail

# Submit GPU stage (Hydrus)
GPU_JOB=$(sbatch --parsable <<'EOF'
#!/bin/bash
#SBATCH --job-name=gen-model
#SBATCH --partition=Hydrus
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=slurm-%x-%j.out

bash run_gen_model.sh
EOF
)
echo "Submitted GPU job: ${GPU_JOB}"

# Submit CPU stage (Apus) after GPU completion
CPU_JOB=$(sbatch --parsable --dependency=afterok:${GPU_JOB} <<'EOF'
#!/bin/bash
#SBATCH --job-name=gen-energy
#SBATCH --partition=Apus
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=96
#SBATCH --time=120:00:00
#SBATCH --mem-per-cpu=2G
#SBATCH --output=slurm-%x-%j.out

bash run_energy.sh
EOF
)
echo "Submitted CPU job (after GPU): ${CPU_JOB}"