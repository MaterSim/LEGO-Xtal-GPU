#!/usr/bin/env bash
set -euo pipefail

CONFIG_FILE=${CONFIG_FILE:-config/run_gen_model.yaml}

eval "$(python - <<'PY'
import os, sys, yaml, shlex

cfg_path = os.environ.get('CONFIG_FILE', 'config/run_gen_model.yaml')
if not os.path.isfile(cfg_path):
    sys.stderr.write(f"Config not found: {cfg_path}\n")
    sys.exit(1)

with open(cfg_path, 'r', encoding='utf-8') as f:
    cfg = yaml.safe_load(f) or {}

def g(path, default):
    d = cfg
    for k in path:
        if not isinstance(d, dict) or k not in d:
            return default
        d = d[k]
    return d

items = {
    'DATA': g(['data'], './data/train/train-v4.csv'),
    'DIS_MODEL': g(['discrete', 'model'], 'GAN'),
    'DIS_EPOCHS': g(['discrete', 'epochs'], 10),
    'DIS_BATCH': g(['discrete', 'batch'], 500),
    'DIS_SAMPLE': g(['discrete', 'sample'], 100),
    'CVAE_HIDDEN': g(['cvae', 'hidden'], 1024),
    'CVAE_BATCH': g(['cvae', 'batch'], 2048),
    'CVAE_EPOCHS': g(['cvae', 'epochs'], 10),
    'CVAE_OUT': g(['cvae', 'out'], 'models/VAE_stage2.pt'),
    'CVAE_EMBED': g(['cvae', 'embed'], 128),
    'LATENT_DIM': g(['cvae', 'embed'], 128),
    'RELAX_TYPE': g(['relax', 'type'], 'single'),
    'RELAX_NUM': g(['relax', 'num'], 100),
    'RELAX_LOG_DIR': g(['relax', 'log_dir'], 'results'),
}

for k, v in items.items():
    print(f"{k}={shlex.quote(str(v))}")
PY
)"

# Stage 1: discrete model (GAN or VAE)
mkdir -p data/sample
DIS_OUT=data/sample/${DIS_MODEL}_dis${DIS_SAMPLE}.csv

train_Discrete=false

if [ "$train_Discrete" = true ] ; then
    echo "Training Discrete model..."
    python 1_train_Discrete.py \
        --data "$DATA" \
        --model "$DIS_MODEL" \
        --epochs "$DIS_EPOCHS" \
        --nbatch "$DIS_BATCH" \
        --sample "$DIS_SAMPLE" \
        --output_file "$DIS_OUT"
else
    echo "Skipping Discrete training"
fi



# Stage 2: continuous CVAE 
mkdir -p models


train_CVAE=false

if [ "$train_CVAE" = true ] ; then
    echo "Training CVAE model..."
    python 2_train_CVAE.py \
        --data "$DATA" \
        --output "$CVAE_OUT" \
        --embedding-dim "$CVAE_EMBED" \
        --hidden-dim "$CVAE_HIDDEN" \
        --batch-size "$CVAE_BATCH" \
        --epochs "$CVAE_EPOCHS"
else
    echo "Skipping CVAE training"
fi


# Stage 3: sample full structures using discrete CSV + trained CVAE
mkdir -p data/sample
NAME="synthetic_${DIS_MODEL}_dis${DIS_SAMPLE}_CVAE_hd${CVAE_HIDDEN}_e${CVAE_EPOCHS}"
FULL_OUT=data/sample/${NAME}.csv

python Gen_Model/sample_CVAE.py \
	--dis-csv "$DIS_OUT" \
	--model-path "$CVAE_OUT" \
	--latent-dim "$LATENT_DIM" \
	--output "$FULL_OUT"

echo "Full sampled data saved to $FULL_OUT"
echo "Completed sampling full structures using discrete + CVAE model at $(date)"

# GPU Relaxation Stage
mkdir -p "${RELAX_LOG_DIR}/${NAME}_type${RELAX_TYPE}"
LOG_FILE="${RELAX_LOG_DIR}/${NAME}_type${RELAX_TYPE}/log-${NAME}-gpu.txt"
exec > >(tee -a "$LOG_FILE") 2>&1

if [[ "${RELAX_TYPE}" == "multi" ]]; then
    echo "Running multi-optimization relaxation..."
    python a-multi_opt.py \
        --dis-csv "$DIS_OUT" \
        --model-path "$CVAE_OUT" \
        --output-dir "${RELAX_LOG_DIR}/${NAME}_type${RELAX_TYPE}/" \
        --batch-size 1000 \
        --initial-steps 1000 \
        --run-steps 500 \
        --latent-dim "$LATENT_DIM" \
        --total-runs 3
    echo "Multi-optimization completed at $(date)"
else
    echo "Running single-pass GPU pre-relaxation..."
    python 3_relax.py --data_file "${FULL_OUT}" \
     --num "${RELAX_NUM}" \
     --results_dir "${RELAX_LOG_DIR}/${NAME}_type${RELAX_TYPE}" \
     --steps 1000 \
     --batch_size 1000
    echo "GPU relaxation completed at $(date)"
fi
