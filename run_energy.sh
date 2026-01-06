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
    'RELAX_NUM': g(['relax', 'num'], 100),
    'RELAX_LOG_DIR': g(['relax', 'log_dir'], 'results'),
    'TOPO_CHUNK': g(['topology', 'chunk_size'], 1000),
    'TOPO_OVERWRITE': g(['topology', 'overwrite'], False),
    'TOPO_WARMUP': g(['topology', 'warmup'], True),
    'ENERGY_MIN': g(['energy', 'min'], -9.4),
    'ENERGY_MAX': g(['energy', 'max'], -8.8),
    'MACE_STEPS_SMALL': g(['mace', 'steps_small'], 250),
    'MACE_STEPS_MEDIUM': g(['mace', 'steps_medium'], 100),
    'MACE_STEPS_LARGE': g(['mace', 'steps_large'], 50),
    'ENERGY_CPU': g(['run_energy', 'cpu'], 96),
    'ENERGY_RANK': g(['run_energy', 'rank'], 0),
    'SKIP_CREATE': g(['run_energy', 'skip_create'], False),
    'SKIP_TOPOLOGY': g(['run_energy', 'skip_topology'], False),
    'SKIP_GULP': g(['run_energy', 'skip_gulp'], False),
    'SKIP_MACE': g(['run_energy', 'skip_mace'], False),
}

for k, v in items.items():
    print(f"{k}={shlex.quote(str(v))}")
PY
)"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export BLIS_NUM_THREADS=1
export MKL_DYNAMIC=FALSE
export OPENBLAS_MAIN_FREE=1

NAME="synthetic_${DIS_MODEL}_dis${DIS_SAMPLE}_CVAE_hd${CVAE_HIDDEN}_e${CVAE_EPOCHS}"
CPU=${CPU:-96}

python 4_energy.py \
    --name "${NAME}" \
    --cpu "${CPU}"
