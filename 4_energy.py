import argparse
from pyxtal.db import database_topology
import shutil
import os
import pickle
from lego_torch.calc_topology import run_topology_pipeline
from lego_torch.analyze_results_to_db import analyze_crystal_results
import time

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Post-processing pipeline for MOF database relaxation and energy analysis")
    
    # Required arguments
    parser.add_argument("--name", "-n", required=True,
                       help="Results subdirectory name under 'results/' (e.g., 'TVAE_v4-...')")
    
    # Optional arguments
    parser.add_argument("--cpu", "-c", type=int, default=96,
                       help="Number of CPU cores to use (default: 96)")
    parser.add_argument("--rank", type=int, default=0,
                       help="Rank identifier for unique database naming (default: 0)")
    parser.add_argument("--chunk-size", type=int, default=1000,
                       help="Chunk size for topology processing (default: 1000)")
    parser.add_argument("--overwrite", action="store_true",
                       help="Overwrite existing topology calculations")
    parser.add_argument("--warmup", action="store_true", default=True,
                       help="Enable warmup for topology processing (default: True)")
    
    # Energy analysis parameters
    parser.add_argument("--energy-min", type=float, default=-9.4,
                       help="Minimum energy threshold for analysis (default: -9.4)")
    parser.add_argument("--energy-max", type=float, default=-8.8,
                       help="Maximum energy threshold for analysis (default: -8.8)")
    
    # MACE calculation parameters
    parser.add_argument("--mace-steps-small", type=int, default=250,
                       help="MACE steps for structures 0-100 atoms (default: 250)")
    parser.add_argument("--mace-steps-medium", type=int, default=100,
                       help="MACE steps for structures 100-200 atoms (default: 100)")
    parser.add_argument("--mace-steps-large", type=int, default=50,
                       help="MACE steps for structures 200-1000 atoms (default: 50)")
    
    # Skip options
    parser.add_argument("--skip-create", action="store_true",
                        help="Skip MACE energy calculation")
    parser.add_argument("--skip-topology", action="store_true",
                       help="Skip topology processing step")
    parser.add_argument("--skip-gulp", action="store_true",
                       help="Skip GULP energy calculation")
    parser.add_argument("--skip-mace", action="store_true",
                       help="Skip MACE energy calculation")
    
    args = parser.parse_args()

    # Resolve results directory path and ensure it exists
    name = os.path.join('results', args.name)
    os.makedirs(name, exist_ok=True)
    cpu = args.cpu
    rank = args.rank

    print(f"Starting post-relaxation pipeline...")
    print(f"Results directory: {name}")
    print(f"CPU cores: {cpu}")
    print(f"Rank: {rank}")
    dirname = name
    if not args.skip_create:
        print("\nAnalyzing crystal structure optimization results...")
        # Load combined results
        combined_results = pickle.load(open(f"{dirname}/combined_results.pkl", 'rb'))
        results, valid_xtal_index = analyze_crystal_results(combined_results, output_dir=dirname, verbose=True)
        print(f"Analysis results saved to {dirname}")
        # Start a fresh metrics file
        with open(f'{name}/metric.txt', 'w') as f:
            f.write(f"Total structures: {results['total_structures']}\n")
            f.write(f"Valid structures: {results['valid_structures']}\n")

    # Track counts safely for later metrics
    initial_count = 0
    unique_count = 0

    if not args.skip_topology:
        print("\nInitializing database and updating topology...")
        db = database_topology(f"{name}/mof-0.db")
        initial_count = db.db.count()
        print(f"Initial structures in database: {initial_count}")
        t1 = time.time()
        mp = True
        if mp == True:
            # Update topology using the pipeline
            run_topology_pipeline(
                in_db=f"{name}/mof-0.db", 
                out_db=f"{name}/mof-1.db", 
                out_dir=f"{name}/chunks", 
                nprocs=cpu, 
                chunk_size=args.chunk_size, 
                overwrite=args.overwrite, 
                warmup=args.warmup)
        else:
            print("  Processing topology without multiprocessing...")
            db.update_row_topology(overwrite=False,prefix=f"{name}/mof-0")
            db.clean_structures_spg_topology(dim=3)
        # Count unique structures after topology update
        t2 = time.time()
        print(f"Topology update completed in {t2 - t1:.2f} seconds")
    else:
        print("\nSkipping topology processing...")

    if not args.skip_gulp:
        print("\nCalculating GULP energies...")
        # Prefer mof-1.db if it exists, otherwise fall back to mof-0.db
        gulp_db_path = f"{name}/mof-1.db" if os.path.exists(f"{name}/mof-1.db") else f"{name}/mof-0.db"
        if not os.path.exists(gulp_db_path):
            raise FileNotFoundError(f"No database found for GULP in {name} (expected {name}/mof-0.db or {name}/mof-1.db)")
        db = database_topology(gulp_db_path)
        db.update_row_energy('GULP', ncpu=cpu, calc_folder=f"{name}/gulp_0")
        # Write unique DB with rank suffix to match downstream selection
        unique_count = db.get_db_unique(f'{name}/unique_{rank}.db')
    else:
        print("\nSkipping GULP energy calculation...")
    with open(f'{name}/metric.txt', 'a+') as f:
        f.write(f' unique_structures: {unique_count:12d}\n')
    if not args.skip_mace:
        print("\nCalculating MACE energies...")
        # Choose best available DB for MACE if unique set not created yet
        candidate_paths = [f"{name}/unique_{rank}.db", f"{name}/mof-1.db", f"{name}/mof-0.db"]
        target_db = next((p for p in candidate_paths if os.path.exists(p)), None)
        if target_db is None:
            raise FileNotFoundError(f"No database found in {name} (expected one of: {', '.join(candidate_paths)})")

        db_final = database_topology(target_db)

        print(f"  Processing structures with 0-100 atoms ({args.mace_steps_small} steps)...")
        db_final.update_row_energy('MACE', N_atoms=(0, 100), steps=args.mace_steps_small, 
                                   ncpu=cpu, overwrite=False, use_relaxed='ff_relaxed')

        print(f"  Processing structures with 100-200 atoms (step 1, {args.mace_steps_medium} steps)...")
        db_final.update_row_energy('MACE', N_atoms=(100, 200), steps=args.mace_steps_medium, 
                                   ncpu=cpu, overwrite=False, use_relaxed='ff_relaxed')

        print(f"  Processing structures with 100-200 atoms (step 2, {args.mace_steps_medium} steps)...")
        db_final.update_row_energy('MACE', N_atoms=(100, 200), steps=args.mace_steps_medium, 
                                   ncpu=cpu, overwrite=False, use_relaxed='ff_relaxed')

        print(f"  Processing structures with 200-1000 atoms (step 1, {args.mace_steps_large} steps)...")
        db_final.update_row_energy('MACE', N_atoms=(200, 1000), steps=args.mace_steps_large, 
                                   ncpu=cpu, overwrite=False, use_relaxed='ff_relaxed')

        print(f"  Processing structures with 200-1000 atoms (step 2, {args.mace_steps_large} steps)...")
        db_final.update_row_energy('MACE', N_atoms=(200, 1000), steps=args.mace_steps_large, 
                                   ncpu=cpu, overwrite=False, use_relaxed='ff_relaxed')
    else:
        print("\nSkipping MACE energy calculation...")
        # Open whichever DB exists for the final analysis step
        candidate_paths = [f"{name}/unique_{rank}.db", f"{name}/mof-1.db", f"{name}/mof-0.db"]
        target_db = next((p for p in candidate_paths if os.path.exists(p)), None)
        if target_db is None:
            raise FileNotFoundError(f"No database found in {name} (expected one of: {', '.join(candidate_paths)})")
        db_final = database_topology(target_db)

    # Energy analysis
    print(f"\nAnalyzing structures with energy range: {args.energy_min} to {args.energy_max}")
    
    attribute = 'mace_energy'
    N_lowE = 0
    N_lowE_cubic = 0
    
    with open(f'{name}/metric.txt', 'a+') as f:
        f.write("\n=== MACE ENERGY ANALYSIS ===\n")
        f.write(f"Energy range: {args.energy_min} to {args.energy_max}\n")
        
        for row in db_final.db.select():
            if hasattr(row, attribute):
                eng = getattr(row, attribute)
                if args.energy_min < eng < args.energy_max:
                    N_lowE += 1
                    if row.space_group_number >= 195:
                        N_lowE_cubic += 1
        f.write(f'N_lowE_all:      {N_lowE:12d}\n')
        f.write(f'N_lowE_cubic:    {N_lowE_cubic:12d}\n')

    print(f"\n=== FINAL RESULTS ===")
    print(f"Low energy structures ({args.energy_min} to {args.energy_max}): {N_lowE}")
    print(f"Low energy cubic structures: {N_lowE_cubic}")
    print(f"Results saved to: {name}/metric.txt")

if __name__ == "__main__":
    main()
