import numpy as np
import os
from pyxtal import pyxtal
from lego_torch.batch_sym import Symmetry
from pyxtal.lego.builder import builder
import time

def analyze_crystal_results(combined_results, loss_threshold=1e6, min_loss=-1, 
                          output_dir="results", verbose=True):
    """
    Analyze crystal structure optimization results from combined_results dictionary.
    
    Args:
        combined_results (dict): Dictionary with keys 'spg', 'wps', 'rep', 'loss'
        loss_threshold (float): Maximum loss threshold for valid structures
        min_loss (float): Minimum loss threshold for valid structures  
        output_dir (str): Directory to save analysis results
        verbose (bool): Whether to print progress information
        
    Returns:
        dict: Analysis results containing validity statistics
    """
    if verbose:
        print(f"Analyzing crystal structures...")
        print(f"  Loss range: {min_loss} < loss < {loss_threshold}")
        print(f"  Output directory: {output_dir}")
    
    wyckoff_csv = os.path.join(os.path.dirname(__file__), "wyckoff_list.csv")
    WP = Symmetry(csv_file=wyckoff_csv)
    
    # Setup reference structure and builder
    xtal_ref = pyxtal()
    xtal_ref.from_prototype('graphite')
    cif_file = xtal_ref.to_pymatgen()
    
    os.makedirs(output_dir, exist_ok=True)
    bu = builder(['C'], [1], rank=0, prefix=f'{output_dir}/mof')
    bu.set_descriptor_calculator(mykwargs={'rcut': 2.1})
    bu.set_reference_enviroments(cif_file)
    bu.set_criteria(CN={'C': [3]})
    
    criteria = {
        "CN": {"C": [3]},
        "cutoff": None
    }

    # Validate input data
    lengths = {key: len(combined_results[key]) for key in ['spg', 'wps', 'rep', 'loss']}
    min_length = min(lengths.values())
    
    if verbose:
        print(f"Data lengths: {lengths}")
        print(f"Processing {min_length} structures...")
    
    valid_entries = 0
    total_in_range = 0
    total_structures = min_length
    errors = 0
    valid_xtal_index = []
    for i in range(min_length):
        loss_val = combined_results['loss'][i]
        
        # Check if loss is in specified range
        if min_loss < loss_val < loss_threshold:
            total_in_range += 1
            try:
                xtal = WP.get_pyxtal_from_spg_wps_rep(
                    combined_results['spg'][i], 
                    combined_results['wps'][i], 
                    combined_results['rep'][i]
                )
                
                if xtal.check_validity(criteria, verbose=False):
                    bu.process_xtal(xtal, sim=[0, loss_val])
                    valid_entries += 1
                    valid_xtal_index.append(i)
                    
            except Exception as e:
                errors += 1
                if verbose and errors <= 5:  # Only show first 5 errors
                    print(f"Error processing entry {i}: {e}")
                elif errors == 6:
                    print(f"... suppressing further error messages")
    
    # Calculate statistics
    results = {
        'total_structures': total_structures,
        'structures_in_loss_range': total_in_range,
        'valid_structures': valid_entries,
        'errors': errors,
        'validity_rate': 100 * valid_entries / total_structures if total_structures > 0 else 0,
    }

    #save valid indices
    np.savetxt(f'{output_dir}/valid_xtal_index.txt', np.array(valid_xtal_index), fmt='%d')
    
    if verbose:
        print(f"\n=== ANALYSIS RESULTS ===")
        print(f"Total structures: {results['total_structures']}")
        print(f"Structures in loss range ({min_loss}, {loss_threshold}): {results['structures_in_loss_range']}")
        print(f"Valid structures: {results['valid_structures']}")
        print(f"Processing errors: {results['errors']}")
        print(f"Validity rate: {results['validity_rate']:.2f}%")

    # write metric.txt with times
    with open(f'{output_dir}/metric.txt', 'w') as f:
        f.write(f"Total structures: {results['total_structures']}\n")
        f.write(f"Structures in loss range ({min_loss}, {loss_threshold}): {results['structures_in_loss_range']}\n")
        f.write(f"Valid structures: {results['valid_structures']}\n")
        f.write(f"Processing errors: {results['errors']}\n")
        f.write(f"Validity rate: {results['validity_rate']:.2f}%\n")


    return results, valid_xtal_index


if __name__ == "__main__":
    # Example usage
    import pickle
    import argparse
    import time

    parser = argparse.ArgumentParser(description="Analyze crystal structure optimization results.")
    parser.add_argument('--start', type=int, default=0, help='Start index of samples')
    parser.add_argument('--batch_size', type=int, default=1000, help='Batch size used during optimization')
    parser.add_argument('--num', type=int, default=100000, help='Total number of samples')
    parser.add_argument('--steps', type=int, default=1000, help='Number of optimization steps (for output dir naming)')
    parser.add_argument('--opt_type', type=str, default='2_stage-opt', help='Type of optimization 2stage-opt,rep-opt')
    parser.add_argument('--ed', type=int, default=128, help='Embedding dimension')
    args = parser.parse_args()
    
    # Load example data (adjust path as needed)
    all_results = []
    batch_size= args.batch_size
    start= args.start
    num= args.num
    end= start + num
    ed= args.ed
    type= args.opt_type
    dirname= f'results_{ed}/combined_results_2stage-opt_latent-128_run-1_batch1000.pkl'
    # Combine all results and create final summary
    print(f"\n=== Final Summary ===")
    print(f"Total samples processed: {num}")

    # Save combined results (all indices, including failed ones)
    combined_results = pickle.load(open(dirname, 'rb'))
    print(f" Shape of keys: {[ (key, len(combined_results[key])) for key in combined_results.keys() ]}")
    #output_dir=f"results_opt_{type}_ed{ed}_{start}to{end}_batchsize{batch_size}_steps{args.steps}"
    output_dir=f"results_latent-and-rep-opt_run-1_ed-{ed}_batch{batch_size}"
    os.makedirs(output_dir, exist_ok=True)
    #print(f"final loss array : {combined_results['loss']}")

    analyze_crystal_results(combined_results,output_dir=output_dir)