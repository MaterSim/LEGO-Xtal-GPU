from datetime import datetime


def write_metrics_file(metrics_data, output_path='results/metric.txt'):
    """
    Write comprehensive metrics to a text file.
    
    Args:
        metrics_data: Dictionary containing metrics information
        output_path: Path to save the metrics file
    """
    with open(output_path, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("MULTI-STAGE OPTIMIZATION METRICS\n")
        f.write("=" * 60 + "\n")
        f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total Runtime: {metrics_data.get('total_runtime', 'N/A')}\n\n")
        
        # Initial setup info
        f.write("INITIAL SETUP:\n")
        f.write("-" * 20 + "\n")
        f.write(f"Input file: {metrics_data.get('input_file', 'N/A')}\n")
        f.write(f"Model path: {metrics_data.get('model_path', 'N/A')}\n")
        f.write(f"Initial samples: {metrics_data.get('initial_samples', 'N/A')}\n")
        f.write(f"Batch size: {metrics_data.get('batch_size', 'N/A')}\n")
        f.write(f"Optimization steps: {metrics_data.get('steps', 'N/A')}\n")
        f.write(f"Latent dimension: {metrics_data.get('latent_dim', 'N/A')}\n\n")
        
        # Initial representation optimization results
        if 'initial_rep_results' in metrics_data:
            initial = metrics_data['initial_rep_results']
            f.write("INITIAL REPRESENTATION OPTIMIZATION:\n")
            f.write("-" * 40 + "\n")
            f.write(f"Total structures processed: {initial.get('total_structures', 'N/A')}\n")
            f.write(f"Valid structures found: {initial.get('valid_structures', 'N/A')}\n")
            f.write(f"Validity rate: {initial.get('validity_rate', 'N/A'):.2f}%\n")
            f.write(f"Processing errors: {initial.get('errors', 'N/A')}\n\n")
        
        # Iterative optimization results
        f.write("ITERATIVE LATENT OPTIMIZATION RESULTS:\n")
        f.write("-" * 45 + "\n")
        total_runs = metrics_data.get('total_runs', 0)
        f.write(f"Total optimization runs: {total_runs}\n\n")
        
        run_results = metrics_data.get('run_results', [])
        total_valid_found = 0
        total_processed = 0
        total_optimization_time = 0
        
        for i, run_data in enumerate(run_results, 1):
            f.write(f"Run {i}:\n")
            f.write(f"  Samples remaining: {run_data.get('samples_remaining', 'N/A')}\n")
            f.write(f"  Valid structures found: {run_data.get('valid_structures', 'N/A')}\n")
            f.write(f"  Validity rate: {run_data.get('validity_rate', 'N/A'):.2f}%\n")
            f.write(f"  Optimization time: {run_data.get('runtime', 'N/A')}\n")
            if run_data.get('error'):
                f.write(f"  Error: {run_data.get('error')}\n")
            f.write("\n")
            
            if run_data.get('valid_structures') != 'N/A':
                total_valid_found += run_data.get('valid_structures', 0)
                total_processed += run_data.get('samples_remaining', 0)
            
            # Extract time in seconds for total calculation
            runtime_str = run_data.get('runtime', '0s')
            if runtime_str != 'N/A' and 's' in runtime_str:
                try:
                    time_seconds = float(runtime_str.replace('s', ''))
                    total_optimization_time += time_seconds
                except:
                    pass
        
        # Add total optimization time summary
        f.write(f"Total time for all optimization runs: {total_optimization_time:.2f}s ({total_optimization_time/60:.2f} minutes)\n\n")
        
        # Summary statistics
        f.write("SUMMARY STATISTICS:\n")
        f.write("-" * 20 + "\n")
        f.write(f"Total valid structures found across all runs: {total_valid_found}\n")
        f.write(f"Final remaining invalid samples: {metrics_data.get('final_remaining_samples', 'N/A')}\n")
        f.write(f"Overall success rate: {(total_valid_found / metrics_data.get('initial_samples', 1)) * 100:.2f}%\n")
        f.write(f"Reduction in invalid samples: {((metrics_data.get('initial_samples', 0) - metrics_data.get('final_remaining_samples', 0)) / metrics_data.get('initial_samples', 1)) * 100:.2f}%\n\n")
        
        # Timing summary
        f.write("TIMING SUMMARY:\n")
        f.write("-" * 15 + "\n")
        total_runtime_str = metrics_data.get('total_runtime', 'N/A')
        f.write(f"Total pipeline runtime: {total_runtime_str}\n")
        if total_optimization_time > 0:
            f.write(f"Time spent on optimization runs: {total_optimization_time:.2f}s ({total_optimization_time/60:.2f} minutes)\n")
            # Calculate percentage of time spent on optimization
            if total_runtime_str != 'N/A' and 's' in total_runtime_str:
                try:
                    total_seconds = float(total_runtime_str.replace('s', ''))
                    optimization_percentage = (total_optimization_time / total_seconds) * 100
                    f.write(f"Optimization time as % of total: {optimization_percentage:.1f}%\n")
                except:
                    pass
            f.write(f"Average time per optimization run: {total_optimization_time/len(run_results):.2f}s\n")
        f.write("\n")
        
    print(f"Metrics written to {output_path}")
