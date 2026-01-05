'''
load pyxtal to optimize using mace 
init_structure.cif
'''

from pyxtal import pyxtal
from ase.io import write
from ase.optimize import FIRE
from ase.constraints import FixSymmetry
from ase.filters import UnitCellFilter
from mace.calculators import mace_mp
import sys
import time

# Load the initial structure using pyxtal
xtal = pyxtal()
xtal.from_seed('diamond_subgroup.cif')
print("Initial structure:")
print(xtal)
print(f"Initial lattice parameters: {xtal.lattice}")

# Get initial ASE atoms
atoms = xtal.to_ase()
print(f"Number of atoms: {len(atoms)}")

# Set up MACE calculator with float64 for geometry optimization
print("\nSetting up MACE calculator (float64 for stability)...")
calc = mace_mp(
    model="medium",
    dispersion=False,
    default_dtype="float64",  # Important for geometry optimization
    device='cpu'  # Use 'cpu' for macOS compatibility
)
atoms.set_calculator(calc)

# Calculate initial energy
print("Calculating initial energy...")
initial_energy = atoms.get_potential_energy()
print(f"Initial energy: {initial_energy:.4f} eV")
print(f"Energy per atom: {initial_energy/len(atoms):.4f} eV/atom")

# Optimize with symmetry constraints
print("\nOptimizing with MACE (with symmetry constraints)...")
print("Progress will be written to optimization.log")
print()
sys.stdout.flush()

start_time = time.time()

try:
    # Apply symmetry constraints
    atoms.set_constraint(FixSymmetry(atoms))
    
    # Use UnitCellFilter to optimize both positions and cell
    ecf = UnitCellFilter(atoms)
    
    # Use FIRE optimizer
    dyn = FIRE(ecf, logfile='optimization.log')
    dyn.run(fmax=0.05, steps=300)
    
    elapsed_time = time.time() - start_time
    
    # Get final energy
    final_energy = atoms.get_potential_energy()
    print(f"\nOptimization successful!")
    print(f"Optimization time: {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)")
    print(f"Final energy: {final_energy:.4f} eV")
    print(f"Energy per atom: {final_energy/len(atoms):.4f} eV/atom")
    print(f"Energy change: {final_energy - initial_energy:.4f} eV")
    print(f"Final lattice parameters: {atoms.cell.cellpar()}")
    
    # Save optimized structure
    write('optimized_structure_maceFF.cif', atoms)
    print("\nOptimized structure saved to optimized_structure_maceFF.cif")
    
    # Load as pyxtal for analysis
    xtal_opt = pyxtal()
    xtal_opt.from_seed('optimized_structure_maceFF.cif')
    print(f"\nOptimized structure:")
    print(xtal_opt)
    
except Exception as e:
    elapsed_time = time.time() - start_time
    print(f"\nOptimization failed with error: {e}")
    print(f"Time elapsed: {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)")