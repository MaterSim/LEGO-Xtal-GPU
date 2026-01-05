# pip install pyxtal
from pyxtal.lego.builder import builder
from pyxtal import pyxtal
from mace.calculators import mace_mp
import time
# Get the graphite reference environment and set up optimizer
xtal = pyxtal()
xtal.from_prototype('graphite')
cif_file = xtal.to_pymatgen()

bu = builder(['C'], [1], db_file='test.db')
bu.set_descriptor_calculator(mykwargs={'rcut': 2.0})
bu.set_reference_enviroments(cif_file)
print(bu)


init_file = 'cifs/init_structure.cif'
t1 = time.time()
xtal = pyxtal()
xtal.from_seed(init_file)
print(xtal)

# Calculate single point energy using MACE
print("\nCalculating initial energy with MACE...")
atoms_init = xtal.to_ase()
calc = mace_mp(
    model="medium",
    dispersion=False,
    default_dtype="float64",
    device='cpu'
)
atoms_init.set_calculator(calc)
initial_energy = atoms_init.get_potential_energy()
print(f"Initial MACE energy: {initial_energy:.4f} eV")
print(f"Initial energy per atom: {initial_energy/len(atoms_init):.4f} eV/atom\n")

xtal_opt, sim1, xs = bu.optimize_xtal(xtal, minimizers=[('Nelder-Mead', 300), ('L-BFGS-B', 200), ('L-BFGS-B', 200)])
t2 = time.time()
print('Time taken for optimization:', t2 - t1, 'seconds')
print(xtal_opt)

# Calculate final energy using MACE
print("\nCalculating final energy with MACE...")
atoms_final = xtal_opt.to_ase()
atoms_final.set_calculator(calc)
final_energy = atoms_final.get_potential_energy()
print(f"Final MACE energy: {final_energy:.4f} eV")
print(f"Final energy per atom: {final_energy/len(atoms_final):.4f} eV/atom")
print(f"Energy change: {final_energy - initial_energy:.4f} eV")
print(f"Energy change per atom: {(final_energy - initial_energy)/len(atoms_final):.4f} eV/atom\n")

xtal_opt.to_file('optimized_structure.cif')

# Do 100 steps relaxation with MACE model
print("\nPerforming additional MACE relaxation (50 steps)...")
from ase.optimize import FIRE
from ase.constraints import FixSymmetry
from ase.filters import UnitCellFilter

t3 = time.time()

try:
    atoms_mace = xtal_opt.to_ase()
    atoms_mace.set_calculator(calc)
    
    # Apply symmetry constraints
    atoms_mace.set_constraint(FixSymmetry(atoms_mace))
    
    # Use UnitCellFilter to optimize both positions and cell
    ecf = UnitCellFilter(atoms_mace)
    
    # Use FIRE optimizer for 100 steps
    dyn = FIRE(ecf, logfile='mace_refine.log')
    dyn.run(fmax=0.05, steps=50)
    
    t4 = time.time()
    
    # Get final MACE energy after refinement
    mace_refined_energy = atoms_mace.get_potential_energy()
    print(f"\nMACE refinement complete!")
    print(f"Refinement time: {t4 - t3:.2f} seconds")
    print(f"MACE refined energy: {mace_refined_energy:.4f} eV")
    print(f"Energy per atom: {mace_refined_energy/len(atoms_mace):.4f} eV/atom")
    print(f"Energy change from SO3: {mace_refined_energy - final_energy:.4f} eV")
    print(f"Total energy change: {mace_refined_energy - initial_energy:.4f} eV")
    
    # Save MACE-refined structure
    from ase.io import write
    write('optimized_structure_SO3_MACE.cif', atoms_mace)
    print("\nMACE-refined structure saved to optimized_structure_SO3_MACE.cif")
    
except Exception as e:
    print(f"\nMACE refinement failed: {e}")
