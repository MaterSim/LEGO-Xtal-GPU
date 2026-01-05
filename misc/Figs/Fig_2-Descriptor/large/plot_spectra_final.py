"""
SO3 Power Spectrum Analysis for Carbon Structures
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from ase.io import read
from pyxtal.lego.SO3 import SO3
from pyxtal import pyxtal
import warnings
import seaborn as sns
warnings.filterwarnings('ignore')


def get_graphite_reference_spectrum(lmax, nmax, rcut, alpha):
    """Compute SO3 power spectrum for graphite sp2 carbon reference"""
    xtal = pyxtal()
    xtal.from_prototype('graphite')
    atoms = xtal.to_ase()
    so3 = SO3(nmax=nmax, lmax=lmax, rcut=rcut, alpha=alpha)
    power_spectrum = so3.compute_p(atoms)
    return np.maximum(power_spectrum[0], 0.01)

def compute_wyckoff_spectra(cif_path, lmax, nmax, rcut, alpha):
    """Compute SO3 power spectrum for each Wyckoff site"""
    atoms = read(str(cif_path))
    xtal = pyxtal()
    xtal.from_seed(str(cif_path))
    
    so3 = SO3(nmax=nmax, lmax=lmax, rcut=rcut, alpha=alpha)
    power_spectrum = so3.compute_p(atoms)
    
    wyckoff_spectra = []
    atom_counter = 0
    for site_idx, site in enumerate(xtal.atom_sites):
        num_atoms = len(site.coords)
        spectrum = np.maximum(power_spectrum[atom_counter], 0.01)
        wyckoff_spectra.append((f"wp{site_idx+1}", spectrum, num_atoms))
        atom_counter += num_atoms
    
    return wyckoff_spectra
def plot_structures(structures, lmax=4, nmax=2, rcut=2.0, alpha=2.0):
    """Plot SO3 power spectra for all structures (one output file per structure)."""
    print("Computing graphite reference...")
    ref_spectrum = get_graphite_reference_spectrum(lmax, nmax, rcut, alpha)
    print("✓ Done\n")

    def _safe_name(s: str) -> str:
        return "".join(ch if ch.isalnum() else "_" for ch in s).strip("_")

    for idx, (name, cif_path) in enumerate(structures.items()):
        cif_path = Path(cif_path)
        print(f"Processing {name}...")

        try:
            wyckoff_spectra = compute_wyckoff_spectra(cif_path, lmax, nmax, rcut, alpha)
            print(f"  {len(wyckoff_spectra)} Wyckoff sites")
            # print(f"  wyckoff_spectra: {wyckoff_spectra}")  # optional noisy debug

            # Cap the max value to 50
            wyckoff_spectra = [(label, np.minimum(spectrum, 50), n_atoms)
                               for label, spectrum, n_atoms in wyckoff_spectra]

            # ---- NEW: create a fresh figure per structure ----
            fig, ax = plt.subplots(1, 1, figsize=(5, 1.9))

            x_indices = np.arange(len(ref_spectrum))

            # Plot graphite reference
            ax.plot(
                x_indices, ref_spectrum,
                marker='s', linestyle='--', linewidth=1, markersize=2,
                label='Ref.', color='gray', alpha=0.8, zorder=1
            )

            # Plot Wyckoff sites
            #colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(wyckoff_spectra)))
            colors = ["#004c4c", "#008080", "#66b2b2", "#b2d8d8"]
            for site_idx, (site_label, spectrum, num_atoms) in enumerate(wyckoff_spectra):
                ax.plot(
                    x_indices, spectrum,
                    marker='o', linestyle='-', linewidth=1, markersize=3,
                    label=f'{site_label}', color=colors[site_idx],
                    alpha=1, zorder=0
                )

            ax.set_xlabel('Coefficient Index', fontsize=10)
            ax.set_ylabel('Power Spectrum', fontsize=10)
            #ax.set_title(f'{name}', fontsize=14)
            if idx == 0:
                ax.legend(loc='upper right', fontsize=8, ncol=1, frameon=True, framealpha=0.5)
            ax.set_yscale('log')

            # Keep your y-limits logic
            if idx == 0:
                ax.set_ylim(0.01, 60)
            else:
                ax.set_ylim(0.01, 8)

            out_name = f"output/SO3_power_spectrum_{idx+1:02d}_{_safe_name(name)}.png"
            fig.tight_layout()
            fig.savefig(out_name, dpi=600, bbox_inches='tight')
            plt.close(fig)

            print(f"  ✓ Saved: {out_name}")
            print(f"  ✓ Done")

        except Exception as e:
            print(f"  ✗ Error: {e}")

    print(f"\n✓ Saved {len(structures)} separate plot files.")
if __name__ == "__main__":
    structures = {
        "Initial": "cifs/init_structure.cif",
        "MACE-FF based Relaxation": "cifs/optimized_structure_maceFF.cif",
        "SO(3) Descriptor based Relaxation": "cifs/optimized_structure.cif",
    }
    
    print("SO(3) Power Spectrum Analysis")
    print("="*60 + "\n")
    plot_structures(structures, lmax=4, nmax=2, rcut=2.0, alpha=2.0)
    print("\n" + "="*60)
    print("Complete!")
