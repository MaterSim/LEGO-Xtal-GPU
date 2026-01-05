import ase.db as db
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

sns.set_style("whitegrid")
sns.set_context("paper", font_scale=1.0)

# Ensure vector text embeds cleanly in PDF
plt.rcParams.update({
    "axes.linewidth": 0.6,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "axes.labelsize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
})
eng_min = -9.355
eng_max2 = 1.5

colors = [ "#105656","#66b2b2", "#7be6e6", "#cad5d5"]

fig, ax = plt.subplots(figsize=(6, 2.6), constrained_layout=True)


# Load energies from both databases
ed=128
db_files = [
    (f"final/run3.db", "Run 1-4", colors[3]),   # Lightest - #D2E5E9
    (f"final/run2.db", "Run 1-3", colors[2]),  # Second lightest - #66b2b2
    (f"final/run1.db", "Run 1-2", colors[1]),  # Second darkest - #1673BB
    (f"final/run0.db", "Run 1", colors[0])  # Darkest - #105656  
]

nbins = 50
bins = np.linspace(0, eng_max2, nbins + 1)

for db_file, label, color in db_files:
    db_conn = db.connect(db_file)
    all_e = [row.mace_energy - eng_min for row in db_conn.select() if hasattr(row, 'mace_energy')]
    data_array = np.array(all_e, dtype=float)
    low = data_array[data_array < eng_max2]
    ax.hist(low, bins=bins, alpha=0.6, color=color, label=f"{label} ({len(low)})")

ax.legend(loc=2, frameon=False)
#ax.set_title(f"TorchAG Multi-stage vs Base Scipy opt")
ax.axvline(0.5, linestyle='--', linewidth=0.4, color='k', alpha=0.8)

ax.grid(False)
ax.set_xlim(-0.005, eng_max2)
ax.set_ylabel("Count")
ax.set_xlabel(r"$\Delta E$ relative to graphite (eV/atom)")
ax.set_ylim(bottom=0)
ax.tick_params(axis="both", which="both",
               bottom=True, top=False, left=True, right=False,
               labelbottom=True, labelleft=True)

ax.tick_params(which="major", length=2.5, width=0.3)

fig.savefig(f"Fig-6_multi_opt.pdf", dpi=300)
