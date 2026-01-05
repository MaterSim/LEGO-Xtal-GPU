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
    "axes.labelsize": 10,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
})

eng_min = -9.355
eng_max2 = 1.5

colors = ["#66b2b2", "#105656"]

fig, ax = plt.subplots(figsize=(4.4, 1.9), constrained_layout=True)
# Load energies from both databases
ed=128
db_files = [
    #(f"TG-2tsg-ed{ed}.db", f"TorchAG 2stg", colors[0]),
    (f"CVAE_2_100.db", f"CVAE ", colors[0]),
    (f"VAE_2_100.db", f"Base VAE ", colors[1])
]

nbins = 50
bins = np.linspace(0, eng_max2, nbins + 1)

for db_file, label, color in db_files:
    db_conn = db.connect(db_file)
    all_e = [row.mace_energy - eng_min for row in db_conn.select() if hasattr(row, 'mace_energy')]
    data_array = np.array(all_e, dtype=float)
    low = data_array[data_array < eng_max2]
    ax.hist(low, bins=bins, alpha=0.6, color=color, 
    label=f"{label}: {len(low)}"
    #label=f"{label}"
    )
    

# Cutoff threshold for low-energy selection
ax.axvline(0.5, linestyle='--', linewidth=0.4, color='k', alpha=0.8)

ax.legend(loc="upper left", frameon=False)
#ax.set_title(f"TorchAG  vs Base Scipy opt")
ax.grid(False)
ax.set_xlim(-0.005, eng_max2)
ax.set_ylabel("Count")
ax.set_xlabel(r"$\Delta E$ relative to graphite (eV/atom)")
ax.set_ylim(bottom=0)

ax.tick_params(axis="both", which="both",
               bottom=True, top=False, left=True, right=False,
               labelbottom=True, labelleft=True)

ax.tick_params(which="major", length=2.5, width=0.3)

fig.savefig("Fig-4_CVAE-vs-VAE.pdf", bbox_inches="tight")
