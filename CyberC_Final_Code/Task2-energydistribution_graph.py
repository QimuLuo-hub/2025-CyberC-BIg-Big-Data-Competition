import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from matplotlib import rcParams
rcParams['font.family'] = 'cmr10'
rcParams['axes.formatter.use_mathtext'] = True  # Use mathtext for tick labels

rcParams['text.usetex'] = True
outdir = "Task2-Figures"
os.makedirs(outdir, exist_ok=True)
# -------------------------------
# load data
# -------------------------------
df = pd.read_excel('atom_energy_summary.xlsx')
energies = df['total energy']

# -------------------------------
# statistics metrics
# -------------------------------
mean_energy = energies.mean()
std_energy = energies.std()
min_energy = energies.min()
max_energy = energies.max()
median_energy = energies.median()

# -------------------------------
# histogram + KDE + shaded mark
# -------------------------------
plt.figure(figsize=(8,6))
sns.histplot(energies, bins=30, kde=True, color='skyblue', edgecolor='black')

# mark mean and lowest
plt.axvline(min_energy, color='blue', linestyle='--', label=f'Min: {min_energy:.2f}')
plt.axvline(mean_energy, color='orange', linestyle='--', label=f'Mean: {mean_energy:.2f}')
plt.axvline(median_energy, color='green', linestyle='--', label=f'Median: {median_energy:.2f}')

# ±1 std dev
# plt.axvline(mean_energy + std_energy, color='orange', linestyle=':', label=f'+1 Std: {mean_energy + std_energy:.2f} eV')
# plt.axvline(mean_energy - std_energy, color='orange', linestyle=':', label=f'-1 Std: {mean_energy - std_energy:.2f} eV')

# shaded area
plt.axvspan(min_energy, mean_energy - std_energy, color='blue', alpha=0.2, label='Low-energy region')
plt.axvspan(mean_energy + std_energy, max_energy, color='red', alpha=0.2, label='High-energy tail')

plt.xlabel('Total Energy')
plt.ylabel('Count')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(outdir, "energy_distribution_updated.png"), dpi=300)
plt.close()
