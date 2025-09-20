import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis


df = pd.read_excel("atom_energy_summary.xlsx")

# statistics
mean_energy = df["total energy"].mean()
median_energy = df["total energy"].median()
min_energy = df["total energy"].min()
max_energy = df["total energy"].max()
var_energy = df["total energy"].var()
std_energy = df["total energy"].std()
range_energy = max_energy - min_energy
skew_energy = skew(df["total energy"])
kurt_energy = kurtosis(df["total energy"])

print("Mean:", mean_energy)
print("Median:", median_energy)
print("Min:", min_energy)
print("Max:", max_energy)
print("Variance:", var_energy)
print("Std Dev:", std_energy)
print("Range:", range_energy)
print("Skewness:", skew_energy)
print("Kurtosis:", kurt_energy)

#find  file with the lowest energy
min_file = df.loc[df["total energy"].idxmin(), "id"]
print("Lowest energy structure:", min_file, "Energy:", min_energy)
