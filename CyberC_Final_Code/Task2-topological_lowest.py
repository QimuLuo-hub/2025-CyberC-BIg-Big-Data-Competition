import numpy as np
from scipy.spatial.distance import pdist, squareform
import networkx as nx
import matplotlib.pyplot as plt
import os

from matplotlib import rcParams
rcParams['font.family'] = 'cmr10'
rcParams['axes.formatter.use_mathtext'] = True  # Use mathtext for tick labels
rcParams['text.usetex'] = True
# -----------------------------
# 1. read xyz
# -----------------------------
def read_xyz(filename):
    """
    Read xyz file and return atomic coordinate array (N,3)
    """
    coords = []
    with open(filename, 'r') as f:
        lines = f.readlines()
        for line in lines[2:]:  # 前两行是原子数量和注释
            parts = line.split()
            if len(parts) >= 4:
                x, y, z = map(float, parts[1:4])
                coords.append([x, y, z])
    return np.array(coords)


xyz_file = "data/Au20_centered/350_centered.xyz"
coords = read_xyz(xyz_file)
print("load coords:\n", coords)

# -----------------------------
# 2. Constructing the adjacency matrix
# -----------------------------
cutoff = 2.834  # Å，Au-Au 键长阈值
dist_matrix = squareform(pdist(coords))
adj_matrix = (dist_matrix < cutoff) & (dist_matrix > 0)
G = nx.from_numpy_array(adj_matrix)

# -----------------------------
# 3. Topology index calculation
# -----------------------------
# Node degree distribution (coordination number)
degrees = [d for n, d in G.degree()]
print("Coordination number per atom:", degrees)
print("average coordination number:", np.mean(degrees))
print("Coordination number standard deviation:", np.std(degrees))

# Figure diameter
try:
    diameter = nx.diameter(G)
except nx.NetworkXError:
    diameter = np.nan  # graph is not connected
print("Figure diameter:", diameter)

# Average clustering coefficient
avg_clustering = nx.average_clustering(G)
print("Average clustering coefficient:", avg_clustering)
# -----------------------------
# local topology index
# -----------------------------
# Local clustering coefficient of each atom
local_clustering = nx.clustering(G)

print("\n=== local topology index ===")
for i in range(len(coords)):
    print(f"Atom {i}: Coordination number={degrees[i]}, Local clustering coefficient={local_clustering[i]:.3f}")
# -----------------------------
# 4. Find triangle and quadrilateral rings
# -----------------------------
# triangular ring
triangles = [c for c in nx.cycle_basis(G) if len(c) == 3]
print("Number of triangle rings:", len(triangles))

# quadrilateral ring
quads = [c for c in nx.cycle_basis(G) if len(c) == 4]
print("Number of quadrilateral rings:", len(quads))

# The number of rings each atom participates in
atom_ring_count = {i: 0 for i in G.nodes()}
for cycle in nx.cycle_basis(G):
    for node in cycle:
        atom_ring_count[node] += 1
print("\n=== The number of rings each atom participates in: ===")
for i in range(len(coords)):
    print(f"Atom {i}: Coordination number={degrees[i]}, The number of rings each atom participates in:={atom_ring_count[i]:}")
