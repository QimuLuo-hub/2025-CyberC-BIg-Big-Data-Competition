import numpy as np
import os
from glob import glob

folder = "centered_Au20"  
buffer = 0.2    
percentile = 95  

def nearest_neighbor_distances(coords):
    num_atoms = coords.shape[0]
    nn_distances = []
    for i in range(num_atoms):
        dists = []
        for j in range(num_atoms):
            if i != j:
                dist = np.linalg.norm(coords[i] - coords[j])
                dists.append(dist)
        nn_distances.append(min(dists)) 
    return nn_distances


all_nn_distances = []

xyz_files = glob(os.path.join(folder, "*.xyz"))
if not xyz_files:
    raise Exception(f" {folder} couln't find any xyz files.")

for file_path in xyz_files:
    with open(file_path, 'r') as f:
        lines = f.readlines()

    num_atoms = int(lines[0].strip())
    data_lines = lines[2:2+num_atoms]

    coords = []
    for line in data_lines:
        parts = line.split()
        coords.append([float(parts[1]), float(parts[2]), float(parts[3])])
    coords = np.array(coords)

    nn_dists = nearest_neighbor_distances(coords)
    all_nn_distances.extend(nn_dists)


all_nn_distances = np.array(all_nn_distances)

mean_dist = np.mean(all_nn_distances)
std_dist = np.std(all_nn_distances)
max_dist = np.max(all_nn_distances)
percentile_dist = np.percentile(all_nn_distances, percentile)

cutoff = percentile_dist + buffer

print(f"Nearest-neighbor distance distribution statistics:")
print(f"  Mean = {mean_dist:.3f} Å")
print(f"  Standard deviation = {std_dist:.3f} Å")
print(f"  Maximum = {max_dist:.3f} Å")
print(f"  {percentile}th percentile = {percentile_dist:.3f} Å")
print(f"\nRecommended cutoff = {cutoff:.3f} Å ( = {percentile}th percentile + buffer {buffer})")
