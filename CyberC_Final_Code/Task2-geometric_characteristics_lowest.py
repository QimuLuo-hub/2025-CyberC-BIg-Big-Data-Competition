import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import numpy as np
import itertools

folder = "data/Au20_centered"
file_name = "350_centered.xyz"
file_path = os.path.join(folder, file_name)

if not os.path.exists(file_path):
    raise Exception(f"file {file_path} not exists！")

print("selected file:", file_path)


with open(file_path, 'r') as f:
    lines = f.readlines()

num_atoms = int(lines[0].strip())
energy = float(lines[1].strip())
data_lines = lines[2:2+num_atoms]

atoms = []
coords = []

for line in data_lines:
    parts = line.split()
    atoms.append(parts[0])
    coords.append([float(parts[1]), float(parts[2]), float(parts[3])])

coords = np.array(coords)

# -------------------------------------------
def analyze_cluster(coords, bond_cutoff=2.834):
    num_atoms = coords.shape[0]

    # -------------------------------
    # 1. atom pair distance in cluster
    # -------------------------------
    all_distances = []
    for i in range(num_atoms):
        for j in range(i+1, num_atoms):
            dist = np.linalg.norm(coords[i] - coords[j])
            all_distances.append(dist)
    all_distances = np.array(all_distances)

    # cluster statistics
    cluster_stats = {
        "min_dist": all_distances.min(),
        "max_dist": all_distances.max(),
        "mean_dist": all_distances.mean(),
        "std_dist": all_distances.std()
    }

    # -------------------------------
    # 2. bond length (distance < bond_cutoff)
    # -------------------------------
    bond_distances = all_distances[all_distances < bond_cutoff]
    bond_stats = {
        "min_bond": bond_distances.min(),
        "max_bond": bond_distances.max(),
        "mean_bond": bond_distances.mean(),
        "std_bond": bond_distances.std(),
        "num_bonds": len(bond_distances)
    }

    # -------------------------------
    # bond angle
    # -------------------------------
    # (a) initial angle: 3 atoms combination
    all_angles = []
    for i in range(num_atoms):
        for j, k in itertools.combinations(range(num_atoms), 2):
            if i != j and i != k:
                v1 = coords[j] - coords[i]
                v2 = coords[k] - coords[i]
                cos_theta = np.dot(v1, v2) / (np.linalg.norm(v1)*np.linalg.norm(v2))
                angle = np.arccos(np.clip(cos_theta, -1.0, 1.0)) * 180/np.pi
                all_angles.append(angle)
    all_angles = np.array(all_angles)

    angle_stats_all = {
        "min_angle": all_angles.min(),
        "max_angle": all_angles.max(),
        "mean_angle": all_angles.mean(),
        "std_angle": all_angles.std()
    }

    # angle: only consider neighbor
    neighbor_list = []
    for i in range(num_atoms):
        neighbors = []
        for j in range(num_atoms):
            if i != j:
                dist = np.linalg.norm(coords[i] - coords[j])
                if dist < bond_cutoff:
                    neighbors.append(j)
        neighbor_list.append(neighbors)

    filtered_angles = []
    for i, neighbors in enumerate(neighbor_list):
        for j, k in itertools.combinations(neighbors, 2):
            v1 = coords[j] - coords[i]
            v2 = coords[k] - coords[i]
            cos_theta = np.dot(v1, v2) / (np.linalg.norm(v1)*np.linalg.norm(v2))
            angle = np.arccos(np.clip(cos_theta, -1.0, 1.0)) * 180/np.pi
            filtered_angles.append(angle)
    filtered_angles = np.array(filtered_angles)

    angle_stats_filtered = {
        "min_angle": filtered_angles.min(),
        "max_angle": filtered_angles.max(),
        "mean_angle": filtered_angles.mean(),
        "std_angle": filtered_angles.std()
    }

    # -------------------------------
    # 4. coordination_numbers
    # -------------------------------
    coordination_numbers = np.zeros(num_atoms, dtype=int)
    for i in range(num_atoms):
        for j in range(num_atoms):
            if i != j:
                dist = np.linalg.norm(coords[i] - coords[j])
                if dist < bond_cutoff:
                    coordination_numbers[i] += 1

    return {
        "cluster_stats": cluster_stats,
        "bond_stats": bond_stats,
        "angle_stats_all": angle_stats_all,
        "angle_stats_filtered": angle_stats_filtered,
        "coordination_numbers": coordination_numbers
    }


# ===============================
# exampe
# ===============================
# coords = np.random.rand(20, 3) * 10

results = analyze_cluster(coords, bond_cutoff=2.834)

print("\n=== All atom-pair distances within the cluster (global geometry) ===")
print(results["cluster_stats"])

print("\n=== Bond length statistics (cutoff=2.85 Å) ===")
print(results["bond_stats"])

print("\n=== Preliminary bond angle statistics (all triplets) ===")
print(results["angle_stats_all"])

print("\n=== Filtered bond angle statistics (neighbor triplets) ===")
print(results["angle_stats_filtered"])

print("\n=== Atomic coordination number ===")
print(results["coordination_numbers"])