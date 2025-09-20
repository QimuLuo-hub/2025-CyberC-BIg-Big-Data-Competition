'''
Isomers Illustrations
'''

import re
import os
import numpy as np
import pyvista as pv
from matplotlib.colors import LinearSegmentedColormap
from scipy.spatial import cKDTree
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib import rcParams
rcParams['font.family'] = 'cmr10'
rcParams['axes.formatter.use_mathtext'] = True  

rcParams['text.usetex'] = True
distance_threshold = 2.834
sphere_radius = 0.45
bond_radius = 0.06
sphere_res = 64
cyl_res = 36
try_enable_depth_peeling = True
change_threshold = 0.05
screenshot_size = (4000, 4000)

color_list = ["#ffffb2", "#ffb732", "#d95f02"]
bond_cmap = LinearSegmentedColormap.from_list('bond_by_closeness', color_list)

def extract_index(filename):
    base = os.path.splitext(os.path.basename(filename))[0]
    m = re.match(r"(\d+)", base)
    return int(m.group(1)) if m else base

def length_to_color(lengths, threshold):
    dev = np.abs(lengths - threshold)
    max_dev = dev.max() if len(dev) > 0 else 1.0
    t = dev / max_dev if max_dev > 0 else np.zeros_like(dev)
    closeness = 1.0 - t
    colors = [bond_cmap(c)[:3] for c in closeness]
    return colors

def read_xyz(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    num_atoms = int(lines[0].strip())
    try:
        energy = float(lines[1].strip())
    except:
        energy = np.nan
    coords = np.array([list(map(float, ln.split()[1:4])) for ln in lines[2:2+num_atoms]])
    return coords, energy

def _format_label(name_or_index, energy):
    if name_or_index is None:
        label_prefix = ""
    elif isinstance(name_or_index, (int, np.integer)):
        label_prefix = f"{int(name_or_index)}"
    else:
        label_prefix = str(name_or_index)
    if np.isnan(energy):
        e_str = "E=N/A"
    else:
        e_str = f"E={energy:.6f}"
    if label_prefix == "":
        return e_str
    else:
        return f"{label_prefix}: {e_str}"

def compare_xyz(file_orig, file_perturbed, out_root="new_image_compare",
                distance_threshold=2.834, sphere_radius=0.45, bond_radius=0.06,
                sphere_res=64, cyl_res=36, change_threshold=0.05,
                screenshot_size=(4000,4000), interactive=False,
                energy_orig=None, energy_pert=None,
                index_orig=None, index_pert=None,
                legend_loc='upper right', legend_fontsize=18):

    os.makedirs(out_root, exist_ok=True)

    coords_orig, energy_in_file_orig = read_xyz(file_orig)
    coords_pert, energy_in_file_pert = read_xyz(file_perturbed)

    energy_orig = energy_orig if energy_orig is not None else energy_in_file_orig
    energy_pert = energy_pert if energy_pert is not None else energy_in_file_pert

    n = coords_orig.shape[0]

    bonds, lengths, mids = [], [], []
    for i in range(n):
        for j in range(i+1, n):
            L = float(np.linalg.norm(coords_orig[i] - coords_orig[j]))
            if L < distance_threshold:
                bonds.append((i, j))
                lengths.append(L)
                mids.append(0.5 * (coords_orig[i] + coords_orig[j]))
    lengths = np.array(lengths) if len(lengths) > 0 else np.array([])
    colors_rgb = length_to_color(lengths, distance_threshold) if len(lengths) > 0 else []

    tree_pert = cKDTree(coords_pert)
    distances, indices = tree_pert.query(coords_orig, k=1)
    changed = distances > change_threshold

    plotter = pv.Plotter(off_screen=not interactive, window_size=screenshot_size)
    plotter.set_background('white')
    if try_enable_depth_peeling:
        try:
            plotter.enable_depth_peeling()
        except Exception:
            pass

    center = coords_orig.mean(axis=0)
    scale = np.linalg.norm(coords_orig.max(axis=0) - coords_orig.min(axis=0))
    cam_offset = np.array([1.0, 1.0, 0.6])
    cam_pos = center + cam_offset / np.linalg.norm(cam_offset) * max(scale, 1.0) * 3.0
    plotter.camera_position = [
        (20.167158764024943, 19.355075867195495, -15.122403493733618),
        (0.1813200993705949, 0.13044064535717292, 0.4335708690133987),
        (0.34706448907133985, 0.34488807838486, 0.8721229581047207)
    ]

    if len(bonds) > 0:
        mids_arr = np.array(mids)
        mid_dists = np.linalg.norm(mids_arr - cam_pos, axis=1)
        order = np.argsort(mid_dists)[::-1]
        for k in order:
            i, j = bonds[k]
            p0, p1 = coords_orig[i], coords_orig[j]
            center_cyl = mids[k]
            direction = p1 - p0
            height = np.linalg.norm(direction)
            if height == 0:
                continue
            if changed[i] or changed[j]:
                cyl_color = (1.0, 0.96, 0.17)
                opacity = 0.2
            else:
                cyl_color = tuple(colors_rgb[k]) if len(colors_rgb) > k else (0.8, 0.5, 0.2)
                opacity = 0.98
            cyl = pv.Cylinder(center=tuple(center_cyl), direction=direction/height,
                              radius=bond_radius, height=height, resolution=cyl_res)
            plotter.add_mesh(cyl, color=cyl_color, opacity=opacity,
                             specular=0.9, diffuse=1.0, smooth_shading=True)

    for i in range(n):
        if changed[i]:
            pos_i = coords_pert[indices[i]]
            for j in range(n):
                if i == j:
                    continue
                pos_j = coords_pert[indices[j]] if changed[j] else coords_orig[j]
                if (i, j) in bonds:
                    idx = bonds.index((i, j))
                elif (j, i) in bonds:
                    idx = bonds.index((j, i))
                else:
                    continue
                L0 = lengths[idx]
                direction = pos_j - pos_i
                norm_dir = np.linalg.norm(direction)
                if norm_dir == 0:
                    continue
                direction_unit = direction / norm_dir
                center_cyl = 0.5 * (pos_i + pos_j) + 0.5 * (L0 - norm_dir) * direction_unit
                height = L0
                cyl = pv.Cylinder(center=tuple(center_cyl), direction=direction_unit,
                                  radius=bond_radius * 0.8, height=height, resolution=cyl_res)
                plotter.add_mesh(cyl, color=(1.0, 0.0, 0.0), opacity=0.9,
                                 specular=0.8, diffuse=1.0, smooth_shading=True)

    dists_atoms = np.linalg.norm(coords_orig - cam_pos, axis=1)
    atom_order = np.argsort(dists_atoms)[::-1]

    for i in atom_order:
        if changed[i]:
            pos_orig = coords_orig[i]
            sph = pv.Sphere(radius=sphere_radius, center=tuple(pos_orig),
                            theta_resolution=sphere_res, phi_resolution=sphere_res)
            plotter.add_mesh(sph, color=(1.0, 0.96, 0.17), opacity=0.4,
                             specular=0.9, specular_power=30, diffuse=1.0, smooth_shading=True)
            pos_pert = coords_pert[indices[i]]
            sph2 = pv.Sphere(radius=sphere_radius, center=tuple(pos_pert),
                             theta_resolution=sphere_res, phi_resolution=sphere_res)
            plotter.add_mesh(sph2, color=(1.0, 0.0, 0.0), opacity=1.0,
                             specular=0.9, specular_power=30, diffuse=1.0, smooth_shading=True)
        else:
            pos = coords_orig[i]
            sph = pv.Sphere(radius=sphere_radius, center=tuple(pos),
                            theta_resolution=sphere_res, phi_resolution=sphere_res)
            plotter.add_mesh(sph, color=(1.0, 0.96, 0.17), opacity=1.0,
                             specular=0.9, specular_power=30, diffuse=1.0, smooth_shading=True)

    plotter.add_light(pv.Light(position=(0, 0, 50), focal_point=center, color='white', intensity=1.2))

    out_file = f"compare_{os.path.splitext(os.path.basename(file_orig))[0]}_{os.path.splitext(os.path.basename(file_perturbed))[0]}.png"
    img = plotter.screenshot(return_img=True, window_size=screenshot_size)
    plotter.close()

    im = Image.fromarray(img)

    fig, ax = plt.subplots(figsize=(10, 10), dpi=200)
    ax.imshow(im)
    ax.axis('off')


    label_orig = _format_label(extract_index(file_orig), energy_orig)
    label_pert = _format_label(extract_index(file_perturbed), energy_pert)
    legend_elements = [
        Patch(facecolor='yellow', edgecolor='black', label=label_orig),
        Patch(facecolor='red', edgecolor='black', label=label_pert)
    ]
    ax.legend(
        handles=legend_elements,
        loc='upper right',
        bbox_to_anchor=(0.98, 0.9),
        fontsize=legend_fontsize,
        framealpha=0.9
    )
    out_path = os.path.join(out_root, out_file)
    os.makedirs(out_root, exist_ok=True)
    fig.savefig(out_path, bbox_inches='tight', pad_inches=0.05)
    plt.close(fig)

    print("Saved:", out_path)
    return out_path



if __name__ == "__main__":
    #group 1: Isoenergetic isomers
    # file_orig = "centered_Au20/84_centered.xyz"
    # file_perturbed = "centered_Au20/832_centered.xyz"

    #group 2: Locally perturbed isomers
    file_orig = "centered_Au20/936_centered.xyz"
    file_perturbed = "centered_Au20/825_centered.xyz"
    output_image = compare_xyz(file_orig, file_perturbed, out_root="one_group_compare")
    import sys
    if len(sys.argv) >= 3:
        compare_xyz(sys.argv[1], sys.argv[2])
    else:
        print("Usage: python compare_visualizer.py file_orig.xyz file_perturbed.xyz")
