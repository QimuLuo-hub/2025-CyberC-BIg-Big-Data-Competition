import os
import numpy as np
import pyvista as pv
from matplotlib.colors import LinearSegmentedColormap

raw_folder = "centered_Au20"
file_name = "350_centered.xyz"
out_root = "example_cluster_visualization"
distance_threshold = 2.834
sphere_radius = 0.45
bond_radius = 0.06
sphere_res = 64
cyl_res = 36
try_enable_depth_peeling = True


screenshot_size = (4000, 4000)  

interactive = True

os.makedirs(out_root, exist_ok=True)


color_list = ["#ffffb2", "#ffb732", "#d95f02"]  
bond_cmap = LinearSegmentedColormap.from_list('bond_by_closeness', color_list)

def length_to_color(lengths, threshold):
    dev = np.abs(lengths - threshold)
    max_dev = dev.max() if len(dev) > 0 else 1.0
    t = dev / max_dev if max_dev > 0 else np.zeros_like(dev)
    closeness = 1.0 - t
    colors = [bond_cmap(c)[:3] for c in closeness]
    return colors

infile = os.path.join(raw_folder, file_name)
if not os.path.exists(infile):
    raise FileNotFoundError(f"{infile} not found.")

with open(infile, 'r') as f:
    lines = f.readlines()
num_atoms = int(lines[0].strip())
coords = np.array([list(map(float, ln.split()[1:4])) for ln in lines[2:2+num_atoms]])

n = coords.shape[0]
bonds, lengths, mids = [], [], []
for i in range(n):
    for j in range(i+1, n):
        L = float(np.linalg.norm(coords[i]-coords[j]))
        if L < distance_threshold:
            bonds.append((i,j))
            lengths.append(L)
            mids.append(0.5*(coords[i]+coords[j]))
lengths = np.array(lengths) if len(lengths) > 0 else np.array([])
colors_rgb = length_to_color(lengths, distance_threshold) if len(lengths)>0 else []

plotter = pv.Plotter(off_screen=not interactive, window_size=screenshot_size)
plotter.set_background('white')  
if try_enable_depth_peeling:
    try:
        plotter.enable_depth_peeling()
    except Exception:
        pass

center = coords.mean(axis=0)
scale = np.linalg.norm(coords.max(axis=0)-coords.min(axis=0))
cam_offset = np.array([1.0,1.0,0.6])
cam_pos = center + cam_offset/np.linalg.norm(cam_offset) * max(scale,1.0)*3.0


plotter.camera_position = [(10.395857086577152, -10.733430431566607, -10.512906672967262),
 (0.31445765640961904, 0.13844373071955185, 0.28101150803587116),
 (0.8353481675415095, 0.3867107160955892, 0.39070226649186324)]

if len(bonds) > 0:
    mids_arr = np.array(mids)
    mid_dists = np.linalg.norm(mids_arr - cam_pos, axis=1)
    order = np.argsort(mid_dists)[::-1]
    for k in order:
        i,j = bonds[k]
        p0,p1 = coords[i], coords[j]
        center_cyl = mids[k]
        direction = p1-p0
        height = np.linalg.norm(direction)
        if height==0: continue
        cyl = pv.Cylinder(center=tuple(center_cyl), direction=direction/height,
                          radius=bond_radius, height=height, resolution=cyl_res)
        plotter.add_mesh(cyl, color=tuple(colors_rgb[k]), opacity=0.98,
                         specular=0.9, diffuse=1.0, smooth_shading=True)

dists_atoms = np.linalg.norm(coords - cam_pos, axis=1)
atom_order = np.argsort(dists_atoms)[::-1]
for i in atom_order:
    sph = pv.Sphere(radius=sphere_radius, center=tuple(coords[i]),
                    theta_resolution=sphere_res, phi_resolution=sphere_res)
    gold_rgb = (1.0, 0.96, 0.17)  
    plotter.add_mesh(sph, color=gold_rgb, opacity=1.0,
                     specular=0.9, specular_power=30, diffuse=1.0, smooth_shading=True)
    wire = pv.Sphere(radius=sphere_radius*1.001, center=tuple(coords[i]),
                     theta_resolution=20, phi_resolution=10)
    plotter.add_mesh(wire, color='black', style='wireframe',
                     opacity=0.12, line_width=0.4)

plotter.add_light(pv.Light(position=(0,0,50), focal_point=center, color='white', intensity=1.2))

out_path = os.path.join(out_root, f"{os.path.splitext(file_name)[0]}_highres.png")

if interactive:
    plotter.show()  
plotter.screenshot(out_path, window_size=screenshot_size)  

print("Current camera position:", plotter.camera_position)
print("Done. Saved:", out_path)
