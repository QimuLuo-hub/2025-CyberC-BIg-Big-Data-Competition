from __future__ import annotations
import os, json, argparse
from typing import Tuple, Dict
import numpy as np
import torch
import numpy as np, json, csv, os

from typing import List


"""
Task 3 – Step 1: Load trained model + find global lowest-energy structure
"""
from Task1_classical import read_xyz_with_energy, numeric_key
from Task1_gnn_weighted import ResidualGNN, AuClusterGraphDataset

def load_model(model_path: str):
    ckpt = torch.load(model_path, map_location='cpu')
    if not isinstance(ckpt, dict) or 'model_state' not in ckpt or 'args' not in ckpt:
        raise RuntimeError("Checkpoint缺少'model_state'或'args'。请用训练脚本重新保存。")
    a: Dict = ckpt['args']
    state = ckpt['model_state']
    # Infer global_dim from checkpoint if a final head was used during training
    inferred_global_dim = 0
    if isinstance(state, dict) and 'final.0.weight' in state:
        w = state['final.0.weight']  # shape: (hidden, 1 + global_dim)
        try:
            in_features = w.shape[1]
        except Exception:
            in_features = int(w.size(1))  # for tensors without .shape
        inferred_global_dim = max(0, in_features - 1)

    #use the model for training（in_dim=rbf；edge_dim=rbf + (use_angle? angle_ka:0)）
    in_dim = int(a.get('rbf', 32))
    edge_dim = in_dim + (int(a.get('angle_ka', 0)) if a.get('use_angle', False) else 0)
    hidden = int(a.get('hidden', 64))
    layers = int(a.get('layers', 4))
    # Use inferred global_dim if available; otherwise fall back to 0
    global_dim = inferred_global_dim

    model = ResidualGNN(in_dim=in_dim, edge_dim=edge_dim, hidden=hidden, layers=layers, global_dim=global_dim)
    model.load_state_dict(state, strict=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device).eval()
    return model, device, a


def scan_lowest_energy(folder: str) -> Tuple[str, float, np.ndarray]:
    files = sorted([f for f in os.listdir(folder) if f.endswith('.xyz')], key=numeric_key)
    if not files:
        raise FileNotFoundError(f"no .xyz file: {folder}")
    best_file, best_E, best_R = None, float('inf'), None
    for fname in files:
        E_label, R = read_xyz_with_energy(os.path.join(folder, fname))
        if E_label < best_E:
            best_file, best_E, best_R = fname, float(E_label), np.asarray(R, dtype=np.float64)
    return best_file, best_E, best_R

# ------------------------------
# Kabsch alignment (for RMSD reporting only)
# ------------------------------

def kabsch_align(P: np.ndarray, Q: np.ndarray) -> tuple[np.ndarray, float]:
    """Align Q onto P (both N×3) via optimal rotation (no scaling), return (Q_aligned, rmsd)."""
    assert P.shape == Q.shape and P.shape[1] == 3
    Pc = P - P.mean(axis=0, keepdims=True)
    Qc = Q - Q.mean(axis=0, keepdims=True)
    C = Qc.T @ Pc
    V, S, Wt = np.linalg.svd(C)
    d = np.sign(np.linalg.det(V @ Wt))
    D = np.eye(3); D[2, 2] = d
    R = V @ D @ Wt
    Q_aligned = Qc @ R
    diff = Pc - Q_aligned
    rmsd = float(np.sqrt((diff * diff).sum() / P.shape[0]))
    return Q_aligned + P.mean(axis=0, keepdims=True), rmsd


def rmsd_after_align(P: np.ndarray, Q: np.ndarray) -> float:
    _, r = kabsch_align(P, Q)
    return r


def min_pair_distance(R: np.ndarray) -> float:
    n = R.shape[0]
    m = float("inf")
    for i in range(n):
        for j in range(i + 1, n):
            d = float(np.linalg.norm(R[i] - R[j]))
            if d < m:
                m = d
    return m


def random_local_perturb(R_ref: np.ndarray, m: float, k_atoms: int, rng: np.random.Generator | None = None) -> np.ndarray:
    """Perturb `k_atoms` random atoms with 3D Gaussian noise of std=m; others unchanged."""
    rng = rng or np.random.default_rng()
    R = R_ref.copy()
    n = R.shape[0]
    k = min(max(1, int(k_atoms)), n)
    idx = rng.choice(n, size=k, replace=False)
    disp = rng.normal(loc=0.0, scale=m, size=(k, 3))
    R[idx] += disp
    return R



def write_xyz(path: str, coords: np.ndarray, comment: str = ""):
    N = coords.shape[0]
    with open(path, 'w') as f:
        f.write(f"{N}\n")
        f.write(comment + "\n")
        for i in range(N):
            x, y, z = coords[i]
            f.write(f"Au {x:.8f} {y:.8f} {z:.8f}\n")

# Build graph (edge_index, edge_attr) and node features x exactly like dataset caching does
def graph_with_node_x(ds: AuClusterGraphDataset, R: np.ndarray):
    edge_index, edge_attr = ds.build_graph(R)
    N = R.shape[0]
    rbf = ds.rbf
    x = torch.zeros((N, rbf), dtype=torch.float32)
    if ds.use_angle:
        for e in range(edge_attr.shape[0]):
            j = int(edge_index[1, e])
            x[j] += edge_attr[e, :rbf]
    else:
        for e in range(edge_attr.shape[0]):
            j = int(edge_index[1, e])
            x[j] += edge_attr[e]
    return x, edge_index, edge_attr

# Try to get baseline (Gupta/SC) energy from the dataset helper
def try_baseline_energy(ds: AuClusterGraphDataset, R: np.ndarray) -> float | None:
    # Prefer dedicated methods if the dataset exposes them
    for name in ['calc_baseline_energy', 'baseline_energy', 'energy_low', 'energy']:
        fn = getattr(ds, name, None)
        if callable(fn):
            try:
                return float(fn(R))
            except Exception:
                pass
    # Or a nested baseline object with an energy-like call
    bl = getattr(ds, 'baseline', None)
    if bl is not None:
        for name in ['energy', '__call__']:
            fn = getattr(bl, name, None)
            if callable(fn):
                try:
                    return float(fn(R))
                except Exception:
                    pass
    return None

def main():
    ap = argparse.ArgumentParser(description='Task 3 – Step 1')
    ap.add_argument('--model_path', type=str, default='out/gnn_delta.pt')
    # ap.add_argument('--folder', type=str, default='data/Au20_OPT_1000', help='folder include Au20 .xyz ')
    ap.add_argument('--folder', type=str, default='data/Au20_centered',help='folder include Au20 .xyz')

    ap.add_argument('--out_json', type=str, default='out/task3_step1_ref.json')
    # Step 2 arguments
    ap.add_argument('--seed', type=int, default=2025)
    ap.add_argument('--magnitudes', type=float, nargs='+', default=[0.01, 0.02, 0.05, 0.10], help='Perturbation std levels (Å)')
    ap.add_argument('--num_per_level', type=int, default=200, help='Samples per magnitude level')
    ap.add_argument('--k_atoms', type=int, default=4, help='#atoms to perturb per sample')
    ap.add_argument('--min_dist', type=float, default=1.9, help='Reject structures with any pair distance < min_dist (Å)')
    ap.add_argument('--out_csv', type=str, default='out/task3_step2_samples.csv')
    ap.add_argument('--save_xyz_dir', type=str, default=None, help='Optional folder to save perturbed xyz files')
    ap.add_argument('--rmsd_preview', type=int, default=5, help='Print first N RMSD values per level for inspection (0 to disable)')
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out_json) or '.', exist_ok=True)

    # 1) load model
    model, device, a = load_model(args.model_path)
    print("[Step1] loaded model already：",
          f"rbf={a.get('rbf')} hidden={a.get('hidden')} layers={a.get('layers')} "
          f"use_angle={a.get('use_angle')} angle_ka={a.get('angle_ka')} angle_pool={a.get('angle_pool')}")

    # confirm the loaded model
    print(f"[Check] Loaded model class: {model.__class__.__name__}")
    if hasattr(model, 'layers'):
        print(f"[Check] #GNN layers = {len(model.layers)} (ckpt says {a.get('layers')})")

    # 2) lowest energy
    ref_file, ref_E, ref_R = scan_lowest_energy(args.folder)
    print(f"[Step1] lowest energy：{ref_file} | E_label = {ref_E:.6f}")

    # save result
    out = {
        'model_path': args.model_path,
        'folder': args.folder,
        'ref_file': ref_file,
        'ref_E_label': ref_E,
        'ref_coords': ref_R.tolist(),
        'model_hparams': {
            'rbf': a.get('rbf'),
            'hidden': a.get('hidden'),
            'layers': a.get('layers'),
            'use_angle': a.get('use_angle'),
            'angle_ka': a.get('angle_ka'),
            'angle_beta': a.get('angle_beta'),
            'angle_pool': a.get('angle_pool'),
        }
    }
    with open(args.out_json, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"[Step1] saved to {args.out_json}")

    # Helper dataset for graph construction and (optional) CSV global features — matches training hyperparams
    baseline = str(a.get('baseline', 'gupta')).lower()
    if baseline == 'gupta':
        params = dict(A=a.get('A'), p=a.get('p'), xi=a.get('xi'), q=a.get('q'), r0=a.get('r0'))
    else:
        params = dict(eps=a.get('eps'), a=a.get('a'), n=a.get('n'), m=a.get('m'), c=a.get('c'))

    ds_helper = AuClusterGraphDataset(
        folder=args.folder, baseline=baseline, params=params, split='train',
        train_ratio=a.get('train_ratio', 0.8), val_ratio=a.get('val_ratio', 0.1), test_ratio=a.get('test_ratio', 0.1),
        cutoff=a.get('cutoff', 6.0), rbf=a.get('rbf', 32), align_on_train=True,
        csv_feats_path=a.get('csv_feats_path'), csv_feats_key=a.get('csv_feats_key'), csv_feat_cols=a.get('csv_feat_cols'), eig_max=a.get('eig_max'),
        use_angle=a.get('use_angle', False), angle_ka=a.get('angle_ka', 8), angle_beta=a.get('angle_beta', 2.0), angle_pool=a.get('angle_pool', 'mean'),
        seed=a.get('seed', 42)
    )

    # Step 3 prep: build reference graph and predict ΔÊ(ref)
    x_ref, eidx_ref, eattr_ref = graph_with_node_x(ds_helper, ref_R)
    g_ref = None
    if ds_helper.csv_feat_dim > 0 and ref_file in ds_helper.csv_feat_map:
        g_ref = torch.from_numpy(ds_helper.csv_feat_map[ref_file])
    with torch.no_grad():
        dE_ref = model(x_ref, eidx_ref, eattr_ref, g_ref)
        dE_ref = float(dE_ref.squeeze().cpu().item())

    # Also compute true ΔE(ref) using ds_helper's tables
    # Try to find ref_file in train, val, or test (should be in one split)
    found_row = None
    for df in [ds_helper.df_train, ds_helper.df_val, ds_helper.df_test]:
        row = df[df["file"] == ref_file]
        if len(row) > 0:
            found_row = row.iloc[0]
            break
    if found_row is not None:
        E_low = float(found_row["E_low"])
        E_low_ref = E_low  # keep for Step 3/4 total-energy delta
        E_label = float(found_row["E_label"])
        E_low_aligned = ds_helper.s * E_low + ds_helper.b
        delta_true_ref = E_label - E_low_aligned
        print(f"[Step3] predicted delta for ref = {dE_ref:.6f} eV | real delta for ref = {delta_true_ref:.6f} eV")
    else:
        print(f"[Step3] predicted delta for ref = {dE_ref:.6f} eV | real delta: ref_file not found in dataset splits")
        E_low_ref = None

    # ------------------------------
    # Step 2: Generate perturbations around the reference structure
    # ------------------------------
    rng = np.random.default_rng(args.seed)
    rows: List[dict] = []

    if args.save_xyz_dir:
        os.makedirs(args.save_xyz_dir, exist_ok=True)

    for m in args.magnitudes:
        kept = 0
        trials = 0
        while kept < args.num_per_level and trials < args.num_per_level * 50:
            trials += 1
            Rp = random_local_perturb(ref_R, m=float(m), k_atoms=args.k_atoms, rng=rng)
            dmin = min_pair_distance(Rp)
            if dmin < args.min_dist:
                continue  # reject physically implausible overlap
            rmsd = rmsd_after_align(ref_R, Rp)

            # --- Step 3: model inference for ΔΔÊ ---
            x_p, eidx_p, eattr_p = graph_with_node_x(ds_helper, Rp)
            g_p = g_ref if g_ref is not None else None  # CSV globals are reference-level; reuse
            with torch.no_grad():
                dE_p = model(x_p, eidx_p, eattr_p, g_p)
                dE_p = float(dE_p.squeeze().cpu().item())
            dE_change = dE_p - dE_ref  # ΔΔÊ

            # Total predicted energy change: s*(E_low(pert)-E_low(ref)) + ΔΔÊ
            dE_total_change = None
            if E_low_ref is not None:
                E_low_pert = try_baseline_energy(ds_helper, Rp)
                if E_low_pert is not None:
                    dE_total_change = ds_helper.s * (E_low_pert - E_low_ref) + dE_change

            # Stability metrics (per-Å response)
            stiffness_resid = abs(dE_change) / rmsd if rmsd > 0 else float('nan')
            stiffness_total = (abs(dE_total_change) / rmsd) if (rmsd > 0 and dE_total_change is not None) else float('nan')

            rows.append(dict(level=float(m), sample_id=kept, rmsd=rmsd, min_dist=dmin,
                              deltaE_change=dE_change,
                              deltaE_total_change=dE_total_change if dE_total_change is not None else float('nan'),
                              stiffness_resid=stiffness_resid,
                              stiffness_total=stiffness_total))

            # Save one .xyz for illustration if m == 0.10 and kept == 0
            if float(m) == 0.10 and kept == 0:
                write_xyz(f"example_m={m}.xyz", Rp, comment=f"ref={ref_file} m={m} rmsd={rmsd:.6f}")

            if args.save_xyz_dir:
                out_name = f"{os.path.splitext(ref_file)[0]}_m{m:.3f}_{kept:04d}.xyz"
                write_xyz(os.path.join(args.save_xyz_dir, out_name), Rp, comment=f"ref={ref_file} m={m} rmsd={rmsd:.6f}")
            kept += 1

    # Quick RMSD stats and preview
    if args.rmsd_preview > 0:
        print("[RMSD] Per-level stats and first samples:")
        from math import inf
        for m in args.magnitudes:
            mv = float(m)
            rmsd_vals = [r['rmsd'] for r in rows if float(r['level']) == mv]
            if not rmsd_vals:
                continue
            rmsd_vals_sorted = sorted(rmsd_vals)
            mean_r = float(np.mean(rmsd_vals))
            min_r = float(rmsd_vals_sorted[0])
            max_r = float(rmsd_vals_sorted[-1])
            print(f"  m={m}: mean_RMSD={mean_r:.6f} Å | min={min_r:.6f} | max={max_r:.6f} | n={len(rmsd_vals)}")
            # preview first N entries (by insertion order)
            preview = [r['rmsd'] for r in rows if float(r['level']) == mv][:args.rmsd_preview]
            print(" preview:", ", ".join(f"{v:.6f}" for v in preview))

    # ------------------------------
    # Step 4: Aggregate MAE / RMSE per magnitude level
    # ------------------------------
    import math
    summary = {}
    for m in args.magnitudes:
        mv = float(m)
        vals = [abs(r['deltaE_change']) for r in rows if float(r['level']) == mv]
        sqs = [r['deltaE_change'] ** 2 for r in rows if float(r['level']) == mv]
        rmsd_vals = [r['rmsd'] for r in rows if float(r['level']) == mv]
        if not vals:
            continue
        mae = float(np.mean(vals))
        rmse = float(math.sqrt(np.mean(sqs)))
        rmsd_mean = float(np.mean(rmsd_vals))
        summary[str(m)] = dict(MAE=mae, RMSE=rmse, mean_RMSD=rmsd_mean, n=len(vals))

    summ_path = os.path.splitext(args.out_csv)[0] + '_summary.json'
    with open(summ_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"[Step4] Summary saved → {summ_path}")
    for m, d in summary.items():
        print(
            f"  m={m}: MAE={d['MAE']:.6f} eV | RMSE={d['RMSE']:.6f} eV | mean_RMSD={d['mean_RMSD']:.6f} Å | n={d['n']}")

    # Save CSV with per-sample stats
    os.makedirs(os.path.dirname(args.out_csv) or '.', exist_ok=True)
    import csv
    with open(args.out_csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['level', 'sample_id', 'rmsd', 'min_dist', 'deltaE_change', 'deltaE_total_change', 'stiffness_resid', 'stiffness_total'])
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"[Step2] Generated {len(rows)} samples → {args.out_csv}")
    if args.save_xyz_dir:
        print(f"[Step2] XYZ files saved under: {args.save_xyz_dir}")

    # ---- Stiffness aggregation across perturbation magnitudes (add after rows are collected) ----

    def aggregate_stiffness(rows, magnitudes, out_prefix: str):
        """
        Aggregate stiffness per magnitude m:
          kappa_resid  = |ΔΔÊ| / RMSD
          kappa_total  = |ΔE_total| / RMSD
        Expects each row to have fields:
          'level', 'stiffness_resid', 'stiffness_total'
        Saves: <out_prefix>_stiffness_summary.json and .csv
        Returns: dict summary
        """

        def summarize(arr: np.ndarray):
            arr = arr[np.isfinite(arr)]
            if arr.size == 0:
                return dict(n=0, mean=float('nan'), median=float('nan'),
                            p10=float('nan'), p90=float('nan'), max=float('nan'))
            return dict(
                n=int(arr.size),
                mean=float(np.mean(arr)),
                median=float(np.median(arr)),
                p10=float(np.percentile(arr, 10)),
                p90=float(np.percentile(arr, 90)),
                max=float(np.max(arr)),
            )

        stats = {}
        for m in magnitudes:
            m_val = float(m)
            lvl = [r for r in rows if float(r['level']) == m_val]
            k_resid = np.array([r['stiffness_resid'] for r in lvl], dtype=float)
            k_total = np.array([r['stiffness_total'] for r in lvl],
                               dtype=float)  # may include NaN if baseline unavailable

            stats[str(m_val)] = {
                'resid': summarize(k_resid),
                'total': summarize(k_total),
            }

        # Save JSON
        json_path = f"{out_prefix}_stiffness_summary.json"
        with open(json_path, 'w') as f:
            json.dump(stats, f, indent=2)
        print(f"[Step4] Stiffness summary saved → {json_path}")

        # Save CSV
        csv_path = f"{out_prefix}_stiffness_summary.csv"
        with open(csv_path, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['m', 'type', 'n', 'mean', 'median', 'p10', 'p90', 'max'])
            for m in magnitudes:
                m_key = str(float(m))
                for typ in ('resid', 'total'):
                    s = stats[m_key][typ]
                    w.writerow([m, typ, s['n'], s['mean'], s['median'], s['p10'], s['p90'], s['max']])
        print(f"[Step4] Stiffness CSV saved → {csv_path}")
        return stats

    # --- call it (use the same prefix as your other summaries) ---
    stiff_stats = aggregate_stiffness(
        rows,
        args.magnitudes,
        os.path.splitext(args.out_csv)[0]  # e.g., 'out/task3_step2_samples'
    )


main()
