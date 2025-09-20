import os
import math
import argparse
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

# Import only functions from classical; the main block there is guarded
from Task1_classical import read_xyz_with_energy, numeric_key, gupta_energy_forces, suttonchen_energy_forces


# ------------------------------
# Utility: deterministic behavior
# ------------------------------
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# ------------------------------
# Dataset & featurization
# ------------------------------
class AuClusterDataset(torch.utils.data.Dataset):
    def __init__(self, folder: str, baseline: str, params: dict, split: str,
                 train_ratio: float = 0.8, val_ratio: float = 0.1, test_ratio: float = 0.1,
                 cutoff: float = 6.0, rbf: int = 32,
                 align_on_train: bool = True,
                 use_forces: bool = False, forces_npz: str = None, scale_forces_with_s: bool = False,
                 csv_feats_path: str | None = None,
                 use_angle: bool = False, angle_ka: int = 8, angle_beta: float = 2.0, angle_pool: str = 'mean',
                 csv_feat_cols: str | None = None):

        super().__init__()
        self.folder = folder
        self.baseline = baseline.lower()
        self.params = params
        self.cutoff = cutoff
        self.rbf = rbf
        self.use_forces = use_forces
        self.scale_forces_with_s = scale_forces_with_s
        self.csv_feats_path = csv_feats_path
        self.csv_feat_cols = csv_feat_cols

        self.use_angle = bool(use_angle)
        self.angle_ka = int(angle_ka)
        self.angle_beta = float(angle_beta)
        self.angle_pool = str(angle_pool)
        # Load XYZ files with natural numeric order
        files = sorted([f for f in os.listdir(folder) if f.endswith('.xyz')], key=numeric_key)

        # Read all structures
        data = []
        for fname in files:
            fpath = os.path.join(folder, fname)
            E_label, R = read_xyz_with_energy(fpath)
            # Compute baseline energy
            if self.baseline == 'gupta':
                E_low, _ = gupta_energy_forces(R, **self.params)
            elif self.baseline in ('suttonchen', 'sc'):
                E_low, _ = suttonchen_energy_forces(R, **self.params)
            else:
                raise ValueError(f"Unknown baseline: {self.baseline}")
            data.append({'file': fname, 'E_label': E_label, 'R': R, 'E_low': E_low})
        df = pd.DataFrame(data)
        # Reproducible shuffle before splitting to avoid biased splits
        rng = np.random.default_rng(42)
        perm = rng.permutation(len(df))
        df = df.iloc[perm].reset_index(drop=True)

        # Optional CSV features (structure-level).
        self.csv_feat_map = {}
        if csv_feats_path:
            feats_df = pd.read_csv(csv_feats_path)
            # normalize column names
            feats_df.columns = [str(c).strip() for c in feats_df.columns]

            # Choose feature columns
            if self.csv_feat_cols is not None:
                # user-specified column list (comma-separated string)
                col_list = [c.strip() for c in self.csv_feat_cols.split(',') if c.strip()]
                missing = [c for c in col_list if c not in feats_df.columns]
                if missing:
                    raise ValueError(f"csv_feat_cols not found in CSV columns: {missing}")
                feat_cols = col_list
            else:
                # default: all numeric columns except the last numeric (treated as energy/target)
                numeric_cols = [c for c in feats_df.columns if np.issubdtype(feats_df[c].dtype, np.number)]
                if len(numeric_cols) < 2:
                    raise ValueError(
                        "CSV must have at least two numeric columns so we can drop the last (energy) and keep features.")
                feat_cols = numeric_cols[:-1]

                # optional eigen trimming (if provided elsewhere)
                if hasattr(self, 'eig_max') and self.eig_max is not None:
                    eig_cols = [c for c in feat_cols if str(c).lower().startswith('eig')]
                    other_cols = [c for c in feat_cols if not str(c).lower().startswith('eig')]
                    eig_cols_sorted = sorted(eig_cols,
                                             key=lambda s: int(''.join(ch for ch in str(s) if ch.isdigit()) or 0))
                    eig_cols_trimmed = eig_cols_sorted[:self.eig_max]
                    feat_cols = other_cols + eig_cols_trimmed

            # Map rows in order: row i -> file i.xyz (keep existing behavior)
            files_sorted = sorted([f for f in os.listdir(folder) if f.endswith('.xyz')], key=numeric_key)
            if len(files_sorted) != len(feats_df):
                raise ValueError(f"Row count in CSV ({len(feats_df)}) != number of .xyz files ({len(files_sorted)}).")
            for fname, (_, r) in zip(files_sorted, feats_df.iterrows()):
                vec = np.asarray([r[c] for c in feat_cols], dtype=np.float32)
                self.csv_feat_map[fname] = vec

            self.csv_feat_dim = len(feat_cols)
        else:
            self.csv_feat_dim = 0


        # Train/val/test split
        n = len(df)
        if not (0.0 < train_ratio < 1.0):
            raise ValueError("train_ratio must be in (0,1)")
        if val_ratio < 0 or test_ratio < 0 or (train_ratio + val_ratio + test_ratio > 1.0 + 1e-8):
            raise ValueError("val_ratio and test_ratio must be >=0 and train+val+test <= 1")
        n_train = int(round(train_ratio * n))
        n_val = int(round(val_ratio * n))
        idx_train_end = n_train
        idx_val_end = n_train + n_val
        self.df_train = df.iloc[:idx_train_end].reset_index(drop=True)
        self.df_val = df.iloc[idx_train_end:idx_val_end].reset_index(drop=True)
        self.df_test = df.iloc[idx_val_end:].reset_index(drop=True)
        # Fit alignment (s, b) on train only
        if align_on_train:
            x = self.df_train['E_low'].values
            y = self.df_train['E_label'].values
            A = np.vstack([x, np.ones_like(x)]).T
            s, b = np.linalg.lstsq(A, y, rcond=None)[0]
            self.s, self.b = float(s), float(b)
        else:
            self.s, self.b = 1.0, 0.0
        # Build examples for the chosen split
        if split == 'train':
            self.table = self.df_train
        elif split == 'val':
            self.table = self.df_val
        elif split == 'test':
            self.table = self.df_test
        else:
            raise ValueError("split must be 'train', 'val', or 'test'")
        # Precompute RBF centers
        self.mu = np.linspace(0.0, cutoff, rbf, dtype=np.float32)
        self.beta = np.float32(1.0)  # width; can be tuned

        # Forces loading if requested
        self.forces_map = {}
        if self.use_forces:
            if not forces_npz:
                raise ValueError("--use_forces requires --forces_npz path to an .npz mapping filename->(N,3) array")
            fm = np.load(forces_npz, allow_pickle=True)
            # support both dict-like and arrays; if not dict, assume 'files' and 'forces' arrays
            if isinstance(fm, np.lib.npyio.NpzFile) and set(fm.files) == {"files","forces"}:
                files_arr = fm["files"]
                forces_arr = fm["forces"]
                for k, F in zip(files_arr, forces_arr):
                    self.forces_map[str(k)] = np.array(F, dtype=np.float32)
            else:
                for k in fm.files:
                    self.forces_map[str(k)] = np.array(fm[k], dtype=np.float32)

        # --------------------------------------------------------------
        # Cache: precompute fingerprints (and residual forces if needed)
        # --------------------------------------------------------------
        self.fp_cache = {}
        self.force_res_cache = {} if self.use_forces else None
        for _, row in self.table.iterrows():
            fname = row['file']
            R = row['R']
            # fingerprints once
            fp = self.radial_fingerprint(R)  # (N, K)
            self.fp_cache[fname] = fp
            # optional residual forces once
            if self.use_forces:
                if self.baseline == 'gupta':
                    _, F_low = gupta_energy_forces(R, **self.params)
                else:
                    _, F_low = suttonchen_energy_forces(R, **self.params)
                F_low = F_low.astype(np.float32)
                if fname not in self.forces_map:
                    raise KeyError(f"DFT forces for {fname} not found in {len(self.forces_map)}-entry map")
                F_dft = self.forces_map[fname].astype(np.float32)
                F_res = (self.s * F_low) if self.scale_forces_with_s else F_low
                # residual = F_dft - baseline (scaled or not)
                F_res = F_dft - F_res
                self.force_res_cache[fname] = torch.from_numpy(F_res)
    def __len__(self):
        return len(self.table)

    def radial_fingerprint(self, R: np.ndarray) -> torch.Tensor:
        """Per-atom fingerprints: radial RBF sums + optional angular (three-body) pooled features.
           Returns (N, rbf + (angle_ka if use_angle else 0))."""
        N = R.shape[0]
        dR = R[None, :, :] - R[:, None, :]
        rij = np.linalg.norm(dR + 1e-18, axis=-1)
        np.fill_diagonal(rij, np.inf)
        within = (rij < self.cutoff)

        # --- Radial RBF ---
        feats_r = np.zeros((N, self.rbf), dtype=np.float32)
        for i in range(N):
            nbr = within[i, :]
            r_sel = rij[i, nbr]
            if r_sel.size == 0:
                continue
            diff = r_sel[:, None] - self.mu[None, :]
            vals = np.exp(-self.beta * (diff ** 2))
            feats_r[i, :] = vals.sum(axis=0)

        if not self.use_angle:
            return torch.from_numpy(feats_r)

        # --- Angular pooled features in cos(theta) ---
        u = np.zeros_like(dR)
        mask = np.isfinite(rij)
        u[mask] = dR[mask] / (rij[mask][..., None] + 1e-18)

        feats_a = np.zeros((N, self.angle_ka), dtype=np.float32)
        mu_a = np.linspace(-1.0, 1.0, self.angle_ka, dtype=np.float32)
        beta_a = np.float32(self.angle_beta)

        for i in range(N):
            nbr_idx = np.where(within[i, :])[0]
            if len(nbr_idx) <= 1:
                continue
            buf = []
            for a in range(len(nbr_idx)):
                j = nbr_idx[a]
                uij = u[i, j]
                for b in range(a + 1, len(nbr_idx)):
                    k = nbr_idx[b]
                    uik = u[i, k]
                    c = float(np.clip(np.dot(uij, uik), -1.0, 1.0))  # cos(theta)
                    phi = np.exp(-beta_a * (c - mu_a) ** 2)  # (K_a,)
                    buf.append(phi)
            if not buf:
                continue
            Phi = np.stack(buf, axis=0)  # (#pairs, K_a)
            ang = Phi.sum(axis=0) if self.angle_pool == 'sum' else Phi.mean(axis=0)
            feats_a[i, :] = ang.astype(np.float32)

        feats = np.concatenate([feats_r, feats_a], axis=1)
        return torch.from_numpy(feats)

    def __getitem__(self, idx: int):
        row = self.table.iloc[idx]
        fname = row['file']
        E_label = float(row['E_label'])
        E_low = float(row['E_low'])
        E_low_aligned = self.s * E_low + self.b
        delta = E_label - E_low_aligned

        # fetch cached features
        fp = self.fp_cache[fname]
        sample = {
            'file': fname,
            'fp': fp,  # (N,K) already a torch.Tensor
            'target': torch.tensor([delta], dtype=torch.float32),
        }
        if self.use_forces:
            sample['F_target'] = self.force_res_cache[fname]  # (N,3) torch.Tensor
        if self.csv_feat_dim > 0:
            if fname not in self.csv_feat_map:
                raise KeyError(f"CSV features for {fname} not found in {self.csv_feats_path}")
            sample['global'] = torch.from_numpy(self.csv_feat_map[fname])  # (D,)
        return sample

# ------------------------------
# Model: atomic MLP summed to total, with optional force and global features
# ------------------------------
class AtomicResidualMLP(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 128, depth: int = 3, dropout: float = 0.0,
                 use_force_head: bool = False, global_dim: int = 0):
        super().__init__()
        # atomic tower
        layers = []
        d = in_dim
        for _ in range(depth):
            layers += [nn.Linear(d, hidden), nn.SiLU()]
            if dropout > 0:
                layers += [nn.Dropout(dropout)]
            d = hidden
        self.atom_mlp = nn.Sequential(*layers)
        # heads
        # energy residual: sum of atomic scalars -> optionally concat global -> final scalar
        self.energy_head = nn.Linear(hidden, 1)
        self.use_force_head = use_force_head
        if use_force_head:
            self.force_head = nn.Linear(hidden, 3)  # per-atom residual force
        # final mixer for global features
        self.global_dim = global_dim
        if global_dim > 0:
            self.final_mlp = nn.Sequential(nn.Linear(1 + global_dim, hidden), nn.SiLU(), nn.Linear(hidden, 1))
        else:
            self.final_mlp = None

    def forward(self, fp: torch.Tensor, global_vec: torch.Tensor | None = None):
        # fp: (N,K)
        h = self.atom_mlp(fp)          # (N,hidden)
        atom_e = self.energy_head(h)   # (N,1)
        total_e = atom_e.sum(dim=0)    # (1,)
        if self.final_mlp is not None and global_vec is not None:
            total_e = self.final_mlp(torch.cat([total_e.view(1), global_vec.view(-1)], dim=0)).view(1)
        if self.use_force_head:
            atom_f = self.force_head(h)  # (N,3)
            return total_e, atom_f
        else:
            return total_e, None

# ------------------------------
# Training / evaluation
# ------------------------------


def run_training(args):
    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    set_seed(args.seed)

    # Baseline parameters
    if args.baseline.lower() == 'gupta':
        params = dict(A=args.A, p=args.p, xi=args.xi, q=args.q, r0=args.r0)
    else:
        params = dict(eps=args.eps, a=args.a, n=args.n, m=args.m, c=args.c)

    ds_train = AuClusterDataset(
        args.folder, args.baseline, params, split='train',
        train_ratio=args.train_ratio, val_ratio=args.val_ratio, test_ratio=args.test_ratio,
        cutoff=args.cutoff, rbf=args.rbf, align_on_train=True,
        use_forces=args.use_forces, forces_npz=args.forces_npz, scale_forces_with_s=args.scale_forces_with_s,
        use_angle=args.use_angle, angle_ka=args.angle_ka, angle_beta=args.angle_beta, angle_pool=args.angle_pool,
        csv_feats_path=args.csv_feats_path, csv_feat_cols=args.csv_feat_cols)
    ds_train.eig_max = args.eig_max

    ds_val = AuClusterDataset(
        args.folder, args.baseline, params, split='val',
        train_ratio=args.train_ratio, val_ratio=args.val_ratio, test_ratio=args.test_ratio,
        cutoff=args.cutoff, rbf=args.rbf, align_on_train=True,
        use_forces=args.use_forces, forces_npz=args.forces_npz, scale_forces_with_s=args.scale_forces_with_s,
        use_angle=args.use_angle, angle_ka=args.angle_ka, angle_beta=args.angle_beta, angle_pool=args.angle_pool,
        csv_feats_path=args.csv_feats_path, csv_feat_cols=args.csv_feat_cols)
    ds_val.eig_max = args.eig_max

    ds_test = AuClusterDataset(
        args.folder, args.baseline, params, split='test',
        train_ratio=args.train_ratio, val_ratio=args.val_ratio, test_ratio=args.test_ratio,
        cutoff=args.cutoff, rbf=args.rbf, align_on_train=True,
        use_forces=args.use_forces, forces_npz=args.forces_npz, scale_forces_with_s=args.scale_forces_with_s,
        use_angle=args.use_angle, angle_ka=args.angle_ka, angle_beta=args.angle_beta, angle_pool=args.angle_pool,
        csv_feats_path=args.csv_feats_path, csv_feat_cols=args.csv_feat_cols)
    ds_test.eig_max = args.eig_max

    # Ensure all splits use the SAME alignment learned on train
    ds_val.s, ds_val.b = ds_train.s, ds_train.b
    ds_test.s, ds_test.b = ds_train.s, ds_train.b

    # ---- Outlier stats from TRAIN (μ, σ) ----
    def _residual_of_row(row, s, b):
        return float(row['E_label'] - (s*row['E_low'] + b))

    if args.outlier_mode == 'residual':
        train_vals = np.array([
            _residual_of_row(r, ds_train.s, ds_train.b)
            for _, r in ds_train.df_train.iterrows()
        ], dtype=float)
    else:  # 'total'
        train_vals = ds_train.df_train['E_label'].to_numpy(dtype=float)

    mu = float(train_vals.mean())
    sigma = float(train_vals.std(ddof=0) + 1e-12)

    def sample_weight(E_low, E_label, s, b):
        if args.outlier_mode == 'residual':
            val = float(E_label - (s*E_low + b))
        else:
            val = float(E_label)
        z = abs((val - mu) / sigma)
        return float(args.outlier_w if z >= args.outlier_z else 1.0)

    print(f"[Outlier] mode={args.outlier_mode}, z_thr={args.outlier_z}, w_out={args.outlier_w}; "
          f"TRAIN μ={mu:.4f}, σ={sigma:.4f}")

    if args.residual_model == 'mlp':
        use_force_head = bool(args.use_forces)
        global_dim = ds_train.csv_feat_dim

        in_dim = ds_train.rbf + (ds_train.angle_ka if ds_train.use_angle else 0)
        model = AtomicResidualMLP(in_dim=in_dim, hidden=args.hidden, depth=args.depth, dropout=args.dropout,
                                  use_force_head=use_force_head, global_dim=global_dim).to(device)
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        # AMP GradScaler
        scaler = torch.cuda.amp.GradScaler(enabled=args.amp and (device.type == 'cuda'))

        def step_epoch(ds, train: bool):
            model.train() if train else model.eval()
            mse = nn.MSELoss()
            abs_errs, sq_errs = [], []
            total_loss, count = 0.0, 0

            accum = max(1, args.accum)
            if train:
                optimizer.zero_grad(set_to_none=True)

            for i in range(len(ds)):
                sample = ds[i]
                fp = sample['fp'].to(device)
                y = sample['target'].to(device)
                g = sample.get('global')
                g = g.to(device) if g is not None else None

                if train:
                    with torch.cuda.amp.autocast(enabled=args.amp and (device.type == 'cuda')):
                        yhat, fhat = model(fp, g)
                        # weighted residual loss using TRAIN-based 3σ rule
                        row = ds.table.iloc[i]
                        w = sample_weight(float(row['E_low']), float(row['E_label']), ds_train.s, ds_train.b)
                        loss_e = w * mse(yhat, y)
                        loss = loss_e
                        if args.use_forces and 'F_target' in sample:
                            F = sample['F_target'].to(device)
                            loss_F = mse(fhat, F)
                            loss = loss + args.w_forces * loss_F
                        loss = loss / accum
                    scaler.scale(loss).backward()
                    if ((i + 1) % accum) == 0:
                        scaler.step(optimizer)
                        scaler.update()
                        optimizer.zero_grad(set_to_none=True)
                else:
                    with torch.no_grad():
                        yhat, fhat = model(fp, g)
                        row = ds.table.iloc[i]
                        w = sample_weight(float(row['E_low']), float(row['E_label']), ds_train.s, ds_train.b)
                        loss_e = w * mse(yhat, y)
                        loss = loss_e
                        if args.use_forces and 'F_target' in sample:
                            F = sample['F_target'].to(device)
                            loss_F = mse(fhat, F)
                            loss = loss + args.w_forces * loss_F

                total_loss += float(loss.item())
                err = float(torch.abs(yhat - y).item())
                abs_errs.append(err)
                sq_errs.append(err**2)
                count += 1

            # flush tail grads if dataset size not divisible by accum
            if train and ((len(ds) % accum) != 0):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            mae = float(np.mean(abs_errs)) if abs_errs else np.nan
            rmse = float(np.sqrt(np.mean(sq_errs))) if sq_errs else np.nan
            avg_loss = total_loss / max(count, 1)
            return avg_loss, mae, rmse

        best_val = float('inf')
        history = []
        for epoch in range(1, args.epochs + 1):
            train_loss, train_mae, train_rmse = step_epoch(ds_train, train=True)
            val_loss, val_mae, val_rmse = step_epoch(ds_val, train=False)
            history.append(dict(epoch=epoch, train_mae=train_mae, val_mae=val_mae))
            if epoch % max(1, args.print_every) == 0:
                print(
                    f"Epoch {epoch:03d} | train MAE={train_mae:.4f} RMSE={train_rmse:.4f} eV | "
                    f"val MAE={val_mae:.4f} RMSE={val_rmse:.4f} eV"
                )
            if val_mae < best_val:
                best_val = val_mae
                os.makedirs(os.path.dirname(args.model_out) or '.', exist_ok=True)
                torch.save({'model_state': model.state_dict(),
                            'args': vars(args)}, args.model_out)

        # Reload the best validation checkpoint before final evaluation
        if os.path.isfile(args.model_out):
            ckpt = torch.load(args.model_out, map_location=device)
            if isinstance(ckpt, dict) and 'model_state' in ckpt:
                model.load_state_dict(ckpt['model_state'])
                model.eval()
                print(f"Loaded best checkpoint from {args.model_out} (based on validation MAE)")
        # Final evaluation & CSV dump
        rows = []
        for split_name, ds in [('train', ds_train), ('val', ds_val), ('test', ds_test)]:
            for i in range(len(ds)):
                sample = ds[i]
                with torch.no_grad():
                    g = sample.get('global')
                    g = g.to(device) if g is not None else None
                    delta_pred, _ = model(sample['fp'].to(device), g)
                    delta_pred = float(delta_pred.cpu().item())
                # Recover baseline-aligned energy using TRAIN alignment (shared earlier)
                E_low = float(ds.table.iloc[i]['E_low'])
                E_label = float(ds.table.iloc[i]['E_label'])
                E_low_aligned = ds_train.s * E_low + ds_train.b
                E_pred = E_low_aligned + delta_pred
                err_total = E_label - E_pred
                n_atoms = int(ds.table.iloc[i]['R'].shape[0]) if 'R' in ds.table.columns else 20
                rows.append(dict(
                    split=split_name,
                    file=ds.table.iloc[i]['file'],
                    delta_true=float(sample['target'].item()),
                    delta_pred=float(delta_pred),
                    E_label=E_label,
                    E_low=E_low,
                    E_low_aligned=E_low_aligned,
                    E_pred=E_pred,
                    err_total=err_total,
                    err_per_atom=err_total / n_atoms
                ))
        df_out = pd.DataFrame(rows)
        # Residual errors
        df_out['abs_err_delta'] = (df_out['delta_true'] - df_out['delta_pred']).abs()
        df_out['sq_err_delta'] = (df_out['delta_true'] - df_out['delta_pred'])**2
        # Total energy errors (already computed): err_total, err_per_atom
        df_out['abs_err_total'] = df_out['err_total'].abs()
        df_out['sq_err_total'] = df_out['err_total']**2

        os.makedirs(os.path.dirname(args.csv_out) or '.', exist_ok=True)
        df_out.to_csv(args.csv_out, index=False)
        print(f"Wrote {args.csv_out}")

        # Print metrics per split: residual and final energy, per cluster and per atom (with R^2)
        for split_name in ['train','val','test']:
            sub = df_out[df_out['split'] == split_name]
            if len(sub) == 0:
                continue
            # Residual (ΔE)
            y_true = sub['delta_true'].to_numpy(dtype=float)
            y_pred = sub['delta_pred'].to_numpy(dtype=float)
            mae_delta = float(np.mean(np.abs(y_true - y_pred)))
            rmse_delta = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
            ss_res = float(np.sum((y_true - y_pred) ** 2))
            ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
            r2_delta = 1.0 - ss_res / ss_tot if ss_tot > 0.0 else float('nan')

            # Total energy
            t_true = sub['E_label'].to_numpy(dtype=float)
            t_pred = sub['E_pred'].to_numpy(dtype=float)
            mae_total = float(np.mean(np.abs(t_true - t_pred)))
            rmse_total = float(np.sqrt(np.mean((t_true - t_pred) ** 2)))
            ss_res_t = float(np.sum((t_true - t_pred) ** 2))
            ss_tot_t = float(np.sum((t_true - t_true.mean()) ** 2))
            r2_total = 1.0 - ss_res_t / ss_tot_t if ss_tot_t > 0.0 else float('nan')

            # Per-atom
            mae_atom = float(np.mean(np.abs(sub['err_per_atom'])))
            rmse_atom = float(np.sqrt(np.mean((sub['err_per_atom']) ** 2)))

            print(
                f"{split_name.title()} | Residual ΔE: MAE={mae_delta:.4f} RMSE={rmse_delta:.4f} R2={r2_delta:.4f} eV  | "
                f"Total Energy: MAE={mae_total:.4f} RMSE={rmse_total:.4f} R2={r2_total:.4f} eV  "
                f"(per-atom MAE={mae_atom:.4f}, RMSE={rmse_atom:.4f})"
            )

# ------------------------------
# CLI
# ------------------------------

def build_argparser():
    ap = argparse.ArgumentParser(description="Delta-learning trainer for Au20 clusters (residual over classical baseline)")
    ap.add_argument('--folder', required=True, help='Folder of .xyz files')
    ap.add_argument('--baseline', choices=['gupta','suttonchen','sc'], default='gupta')
    ap.add_argument('--train_ratio', type=float, default=0.8)
    ap.add_argument('--val_ratio', type=float, default=0.1)
    ap.add_argument('--test_ratio', type=float, default=0.1)
    ap.add_argument('--cutoff', type=float, default=6.0)
    ap.add_argument('--rbf', type=int, default=32)
    ap.add_argument('--hidden', type=int, default=64)
    ap.add_argument('--depth', type=int, default=3)
    ap.add_argument('--dropout', type=float, default=0.0)
    ap.add_argument('--epochs', type=int, default=200)
    ap.add_argument('--print_every', type=int, default=10)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--weight_decay', type=float, default=0.0)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--cpu', action='store_true')
    ap.add_argument('--csv_out', default='out/delta_eval.csv')
    ap.add_argument('--model_out', default='out/delta_model.pt')
    # Gupta params
    ap.add_argument('--A', type=float, default=0.2061)
    ap.add_argument('--p', type=float, default=10.229)
    ap.add_argument('--xi', type=float, default=1.790)
    ap.add_argument('--q', type=float, default=4.036)
    ap.add_argument('--r0', type=float, default=2.88)
    # SC params
    ap.add_argument('--eps', type=float, default=1.0)
    ap.add_argument('--a', type=float, default=4.08)
    ap.add_argument('--n', type=int, default=10)
    ap.add_argument('--m', type=int, default=8)
    ap.add_argument('--c', type=float, default=34.408)
    # Residual model and force/global options
    ap.add_argument('--residual_model', choices=['mlp'], default='mlp')
    ap.add_argument('--use_forces', action='store_true', help='Use DFT forces for residual force loss')
    ap.add_argument('--forces_npz', type=str, default=None, help='Path to .npz with DFT forces mapping filename->(N,3)')
    ap.add_argument('--w_forces', type=float, default=0.05, help='Weight for force loss term')
    ap.add_argument('--scale_forces_with_s', action='store_true', help='(Not recommended) multiply baseline forces by s in residual')
    ap.add_argument('--csv_feats_path', type=str, default=None,
                    help='Path to CSV with structure-level features; last numeric column is treated as non-feature (real energy) and dropped')
    ap.add_argument('--csv_feat_cols', type=str, default=None,
                    help='Comma-separated column names to use as global features (overrides auto numeric selection).')

    ap.add_argument('--eig_max', type=int, default=None,
                    help='Optional: use only the first N eigenvalue features (columns named eig1..eigN). '
                         'If None, use all available eigenvalue columns.')

    #angle
    ap.add_argument('--use_angle', action='store_true',
                    help='Enable angular (three-body) pooled features around each atom')
    ap.add_argument('--angle_ka', type=int, default=8,
                    help='Number of angular Gaussian basis functions over cos(theta)')
    ap.add_argument('--angle_beta', type=float, default=2.0,
                    help='Width of angular Gaussian basis (larger=broader)')
    ap.add_argument('--angle_pool', choices=['mean', 'sum'], default='mean',
                    help='Pooling over angle pairs per atom')
    # Speed options
    ap.add_argument('--amp', action='store_true', help='Enable CUDA AMP mixed precision for faster training')
    ap.add_argument('--accum', type=int, default=1, help='Gradient accumulation steps to simulate larger batch size')
    ap.add_argument('--compile', action='store_true',
                    help='Use torch.compile if available (PyTorch 2.x)')

    # Outlier down-weighting (based on TRAIN residuals ΔE)
    ap.add_argument('--outlier_z', type=float, default=3.0,
                    help='Z-score threshold on TRAIN residuals |ΔE-μ|/σ above which samples are down-weighted')
    ap.add_argument('--outlier_w', type=float, default=0.2,
                    help='Loss weight applied to outliers (0<w<=1). 1 means no down-weighting')
    ap.add_argument('--outlier_mode', choices=['residual','total'], default='residual',
                    help='Use residual (E_label - (s*E_low+b)) or total E_label to compute z-score')
    return ap



if __name__ == '__main__':
    args = build_argparser().parse_args([
        # energy-only-MLP
        "--folder", "data/Au20_OPT_1000",
        # "--folder", "data/Au20_centered",

        "--baseline", "gupta",
        "--residual_model", "mlp",
        "--val_ratio", "0.1", "--test_ratio", "0.1",
        "--epochs", "500",
        "--print_every", "10",
        "--cutoff", "6.0",
        "--rbf", "32",
        "--hidden", "64",
        "--depth", "4",

        # '--csv_feats_path', 'data/Au20_CM_stat_features.csv',
        # "--eig_max", "5",
        "--csv_out", "out/delta_eval.csv",
        "--model_out", "out/delta_model.pt",
        #angle
        "--use_angle","--angle_ka","8","--angle_beta","2.0","--angle_pool","mean",
        # "--amp", "--accum", "8",

        #speed
        "--amp",
        "--accum", "8",
        "--compile",

        #global
    "--csv_feats_path", "data/Au20_CM_stat_features_v0.csv",
     "--csv_feat_cols", "CN_mean, centroid_max, dist_min, dist_max",

    # residual outlier down-weighting (same as gnn_weighted)
          "--outlier_z", "3",
        "--outlier_w", "0.2",
        "--outlier_mode", "residual",
    ])
    run_training(args)
