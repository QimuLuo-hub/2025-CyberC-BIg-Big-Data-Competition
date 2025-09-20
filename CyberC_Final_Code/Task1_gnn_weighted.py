import os
import argparse
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from Task1_classical import read_xyz_with_energy, numeric_key, gupta_energy_forces, suttonchen_energy_forces

# ------------------------------
# Utils
# ------------------------------
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# ------------------------------
# Dataset with graph building (+ angle) + CSV + caching
# ------------------------------
class AuClusterGraphDataset(torch.utils.data.Dataset):
    def __init__(self, folder: str, baseline: str, params: dict, split: str,
                 train_ratio: float = 0.8, val_ratio: float = 0.1, test_ratio: float = 0.1,
                 cutoff: float = 6.0, rbf: int = 32, align_on_train: bool = True,
                 csv_feats_path: str | None = None, csv_feats_key: str | None = None, csv_feat_cols: list | str | None = None,
                 eig_max: int | None = None,
                 use_angle: bool = False, angle_ka: int = 8, angle_beta: float = 2.0, angle_pool: str = 'mean',
                 seed: int = 42):
        super().__init__()
        self.folder = folder
        self.baseline = baseline.lower()
        self.params = params
        self.cutoff = float(cutoff)
        self.rbf = int(rbf)
        self.csv_feats_path = csv_feats_path
        self.csv_feats_key = csv_feats_key
        self.csv_feat_cols = csv_feat_cols
        self.eig_max = eig_max
        self.use_angle = bool(use_angle)
        self.angle_ka = int(angle_ka)
        self.angle_beta = float(angle_beta)
        self.angle_pool = str(angle_pool)
        self.seed = int(seed)

        files = sorted([f for f in os.listdir(folder) if f.endswith('.xyz')], key=numeric_key)
        data = []
        for fname in files:
            E_label, R = read_xyz_with_energy(os.path.join(folder, fname))
            if self.baseline == 'gupta':
                E_low, _ = gupta_energy_forces(R, **self.params)
            elif self.baseline in ('suttonchen', 'sc'):
                E_low, _ = suttonchen_energy_forces(R, **self.params)
            else:
                raise ValueError(f"Unknown baseline: {self.baseline}")
            data.append({'file': fname, 'E_label': E_label, 'R': R, 'E_low': E_low})
        df = pd.DataFrame(data)

        # reproducible shuffle before splitting (avoid biased splits)
        rng = np.random.default_rng(self.seed)
        df = df.iloc[rng.permutation(len(df))].reset_index(drop=True)

        # splits
        n = len(df)
        n_train = int(round(train_ratio * n))
        n_val = int(round(val_ratio * n))
        self.df_train = df.iloc[:n_train].reset_index(drop=True)
        self.df_val   = df.iloc[n_train:n_train+n_val].reset_index(drop=True)
        self.df_test  = df.iloc[n_train+n_val:].reset_index(drop=True)

        # alignment (train only)
        if align_on_train:
            x = self.df_train['E_low'].to_numpy()
            y = self.df_train['E_label'].to_numpy()
            A = np.vstack([x, np.ones_like(x)]).T
            s, b = np.linalg.lstsq(A, y, rcond=None)[0]
            self.s, self.b = float(s), float(b)
        else:
            self.s, self.b = 1.0, 0.0

        # table by split
        if split == 'train':
            self.table = self.df_train
        elif split == 'val':
            self.table = self.df_val
        elif split == 'test':
            self.table = self.df_test
        else:
            raise ValueError("split must be 'train','val','test'")

        # RBF basis (radial)
        self.mu = np.linspace(0.0, self.cutoff, self.rbf, dtype=np.float32)
        self.beta = np.float32(1.0)
        # edge feature dimension = radial + (optional) angular
        self.edge_feat_dim = self.rbf + (self.angle_ka if self.use_angle else 0)

        # optional CSV global features: drop last numeric col (assume energy)
        self.csv_feat_map = {}
        if csv_feats_path:
            feats_df = pd.read_csv(csv_feats_path)
            feats_df.columns = [str(c).strip() for c in feats_df.columns]
            # Choose feature columns
            feat_cols = None
            if self.csv_feat_cols is not None:
                # allow comma-separated string or list
                if isinstance(self.csv_feat_cols, str):
                    col_list = [c.strip() for c in self.csv_feat_cols.split(',') if str(c).strip()]
                else:
                    col_list = [str(c).strip() for c in self.csv_feat_cols]
                missing = [c for c in col_list if c not in feats_df.columns]
                if missing:
                    raise ValueError(f"csv_feat_cols not found in CSV columns: {missing}")
                feat_cols = col_list
            else:
                # default: all numeric columns except the last numeric (assumed energy)
                num_cols = [c for c in feats_df.columns if np.issubdtype(feats_df[c].dtype, np.number)]
                if len(num_cols) < 2:
                    raise ValueError("CSV must have >=2 numeric columns; last is treated as energy to drop.")
                feat_cols = num_cols[:-1]
                # optional eigen trimming only applies when columns are auto-selected
                if self.eig_max is not None:
                    eig_cols = [c for c in feat_cols if str(c).lower().startswith('eig')]
                    other_cols = [c for c in feat_cols if not str(c).lower().startswith('eig')]
                    eig_cols_sorted = sorted(eig_cols, key=lambda s: int(''.join(ch for ch in str(s) if ch.isdigit()) or 0))
                    feat_cols = other_cols + eig_cols_sorted[:self.eig_max]

            key_col = None
            if self.csv_feats_key and self.csv_feats_key in feats_df.columns:
                key_col = self.csv_feats_key
            else:
                for cand in ['file','filename','name','id','fname','stem','structure','structure_id','index','idx']:
                    if cand in feats_df.columns:
                        key_col = cand; break

            files_sorted = sorted([f for f in os.listdir(folder) if f.endswith('.xyz')], key=numeric_key)
            file_set = set(files_sorted)

            def norm_name(v: str) -> str:
                s = str(v)
                try:
                    num = int(s)
                    return f"{num}.xyz"
                except Exception:
                    pass
                return s if s.endswith('.xyz') else s + '.xyz'

            if key_col is not None:
                for _, r in feats_df.iterrows():
                    fname = norm_name(r[key_col])
                    if fname in file_set:
                        self.csv_feat_map[fname] = np.asarray([r[c] for c in feat_cols], dtype=np.float32)
                if not self.csv_feat_map:
                    raise ValueError("CSV key present but no filenames matched; check --csv_feats_key or stems.")
            else:
                if len(feats_df) != len(files_sorted):
                    raise ValueError("CSV lacks key and row count != .xyz count; cannot map by order.")
                for fname, (_, r) in zip(files_sorted, feats_df.iterrows()):
                    self.csv_feat_map[fname] = np.asarray([r[c] for c in feat_cols], dtype=np.float32)
            self.csv_feat_dim = len(next(iter(self.csv_feat_map.values()))) if self.csv_feat_map else 0
        else:
            self.csv_feat_dim = 0

        # ---------- CACHE graphs + node features for this split ----------
        self.cached = []
        for _, row in self.table.iterrows():
            R = row['R']
            edge_index, edge_attr = self.build_graph(R)
            # node init: sum of radial RBF from incoming edges (keep dim = rbf)
            N = R.shape[0]
            x = torch.zeros((N, self.rbf), dtype=torch.float32)
            if self.use_angle:
                for e in range(edge_attr.shape[0]):
                    j = int(edge_index[1, e])
                    x[j] += edge_attr[e, :self.rbf]
            else:
                for e in range(edge_attr.shape[0]):
                    j = int(edge_index[1, e])
                    x[j] += edge_attr[e]
            self.cached.append((
                row['file'],
                x, edge_index, edge_attr,
                float(row['E_low']), float(row['E_label']),
                R
            ))

    def __len__(self): return len(self.table)

    def build_graph(self, R: np.ndarray):
        R = np.asarray(R, dtype=np.float32)
        N = R.shape[0]
        dR = R[None, :, :] - R[:, None, :]
        rij = np.linalg.norm(dR + 1e-18, axis=-1)
        np.fill_diagonal(rij, np.inf)

        # unit vectors (for angle)
        u = None
        if self.use_angle:
            u = np.zeros_like(dR)
            mask = np.isfinite(rij)
            u[mask] = dR[mask] / (rij[mask][..., None] + 1e-18)

        src, dst, feats_r = [], [], []
        for i in range(N):
            for j in range(N):
                if i == j: continue
                dij = rij[i, j]
                if dij < self.cutoff:
                    src.append(i); dst.append(j)
                    diff = dij - self.mu
                    feats_r.append(np.exp(-self.beta * (diff ** 2)))
        if len(src) == 0:
            # fully connect fallback
            for i in range(N):
                for j in range(N):
                    if i == j: continue
                    dij = rij[i, j]
                    src.append(i); dst.append(j)
                    diff = dij - self.mu
                    feats_r.append(np.exp(-self.beta * (diff ** 2)))

        edge_index = torch.tensor([src, dst], dtype=torch.long)
        feats_r = np.asarray(feats_r, dtype=np.float32)  # (E, rbf)

        if not self.use_angle:
            edge_attr = torch.tensor(feats_r, dtype=torch.float32)
            return edge_index, edge_attr

        # Angular pooled features (line-graph style) at destination node
        E = len(src)
        feats_a = np.zeros((E, self.angle_ka), dtype=np.float32)
        mu_a = np.linspace(-1.0, 1.0, self.angle_ka, dtype=np.float32)
        beta_a = np.float32(self.angle_beta)

        incoming = [[] for _ in range(N)]
        for e in range(E):
            i, j = src[e], dst[e]
            incoming[j].append((e, i))

        for j in range(N):
            inc = incoming[j]
            if len(inc) <= 1: continue
            for idx_e, i in inc:
                buf = []
                u_ji = u[j, i]
                for idx_e2, k in inc:
                    if k == i: continue
                    u_jk = u[j, k]
                    c = float(np.clip(np.dot(u_ji, u_jk), -1.0, 1.0))
                    phi = np.exp(-beta_a * (c - mu_a) ** 2)  # (K_a,)
                    buf.append(phi)
                if not buf: continue
                Phi = np.stack(buf, axis=0)
                ang = Phi.sum(axis=0) if self.angle_pool == 'sum' else Phi.mean(axis=0)
                feats_a[idx_e] = ang.astype(np.float32)

        feats = np.concatenate([feats_r, feats_a], axis=1)  # (E, rbf+Ka)
        edge_attr = torch.tensor(feats, dtype=torch.float32)
        return edge_index, edge_attr

    def __getitem__(self, idx: int):
        fname, x, edge_index, edge_attr, E_low, E_label, R = self.cached[idx]
        E_low_aligned = self.s * E_low + self.b
        delta = E_label - E_low_aligned
        sample = {
            'file': fname,
            'x': x, 'edge_index': edge_index, 'edge_attr': edge_attr,
            'target': torch.tensor([delta], dtype=torch.float32),
            'E_low': E_low, 'E_label': E_label,
            'n_atoms': R.shape[0],
        }
        if self.csv_feat_map:
            if fname not in self.csv_feat_map:
                raise KeyError(f"CSV features for {fname} not found in {self.csv_feats_path}")
            sample['global'] = torch.from_numpy(self.csv_feat_map[fname])
        return sample

    # --- Public: baseline energy for arbitrary coordinates (used by sensitivity_gnn) ---
    def calc_baseline_energy(self, R: np.ndarray) -> float:
        """Compute baseline (Gupta or Sutton-Chen) energy for coordinates R using the
        same parameters used to build the dataset. Returns total energy (float)."""
        R = np.asarray(R, dtype=np.float64)
        if self.baseline == 'gupta':
            E, _ = gupta_energy_forces(R, **self.params)
            return float(E)
        elif self.baseline in ('suttonchen', 'sc'):
            E, _ = suttonchen_energy_forces(R, **self.params)
            return float(E)
        else:
            raise ValueError(f"Unknown baseline: {self.baseline}")

# ------------------------------
# Model (MPNN/GGNN style)
# ------------------------------
class GNNLayer(nn.Module):
    def __init__(self, node_dim: int, edge_dim: int, hidden: int):
        super().__init__()
        self.msg_mlp = nn.Sequential(
            nn.Linear(node_dim + edge_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
        )
        self.upd = nn.GRUCell(input_size=hidden, hidden_size=node_dim)

    def forward(self, x, edge_index, edge_attr):
        src, dst = edge_index
        m_in = torch.cat([x[src], edge_attr], dim=-1)  # (E, node_dim+edge_dim)
        m = self.msg_mlp(m_in)                         # (E, hidden)
        N = x.size(0)
        agg = torch.zeros((N, m.size(-1)), device=x.device)
        agg.index_add_(0, dst, m)
        return self.upd(agg, x)

class ResidualGNN(nn.Module):
    def __init__(self, in_dim: int, edge_dim: int, hidden: int = 128, layers: int = 4, global_dim: int = 0):
        super().__init__()
        self.embed = nn.Linear(in_dim, hidden)
        self.layers = nn.ModuleList([GNNLayer(hidden, edge_dim, hidden) for _ in range(layers)])
        self.head_atom = nn.Sequential(nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, 1))
        self.global_dim = global_dim
        self.final = nn.Sequential(nn.Linear(1 + global_dim, hidden), nn.SiLU(), nn.Linear(hidden, 1)) if global_dim > 0 else None

    def forward(self, x, edge_index, edge_attr, global_vec=None):
        h = F.silu(self.embed(x))
        for layer in self.layers:
            h = layer(h, edge_index, edge_attr)
        atom_e = self.head_atom(h)     # (N,1)
        total = atom_e.sum(dim=0)      # (1,)
        if self.final is not None and global_vec is not None:
            total = self.final(torch.cat([total.view(1), global_vec.view(-1)], dim=0)).view(1)
        return total

# ------------------------------
# Train + Eval (AMP + accum + compile + cached data)
# ------------------------------
def train_and_eval(args):
    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    set_seed(args.seed)

    # baseline params
    if args.baseline.lower() == 'gupta':
        params = dict(A=args.A, p=args.p, xi=args.xi, q=args.q, r0=args.r0)
    else:
        params = dict(eps=args.eps, a=args.a, n=args.n, m=args.m, c=args.c)

    # datasets (cache happens inside)
    ds_train = AuClusterGraphDataset(args.folder, args.baseline, params, split='train',
        train_ratio=args.train_ratio, val_ratio=args.val_ratio, test_ratio=args.test_ratio,
        cutoff=args.cutoff, rbf=args.rbf, align_on_train=True,
        csv_feats_path=args.csv_feats_path, csv_feats_key=args.csv_feats_key, csv_feat_cols=args.csv_feat_cols, eig_max=args.eig_max,
        use_angle=args.use_angle, angle_ka=args.angle_ka, angle_beta=args.angle_beta, angle_pool=args.angle_pool,
        seed=args.seed)

    ds_val = AuClusterGraphDataset(args.folder, args.baseline, params, split='val',
        train_ratio=args.train_ratio, val_ratio=args.val_ratio, test_ratio=args.test_ratio,
        cutoff=args.cutoff, rbf=args.rbf, align_on_train=True,
        csv_feats_path=args.csv_feats_path, csv_feats_key=args.csv_feats_key, csv_feat_cols=args.csv_feat_cols, eig_max=args.eig_max,
        use_angle=args.use_angle, angle_ka=args.angle_ka, angle_beta=args.angle_beta, angle_pool=args.angle_pool,
        seed=args.seed)
    ds_test  = AuClusterGraphDataset(args.folder, args.baseline, params, split='test',
        train_ratio=args.train_ratio, val_ratio=args.val_ratio, test_ratio=args.test_ratio,
        cutoff=args.cutoff, rbf=args.rbf, align_on_train=True,
        csv_feats_path=args.csv_feats_path, csv_feats_key=args.csv_feats_key, csv_feat_cols=args.csv_feat_cols, eig_max=args.eig_max,
        use_angle=args.use_angle, angle_ka=args.angle_ka, angle_beta=args.angle_beta, angle_pool=args.angle_pool,
        seed=args.seed)
    # share alignment
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

    global_dim = ds_train.csv_feat_dim
    model = ResidualGNN(in_dim=args.rbf, edge_dim=ds_train.edge_feat_dim, hidden=args.hidden, layers=args.layers, global_dim=global_dim).to(device)
    if args.compile and hasattr(torch, 'compile'):
        try:
            model = torch.compile(model)
        except Exception:
            pass

    opt = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    mse = nn.MSELoss()
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp and (device.type == 'cuda'))

    def step_epoch(ds, train: bool):
        model.train() if train else model.eval()
        abs_errs, sq_errs = [], []
        total_loss, count = 0.0, 0
        accum = max(1, args.accum)
        if train:
            opt.zero_grad(set_to_none=True)
        for i in range(len(ds)):
            s = ds[i]
            x = s['x'].to(device)
            eidx = s['edge_index'].to(device)
            eattr = s['edge_attr'].to(device)
            y = s['target'].to(device)
            g = s.get('global'); g = g.to(device) if g is not None else None

            if train:
                with torch.cuda.amp.autocast(enabled=args.amp and (device.type == 'cuda')):
                    yhat = model(x, eidx, eattr, g)
                    w = sample_weight(float(s['E_low']), float(s['E_label']), ds_train.s, ds_train.b)
                    loss = w * mse(yhat, y) / accum
                # with torch.cuda.amp.autocast(enabled=args.amp and (device.type == 'cuda')):
                #     yhat = model(x, eidx, eattr, g)
                #     loss = mse(yhat, y) / accum
                scaler.scale(loss).backward()
                if ((i + 1) % accum) == 0:
                    scaler.step(opt); scaler.update()
                    opt.zero_grad(set_to_none=True)
            else:
                with torch.no_grad():
                    yhat = model(x, eidx, eattr, g)
                    w = sample_weight(float(s['E_low']), float(s['E_label']), ds_train.s, ds_train.b)
                    loss = w * mse(yhat, y)
                # with torch.no_grad():
                #     yhat = model(x, eidx, eattr, g)
                #     loss = mse(yhat, y)

            total_loss += float(loss.item())
            err = float(torch.abs(yhat - y).item())
            abs_errs.append(err); sq_errs.append(err**2); count += 1

        if train and ((len(ds) % accum) != 0):
            scaler.step(opt); scaler.update()
            opt.zero_grad(set_to_none=True)

        mae = float(np.mean(abs_errs)) if abs_errs else np.nan
        rmse = float(np.sqrt(np.mean(sq_errs))) if sq_errs else np.nan
        return total_loss / max(count, 1), mae, rmse

    best_val = float('inf')
    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_mae, tr_rmse = step_epoch(ds_train, True)
        va_loss, va_mae, va_rmse = step_epoch(ds_val, False)
        if epoch % max(1, args.print_every) == 0:
            print(f"Epoch {epoch:03d} | train MAE={tr_mae:.4f} RMSE={tr_rmse:.4f} eV | "
                  f"val MAE={va_mae:.4f} RMSE={va_rmse:.4f} eV")
        if va_mae < best_val:
            best_val = va_mae
            os.makedirs(os.path.dirname(args.model_out) or '.', exist_ok=True)
            torch.save({'model_state': model.state_dict(), 'args': vars(args)}, args.model_out)

    # reload best (by val MAE)
    if os.path.isfile(args.model_out):
        ckpt = torch.load(args.model_out, map_location=device)
        if isinstance(ckpt, dict) and 'model_state' in ckpt:
            model.load_state_dict(ckpt['model_state']); model.eval()
            print(f"Loaded best checkpoint from {args.model_out} (based on validation MAE)")

    # final eval + CSV (residual + total) + R²
    rows = []
    for split_name, ds in [('train', ds_train), ('val', ds_val), ('test', ds_test)]:
        for i in range(len(ds)):
            s = ds[i]
            with torch.no_grad():
                yhat = model(s['x'].to(device), s['edge_index'].to(device), s['edge_attr'].to(device),
                             s.get('global').to(device) if s.get('global') is not None else None)
                delta_pred = float(yhat.cpu().item())
            E_low = float(s['E_low']); E_lab = float(s['E_label'])
            E_low_aln = ds_train.s * E_low + ds_train.b
            E_pred = E_low_aln + delta_pred
            err = E_lab - E_pred
            rows.append(dict(
                split=split_name, file=ds.table.iloc[i]['file'],
                delta_true=E_lab - E_low_aln, delta_pred=delta_pred,
                E_label=E_lab, E_low=E_low, E_low_aligned=E_low_aln,
                E_pred=E_pred, err_total=err, err_per_atom=err/float(s['n_atoms'])
            ))
    df_out = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.csv_out) or '.', exist_ok=True)
    df_out.to_csv(args.csv_out, index=False)
    print(f"Wrote {args.csv_out}")

    for split in ['train', 'val', 'test']:
        sub = df_out[df_out['split'] == split]
        if len(sub) == 0: continue
        # Residual
        y_true = sub['delta_true'].to_numpy(dtype=float)
        y_pred = sub['delta_pred'].to_numpy(dtype=float)
        mae_delta = float(np.mean(np.abs(y_true - y_pred)))
        rmse_delta = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
        ss_res = float(np.sum((y_true - y_pred) ** 2))
        ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
        r2_delta = 1.0 - ss_res/ss_tot if ss_tot > 0.0 else float('nan')
        # Total
        t_true = sub['E_label'].to_numpy(dtype=float)
        t_pred = sub['E_pred'].to_numpy(dtype=float)
        mae_total = float(np.mean(np.abs(t_true - t_pred)))
        rmse_total = float(np.sqrt(np.mean((t_true - t_pred) ** 2)))
        ss_res_t = float(np.sum((t_true - t_pred) ** 2))
        ss_tot_t = float(np.sum((t_true - t_true.mean()) ** 2))
        r2_total = 1.0 - ss_res_t/ss_tot_t if ss_tot_t > 0.0 else float('nan')
        # Per-atom
        mae_atom = float(np.mean(np.abs(sub['err_per_atom'])))
        rmse_atom = float(np.sqrt(np.mean((sub['err_per_atom']) ** 2)))
        print(
            f"{split.title()} | Residual ΔE: MAE={mae_delta:.4f} RMSE={rmse_delta:.4f} R2={r2_delta:.4f} eV  | "
            f"Total Energy: R2={r2_total:.4f} eV  "
            f"(per-atom MAE={mae_atom:.4f}, RMSE={rmse_atom:.4f})"
        )



# ------------------------------
# CLI
# ------------------------------
def build_argparser():
    ap = argparse.ArgumentParser(description="Fast residual-learning GNN for Au20 clusters (classical baseline) with angle features")
    ap.add_argument('--folder', required=True)
    ap.add_argument('--baseline', choices=['gupta','suttonchen','sc'], default='gupta')
    ap.add_argument('--train_ratio', type=float, default=0.8)
    ap.add_argument('--val_ratio', type=float, default=0.1)
    ap.add_argument('--test_ratio', type=float, default=0.1)
    ap.add_argument('--cutoff', type=float, default=6.0)
    ap.add_argument('--rbf', type=int, default=32)
    ap.add_argument('--hidden', type=int, default=64)
    ap.add_argument('--layers', type=int, default=4)
    ap.add_argument('--epochs', type=int, default=500)
    ap.add_argument('--print_every', type=int, default=10)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--weight_decay', type=float, default=0.0)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--cpu', action='store_true')
    ap.add_argument('--csv_out', default='out/gnn_delta_eval.csv')
    ap.add_argument('--model_out', default='out/gnn_delta.pt')

    # Gupta params
    ap.add_argument('--A', type=float, default=0.2061)
    ap.add_argument('--p', type=float, default=10.229)
    ap.add_argument('--xi', type=float, default=1.790)
    ap.add_argument('--q', type=float, default=4.036)
    ap.add_argument('--r0', type=float, default=2.88)

    # Sutton-Chen params
    ap.add_argument('--eps', type=float, default=1.0)
    ap.add_argument('--a', type=float, default=4.08)
    ap.add_argument('--n', type=int, default=10)
    ap.add_argument('--m', type=int, default=8)
    ap.add_argument('--c', type=float, default=34.408)

    # CSV features
    ap.add_argument('--csv_feats_path', type=str, default=None,
                    help='Path to CSV of structure-level features; last numeric col is dropped (energy).')
    ap.add_argument('--csv_feats_key', type=str, default=None,
                    help='Column name in CSV to match filenames; else tries common names or maps by order.')
    ap.add_argument('--csv_feat_cols', type=str, default=None,
                    help='Comma-separated column names to use as features from the CSV (overrides auto numeric selection).')
    ap.add_argument('--eig_max', type=int, default=None,
                    help='If CSV has eig1..eigK, keep only first N eigenvalues.')

    # Angle / three-body
    ap.add_argument('--use_angle', action='store_true',
                    help='Enable angular (three-body) features via pooled line-graph angles at destination nodes')
    ap.add_argument('--angle_ka', type=int, default=8,
                    help='Number of angular Gaussian basis functions over cos(theta) in [-1,1]')
    ap.add_argument('--angle_beta', type=float, default=2.0, help='Width of angular Gaussian basis')
    ap.add_argument('--angle_pool', choices=['mean', 'sum'], default='mean',
                    help='Pooling over third neighbors for each edge angle features')

    # Speed
    ap.add_argument('--amp', action='store_true', help='Enable CUDA AMP mixed precision')
    ap.add_argument('--accum', type=int, default=1, help='Gradient accumulation steps')
    ap.add_argument('--compile', action='store_true', help='Use torch.compile if available (PyTorch 2.x)')

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
        "--folder","data/Au20_OPT_1000",
        # "--folder", "data/Au20_centered",

        "--baseline","gupta",
        # "--baseline", "sc",

        "--epochs","500",
        "--cutoff","6.0",

        "--rbf","32",
        "--hidden","64",
        "--layers","4",

        "--val_ratio","0.1","--test_ratio","0.1",
        "--csv_out","out/gnn_delta_eval.csv","--model_out","out/gnn_delta.pt",
        # angle (optional)
        "--use_angle","--angle_ka","8","--angle_beta","2.0","--angle_pool","mean",
        # speed (optional)
        "--amp","--accum","8","--compile",

        #global
        "--csv_feats_path", "data/Au20_CM_stat_features_v0.csv",#full with 78.xyz
        "--csv_feats_key", "file",
        # "--csv_feat_cols", "CN_mean, centroid_max, dist_min, dist_max",
        "--csv_feat_cols", "CN_mean, centroid_max, dist_min, dist_max",

        #residual
        "--outlier_z", "3",
        "--outlier_w", "0.2",
        "--outlier_mode", "residual",
    ])
    train_and_eval(args)

