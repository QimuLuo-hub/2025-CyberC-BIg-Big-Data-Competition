import os
import numpy as np
import pandas as pd

import math
from matplotlib import rcParams
rcParams['font.family'] = 'cmr10'
rcParams['axes.formatter.use_mathtext'] = True  # Use mathtext for tick labels

rcParams['text.usetex'] = True


def pairwise_vectors(R):
    """
    R: (N,3) array of Cartesian coordinates (Å)
    returns:
        rij_vec: (N,N,3) vector from i->j
        rij:     (N,N)   distance matrix (0 on diagonal)
    """
    R = np.asarray(R, dtype=float)
    rij_vec = R[:, None, :] - R[None, :, :]
    rij = np.linalg.norm(rij_vec + 1e-18, axis=-1)
    np.fill_diagonal(rij, 0.0)
    return rij_vec, rij

# ---------------------------------------------------
# Gupta potential (second-moment tight-binding)
# ---------------------------------------------------

def gupta_energy_forces(R, A, p, xi, q, r0):
    """
    E = sum_i [ sum_{j!=i} A * exp(-p * r_ij / r0)
                - sqrt( sum_{j!=i} xi^2 * exp(-2q * r_ij / r0) ) ]
    Returns (E_total[eV], F[N,3] in eV/Å).
    """
    rij_vec, rij = pairwise_vectors(R)
    with np.errstate(over='ignore', under='ignore', divide='ignore', invalid='ignore'):
        exp_rep = np.exp(-p * rij / r0);      np.fill_diagonal(exp_rep, 0.0)
        E_rep = A * np.sum(exp_rep)

        exp_att = np.exp(-2.0 * q * rij / r0); np.fill_diagonal(exp_att, 0.0)
        S_i = np.sum((xi**2) * exp_att, axis=1)
        sqrt_S = np.sqrt(np.clip(S_i, 1e-30, None))
        E_att = np.sum(sqrt_S)

        E = float(E_rep - E_att)

        e_ij = np.zeros_like(rij_vec)
        mask = rij > 0
        e_ij[mask] = rij_vec[mask] / rij[mask][:, None]

        dErep_drij = A * exp_rep * (-p / r0)
        F_rep = -np.sum(dErep_drij[:, :, None] * e_ij, axis=1)

        inv_2sqrt = 0.5 / np.clip(sqrt_S, 1e-30, None)
        dSi_drij = (xi**2) * exp_att * (-2.0 * q / r0)
        d_sqrtSi_drij = inv_2sqrt[:, None] * dSi_drij

        F_att_self = -np.sum(d_sqrtSi_drij[:, :, None] * e_ij, axis=1)
        F_att_nei  =  np.sum(d_sqrtSi_drij[:, :, None] * e_ij, axis=0)
        F_att = F_att_self + F_att_nei

        F = F_rep + F_att
        return E, F

# ---------------------------------------------------
# Sutton–Chen potential (10–8 variant for Au)
# ---------------------------------------------------

def suttonchen_energy_forces(R, eps, a, n, m, c):
    """
    E = eps * sum_i [ 1/2 * sum_{j!=i} (a/r_ij)^n - c * sqrt( rho_i ) ]
    where rho_i = sum_{j!=i} (a/r_ij)^m
    Returns (E_total[eV], F[N,3] in eV/Å).
    """
    rij_vec, rij = pairwise_vectors(R)
    with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
        mask = rij > 0
        pow_n = np.zeros_like(rij); pow_m = np.zeros_like(rij)
        pow_n[mask] = (a / rij[mask]) ** n
        pow_m[mask] = (a / rij[mask]) ** m

        pair_sum = np.sum(pow_n)
        rho_i = np.sum(pow_m, axis=1)
        E = eps * (0.5 * pair_sum - c * np.sum(np.sqrt(np.clip(rho_i, 1e-30, None))))
        E = float(E)

        e_ij = np.zeros_like(rij_vec)
        e_ij[mask] = rij_vec[mask] / rij[mask][:, None]

        d_pow_n_drij = np.zeros_like(rij)
        d_pow_n_drij[mask] = -n * (a ** n) * (rij[mask] ** (-n - 1))
        F_pair = -0.5 * eps * np.sum(d_pow_n_drij[:, :, None] * e_ij, axis=1) \
                 -0.5 * eps * np.sum(d_pow_n_drij[:, :, None] * (-e_ij), axis=0)

        inv_2sqrt_rho = 0.5 / np.sqrt(np.clip(rho_i, 1e-30, None))
        d_rho_drij = np.zeros_like(rij)
        d_rho_drij[mask] = -m * (a ** m) * (rij[mask] ** (-m - 1))
        d_sqrt_drij = (inv_2sqrt_rho[:, None]) * d_rho_drij

        F_many_self = -(-eps * c) * np.sum(d_sqrt_drij[:, :, None] * e_ij, axis=1)
        F_many_nei  = -(-eps * c) * np.sum(d_sqrt_drij[:, :, None] * (-e_ij), axis=0)
        F_many = F_many_self + F_many_nei

        F = F_pair + F_many
        return E, F

def read_xyz_with_energy(path):
    with open(path, "r") as f:
        lines = [ln.strip() for ln in f.readlines()]

    N = int(lines[0])  # number of atoms, first line
    E_label = float(lines[1].split()[0])  # total energy, second line
    coords = []
    for i in range(N): #for all coords (20 intotal)
        toks = lines[2 + i].split()
        coords.append([float(toks[1]), float(toks[2]), float(toks[3])])
    R = np.array(coords, dtype=float) #obtain the set of the cluster (20 *3)
    return E_label, R


import re
#    Extracts leading integer from filename like '123.xyz'.
def numeric_key(fname):
    m = re.match(r"(\d+)", fname)
    return int(m.group(1)) if m else fname

def load_dataset(folder):
    rows = []
    # Natural sort by numeric index
    files = sorted(os.listdir(folder), key=numeric_key)

    for fname in files:
        if not fname.endswith(".xyz"):
            continue
        fpath = os.path.join(folder, fname)
        try:
            E_label, R = read_xyz_with_energy(fpath)
            rows.append({"file": fname, "E_label": E_label, "R": R})
        except Exception as e:
            print(f"Failed to read {fname}: {e}")
    return pd.DataFrame(rows)


def linear_examine(E_method, E_lbl):
    A = np.vstack([E_method, np.ones_like(E_method)]).T
    s, b = np.linalg.lstsq(A, E_lbl, rcond=None)[0]
    E_fit = s * E_method + b

    # Pearson
    r = np.corrcoef(E_method, E_lbl)[0, 1]

    # R^2
    r2 = r2_score(E_lbl, E_fit)


    return r, r2

if __name__ == "__main__":
    folder = "data/Au20_OPT_1000"  # adjust path if needed
    df = load_dataset(folder)

    # ---- Parameter sets (PLACEHOLDERS; replace with literature values) ----
    # Gupta (e.g., Cleri–Rosato-style)
    GUPTA = dict(A=0.2061, p=10.229, xi=1.790, q=4.036, r0=2.88)
    # Sutton–Chen 10–8 for Au (use real length scale a in Å; eps is an energy scale)
    SC = dict(eps=1.0, a=4.08, n=10, m=8, c=34.408)

    # ---- Compute classical energies & Fmax for each structure ----
    E_g_list, Fmax_g_list = [], []
    E_sc_list, Fmax_sc_list = [], []

    for _, row in df.iterrows():
        R = row["R"]
        Eg, Fg = gupta_energy_forces(R, **GUPTA)
        Esc, Fsc = suttonchen_energy_forces(R, **SC)

        E_g_list.append(Eg)
        Fmax_g_list.append(float(np.max(np.linalg.norm(Fg, axis=1))))

        E_sc_list.append(Esc)
        Fmax_sc_list.append(float(np.max(np.linalg.norm(Fsc, axis=1))))

    # Add columns
    df["E_gupta"] = E_g_list
    df["Fmax_gupta"] = Fmax_g_list
    # max (among 20 atoms) atomic force predicted by the Gupta potential for that structure

    df["E_sc"] = E_sc_list
    df["Fmax_sc"] = Fmax_sc_list

    df["Delta_gupta"] = df["E_label"] - df["E_gupta"]
    df["Delta_sc"] = df["E_label"] - df["E_sc"]

    # ---- Align classical energies to DFT labels (offset/scale) ----
    # We fit s,b in:  E_aligned = s * E_low + b
    # This puts classical energies on the same reference as DFT totals, enabling fair comparison.
    import numpy as np

    def fit_scale_offset(x, y):
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        A = np.vstack([x, np.ones_like(x)]).T
        s, b = np.linalg.lstsq(A, y, rcond=None)[0]
        return float(s), float(b)

    # Fit on all rows (you can change to a train split if desired)
    s_g, b_g = fit_scale_offset(df["E_gupta"].values, df["E_label"].values)
    s_sc, b_sc = fit_scale_offset(df["E_sc"].values, df["E_label"].values)

    # Create aligned energies and residuals
    df["E_gupta_aligned"] = s_g * df["E_gupta"] + b_g
    df["Delta_gupta_aligned"] = df["E_label"] - df["E_gupta_aligned"]

    df["E_sc_aligned"] = s_sc * df["E_sc"] + b_sc
    df["Delta_sc_aligned"] = df["E_label"] - df["E_sc_aligned"]

    print(f"Gupta alignment: s={s_g:.6f}, b={b_g:.6f}")
    print(f"Sutton-Chen alignment: s={s_sc:.6f}, b={b_sc:.6f}")

    print(df.head(10))

    for name, col in [("Gupta", "Delta_gupta_aligned"), ("Sutton-Chen", "Delta_sc_aligned")]:
        mae = float(np.abs(df[col]).mean())
        rmse = float(np.sqrt((df[col]**2).mean()))
        print(f"{name} after alignment -> MAE={mae:.4f} eV, RMSE={rmse:.4f} eV")
        #this is eV/cluster not per atom!
    os.makedirs("out", exist_ok=True)
    cols = [c for c in df.columns if c != "R"]
    df[cols].to_csv("out/classical_summary.csv", index=False)

    import pandas as pd
    # Load predictions
    preds = pd.read_csv("out/gnn_delta_eval.csv")
    if "residual" not in preds.columns:
        preds["residual"] = preds["delta_true"] - preds["delta_pred"]

    # Save full residuals
    preds.to_csv("out/test_residuals.csv", index=False)
    print("Saved residuals to out/test_residuals.csv, shape=", preds.shape)

    import pandas as pd
    import matplotlib.pyplot as plt

    # load the summary dataframe you already have
    df = pd.read_csv("out/classical_summary.csv")

    # 方法 2: 归一化 + 同轴画
    E_lbl = df["E_label"].values
    E_gupt = df["E_gupta"].values

    # 线性归一化到 [0,1]
    E_lbl_norm = (E_lbl - E_lbl.min()) / (E_lbl.max() - E_lbl.min())
    E_gupt_norm = (E_gupt - E_gupt.min()) / (E_gupt.max() - E_gupt.min())

    plt.figure(figsize=(6, 6))
    plt.scatter(E_gupt_norm, E_lbl_norm, alpha=0.7)
    plt.xlabel("Normalized E_gupta")
    plt.ylabel("Normalized E_label")
    # plt.title("Normalized E_label vs E_gupta")
    # 画参考线 y=x
    minv = min(E_gupt_norm.min(), E_lbl_norm.min())
    maxv = max(E_gupt_norm.max(), E_lbl_norm.max())
    plt.plot([minv, maxv], [minv, maxv], 'k--')
    plt.grid(True, linestyle=':', alpha=0.5)
    plt.tight_layout()
    plt.savefig("out/Elabel_vs_Egupta_normalized.png", dpi=200)
    plt.show()

    from sklearn.metrics import r2_score, mean_squared_error
    import numpy as np

    E_lbl = df["E_label"].values
    E_gupt = df["E_gupta"].values

    E_sc = df["E_sc"].values

    r_gu, r2_gu  = linear_examine(E_gupt, E_sc)
    r_sc, r2_sc = linear_examine(E_sc, E_gupt)

    # print(f"Linear fit: y = {s:.3f} x + {b:.3f}")
    print(f"Pearson r_gu = {r_gu:.6f}, r2_gu = {r2_gu:.6f}")
    print(f"Pearson r_sc = {r_sc:.6f}, r2_sc = {r2_sc:.6f}")

    # print(f"R^2 = {r2:.3f}")
    # print(f"RMSE = {rmse:.3f}")


