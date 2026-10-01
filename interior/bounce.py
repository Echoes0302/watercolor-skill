"""地上光斑往上反的光：每个面上一点收到多少（窗所在那面墙得不到窗的直射，只吃这个）。"""
import numpy as np
from scene import *

def patch_samples(step=0.04):
    X, Z = np.meshgrid(np.arange(-1.5, RX, step), np.arange(-0.5, BZ, step))
    X, Z = X.ravel(), Z.ravel()
    s = (BZ - Z) / -SUN[2]
    lit = window_pass(X - s * SUN[0], 0 - s * SUN[1], np.abs(s)) * (s > 0)
    k = lit > 0.05
    return X[k], Z[k], lit[k] * step * step

def bounce_on(plane, U, V):
    """plane: 'ceil'(U=X,V=Z) / 'back'(U=X,V=Y) / 'right'(U=Z,V=Y)。返回同形状的辐照度（相对值）。"""
    qx, qz, w = patch_samples()
    U = np.asarray(U, np.float64); V = np.asarray(V, np.float64)
    out = np.zeros(U.shape, np.float64)
    for i in range(0, len(qx), 400):
        x, z, ww = qx[i:i + 400], qz[i:i + 400], w[i:i + 400]
        if plane == "ceil":
            dx, dy, dz = x - U[..., None], -CY, z - V[..., None]
            r2 = dx * dx + dy * dy + dz * dz
            out += (ww * CY * CY / r2 ** 2).sum(-1)
        elif plane == "back":
            dx, dy, dz = x - U[..., None], -V[..., None], z - BZ
            r2 = dx * dx + dy * dy + dz * dz
            out += (ww * (BZ - z) * V[..., None] / r2 ** 2).sum(-1)
        else:
            dx, dy, dz = x - RX, -V[..., None], z - U[..., None]
            r2 = dx * dx + dy * dy + dz * dz
            out += (ww * (RX - x) * V[..., None] / r2 ** 2).sum(-1)
    return out / np.pi

def grid_sample(G, u0, u1, v0, v1, U, V):
    """G 是在 [u0,u1]×[v0,v1] 上的规则网格，双线性取样。"""
    gh, gw = G.shape
    fu = np.clip((U - u0) / (u1 - u0) * (gw - 1), 0, gw - 1.001)
    fv = np.clip((V - v0) / (v1 - v0) * (gh - 1), 0, gh - 1.001)
    i, j = fv.astype(int), fu.astype(int); a, b = fv - i, fu - j
    return (G[i, j] * (1 - a) * (1 - b) + G[i, j + 1] * (1 - a) * b + G[i + 1, j] * a * (1 - b) + G[i + 1, j + 1] * a * b)

if __name__ == "__main__":
    qx, qz, w = patch_samples(); print("samples", len(qx), "patch X", qx.min(), qx.max(), "Z", qz.min(), qz.max())
    for name, (u0, u1, v0, v1) in {"ceil": (-1.5, RX, 0, BZ), "back": (-1.5, RX, 0, CY), "right": (0, BZ, 0, CY)}.items():
        uu, vv = np.meshgrid(np.linspace(u0, u1, 6), np.linspace(v0, v1, 5))
        B = bounce_on(name, uu, vv)
        print(name); print(np.round(B, 3))
