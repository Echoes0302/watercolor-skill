"""茶馆一角：一扇窗、一张桌、一把拉开的竹椅、一碗还在冒气的茶。相机和光在这里。"""
import math
import numpy as np

W, H = 900, 1150
F, PX, HZ = 950.0, 560.0, 520.0
EYE = 1.35
CAM = np.array([1.9, EYE, -0.3])
YAW = math.radians(14)
BZ, RX, CY = 4.0, 3.9, 2.65          # 后墙 Z、右墙 X、天花板高
LX = -3.0                             # 左边很远（画面外）
R_ = np.array([math.cos(YAW), 0, -math.sin(YAW)])
F_ = np.array([math.sin(YAW), 0, math.cos(YAW)])
NEAR = 0.3

WIN = (1.15, 2.2, 1.05, 2.2)          # 窗：X0, X1, Y0, Y1（在后墙上）
SUN = np.array([0.08, -0.62, -0.78]); SUN /= np.linalg.norm(SUN)   # 太阳从窗进来，往我们这边、往下、略往左

def cam(p):
    d = np.asarray(p, float) - CAM
    return d @ R_, d[1], d @ F_

def pr(p):
    x, y, z = cam(p); z = max(z, 1e-3)
    return (PX + F * x / z, HZ - F * y / z)

def depth_of(p):
    return cam(p)[2]

def clip_poly(pts3):
    cs = [np.array(cam(p)) for p in pts3]
    out = []
    for i in range(len(cs)):
        a, b = cs[i], cs[(i + 1) % len(cs)]
        ina, inb = a[2] >= NEAR, b[2] >= NEAR
        if ina: out.append(a)
        if ina != inb:
            t = (NEAR - a[2]) / (b[2] - a[2]); out.append(a + (b - a) * t)
    return [(PX + F * c[0] / c[2], HZ - F * c[1] / c[2]) for c in out]

def rays():
    yy, xx = np.mgrid[:H, :W].astype(np.float32)
    a = (xx - PX) / F; b = -(yy - HZ) / F
    return a * R_[0] + F_[0], b, a * R_[2] + F_[2]

def depth_map():
    dx, dy, dz = rays()
    T = np.full(dx.shape, 1e9, np.float32); K = np.zeros(dx.shape, np.int8)
    def take(t, k, ok):
        nonlocal T, K
        m = ok & (t > 0.05) & (t < T)
        T = np.where(m, t, T); K = np.where(m, k, K)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = -CAM[1] / dy; X, Z = CAM[0] + t * dx, CAM[2] + t * dz
        take(t, 1, (dy < 0) & (Z <= BZ) & (X <= RX))                  # 地
        t = (BZ - CAM[2]) / dz; X, Y = CAM[0] + t * dx, CAM[1] + t * dy
        take(t, 2, (dz > 0) & (Y >= 0) & (Y <= CY) & (X <= RX))       # 后墙
        t = (RX - CAM[0]) / dx; Y, Z = CAM[1] + t * dy, CAM[2] + t * dz
        take(t, 3, (dx > 0) & (Y >= 0) & (Y <= CY) & (Z <= BZ))       # 右墙
        t = (CY - CAM[1]) / dy; X, Z = CAM[0] + t * dx, CAM[2] + t * dz
        take(t, 4, (dy > 0) & (Z <= BZ) & (X <= RX))                  # 天花板
    T = np.where(T > 1e8, 20, T)
    return T, K, (dx, dy, dz)

def S(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0 + 1e-9), 0, 1); return t * t * (3 - 2 * t)

def window_pass(Xw, Yw, s, bars=True):
    """后墙上 (Xw, Yw) 透不透光：窗洞 + 窗棂（竖三根、横一根）。s 越大半影越宽。"""
    soft = 0.01 + 0.025 * s
    x0, x1, y0, y1 = WIN
    m = S(x0 - soft, x0 + soft, Xw) * (1 - S(x1 - soft, x1 + soft, Xw)) * S(y0 - soft, y0 + soft, Yw) * (1 - S(y1 - soft, y1 + soft, Yw))
    if bars:
        for bx in np.linspace(x0, x1, 5)[1:-1]:
            m = m * (1 - 0.9 * (1 - S(0.025 - soft * 0.5, 0.025 + soft * 0.5, np.abs(Xw - bx))))
        m = m * (1 - 0.9 * (1 - S(0.025 - soft * 0.5, 0.025 + soft * 0.5, np.abs(Yw - (y0 + y1) * 0.52))))
    return m

def light_on_plane(Yp, dirs):
    """水平面 Y=Yp 上每个像素被窗里的太阳照到多少。"""
    dx, dy, dz = dirs
    with np.errstate(divide="ignore", invalid="ignore"):
        t = (Yp - CAM[1]) / np.where(np.abs(dy) < 1e-6, 1e-6, dy)
    X, Z = CAM[0] + t * dx, CAM[2] + t * dz
    s = (BZ - Z) / -SUN[2]
    Xw, Yw = X - s * SUN[0], Yp - s * SUN[1]
    return np.nan_to_num(window_pass(Xw, Yw, np.abs(s))) * (s > 0) * (t > 0)

def light_at(p):
    X, Y, Z = p
    s = (BZ - Z) / -SUN[2]
    return float(window_pass(np.array([X - s * SUN[0]]), np.array([Y - s * SUN[1]]), np.array([abs(s)]))[0]) if s > 0 else 0.0

def to_floor(p):
    u = p[1] / -SUN[1]
    return (p[0] + u * SUN[0], 0.0, p[2] + u * SUN[2])

def box(cx, cz, w, d, y0, y1, rot=0.0):
    c, s = math.cos(math.radians(rot)), math.sin(math.radians(rot))
    out = []
    for y in (y0, y1):
        for ux, uz in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
            lx, lz = ux * w / 2, uz * d / 2
            out.append((cx + lx * c - lz * s, y, cz + lx * s + lz * c))
    return out

def hull2(pts):
    pts = sorted(set((round(a, 3), round(b, 3)) for a, b in pts))
    if len(pts) < 3: return pts
    def cr(o, a, b): return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    lo, up = [], []
    for p in pts:
        while len(lo) >= 2 and cr(lo[-2], lo[-1], p) <= 0: lo.pop()
        lo.append(p)
    for p in reversed(pts):
        while len(up) >= 2 and cr(up[-2], up[-1], p) <= 0: up.pop()
        up.append(p)
    return lo[:-1] + up[:-1]

def shadow_poly(corners):
    fl = [to_floor(p) for p in corners]          # 悬空的东西（桌面）影子只是它自己压下去那块，不连到脚下
    return [(a, 0.0, b) for a, b in hull2([(p[0], p[2]) for p in fl])]
