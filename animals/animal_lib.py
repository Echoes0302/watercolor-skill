"""动物（Penovác 那一路）：纸先湿，墨下去自己往纤维里爬——毛不是画的，是洇出来的边。
bleed()  给一个剪影蒙版，长出一圈顺着法线方向的毛刺边（LIC：白噪声沿法线方向积分成一根根纤维）。
"""
import os, sys
import numpy as np
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))       # skill 根目录（animals/ 的上一层）
for _p in (_ROOT, os.path.expanduser("~/.claude/skills/watercolor")):
    if os.path.exists(os.path.join(_p, "watercolor_lib.py")):
        sys.path.insert(0, _p); break
import watercolor_lib as wc

S = wc.smoothstep


def sample(a, x, y):
    """双线性取样，坐标出界就夹住。"""
    H, W = a.shape
    x = np.clip(x, 0, W - 1.001); y = np.clip(y, 0, H - 1.001)
    x0 = np.floor(x).astype(np.int32); y0 = np.floor(y).astype(np.int32)
    fx = (x - x0).astype(np.float32); fy = (y - y0).astype(np.float32)
    a00 = a[y0, x0]; a01 = a[y0, x0 + 1]; a10 = a[y0 + 1, x0]; a11 = a[y0 + 1, x0 + 1]
    return (a00 * (1 - fx) + a01 * fx) * (1 - fy) + (a10 * (1 - fx) + a11 * fx) * fy


def outward_normal(soft):
    gy, gx = np.gradient(soft)
    n = np.hypot(gx, gy) + 1e-6
    return (-gx / n).astype(np.float32), (-gy / n).astype(np.float32)


def fibers(nx, ny, reach, rng, width=0.6, steps=None):
    """沿法线方向把白噪声积分成一根根纤维（线积分卷积）。返回均值 0、方差 1 的场。"""
    H, W = nx.shape
    w = rng.standard_normal((H, W)).astype(np.float32)
    if width > 0:
        w = wc.blur(w, width)
    yy, xx = np.mgrid[:H, :W].astype(np.float32)
    steps = steps or max(4, int(reach / 1.5))
    acc = np.zeros((H, W), np.float32)
    for k in range(-steps, steps + 1):
        acc += sample(w, xx + k * 1.5 * nx, yy + k * 1.5 * ny)
    acc /= (2 * steps + 1)
    return (acc - acc.mean()) / (acc.std() + 1e-6)


def bleed(mask, reach=16, hair=0.22, rng=None, width=0.6):
    """剪影 → 洇开的覆盖图。reach 是毛往外爬多远（像素），hair 是毛刺有多扎。
    返回 (cov, soft)：cov 0..1（边上是一根根往外爬的毛），soft 是模糊过的剪影（边在 0.5）。"""
    rng = rng or np.random.default_rng(0)
    soft = wc.blur(mask.astype(np.float32), reach * 0.45)
    nx, ny = outward_normal(wc.blur(mask.astype(np.float32), reach * 0.9))
    fib = fibers(nx, ny, reach, rng, width=width)
    band = np.clip(4 * soft * (1 - soft), 0, 1)                 # 只在边上一圈起毛
    v = soft + hair * fib * band
    cov = S(0.38, 0.62, v)
    thin = 0.45 + 0.55 * S(0.5, 0.85, soft)                     # 爬出去的毛是稀的墨，浅
    return (cov * thin).astype(np.float32), soft


def spline(pts, n=24, closed=True):
    """Catmull-Rom：几个控制点连成一条顺的曲线（剪影轮廓用）。"""
    P = np.asarray(pts, np.float64)
    if closed:
        P = np.vstack([P[-1], P, P[0], P[1]])
    else:
        P = np.vstack([P[0], P, P[-1]])
    out = []
    for i in range(1, len(P) - 2):
        p0, p1, p2, p3 = P[i - 1], P[i], P[i + 1], P[i + 2]
        for t in np.linspace(0, 1, n, endpoint=False):
            t2, t3 = t * t, t * t * t
            out.append(0.5 * ((2 * p1) + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t2 + (-p0 + 3 * p1 - 3 * p2 + p3) * t3))
    if not closed:
        out.append(P[-2])
    return [tuple(p) for p in out]


def poly_mask(pts, W=None, H=None):
    from PIL import Image, ImageDraw
    W = W or wc.W; H = H or wc.H
    im = Image.new("L", (W, H), 0)
    ImageDraw.Draw(im).polygon([(float(x), float(y)) for x, y in pts], fill=255)
    return np.asarray(im, np.float32) / 255.0


def ink_drops(drops, H=None, W=None):
    """几滴墨落在湿纸上各自化开：[(x, y, 半径, 浓度), ...] → 浓度图（不是剪影，是墨在剪影里怎么分布）。"""
    H = H or wc.H; W = W or wc.W
    yy, xx = np.mgrid[:H, :W].astype(np.float32)
    d = np.zeros((H, W), np.float32)
    for x, y, r, a in drops:
        d = np.maximum(d, a * np.exp(-(((xx - x) ** 2 + (yy - y) ** 2) / (2 * r * r))))
    return d


def fine_stroke(pts, w0, w1=None, taper=True, aa=3, rough=0.1, step=0.5, rng=None):
    """细线专用的 stroke_mask：wc.stroke_mask 每 2px 盖一个圆，线宽不到 2px 就串成一颗颗珠子（胡子、眼线都翻过）。
    这里每 step 像素盖一个，细线也是连着的。"""
    import math
    from PIL import Image, ImageDraw
    rng = rng or np.random.default_rng(0)
    W, H = wc.W, wc.H
    w1 = w0 if w1 is None else w1
    im = Image.new("L", (W * aa, H * aa), 0); d = ImageDraw.Draw(im)
    seg = []
    for i in range(len(pts) - 1):
        a, b = pts[i], pts[i + 1]
        L = max(1, int(math.hypot(b[0] - a[0], b[1] - a[1]) / step))
        seg += [(a[0] + (b[0] - a[0]) * k / L, a[1] + (b[1] - a[1]) * k / L) for k in range(L)]
    seg.append(pts[-1])
    n = len(seg)
    for i, (x, y) in enumerate(seg):
        t = i / max(1, n - 1)
        w = w0 + (w1 - w0) * t
        if taper:
            w *= min(1, 0.35 + 2.2 * min(t, 1 - t) + 0.3)
        w *= 1 + rng.uniform(-rough, rough)
        r = max(0.3, w / 2) * aa
        d.ellipse([x * aa - r, y * aa - r, x * aa + r, y * aa + r], fill=255)
    im = im.resize((W, H), Image.LANCZOS)
    return np.clip(np.asarray(im, np.float32) / 255.0, 0, 1)


def arc(p0, p1, bulge, n=16):
    """两点之间一条弧（二次贝塞尔），bulge 是中点往法线方向鼓多少像素（正 = 往左手边）。"""
    (x0, y0), (x1, y1) = p0, p1
    mx, my = (x0 + x1) / 2, (y0 + y1) / 2
    dx, dy = x1 - x0, y1 - y0; L = (dx * dx + dy * dy) ** 0.5 + 1e-6
    cx, cy = mx + (-dy / L) * bulge * 2, my + (dx / L) * bulge * 2
    t = np.linspace(0, 1, n)
    return [((1 - s) ** 2 * x0 + 2 * (1 - s) * s * cx + s * s * x1, (1 - s) ** 2 * y0 + 2 * (1 - s) * s * cy + s * s * y1) for s in t]
