"""城市水彩引擎（Zbukvic 那一路）——颜料按吸光度一层层叠，不是 alpha 盖；纸有高度，颜料会沉、干笔会挂。

核心：
  Paper.D  (H,W,3) 光学密度。最终 = 纸色 * exp(-D)。叠得越多越深、透明，跟真水彩一个道理。
  wash()   Tyler Hobbs 式：底形递归变形，再叠几十层轻微变形的透明层 → 边自然虚化；
           每个顶点可给自己的方差（上限 0.09，再大自交出碎玻璃）；fade / wet_map 控制哪边化开。
  wet()    湿画法：大块、大模糊、低频水痕，可带回流花（bloom）。
  dry()    干笔：颜料只挂在纸纹凸起处，留白色碎闪。
  lift()   擦白 / 留白。
  blur2()  竖横不同的模糊（湿地倒影）。
  grain_from_profile() / spectral_texture()  颜料颗粒：从真画借功率谱、随机相位——只借质感不借画面。

依赖只有 numpy + pillow。
"""
import math
import random

import numpy as np
from PIL import Image, ImageDraw

W, H = 900, 1150
PAPER = np.array([241, 238, 230], dtype=np.float32) / 255.0
_fft_cache = {}
rng = np.random.default_rng(7)
random.seed(7)


def set_size(w, h):
    """改画布尺寸（在建 Paper 之前调）。脚本里 `from watercolor_lib import W, H` 是拷贝，改完请重新读 wc.W / wc.H。"""
    global W, H
    W, H = int(w), int(h)
    _fft_cache.clear()


def set_seed(s):
    global rng
    rng = np.random.default_rng(s)
    random.seed(s)


# ---------- 基础 ----------


def blur(a, sigma):
    if sigma <= 0.3:
        return a
    h, w = a.shape
    key = (h, w, round(sigma, 2))
    if key not in _fft_cache:
        fy = np.fft.fftfreq(h)[:, None]
        fx = np.fft.rfftfreq(w)[None, :]
        _fft_cache[key] = np.exp(-2 * (math.pi ** 2) * (sigma ** 2) * (fx ** 2 + fy ** 2))
    pad = int(min(3 * sigma, 200))
    if pad:
        ap = np.pad(a, pad, mode="edge")
        hh, ww = ap.shape
        k2 = (hh, ww, round(sigma, 2))
        if k2 not in _fft_cache:
            fy = np.fft.fftfreq(hh)[:, None]
            fx = np.fft.rfftfreq(ww)[None, :]
            _fft_cache[k2] = np.exp(-2 * (math.pi ** 2) * (sigma ** 2) * (fx ** 2 + fy ** 2))
        out = np.fft.irfft2(np.fft.rfft2(ap) * _fft_cache[k2], s=ap.shape)
        return out[pad:-pad, pad:-pad]
    return np.fft.irfft2(np.fft.rfft2(a) * _fft_cache[key], s=a.shape)


def blur2(a, sy, sx):
    """竖横不同的模糊（倒影：竖着拉长、横着只糊一点）。"""
    h, w = a.shape
    pad = int(min(3 * max(sy, sx), 200))
    ap = np.pad(a, pad, mode="edge")
    hh, ww = ap.shape
    fy = np.fft.fftfreq(hh)[:, None]
    fx = np.fft.rfftfreq(ww)[None, :]
    g = np.exp(-2 * (math.pi ** 2) * ((sx ** 2) * fx ** 2 + (sy ** 2) * fy ** 2))
    out = np.fft.irfft2(np.fft.rfft2(ap) * g, s=ap.shape)
    return out[pad:-pad, pad:-pad]


def noise(scale_y, scale_x=None, octaves=4, persistence=0.55):
    """分形噪声 0..1。scale 是最低频那一层一个格子多少像素。"""
    scale_x = scale_x or scale_y
    out = np.zeros((H, W), np.float32)
    amp, tot = 1.0, 0.0
    sy, sx = scale_y, scale_x
    for _ in range(octaves):
        gh, gw = max(2, int(H / sy) + 2), max(2, int(W / sx) + 2)
        g = rng.random((gh, gw)).astype(np.float32)
        im = Image.fromarray((g * 65535).astype(np.uint16)).resize((W, H), Image.BICUBIC)
        out += amp * (np.asarray(im, np.float32) / 65535.0)
        tot += amp
        amp *= persistence
        sy, sx = max(1.5, sy / 2), max(1.5, sx / 2)
    out /= tot
    out = (out - out.min()) / (out.max() - out.min() + 1e-6)
    return out


def poly_mask(pts, aa=2):
    im = Image.new("L", (W * aa, H * aa), 0)
    ImageDraw.Draw(im).polygon([(x * aa, y * aa) for x, y in pts], fill=255)
    if aa > 1:
        im = im.resize((W, H), Image.BILINEAR)
    return np.asarray(im, np.float32) / 255.0


def smoothstep(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0 + 1e-6), 0, 1)
    return t * t * (3 - 2 * t)


def spectral_texture(ref_paths, hp=5.5, seed=0):
    """从真画里借「颗粒的粗细分布」：只要功率谱，相位随机——质感跟着参考走，画面一点不借。
    （思路来自 Vesper 9/28 那版 watercolor_mimic_proof.py）"""
    power = np.zeros((H, W), np.float64)
    n = 0
    for p in ref_paths:
        g = np.asarray(Image.open(p).convert("L").resize((W, H), Image.LANCZOS), np.float32) / 255.0
        res = g - blur(g, hp)
        power += np.abs(np.fft.fft2(res)) ** 2
        n += 1
    mag = np.sqrt(power / max(1, n) + 1e-12)
    r = np.random.default_rng(seed)
    phase = r.uniform(-np.pi, np.pi, (H, W))
    tex = np.real(np.fft.ifft2(mag * np.exp(1j * phase)))
    return ((tex - tex.mean()) / (tex.std() + 1e-6)).astype(np.float32)


def grain_from_profile(path, seed=0):
    """用存好的径向功率谱（pigment_profile.npy，由参考画算出，只有一条曲线，不含任何画面）合成颜料颗粒。"""
    f, prof = np.load(path)
    fy = np.fft.fftfreq(H)[:, None]; fx = np.fft.fftfreq(W)[None, :]
    r = np.sqrt(fx ** 2 + fy ** 2)
    mag = np.sqrt(np.interp(r, f, prof))
    ph = np.random.default_rng(seed).uniform(-np.pi, np.pi, (H, W))
    tex = np.real(np.fft.ifft2(mag * np.exp(1j * ph)))
    return ((tex - tex.mean()) / (tex.std() + 1e-6)).astype(np.float32)


# ---------- 纸 ----------
class Paper:
    def __init__(self):
        self.D = np.zeros((H, W, 3), np.float32)
        # 冷压水彩纸：一粒一粒的凸起（细）+ 几毫米一片的起伏（中）+ 纸浆的纤维团（粗）
        bumps = noise(4.5, octaves=2, persistence=0.5)
        bumps = smoothstep(0.25, 0.8, bumps)                            # 凸起有平顶，坑是窄的
        mid = noise(16, octaves=2)
        felt = noise(60, octaves=2)
        h = 0.62 * bumps + 0.26 * mid + 0.12 * felt
        self.height = (h - h.min()) / (h.max() - h.min())
        self.grain = self.height                                        # 干笔挂在凸起上
        hb = blur(self.height, 0.8)
        gy, gx = np.gradient(hb)
        lx, ly = -0.7, -0.7                                             # 光从左上来
        self.shade = np.clip(1 + 2.2 * (-(gx * lx + gy * ly)), 0.975, 1.025).astype(np.float32)
        self.hstreak = noise(9, 70, octaves=3, persistence=0.5)       # 横向拖笔纹
        self.vstreak = noise(60, 5, octaves=3, persistence=0.5)       # 竖向（倒影）
        self.lift_mask = np.zeros((H, W), np.float32)

    # 颜色 -> 每通道吸光系数
    @staticmethod
    def K(color):
        c = np.clip(np.array(color, np.float32) / 255.0, 0.02, 1.0)
        return -np.log(c / PAPER.clip(0.02, 1))

    def add(self, cov, color, strength=1.0, granulate=0.0, edge=0.0, edge_r=2.5):
        cov = np.clip(cov, 0, 1)
        k = self.K(color)
        m = cov.copy()
        if granulate:
            m = m * (1 - granulate + 2 * granulate * self.grain)
        if not np.isscalar(edge) or edge:
            e = np.clip(cov - blur(cov, edge_r), 0, 1)
            m = m + edge * e * 2.2
        self.D += m[..., None] * k[None, None, :] * strength

    def lift(self, cov, amount=1.0):
        """擦掉/留白：把已经上的颜色按比例拿掉。"""
        self.D *= (1 - np.clip(cov * amount, 0, 1))[..., None]

    def render(self, tone=None, paper_strength=1.0, pigment_tex=None, tex_amount=0.14):
        # 颜料往坑里沉：同一层颜色，坑里深、凸起上浅
        pool = 1 + paper_strength * 0.16 * (0.5 - self.height)
        if pigment_tex is not None:
            # 颜料自己的颗粒：只长在有颜料的地方，白纸干净
            pool = pool * (1 + tex_amount * np.clip(pigment_tex, -2.5, 2.5))
        img = PAPER[None, None, :] * np.exp(-self.D * pool[..., None])
        # 纸面凸起被左上的光照出的明暗（白纸上也看得见）
        dens = 1 - np.exp(-self.D.mean(axis=2) * 1.5)             # 这里颜料有多浓
        img *= (1 + paper_strength * (self.shade - 1) * (0.25 + 0.75 * dens))[..., None]
        img *= (0.985 + 0.02 * self.height)[..., None]
        img = np.clip(img, 0, 1)
        return Image.fromarray((img * 255).astype(np.uint8), "RGB")


# ---------- 形状变形（Hobbs）----------
def deform(pts, var, depth):
    """pts: [(x,y)]；var: 每个顶点的方差系数（相对边长）。"""
    for _ in range(depth):
        npts, nvar = [], []
        n = len(pts)
        for i in range(n):
            a, va = pts[i], var[i]
            c, vc = pts[(i + 1) % n], var[(i + 1) % n]
            L = math.hypot(c[0] - a[0], c[1] - a[1])
            vm = (va + vc) / 2 * random.uniform(0.7, 1.3)
            bx = (a[0] + c[0]) / 2 + random.gauss(0, vm * L)
            by = (a[1] + c[1]) / 2 + random.gauss(0, vm * L)
            npts += [a, (bx, by)]
            nvar += [va, vm]
        pts, var = npts, nvar
    return pts, var


def wash(paper, pts, color, strength=1.0, var=0.12, layers=30, base_depth=4, layer_depth=3,
         soft=0.8, mottle=0.35, granulate=0.15, edge=0.35, texture_scale=40, fade=None, wet_map=None, wet_r=12):
    """一块湿中带干的色。var 可以是标量或每个顶点一个值（上限 0.09，再大多边形会自交出碎玻璃）。
    fade: (H,W) 乘子，用来让某一边化开（lost edge），比拿大方差硬撕干净。"""
    if np.isscalar(var):
        var = [var] * len(pts)
    var = [min(v, 0.09) for v in var]
    base, bvar = deform(list(pts), list(var), base_depth)
    cov = np.zeros((H, W), np.float32)
    tex = noise(texture_scale, octaves=3)
    for i in range(layers):
        p, _ = deform(base, [v * 0.9 for v in bvar], layer_depth)
        m = poly_mask(p, aa=1)
        # 每层自己的斑驳：只有噪声高于某阈值的地方着色
        thr = rng.uniform(0.15, 0.55) * mottle
        shift_y, shift_x = rng.integers(-30, 30, 2)
        t = np.roll(tex, (shift_y, shift_x), axis=(0, 1))
        cov += m * smoothstep(thr, thr + 0.25, t)
    cov /= layers
    cov = np.clip(cov * 1.35, 0, 1)
    if soft:
        cov = blur(cov, soft)
    if wet_map is not None:
        # 纸湿的地方：颜料往外跑、边软；纸干的地方：边硬、会有沉积边
        spread = blur(cov, wet_r) * (0.85 + 0.3 * noise(50, octaves=2))
        cov = cov * (1 - wet_map) + spread * wet_map
        edge_w = edge * (1 - wet_map)
    else:
        edge_w = edge
    if fade is not None:
        cov = cov * fade
    paper.add(cov, color, strength, granulate=granulate, edge=edge_w)
    return cov


def wet(paper, cov, color, strength=1.0, spread=18, bloom=0.0, granulate=0.1):
    """湿画法：把给定覆盖图晕开，颜料被水冲出低频的深浅。"""
    c = blur(cov, spread)
    n = noise(90, octaves=3)
    c = c * (0.55 + 0.9 * n)
    if bloom:
        # 回流花：一小块水冲进半干的颜料，边缘一圈深、里面淡
        b = noise(55, octaves=4)
        ring = np.exp(-((b - 0.62) / 0.035) ** 2) * 0.9
        c = c * (1 - bloom * smoothstep(0.62, 0.75, b)) + bloom * ring * c
    paper.add(np.clip(c, 0, 1.4), color, strength, granulate=granulate)
    return c


def dry(paper, cov, color, strength=1.0, thresh=0.5, streak="h", sharp=0.08):
    """干笔：颜料只挂在纸纹凸起处。thresh 越高越碎。"""
    s = {"h": paper.hstreak, "v": paper.vstreak, None: paper.grain}[streak]
    tex = 0.55 * s + 0.45 * paper.grain
    t = thresh if np.isscalar(thresh) else thresh
    m = cov * smoothstep(t - sharp, t + sharp, tex)
    paper.add(m, color, strength, granulate=0.0, edge=0.0)
    return m


def stroke_mask(pts, w0, w1=None, taper=True, aa=2, rough=0.15):
    """沿折线画一笔，宽度从 w0 渐到 w1，带点抖。"""
    w1 = w0 if w1 is None else w1
    im = Image.new("L", (W * aa, H * aa), 0)
    d = ImageDraw.Draw(im)
    # 细分
    seg = []
    for i in range(len(pts) - 1):
        a, b = pts[i], pts[i + 1]
        L = max(1, int(math.hypot(b[0] - a[0], b[1] - a[1]) / 2))
        for k in range(L):
            t = k / L
            seg.append((a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t))
    seg.append(pts[-1])
    n = len(seg)
    for i, (x, y) in enumerate(seg):
        t = i / max(1, n - 1)
        w = w0 + (w1 - w0) * t
        if taper:
            w *= min(1, 0.35 + 2.2 * min(t, 1 - t) + 0.3)
        w *= 1 + random.uniform(-rough, rough)
        r = max(0.4, w / 2) * aa
        d.ellipse([x * aa - r, y * aa - r, x * aa + r, y * aa + r], fill=255)
    im = im.resize((W, H), Image.BILINEAR)
    return np.asarray(im, np.float32) / 255.0


def dab(paper, x, y, r, color, strength=1.0, soft=0.8, squash=1.0, rot=0.0):
    """一点：灯、鸟、甩出去的颜料点。"""
    yy, xx = np.ogrid[:H, :W]
    c, s = math.cos(rot), math.sin(rot)
    dx, dy = xx - x, yy - y
    u = (dx * c + dy * s) / r
    v = (-dx * s + dy * c) / (r * squash)
    d = np.sqrt(u * u + v * v)
    m = np.clip(1 - smoothstep(1 - soft * 0.6, 1 + soft * 0.4, d), 0, 1)
    paper.add(m.astype(np.float32), color, strength, edge=0.3, edge_r=1.2)
    return m


def splatter(paper, n, box, color, rmin=0.8, rmax=3.2, strength=1.2):
    x0, y0, x1, y1 = box
    for _ in range(n):
        r = random.uniform(rmin, rmax) ** 1.0
        dab(paper, random.uniform(x0, x1), random.uniform(y0, y1), r, color, strength,
            soft=0.5, squash=random.uniform(0.6, 1.0), rot=random.uniform(0, 3.14))
