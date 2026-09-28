"""雨夜（9/28 她带着磨了六版，「好看！！！」）：只画左边一排楼，右边让给湿马路、一排灯和雾。力气都在光和地上。
她：「还是太规矩了……夜+雨，应该是更多的晕染和变化，以及扭曲。两边建筑不好处理就只画一边，把力气画到光影变化、地面反射、灯光上。」
    「雨夜的光基本上在于地板反光」「橱窗的光可以不是都长一样的」「斜的纹又太工整了，像排线」。
参考：Hassam 雨夜（楼顶化进天、灯往雾里渗、地上一条条碎的竖亮）；林經哲的雨街（光一团团湿着化开、倒影竖着大片晕、积水里清楚积水外碎、满屏留白雨点）。
第一遍：天（湿接湿 + 几笔大斜刷）/ 左边楼一整块（顶化进天）/ 右边雾和秃树 / 地（近深远亮）。缩小看。
第二遍：楼里长东西——五家不一样的店、楼上几点灯、一条青霓虹、红棚子；窗的光在雨里化开（边化 + 一团团湿晕）。
第三遍：地上——楼和店按楼脚翻下来；店的倒影单独一层（狠扭、长拖、横断几截、暖往紫渗）；然后才画灯、车、人（不然会被倒影洗掉）；
        每盏灯拖下来的竖亮（同一张波纹位移场扭，横笔切断，被站在前面的人和车挡住）；车人的深倒影。
最后：很淡的全屏雨丝、灯边上的白雨点和几点深的甩点。
哥看完补了三条（她又校正了一条）：远处那串灯整组化成一团光；近的明确的东西画完整落地，只有远的、跟背景同值的才不完整；焦点附近落几笔硬的书法笔（wc.brush）。
从 skill 根目录跑：python scenes/rainnight.py [seed] [values]      默认 seed 11 = 定稿
"""
import sys, os, math, random, time
import numpy as np
from PIL import Image
HERE = os.path.dirname(os.path.abspath(__file__))
SKILL_DIR = os.path.dirname(HERE)
sys.path.insert(0, SKILL_DIR)
import watercolor_lib as wc

W, H = wc.W, wc.H
t0 = time.time()
SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 11
STAGE = sys.argv[2] if len(sys.argv) > 2 else "full"
wc.set_seed(SEED)
P = wc.Paper()
yy, xx = np.mgrid[:H, :W]
yf, xf = yy.astype(np.float32), xx.astype(np.float32)
S = wc.smoothstep

VP = (540, 578)
HZ = VP[1]

ULTRA = (52, 62, 128)       # 群青
INDIGO = (30, 34, 66)
VIOLET = (96, 80, 132)
DEEP_WARM = (78, 54, 66)    # 暗里掺暖（紫褐），不死黑
FOG = (150, 150, 176)
GLOW = (246, 212, 132)
AMBER = (236, 160, 76)
ORANGE = (226, 120, 58)
RED = (200, 48, 44)
PALE = (250, 240, 206)


def isolated(fn):
    st, rs = random.getstate(), wc.rng.bit_generator.state
    fn()
    random.setstate(st); wc.rng.bit_generator.state = rs


def lineL(x, y0):
    """过消失点、在 x=0 处高 y0 的线。"""
    return HZ + (y0 - HZ) * (VP[0] - np.asarray(x, np.float32)) / VP[0]


def lineR(x, y9):
    """过消失点、在 x=W 处高 y9 的线。"""
    return HZ + (y9 - HZ) * (np.asarray(x, np.float32) - VP[0]) / (W - VP[0])


FOOT0 = 700          # 楼脚在 x=0 处
CURBL0 = 790         # 左边马路牙子
CURBR9 = 706         # 右边马路牙子（x=W 处）
EYE = lambda y: max(1.0, y - HZ)     # 人眼高：站在街上看，人头都在地平线上


def scale_at(y):
    return max(1.0, float(y) - HZ)


def roof_T(hn):
    """楼高 hn 个眼高 → 屋顶线在 x=0 处的 y。"""
    return HZ - (hn - 1) * (FOOT0 - HZ)


def skyline(segs):
    pts = []
    for xa, xb, hn, orn in segs:
        T = roof_T(hn)
        pts.append((xa, float(lineL(xa, T))))
        for kind, u, s in orn:
            x = xa + (xb - xa) * u
            y = float(lineL(x, T))
            f = (VP[0] - x) / VP[0]
            s = s * max(0.2, f)
            if kind == "chim":
                pts += [(x - s * 0.16, y), (x - s * 0.16, y - s * 0.55), (x + s * 0.16, y - s * 0.55), (x + s * 0.16, y)]
            elif kind == "mansard":
                pts += [(x - s * 0.7, y), (x - s * 0.52, y - s * 0.42), (x + s * 0.52, y - s * 0.42), (x + s * 0.7, y)]
            elif kind == "dome":
                for a in np.linspace(math.pi, 0, 9):
                    pts.append((x + s * 0.5 * math.cos(a), y - s * 0.55 * math.sin(a)))
        pts.append((xb, float(lineL(xb, T))))
    return pts


near = np.clip((yf - HZ) / (H - HZ), 0, 1).astype(np.float32)
footL = lineL(xf, FOOT0)
curbL = lineL(xf, CURBL0)
curbR = lineR(xf, CURBR9)
groundL = S(footL - 1, footL + 2, yf) * (xf < VP[0])                     # 左边楼脚以下（人行道 + 街）
groundR = S(HZ - 4, HZ + 10, yf) * (xf >= VP[0])                         # 右边地平线在雾里，软
ground = np.maximum(groundL, groundR).astype(np.float32)
cityglow = np.exp(-(((xf - VP[0] - 60) / 300.0) ** 2 + ((yf - HZ + 20) / 190.0) ** 2)).astype(np.float32)


def backrun(cx, cy, r, color, strength=0.3, lift=0.2, squash=1.0, mask=None):
    """一朵回流花：半干的颜料里掉进一滴水——里面被冲淡，边上一圈颜料堆成菜花边。只放几处，不铺满。"""
    d = np.hypot(xf - cx, (yf - cy) / squash) / r
    n = wc.noise(max(3.0, r * 0.3), octaves=3)
    sh = d + 0.55 * (n - 0.5)
    inside = (1 - S(0.92, 1.0, sh)).astype(np.float32)
    rim = np.exp(-((sh - 0.98) / 0.035) ** 2).astype(np.float32)
    if mask is not None:
        inside, rim = inside * mask, rim * mask
    P.lift(inside * lift, 1.0)
    P.add(rim * S(0.25, 0.5, P.grain + 0.2), color, strength)


# ================= 第一遍：只有大块 =================
# 1. 天：一整片湿的群青，上深，往街尽头和右边雾里亮、暖。湿接湿，不开满天的花
sky = (1 - ground).astype(np.float32)
top = np.clip(1 - yf / 650.0, 0, 1) ** 0.9
P.add((sky * (0.3 + 0.7 * top) * (1 - 0.75 * cityglow)).astype(np.float32), ULTRA, 0.85, granulate=0.08)
wc.wet(P, (sky * top * S(0.35, 0.8, wc.noise(160, 240, octaves=3))).astype(np.float32), INDIGO, strength=0.5, spread=45)
wc.wet(P, (sky * S(0.4, 0.8, wc.noise(130, 110, octaves=3)) * S(0.1, 0.8, top) * (1 - cityglow)).astype(np.float32), VIOLET, strength=0.35, spread=35)
wc.wet(P, (sky * cityglow).astype(np.float32), (214, 170, 140), strength=0.3, spread=26)
P.lift((sky * cityglow * 0.4).astype(np.float32), 1.0)

def _sky_strokes():
    """天上掉几笔大的斜刷：湿接湿，一边化开一边留一道干了的水线；颜色往里掉一点玫瑰紫、一点冷青、一点暖灰。"""
    random.seed(SEED + 20)
    def band(cx, cy, ang, L, w):
        ca, sa = math.cos(ang), math.sin(ang)
        u = (xf - cx) * ca + (yf - cy) * sa
        v = -(xf - cx) * sa + (yf - cy) * ca
        rag = (wc.noise(max(6.0, w * 0.4), max(6.0, w * 0.9), octaves=3) - 0.5) * w * 0.7
        v = v + rag
        across = S(-w, -w * 0.55, v) * (1 - S(w * 0.35, w * 0.6, v))        # 下沿干得快，边硬一点
        along = S(-L / 2, -L / 2 + L * 0.25, u) * (1 - S(L / 2 - L * 0.35, L / 2, u))
        return (across * along).astype(np.float32)
    cols = [(INDIGO, 0.45), (VIOLET, 0.4), ((150, 80, 130), 0.3), ((60, 104, 146), 0.35), ((118, 100, 118), 0.3), (INDIGO, 0.4), ((150, 80, 130), 0.25)]
    for k, (col, stv) in enumerate(cols):
        cx = random.uniform(80, 900); cy = random.uniform(30, 480)
        ang = math.radians(random.uniform(-32, -14))
        m = band(cx, cy, ang, random.uniform(380, 760), random.uniform(50, 120)) * sky * (1 - 0.8 * cityglow)
        P.add(m, col, stv, granulate=0.2, edge=0.3, edge_r=3)
        wc.wet(P, m * 0.6, col, strength=stv * 0.4, spread=16)
    # 几处被冲淡的亮：云缝里透着城市的光，湿的时候用纸巾吸掉
    for k in range(3):
        cx = random.uniform(420, 880); cy = random.uniform(160, 430)
        m = band(cx, cy, math.radians(random.uniform(-28, -12)), random.uniform(260, 460), random.uniform(30, 60)) * sky
        P.lift((wc.blur(m, 5) * 0.3).astype(np.float32), 1.0)
        wc.wet(P, m * 0.5, (210, 170, 150), strength=0.12, spread=10)
    # 一朵很淡的回流花，只一处
    # 雨丝：天上斜斜的一层，很淡
    from PIL import ImageDraw
    im = Image.new("L", (W, H), 0); d = ImageDraw.Draw(im)
    for _ in range(420):
        x0, y0 = random.uniform(0, W + 60), random.uniform(0, 600)
        Lr = random.uniform(20, 70)
        d.line([(x0, y0), (x0 - Lr * 0.22, y0 + Lr)], fill=random.randint(80, 200), width=1)
    rn = wc.blur(np.asarray(im, np.float32) / 255.0, 0.6)
    P.lift((rn * sky * 0.12).astype(np.float32), 1.0)
isolated(_sky_strokes)

# 2. 右边：没有楼，只有一团雾里的树和远处的房子，边全化开，比天深一点点
u = np.clip((xf - VP[0]) / 360.0, 0, 1)
far_top = HZ - 40 - 90 * u - 60 * wc.noise(30, 50, octaves=3) * u
fogmass = (S(far_top - 14, far_top + 14, yf) * (1 - S(HZ + 4, HZ + 30, yf)) * S(VP[0] - 10, VP[0] + 50, xf)).astype(np.float32)
wc.wet(P, fogmass * (0.6 + 0.4 * wc.noise(40, 30, octaves=2)), (80, 82, 124), strength=0.5, spread=6)
wc.wet(P, fogmass * S(0.45, 0.75, wc.noise(50, 40, octaves=2)), DEEP_WARM, strength=0.2, spread=8)

def _trees():
    """右边雾里几棵光秃秃的树：比雾深一点点，树梢化进天里，树脚没进雾——Hassam 那张左边的树。"""
    from PIL import ImageDraw
    random.seed(SEED + 9)
    aa = 2
    im = Image.new("L", (W * aa, H * aa), 0)
    d = ImageDraw.Draw(im)
    def br(x, y, ang, L, w, depth):
        x2, y2 = x + L * math.cos(ang), y - L * math.sin(ang)
        mx = (x + x2) / 2 + random.uniform(-0.1, 0.1) * L
        my = (y + y2) / 2 + random.uniform(-0.1, 0.1) * L
        for (p, q_) in (((x, y), (mx, my)), ((mx, my), (x2, y2))):
            d.line([(p[0] * aa, p[1] * aa), (q_[0] * aa, q_[1] * aa)], fill=255, width=max(1, int(w * aa)))
        if depth == 0:
            return
        n = random.choice([2, 2, 3])
        for k in range(n):
            br(x2, y2, ang + (k - (n - 1) / 2) * random.uniform(0.35, 0.6) + random.uniform(-0.25, 0.25), L * random.uniform(0.62, 0.8), w * 0.6, depth - 1)
    for x0, s_mul in ((692, 1.0), (782, 1.05), (872, 0.95)):
        base = float(lineR(x0, CURBR9)) - 6
        s = scale_at(base)
        hgt = 3.4 * s * s_mul
        br(x0, base, math.pi / 2 + random.uniform(-0.06, 0.06), hgt * 0.36, 0.07 * s, 6)
    m = np.asarray(im.resize((W, H), Image.BILINEAR), np.float32) / 255.0
    m = wc.blur(m, 0.8) * (0.55 + 0.45 * S(0.3, 0.6, P.grain))
    m = m * (1 - 0.7 * S(HZ - 60, HZ + 10, yf)) * (0.55 + 0.45 * wc.noise(40, octaves=2))
    wc.wet(P, m.astype(np.float32), (66, 68, 112), strength=0.55, spread=1.2)
    wc.wet(P, wc.blur(m, 6).astype(np.float32), (90, 88, 128), strength=0.25, spread=6)
isolated(_trees)

# 3. 左边楼：一整块深群青，楼顶化进天（湿度高、边碎）；楼里暗处掺暖
L_segs = [(-60, 140, 12, []),
          (140, 290, 8.5, [("chim", 0.3, 70), ("chim", 0.72, 55)]),
          (290, 380, 10, [("mansard", 0.5, 130)]),
          (380, 440, 7.5, [("chim", 0.55, 70)]),
          (440, 490, 9.5, [("dome", 0.5, 170)]),
          (490, 520, 7, []),
          (520, 540, 8, [])]
sk = skyline(L_segs)
bld = sk + [(540, float(lineL(540, FOOT0))), (-60, float(lineL(-60, FOOT0)))]
roof = np.interp(xf[0], [p[0] for p in sk], [p[1] for p in sk])[None, :]
topzone = np.clip(1 - (yf - roof) / 150.0, 0, 1)                       # 楼顶往下 150px 是湿的
wetB = np.clip(0.12 + 0.6 * topzone * wc.noise(30, 40, octaves=3) + 0.5 * cityglow, 0, 0.95).astype(np.float32)
fadeB = (1 - S(footL - 3, footL + 5, yf)).astype(np.float32)
covB = wc.wash(P, bld, (40, 46, 92), strength=1.0, var=[0.02] * (len(bld) - 2) + [0.03] * 2, layers=28, edge=0.35, granulate=0.1,
               fade=fadeB, wet_map=wetB, wet_r=9)
wc.wet(P, (covB * S(0.45, 0.7, wc.noise(70, 30, octaves=3))).astype(np.float32), DEEP_WARM, strength=0.5, spread=10)
wc.wet(P, (covB * S(0.5, 0.75, wc.noise(90, 20, octaves=3))).astype(np.float32), INDIGO, strength=0.4, spread=8)
# 楼顶有几处干脆化没了：天的颜色湿着冲进楼顶
lost = (covB * topzone * S(0.52, 0.7, wc.noise(24, 36, octaves=3))).astype(np.float32)
lost = wc.blur(lost, 4)
P.lift(lost * 0.35, 1.0)
wc.wet(P, lost, ULTRA, strength=0.2, spread=10)
backrun(250, float(lineL(250, roof_T(8.5))) + 60, 45, INDIGO, 0.4, 0.12, squash=1.3, mask=covB)
# 远处几栋按栋往光里退，每栋浅一档，被城市的暖吃一点
for (xa, xb, hn, _), amt in zip(L_segs[-4:], (0.16, 0.3, 0.44, 0.58)):
    blk = (S(xa - 2, xa + 2, xf) * (1 - S(xb - 2, xb + 2, xf))).astype(np.float32)
    P.lift((covB * blk * amt).astype(np.float32), 1.0)
    P.add((covB * blk * amt).astype(np.float32), (160, 128, 132), 0.25)

# 4. 地：整条街湿的。远处反着街尽头那团暖（亮），往我们这边越来越深（反的是头顶的深天）
gtone = ground * (0.25 + 0.9 * near ** 0.7)
P.add((gtone * (1 - 0.6 * cityglow)).astype(np.float32), (62, 66, 128), 1.0, granulate=0.25)
wc.wet(P, (ground * near * S(0.3, 0.75, wc.noise(70, 220, octaves=3))).astype(np.float32), INDIGO, strength=0.55, spread=16)
wc.wet(P, (ground * S(0.45, 0.72, wc.noise(50, 140, octaves=3))).astype(np.float32), VIOLET, strength=0.35, spread=10)
wc.wet(P, (ground * (1 - near) * np.exp(-(((xf - VP[0] - 40) / 220.0) ** 2))).astype(np.float32), (214, 170, 140), strength=0.3, spread=18)
P.lift((ground * (1 - near) ** 2 * np.exp(-(((xf - VP[0] - 30) / 160.0) ** 2)) * 0.45).astype(np.float32), 1.0)

if STAGE == "values":
    img = P.render()
    out = os.path.join(HERE, "rainnight_values.png")
    img.save(out)
    img.convert("L").resize((64, 82), Image.LANCZOS).resize((256, 328), Image.NEAREST).save(out.replace(".png", "_thumb.png"))
    print("values", out, round(time.time() - t0, 1)); sys.exit()

LITWIN = np.zeros((H, W), np.float32)       # 所有亮着的窗和橱窗，最后一起在雨里化开
GLOWS = {}                                   # 颜色 → 这种光的窗（按亮度加权）：每种光自己晕、自己倒影
RAIN_K = -0.2                                # 雨丝的斜度：往下走一格，往左偏 0.2 格
_sh = np.rint(RAIN_K * (yf - H / 2)).astype(np.int32)
_xs_sh = np.clip(xx + _sh, 0, W - 1)
_xs_un = np.clip(xx - _sh, 0, W - 1)


def rain_blur(a, sy, sx):
    """顺着雨丝的方向糊：先把画面错切成雨丝竖直，竖着拉，再切回去。"""
    b = a[yy, _xs_sh]
    b = wc.blur2(b.astype(np.float32), sy, sx)
    return b[yy, _xs_un]


def add_glow(col, m, w=1.0):
    GLOWS[col] = np.maximum(GLOWS.get(col, np.zeros((H, W), np.float32)), (m * w).astype(np.float32))

LIGHTS = []   # (x, y, 贴地 y, 半径, 颜色, 力度, 倒影长度系数)
DX = ((wc.noise(2.6, 40, octaves=3) - 0.5) * 2 * (1.5 + 16 * near)).astype(np.float32)    # 水面波纹：一横条一横条各自往左右错
_hold = []
isolated(lambda: _hold.append(S(0.36, 0.6, wc.noise(150, 340, octaves=2)).astype(np.float32)))
CALM = _hold[0]                      # 1 = 碎，0 = 一整片平静的镜面：波纹整片整片地没有，不是每处都抖一点（哥）
DX = (DX * (0.12 + 0.88 * CALM)).astype(np.float32)


def glow_at(m, radius, color, amt, lift=0.5):
    """光往湿的暗里渗：高斯 ×（0.4 + 噪声），再让它自己开花；外沿一圈沉积。"""
    g = wc.blur(m.astype(np.float32), radius)
    g = g / (g.max() + 1e-6)
    rag = np.clip(g * (0.4 + 1.0 * wc.noise(max(4.0, radius * 1.4), octaves=3)), 0, 1)
    P.lift((np.clip(rag, 0, 1) ** 0.7 * amt * lift).astype(np.float32), 1.0)
    wc.wet(P, (rag * amt).astype(np.float32), color, strength=0.35, spread=max(1.5, radius * 0.22))


# ================= 第二遍：楼里长东西 =================
# 楼层：几条若有若无的檐口线（湿的，断的），往消失点收
random.seed(SEED + 1)
def _floors():
    for hn in (2.3,):
        T = HZ - (hn - 1) * (FOOT0 - HZ)
        xs0 = np.linspace(-20, 530, 40)
        pts = [(x, float(lineL(x, T))) for x in xs0 if lineL(x, T) > roof[0, int(min(W - 1, max(0, x)))] + 6]
        if len(pts) < 3:
            continue
        m = wc.stroke_mask(pts, 2.2, 0.6, taper=False, rough=0.5) * covB
        m = m * S(0.45, 0.6, 0.6 * P.hstreak + 0.4 * wc.noise(30, 120, octaves=2))
        P.lift((m * 0.35).astype(np.float32), 1.0)
        P.add((np.roll(m, 3, axis=0) * 0.8).astype(np.float32), INDIGO, 0.35)
    # 楼与楼之间：缝左边一条竖着的淡深色，上下断续
    for xa, xb, hn, _ in L_segs[1:]:
        yt = float(roof[0, int(max(0, min(W - 1, xa)))])
        yb_ = float(lineL(xa, FOOT0))
        st = wc.stroke_mask([(xa - 2, yt), (xa - 1, yb_)], 7 * (VP[0] - xa) / VP[0] + 1.5, 2.0, taper=False, rough=0.3)
        P.add((st * S(0.35, 0.6, P.vstreak) * covB).astype(np.float32), INDIGO, 0.5)
isolated(_floors)

# 楼上的窗：大多是比墙深一点的洞，几扇亮着暖灯（稀、不齐、边化开），远的更小更淡
def _windows():
    random.seed(SEED + 2)
    PRES = wc.noise(170, 140, octaves=2)          # 暗窗只留一两组，其余整层整栋没有
    for (xa, xb, hn, _) in L_segs:
        for fl in (2.7, 4.5, 6.3, 8.1, 9.9):
            if fl + 1.0 > hn - 0.4:
                continue
            x = max(xa, -10) + random.uniform(4, 14)
            while x < xb - 6:
                s = scale_at(lineL(x, FOOT0))
                if s < 10:
                    break
                ww, wh = 0.26 * s * random.uniform(0.85, 1.15), 0.95 * s * random.uniform(0.9, 1.05)
                Tt = HZ - (fl + 1.0 - 1) * (FOOT0 - HZ)
                ytop = float(lineL(x, Tt)) + random.uniform(-2, 2)
                q = [(x, ytop), (x + ww, float(lineL(x + ww, Tt))), (x + ww, float(lineL(x + ww, Tt)) + wh * 0.97), (x, ytop + wh)]
                r = random.random()
                if r < 0.1:
                    m = wc.poly_mask(q)
                    col = random.choice([AMBER, GLOW, ORANGE])
                    LITWIN[:] = np.maximum(LITWIN, m)
                    add_glow(col, m, 0.55)
                    P.lift((np.clip(wc.blur(m, 1.4) * (0.75 + 0.35 * wc.noise(4, octaves=2)), 0, 1) * 0.88).astype(np.float32), 1.0)
                    wc.wash(P, q, col, strength=0.75, var=0.05, layers=4, edge=0.25, wet_map=np.full((H, W), 0.7, np.float32), wet_r=2.5)
                    glow_at(m, 0.25 * s + 2, col, 0.45, lift=0.35)
                elif r < 0.55 and PRES[int(min(H - 1, max(0, ytop))), int(min(W - 1, max(0, x)))] > 0.6:
                    wc.wash(P, q, (24, 24, 44), strength=random.uniform(0.3, 0.6), var=0.03, layers=3, edge=0.3,
                            wet_map=np.full((H, W), 0.5, np.float32), wet_r=1.5)
                x += (0.26 * s + 0.55 * s) * random.uniform(0.85, 1.25)
isolated(_windows)

# 一楼：一整排店，玻璃里是暖的，柱子是深的；一家咖啡馆挂红棚子（全画最暖最艳的一块）
SHOP_TOP = HZ - 0.95 * (FOOT0 - HZ)        # 店面玻璃上沿 1.95 个眼高
SHOP_BOT = HZ + 0.9 * (FOOT0 - HZ)         # 下沿离地 0.1 个眼高
SHOPS = []
SHOPQ = []
COOLW = (222, 236, 232)     # 日光灯：偏冷的白
NEON_R = (238, 84, 104)
NEON_T = (96, 214, 204)
def _shops():
    random.seed(SEED + 3)
    rs = random.Random(SEED + 33)             # 每家店长什么样用自己一条随机数，不打乱店的位置
    x = -30.0
    wetm = np.full((H, W), 0.55, np.float32)
    while x < 522:
        s = scale_at(lineL(x, FOOT0))
        ww = (0.9 * s) * random.uniform(0.6, 1.5)
        r = random.random()
        if r < 0.78 and s > 6:
            top_ = SHOP_TOP + random.uniform(-0.25, 0.2) * (FOOT0 - HZ)
            bot_ = SHOP_BOT if random.random() < 0.7 else SHOP_BOT - random.uniform(0.2, 0.5) * (FOOT0 - HZ)
            q = [(x, float(lineL(x, top_))), (x + ww, float(lineL(x + ww, top_))), (x + ww, float(lineL(x + ww, bot_))), (x, float(lineL(x, bot_)))]
            m = wc.poly_mask(q)
            ms = np.clip(wc.blur(m, 1.5) * (0.8 + 0.4 * wc.noise(6, 10, octaves=2)), 0, 1)
            random.choice([0, 1, 2, 3, 4])
            kind = rs.choices(["warm", "amber", "white", "dim", "deep"], weights=[3, 2, 1.3, 1.2, 1.0])[0]
            col = {"warm": GLOW, "amber": AMBER, "white": COOLW, "dim": AMBER, "deep": ORANGE}[kind]
            b = {"warm": rs.uniform(0.85, 1.1), "amber": rs.uniform(0.7, 1.0), "white": rs.uniform(0.9, 1.15), "dim": rs.uniform(0.35, 0.5), "deep": rs.uniform(0.6, 0.85)}[kind]
            LITWIN[:] = np.maximum(LITWIN, m * min(1.0, b + 0.2))
            add_glow(col, m, b)
            P.lift(np.clip(ms * (0.8 + 0.45 * b), 0, 1).astype(np.float32), 1.0)
            wc.wash(P, q, col, strength=0.45 if kind != "white" else 0.25, var=0.035, layers=5, edge=0.25, wet_map=wetm, wet_r=2.5)
            if kind == "dim":
                wc.wet(P, ms.astype(np.float32), (120, 70, 70), strength=0.45, spread=2)       # 快打烊的店：里面暗一截
            # 玻璃里下半截最亮（灯照着地板和货），边上掉一圈橙
            ring = np.clip(ms - wc.blur(ms, 0.12 * s + 1), 0, 1)
            wc.wet(P, (ring * 2).astype(np.float32), ORANGE, strength=0.35, spread=1.5)
            # 里面几块湿的深影：人、货架、一张桌子，不描边
            for k in range(random.randint(1, 3)):
                xk = x + ww * random.uniform(0.15, 0.85)
                hk = random.uniform(0.45, 0.85) * s
                yk = float(lineL(xk, bot_))
                wc.wash(P, [(xk - 0.08 * s, yk - hk), (xk + 0.06 * s, yk - hk * 1.02), (xk + 0.1 * s, yk), (xk - 0.1 * s, yk)], (80, 52, 60),
                        strength=0.5, var=0.06, layers=3, edge=0.15, wet_map=np.full((H, W), 0.8, np.float32), wet_r=2.5)
            if kind != "white":
                wc.wet(P, (ms * (0.4 + 0.6 * np.clip((yf - float(lineL(x, top_))) / (0.9 * s + 1), 0, 1))).astype(np.float32), AMBER, strength=0.35, spread=2)
            else:
                wc.wet(P, (ms * S(0.5, 0.7, wc.noise(8, 20, octaves=2))).astype(np.float32), (150, 190, 190), strength=0.2, spread=2)
            if s > 30 and rs.random() < 0.3:                   # 一条霓虹招牌：在玻璃上沿上面，细、亮、颜色艳
                ncol = rs.choice([NEON_R, NEON_T]); ncol = NEON_T      # 红的留给棚子和尾灯，招牌用青
                ty = top_ - 0.28 * (FOOT0 - HZ)
                x1, x2 = x + ww * rs.uniform(0.05, 0.25), x + ww * rs.uniform(0.6, 0.95)
                nm = wc.stroke_mask([(x1, float(lineL(x1, ty))), (x2, float(lineL(x2, ty)))], 0.07 * s + 1.5, 0.06 * s + 1.2, taper=False, rough=0.2)
                P.lift(np.clip(nm * 1.2, 0, 1).astype(np.float32), 1.0)
                P.add(nm, ncol, 0.55)
                LITWIN[:] = np.maximum(LITWIN, nm)
                add_glow(ncol, nm, 1.0)
            nk = random.choice([0, 1, 1, 2])
            for k in range(nk):                              # 窗框竖档：细、深，上下被光吃掉一点
                xk = x + ww * (k + 1) / (nk + 1) + random.uniform(-2, 2)
                mk = wc.stroke_mask([(xk, float(lineL(xk, top_))), (xk, float(lineL(xk, bot_)))], 0.04 * s + 0.8, 0.035 * s + 0.6, taper=False, rough=0.08)
                P.add((mk * (0.6 + 0.4 * S(0.3, 0.6, P.grain))).astype(np.float32), (58, 42, 60), 0.6)
            glow_at(ms, 0.45 * s + 4, col, 0.8 * b, lift=0.5)
            SHOPS.append((x + ww / 2, float(lineL(x + ww / 2, FOOT0)), ww, col, s))
            SHOPQ.append((x, ww, top_, bot_, s, kind))
        x += ww + 0.2 * s * random.uniform(0.5, 1.8)
isolated(_shops)


def _neon():
    """至少一块霓虹：青的，挂在中段一家店的玻璃上沿——整条街的暖里掺一点冷的艳。"""
    if NEON_R in GLOWS or NEON_T in GLOWS:
        return
    x, ww, top_, bot_, s, kind = min(SHOPQ, key=lambda q: abs(q[0] + q[1] / 2 - 300))
    ty = top_ - 0.22 * (FOOT0 - HZ)
    x1, x2 = x + ww * 0.12, x + ww * 0.78
    nm = wc.stroke_mask([(x1, float(lineL(x1, ty))), (x2, float(lineL(x2, ty)))], 0.08 * s + 1.5, 0.07 * s + 1.2, taper=False, rough=0.25)
    P.lift(np.clip(nm * 1.2, 0, 1).astype(np.float32), 1.0)
    P.add(nm, NEON_T, 0.5)
    LITWIN[:] = np.maximum(LITWIN, nm)
    add_glow(NEON_T, nm, 1.0)
isolated(_neon)
# 红棚子：一段，棚底吃店里的光
def _awning():
    cand = [sh for sh in SHOPS if 120 < sh[0] < 360] or SHOPS
    sx, sy, sw, _, s_ = min(cand, key=lambda sh: abs(sh[0] - 305))
    xa, xb = sx - sw / 2 - 6, sx + sw / 2 + 6
    t1, t2 = SHOP_TOP - 16, SHOP_TOP + 26
    q = [(xa, float(lineL(xa, t1))), (xb, float(lineL(xb, t1))), (xb + 6, float(lineL(xb + 6, t2)) + 8), (xa - 8, float(lineL(xa - 8, t2)) + 14)]
    wc.wash(P, q, (170, 40, 44), strength=0.9, var=0.02, layers=8, edge=0.45, wet_map=np.full((H, W), 0.3, np.float32), wet_r=2)
    wc.wet(P, wc.poly_mask(q) * S(0.5, 0.7, wc.noise(10, 30, octaves=2)), (90, 30, 50), strength=0.4, spread=3)
    edge_ = wc.stroke_mask([(xa - 8, float(lineL(xa - 8, t2)) + 14), (xb + 6, float(lineL(xb + 6, t2)) + 8)], 2.5, 1.5, taper=False, rough=0.3)
    P.lift((edge_ * S(0.3, 0.55, P.grain) * 0.5).astype(np.float32), 1.0)
    P.add(edge_ * 0.6, AMBER, 0.4)
    LIGHTS.append(((xa + xb) / 2, float(lineL((xa + xb) / 2, t2)) + 10, float(lineL((xa + xb) / 2, FOOT0)) + 4, sw * 0.3, RED, 0.35, 0.9))
isolated(_awning)
def _halation():
    """雨夜里窗的光不是一团雾：玻璃的边化开，贴着窗一小圈暖，然后顺着雨丝的方向一道道往下淌（林經哲那张楼面）。"""
    zone = np.clip(wc.blur(LITWIN, 7) * 2.6, 0, 1) * (0.7 + 0.3 * wc.noise(12, octaves=2))
    Db = np.stack([wc.blur(P.D[..., c], 4.5) for c in range(3)], -1)
    P.D = P.D * (1 - zone[..., None]) + Db * zone[..., None]            # 边化掉：像湿纸上颜料往外跑
    g = wc.blur(LITWIN, 4) * 1.0 + wc.blur(LITWIN, 9) * 0.7
    h1 = np.clip((1 - np.exp(-2.2 * g)) * (0.7 + 0.5 * wc.noise(12, octaves=3)), 0, 1).astype(np.float32)
    P.lift((h1 * (0.8 - 0.3 * LITWIN)).astype(np.float32), 1.0)
    wc.wet(P, (h1 * 0.7).astype(np.float32), GLOW, strength=0.22, spread=3)
    rs = random.Random(SEED + 50)
    lit_all = np.zeros((H, W), np.float32)
    ROSE = (238, 130, 146)
    for col, M in GLOWS.items():
        warm = col in (GLOW, AMBER, ORANGE, PALE)
        area = float(M.sum())
        # 1. 几团湿晕：大小不一、边有软有硬，颜色在里面变
        g = wc.blur(M, 9) * 0.9 + wc.blur(M, 28) * 1.3
        blob = (1 - np.exp(-2.4 * g)) * np.clip(0.15 + 1.2 * (0.55 * wc.noise(18, octaves=3) + 0.45 * wc.noise(50, octaves=2)), 0, 1.3)
        blob = np.clip(blob, 0, 1) * (1 - 0.5 * M)
        P.lift((blob * 0.72).astype(np.float32), 1.0)
        wc.wet(P, (blob * 0.8).astype(np.float32), col, strength=0.3, spread=3)
        if warm:
            cz = S(0.55, 0.7, wc.noise(26, octaves=2))
            wc.wet(P, (blob * cz).astype(np.float32), ROSE, strength=0.22, spread=4)        # 暖里掉一点玫瑰
            wc.wet(P, (blob * (1 - cz) * S(0.3, 0.6, wc.noise(20, octaves=2))).astype(np.float32), ORANGE, strength=0.2, spread=4)
        lit_all = np.maximum(lit_all, blob.astype(np.float32))
    from PIL import ImageDraw
    im = Image.new("L", (W, H), 0); d = ImageDraw.Draw(im)
    random.seed(SEED + 21)
    for _ in range(900):
        x0, y0 = random.uniform(-40, 600), random.uniform(120, 740)
        Lr = random.uniform(10, 40)
        d.line([(x0, y0), (x0 + RAIN_K * Lr, y0 + Lr)], fill=random.randint(100, 255), width=1)
    rn = wc.blur(np.asarray(im, np.float32) / 255.0, 0.45)
    P.lift((rn * np.clip(h1 * 1.2 + lit_all * 0.8, 0, 1) * 0.4).astype(np.float32), 1.0)
    warmwin = np.zeros((H, W), np.float32)
    for col, M in GLOWS.items():
        if col in (GLOW, AMBER, ORANGE):
            warmwin = np.maximum(warmwin, M)
    ww_ = np.clip(wc.blur(warmwin, 2), 0, 1)
    wc.wet(P, (ww_ * (0.5 + 0.5 * wc.noise(10, 6, octaves=2))).astype(np.float32), AMBER, strength=0.32, spread=2)   # 玻璃里的暖找回来，中间留白
    wc.wet(P, np.clip(ww_ - wc.blur(ww_, 6), 0, 1).astype(np.float32) * 2, ORANGE, strength=0.3, spread=2)
isolated(_halation)


def _shop_insides():
    """化开以后橱窗里又太空：掉几块湿的深——人影、货架、柜台——边全是软的，像隔着有水的玻璃。"""
    rs = random.Random(SEED + 34)
    for x, ww, top_, bot_, s, kind in SHOPQ:
        if s < 14:
            continue
        for k in range(rs.randint(1, 3)):
            xk = x + ww * rs.uniform(0.12, 0.88)
            hk = rs.uniform(0.35, 0.8) * s
            yk = float(lineL(xk, bot_))
            wk = rs.uniform(0.06, 0.13) * s
            m = wc.poly_mask([(xk - wk * 0.8, yk - hk), (xk + wk * 0.7, yk - hk * 1.03), (xk + wk, yk), (xk - wk, yk)])
            m = wc.blur(m, 0.05 * s + 1.2) * (0.6 + 0.5 * wc.noise(6, 3, octaves=2))
            P.add(np.clip(m, 0, 1).astype(np.float32), (96, 56, 70) if kind != "white" else (70, 80, 96), rs.uniform(0.3, 0.5))
        if rs.random() < 0.6:                            # 柜台 / 货架：一横，淡
            yc = rs.uniform(0.35, 0.6)
            ty = bot_ - yc * (bot_ - top_)
            cm_ = wc.stroke_mask([(x + ww * 0.05, float(lineL(x, ty))), (x + ww * 0.95, float(lineL(x + ww, ty)))], 0.05 * s + 1, 0.04 * s + 1, taper=False, rough=0.3)
            P.add((wc.blur(cm_, 1.2) * 0.8).astype(np.float32), (110, 70, 70) if kind != "white" else (90, 104, 112), 0.35)
isolated(_shop_insides)
D_bld = P.D.copy()          # 楼 + 店的样子，第三遍拿来翻成地上的倒影（灯、车、人各自按自己的脚翻）

# ---- 地上的倒影先铺（灯、车、人后画在上面，不然会被倒影洗掉）----
cuts = S(0.58, 0.66, wc.noise(1.6, 60, octaves=3, persistence=0.5)) * (0.3 + 0.7 * near)      # 雨点打出的横纹，把倒影一截截切开
dens = 0.62 * wc.noise(9, 80, octaves=4, persistence=0.55) + 0.38 * wc.noise(46, 200, octaves=2)
thr_p = 0.5 - 0.05 * near
puddle = (S(thr_p - 0.012, thr_p + 0.012, dens) * ground).astype(np.float32)               # 积水：硬边、不规则，噪声长出来的

# 1. 楼和天的倒影：每一列以自己的楼脚（右边以地平线）为轴翻下来，按同一张波纹位移场取样
axis = np.where(xf[0] < VP[0], lineL(xf[0], FOOT0), float(HZ)).astype(np.float32)[None, :]
xs_ = np.clip(np.rint(xf + DX), 0, W - 1).astype(np.int32)
ys_ = np.clip(np.rint(2 * axis - yf), 0, H - 1).astype(np.int32)
R = D_bld[ys_, xs_]
for c in range(3):
    R[..., c] = wc.blur2(R[..., c].astype(np.float32), 5, 1.0)
below_axis = S(axis + 1, axis + 4, yf)
puddle_s = wc.blur(puddle, 2.5)
refl = (ground * below_axis * (0.5 + 0.25 * puddle_s * near ** 0.5)).astype(np.float32)
refl *= (0.8 + 0.4 * wc.noise(30, 120, octaves=2))
refl = np.clip(refl, 0, 0.85)
P.D = P.D * (1 - refl[..., None]) + R * refl[..., None] * 1.05

# 2. 店门口地上一摊暖：人行道被橱窗照亮，湿的，边被咬碎
spill = np.zeros((H, W), np.float32)
for sx, sy, sw, col, s in SHOPS:
    e = np.exp(-(((xf - sx) / (sw * 0.8 + 0.5 * s)) ** 2 + ((yf - sy - 0.18 * s) / (0.3 * s + 4)) ** 2)) * S(sy - 2, sy + 3, yf) * min(1.0, (s / 70.0) ** 1.5)
    spill = np.maximum(spill, e.astype(np.float32))
spill_n = spill * (0.5 + 0.9 * wc.noise(6, 30, octaves=3))
_ = wc.noise(3, 26, octaves=3)      # 原来这里撒过一层碎白点（像雪/砂石，删了）；噪声留着，它吃掉的随机数定住了后面的地面
P.lift((wc.blur(spill, 5) * 0.15).astype(np.float32), 1.0)
wc.wet(P, (wc.blur(spill, 4) * 0.8).astype(np.float32), AMBER, strength=0.3, spread=4)


def _shop_reflect():
    """店的光落在湿地上：翻下来、被水扭得很狠、拉得很长；积水里清楚，积水外碎；暖色往紫里渗。"""
    DX2 = (wc.noise(3, 46, octaves=3) - 0.5) * 2 * (3 + 34 * near) + (wc.noise(16, 220, octaves=2) - 0.5) * 2 * (2 + 14 * near)
    xs2 = np.clip(np.rint(xf + DX2), 0, W - 1).astype(np.int32)
    tp = 0.5
    pud = S(tp - 0.015, tp + 0.015, 0.6 * wc.noise(12, 100, octaves=4) + 0.4 * wc.noise(55, 230, octaves=2)) * ground
    brk = S(0.36, 0.64, 0.65 * wc.noise(6, 70, octaves=3) + 0.35 * wc.noise(2.2, 30, octaves=2))
    wetk = ((0.6 + 0.4 * pud) * (0.55 + 0.45 * np.maximum(brk, pud))).astype(np.float32)
    vmod = (0.4 + 0.85 * (0.6 * wc.noise(40, 9, octaves=2) + 0.4 * wc.noise(90, 26, octaves=2))).astype(np.float32)   # 竖着拖下来的笔，宽窄不一
    fall = (1 - 0.65 * S(0, 420, yf - axis)).astype(np.float32)
    slab = (0.35 + 0.65 * S(0.3, 0.33, wc.noise(22, 420, octaves=2))).astype(np.float32)      # 一截截横着断开：水到这里没了，边是硬的
    total = np.zeros((H, W), np.float32)
    for col, M in GLOWS.items():
        warm = col in (GLOW, AMBER, ORANGE, PALE)
        srcm = np.clip(M + 0.5 * rain_blur(M, 24, 2.0), 0, 1)
        mir = srcm[ys_, xs2] * below_axis * ground * (xf < VP[0])
        mir = wc.blur2(mir.astype(np.float32), 7, 1.0)
        longm = np.roll(wc.blur2(mir, 42, 1.8), 34, axis=0) * 1.1
        R = np.clip(np.maximum(mir * 1.2, longm) * vmod * fall * slab, 0, 1) * wetk
        P.lift((R * 0.85).astype(np.float32), 1.0)
        P.lift((np.clip((R - 0.4) * 2.2, 0, 1) * 0.9).astype(np.float32), 1.0)            # 芯亮到发白：大胆一点
        wc.wet(P, (R * 0.9).astype(np.float32), col, strength=0.4, spread=2)
        halo_r = np.clip(wc.blur(R, 7) * 1.3 - R, 0, 1)
        wc.wet(P, halo_r.astype(np.float32), ORANGE if warm else col, strength=0.22, spread=4)
        total = np.maximum(total, R)
    side = S(-4, 10, yf - axis) * (1 - S(-10, 40, yf - curbL)) * (xf < VP[0])        # 只在人行道上：光和光之间压深
    gapm = ground * side * (1 - np.clip(wc.blur(total, 3) * 2.5, 0, 1))
    wc.wet(P, gapm.astype(np.float32), INDIGO, strength=0.28, spread=4)                   # 光和光之间压深：对比出来
    rim = np.clip(wc.blur(total, 16) * 1.5 - total, 0, 1) * ground
    wc.wet(P, rim.astype(np.float32), (140, 64, 124), strength=0.3, spread=10)       # 暖往紫里渗
isolated(_shop_reflect)



POSTS = []
# ================= 灯：两边马路牙子上一排，往雾里退；间距不齐、亮度不齐 =================
def lamp(x, base, bright=1.0, col=GLOW, double=False, keep=0.7, draw_post=True):
    s = scale_at(base)
    hgt = 2.55 * s
    top_ = base - hgt
    w0 = max(0.8, 0.045 * s)
    post = wc.stroke_mask([(x, base), (x + random.uniform(-0.6, 0.6), base - hgt * 0.5), (x, top_ + 0.1 * s)], w0 * 1.3, w0 * 0.8, taper=False, rough=0.15)
    cut = top_ + keep * hgt                                   # 只画上面 keep 这么多，下面交给雨和地上的暗（哥：一根路灯画七成）
    fadeP = (1 - S(cut - 0.12 * hgt, cut + 0.04 * hgt, yf)) * (0.75 + 0.25 * S(0.3, 0.6, P.vstreak)) * float(draw_post)
    P.add((post * fadeP * min(1.0, 0.25 + s / 90.0)).astype(np.float32), (26, 26, 44), 1.0)
    POSTS.append((post * fadeP * min(1.0, 0.25 + s / 90.0)).astype(np.float32))
    heads = [(x - 0.2 * s, top_ + 0.14 * s), (x + 0.2 * s, top_ + 0.14 * s)] if double else [(x, top_)]
    if double:
        arm = wc.stroke_mask([(x - 0.2 * s, top_ + 0.2 * s), (x, top_ + 0.08 * s), (x + 0.2 * s, top_ + 0.2 * s)], w0, w0, taper=False)
        P.add(arm.astype(np.float32), (26, 26, 44), 0.9)
    for hx, hy in heads:
        r = max(1.4, 0.085 * s)
        # 灯罩：上面一顶深的小帽子，下面一团亮
        cap = [(hx - r * 1.1, hy - r * 0.9), (hx + r * 1.1, hy - r * 0.9), (hx + r * 0.3, hy - r * 1.7), (hx - r * 0.3, hy - r * 1.7)]
        wc.wash(P, cap, (26, 26, 44), strength=0.9, var=0.02, layers=3, edge=0.3)
        core = np.clip(1 - S(r * 0.6, r * 1.25, np.hypot(xf - hx, (yf - hy) * 0.85)), 0, 1).astype(np.float32)
        P.lift(core, 1.0)
        P.add(core * 0.5, col, 0.35)
        # 光晕：往湿的暗里渗，形是噪声长的；外面再一圈更大更淡的雾
        halo = np.exp(-(np.hypot(xf - hx, yf - hy) / (2.2 * r + 5)) ** 2) * (0.55 + 0.9 * wc.noise(max(3.0, r * 1.2), octaves=3))
        halo2 = np.exp(-(np.hypot(xf - hx, (yf - hy) * 1.1) / (6.5 * r + 16)) ** 2) * (0.6 + 0.8 * wc.noise(max(6.0, r * 3), octaves=3))
        P.lift(np.clip(halo * 0.85 * bright, 0, 1).astype(np.float32), 1.0)
        P.lift(np.clip(halo2 * 0.42 * bright, 0, 1).astype(np.float32), 1.0)
        wc.wet(P, np.clip(halo * 0.9, 0, 1).astype(np.float32), col, strength=0.3 * bright, spread=max(1.0, r * 0.5))
        wc.wet(P, np.clip(halo2 * 0.8, 0, 1).astype(np.float32), AMBER, strength=0.18 * bright, spread=max(2.0, r * 1.2))
        halo3 = np.exp(-(np.hypot(xf - hx, (yf - hy) * 1.2) / (15 * r + 34)) ** 2) * (0.5 + 0.9 * wc.noise(max(10.0, r * 6), octaves=2))
        P.lift(np.clip(halo3 * 0.2 * bright, 0, 1).astype(np.float32), 1.0)        # 雨雾被灯照亮的一大团
        LIGHTS.append((hx, hy, base, r, col, bright, 1.0))


def _lamps():
    random.seed(SEED + 5)
    # 近的几根画出来（各自只画一截），远的那一串不再一根根排队往里退——整组化成街尽头一团光（哥：敢于整组消失）
    # 近的灯是明确的东西：画完整、落地（她：近的也消失掉下半，就像飘着）；只有远的、进了雾的才虚——而且不是每根都一样
    for x, b, kp, po in ((118, 1.0, 1.0, 1.0), (292, 0.85, 1.0, 1.0), (398, 0.9, 0.9, 0.8), (455, 0.6, 0.55, 0.3)):
        lamp(x, float(lineL(x, CURBL0)), b, col=GLOW if b > 0.65 else AMBER, double=(x == 292), keep=kp, draw_post=po)
    for x, b, kp, po in ((812, 0.95, 1.0, 0.9), (700, 0.7, 0.6, 0.35)):
        lamp(x, float(lineR(x, CURBR9)), b, col=GLOW, keep=kp, draw_post=po)
    cx_, cy_ = 548, 566
    blob = np.exp(-(((xf - cx_) / 70.0) ** 2 + ((yf - cy_) / 26.0) ** 2)) * np.clip(0.3 + 1.0 * wc.noise(10, 16, octaves=3), 0, 1.2)
    P.lift(np.clip(blob * 0.8, 0, 1).astype(np.float32), 1.0)
    wc.wet(P, np.clip(blob, 0, 1).astype(np.float32), GLOW, strength=0.25, spread=4)
    wc.wet(P, np.clip(wc.blur(blob, 10) * 1.2 - blob, 0, 1).astype(np.float32), AMBER, strength=0.25, spread=6)
    LIGHTS.append((cx_, cy_ - 8, HZ + 8, 16, GLOW, 0.5, 1.6))
    # 雾里远处几点小灯（窗、车站），没有杆：不齐、不等大
    for x, y, r in ((512, 566, 2.6), (596, 556, 1.7), (842, 530, 2.4), (626, 571, 3.2)):
        core = np.clip(1 - S(r * 0.5, r * 1.3, np.hypot(xf - x, yf - y)), 0, 1).astype(np.float32)
        P.lift(core * 0.9, 1.0); P.add(core * 0.5, GLOW, 0.3)
        halo = np.exp(-(np.hypot(xf - x, yf - y) / (5 * r + 6)) ** 2) * (0.6 + 0.8 * wc.noise(6, octaves=2))
        P.lift(np.clip(halo * 0.4, 0, 1).astype(np.float32), 1.0)
        LIGHTS.append((x, y, HZ + 6, r, GLOW, 0.4, 0.6))
isolated(_lamps)

# ================= 车：一辆往里开（两点红尾灯），一辆远远迎面来（两点白灯）=================
CARS = []
def car(cx, base, away=True, lose=0.0):
    s = scale_at(base)
    w, h = 1.12 * s, 0.86 * s
    x0 = cx - w / 2
    body = [(x0, base), (x0, base - h * 0.52), (x0 + w * 0.14, base - h * 0.62), (x0 + w * 0.22, base - h), (x0 + w * 0.78, base - h),
            (x0 + w * 0.86, base - h * 0.62), (x0 + w, base - h * 0.52), (x0 + w, base)]
    wetb = S(base - h * 0.4, base + 2, yf).astype(np.float32)
    cm = wc.wash(P, body, (30, 30, 50), strength=1.0, var=0.018, layers=10, edge=0.4, wet_map=wetb * 0.9, wet_r=4,
                 fade=((1 - 0.6 * S(base - h * 0.25, base + 4, yf)) *
                       (1 - lose * S(x0 + w * 0.5, x0 + w * 0.95, xf) * S(base - h * 0.8, base - h * 0.3, yf))).astype(np.float32))   # 半边车化进湿地的暗
    wc.wet(P, cm * S(0.5, 0.7, wc.noise(8, octaves=2)), DEEP_WARM, strength=0.35, spread=3)
    # 后窗反着街尽头的光：浅一块，不贴边
    win = [(x0 + w * 0.25, base - h * 0.95), (x0 + w * 0.75, base - h * 0.95), (x0 + w * 0.84, base - h * 0.64), (x0 + w * 0.16, base - h * 0.64)]
    wm = wc.poly_mask(win) * (0.5 + 0.5 * wc.noise(4, 12, octaves=2))
    P.lift((wm * 0.35).astype(np.float32), 1.0)
    P.lift((wc.stroke_mask([(x0 + w * 0.24, base - h + 1), (x0 + w * 0.76, base - h + 1)], 0.03 * s + 0.8, 0.02 * s + 0.6, taper=False) * S(0.3, 0.55, P.grain) * 0.55).astype(np.float32), 1.0)
    for lx in (x0 + w * 0.1, x0 + w * 0.9):
        ly_ = base - h * 0.42
        r = max(1.2, 0.055 * s)
        m = np.clip(1 - S(r * 0.6, r * 1.3, np.hypot((xf - lx) * (0.8 if away else 1.0), yf - ly_)), 0, 1).astype(np.float32)
        P.lift(m, 1.0)
        if away:
            P.add(m, RED, 0.9)
            halo = np.exp(-(np.hypot(xf - lx, yf - ly_) / (3.5 * r + 3)) ** 2) * (0.5 + 0.9 * wc.noise(max(3.0, r * 1.5), octaves=2))
            P.lift(np.clip(halo * 0.5, 0, 1).astype(np.float32), 1.0)
            wc.wet(P, np.clip(halo, 0, 1).astype(np.float32), RED, strength=0.45, spread=max(1.0, r * 0.6))
            LIGHTS.append((lx, ly_, base, r * 1.2, RED, 1.0, 1.25))
        else:
            P.add(m * 0.5, PALE, 0.4)
            halo = np.exp(-(np.hypot(xf - lx, yf - ly_) / (4.5 * r + 4)) ** 2) * (0.6 + 0.8 * wc.noise(max(3.0, r * 1.5), octaves=2))
            P.lift(np.clip(halo * 0.75, 0, 1).astype(np.float32), 1.0)
            wc.wet(P, np.clip(halo, 0, 1).astype(np.float32), PALE, strength=0.2, spread=max(1.0, r * 0.6))
            LIGHTS.append((lx, ly_, base, r * 1.3, PALE, 1.0, 1.4))
    CARS.append((cm, base))


def _cars():
    random.seed(SEED + 6)
    car(648, 694, away=True, lose=0.85)
    car(492, 614, away=False, lose=0.7)
    car(596, 603, away=True, lose=0.9)
isolated(_cars)

# ================= 人：打伞的剪影，站在亮倒影前面 =================
PEOPLE = []
def umbrella_person(x, base, ucol=(30, 30, 50), coat=(38, 36, 56), lean=0.0, walking=True, legs_keep=1.0):
    s = scale_at(base)
    h = 1.02 * s
    torso = [(x - h * 0.05, base - h * 0.85), (x + h * 0.05, base - h * 0.85), (x + h * 0.095, base - h * 0.8), (x + h * 0.08, base - h * 0.58),
             (x + h * 0.1 + lean * h * 0.06, base - h * 0.3), (x - h * 0.1 + lean * h * 0.06, base - h * 0.3), (x - h * 0.08, base - h * 0.58), (x - h * 0.095, base - h * 0.8)]
    wc.wash(P, torso, coat, strength=1.35, var=0.04, layers=5, edge=0.25, wet_map=np.full((H, W), 0.4, np.float32), wet_r=1.5)
    wc.dab(P, x + lean * h * 0.02, base - h * 0.89, h * 0.055, (52, 40, 50), 1.2, soft=0.9)
    legs = np.zeros((H, W), np.float32)
    spread = 0.07 if walking else 0.03
    for dx in (-spread, spread * 0.9):
        legs = np.maximum(legs, wc.stroke_mask([(x + dx * 0.5 * h, base - h * 0.34), (x + dx * h * 1.2, base)], h * 0.075, h * 0.03, taper=False, rough=0.2))
    legs = legs * S(0.25, 0.5, 0.5 * P.vstreak + 0.5 * P.grain) * (1 - 0.5 * S(base - h * 0.1, base + 2, yf)) * (1 - (1 - legs_keep) * S(base - h * 0.3, base - h * 0.05, yf))
    P.add(legs, coat, 1.0)
    ux, uy, ur = x + lean * h * 0.08, base - h * 0.94, h * 0.27
    dome = [(ux + ur * math.cos(a), uy - ur * 0.55 * math.sin(a) + lean * ur * 0.25 * math.cos(a)) for a in np.linspace(0, math.pi, 20)]
    scal = [(ux - ur + k * ur / 3, uy + (0.03 * h if k % 2 else 0)) for k in range(7)]
    um = wc.poly_mask(dome + scal[::-1])
    wc.wash(P, dome + scal[::-1], ucol, strength=1.0, var=0.02, layers=4, edge=0.4)
    rim = np.clip(um - wc.blur(um, 1.2), 0, 1) * (yf < uy - ur * 0.15) * S(0.3, 0.55, P.grain)
    P.lift((rim * 1.3).astype(np.float32), 0.6)
    P.add(wc.stroke_mask([(ux, uy), (ux, uy + ur * 0.55)], 0.012 * h + 0.6, 0.012 * h + 0.6, taper=False), (30, 30, 40), 0.7)
    PEOPLE.append((np.maximum(np.maximum(um, legs), wc.poly_mask(torso)), base))


def _people():
    random.seed(SEED + 7)
    umbrella_person(222, 676, lean=0.2, legs_keep=0.2)
    umbrella_person(252, 672, ucol=(36, 40, 70), walking=False)
    umbrella_person(372, 632, lean=-0.1)
    umbrella_person(430, 752, ucol=(176, 38, 42), coat=(40, 34, 52), lean=0.3, legs_keep=0.35)       # 红伞：横穿马路，站在灯的倒影前面
    umbrella_person(724, 640, ucol=(34, 34, 56))
isolated(_people)
OBJ = np.zeros((H, W), np.float32)             # 站在地上的东西挡住后面的倒影
for m_ in POSTS + [c for c, _ in CARS] + [p for p, _ in PEOPLE]:
    OBJ = np.maximum(OBJ, np.clip(m_ * 1.3, 0, 1))
OCC = (1 - OBJ).astype(np.float32)


# ================= 第三遍：地上。整条街都是湿的，这一块交给颜料自己跑 =================
# 3. 每盏灯拖下来的竖亮：从脚下往我们这边拉，被同一张位移场扭成锯齿，被横纹切成一截截的横笔
dash = wc.noise(3.2, 24, octaves=3)
wob_all = wc.noise(50, 2000, octaves=2) - 0.5
core_tot = np.zeros((H, W), np.float32)
fr_maps = {}
post_ref = np.zeros((H, W), np.float32)
for i, (lx, ly_, base, r, col, bright, lenk) in enumerate(LIGHTS):
    s = scale_at(base)
    hgt = max(4.0, base - ly_)
    L = (2.0 * hgt + 40) * lenk
    t = (yf - base) / L
    prof = S(base - 1, base + 4, yf) * S(-0.02, 0.08, t) * (1 - S(0.3, 1.0, t)) ** 1.3
    wob = np.roll(wob_all, i * 37, axis=0)[:, :1] * (0.6 * r + 0.04 * np.clip(yf - base, 0, None))
    xs = xf - lx - DX * (0.6 + 1.2 * np.clip(t, 0, 1)) - wob
    w = r * 1.05 + 0.02 * np.clip(yf - base, 0, None)
    core = np.exp(-(xs / w) ** 2) * prof * (1 - CALM * 0.8 * (1 - S(0.44, 0.56, np.roll(dash, i * 53, axis=1))))
    fr = np.exp(-(xs / (w * 2.8)) ** 2) * prof
    core_tot = np.maximum(core_tot, (core * bright).astype(np.float32))
    key = tuple(col)
    fr_maps[key] = np.maximum(fr_maps.get(key, np.zeros((H, W), np.float32)), (fr * bright).astype(np.float32))
    if col != RED and col != PALE and hgt > 30:
        pr = np.exp(-(((xf - lx - DX * 1.2 - wob) / (0.03 * s + 0.8)) ** 2)) * S(base, base + 4, yf) * (1 - S(base, base + hgt * 0.9, yf))
        post_ref = np.maximum(post_ref, pr.astype(np.float32))
wet_fac = (0.45 + 0.55 * puddle) * (1 - 0.55 * cuts * CALM) * ground * OCC
P.add((post_ref * wet_fac * S(0.3, 0.6, P.vstreak)).astype(np.float32), (26, 26, 46), 0.6)
for key, fm in fr_maps.items():
    fm = (fm * wet_fac).astype(np.float32)
    P.lift(np.clip(fm * 0.45, 0, 1), 1.0)
    edge_col = {GLOW: ORANGE, AMBER: ORANGE, RED: RED, PALE: (230, 200, 150)}.get(key, ORANGE)
    wc.wet(P, fm, edge_col, strength=0.4 if key != RED else 0.7, spread=3)
core_tot = (core_tot * wet_fac).astype(np.float32)
P.lift(np.clip(core_tot * 1.3, 0, 1), 1.0)
P.add(np.clip(core_tot, 0, 1) * 0.7, GLOW, 0.3)

# 4. 灯脚下一圈被照亮的湿地：不是圆的，是一摊一摊
pools = np.zeros((H, W), np.float32)
for lx, ly_, base, r, col, bright, lenk in LIGHTS:
    s = scale_at(base)
    if col == RED or s < 8:
        continue
    e = np.exp(-(((xf - lx) / (0.9 * s + 6)) ** 2 + ((yf - base - 0.05 * s) / (0.16 * s + 3)) ** 2)) * bright
    pools = np.maximum(pools, e.astype(np.float32))
pool_hard = S(0.5, 0.53, pools * (0.5 + 0.9 * wc.noise(3, 22, octaves=3))) * ground * OCC
P.lift((wc.blur(pools, 4) * 0.2 * ground * OCC).astype(np.float32), 1.0)
wc.wet(P, (wc.blur(pools, 3) * 0.6 * ground).astype(np.float32), AMBER, strength=0.2, spread=4)

# 4b. 地上的冷暖：红灯底下渗一片紫，店的倒影边上发橙，近处湿的深一块（只在前景，框住画面）
redzone = wc.blur(fr_maps.get(RED, np.zeros((H, W), np.float32)), 14)
wc.wet(P, (np.clip(redzone * 2.5, 0, 1) * ground).astype(np.float32), (150, 70, 120), strength=0.3, spread=10)
wc.wet(P, (ground * S(0.55, 0.75, wc.noise(40, 110, octaves=3)) * S(0.2, 0.7, near)).astype(np.float32), (110, 70, 120), strength=0.22, spread=8)
fg = (ground * S(0.55, 1.0, near) * (0.5 + 0.5 * wc.noise(60, 160, octaves=2)) * (1 - np.clip(core_tot * 2, 0, 1))).astype(np.float32)
wc.wet(P, fg, INDIGO, strength=0.35, spread=14)
# 近处几笔宽的、软的亮横笔：反着雾里的天（Hassam 右下角那几笔）
from PIL import ImageDraw as _ID
random.seed(SEED + 10)
imb = Image.new("L", (W, H), 0); db = _ID.Draw(imb)
for _ in range(9):
    y0 = random.uniform(820, 1120); x0 = random.uniform(520, 900); L_ = random.uniform(60, 190)
    db.line([(x0 - L_ / 2, y0), (x0 + L_ / 2, y0 + random.uniform(-4, 4))], fill=random.randint(120, 255), width=random.randint(3, 8))
sweep = wc.blur(np.asarray(imb, np.float32) / 255.0, 1.2) * S(0.35, 0.6, 0.6 * P.hstreak + 0.4 * P.grain) * OCC
P.lift((sweep * 0.3).astype(np.float32), 1.0)

# 5. 车和人的深倒影：以各自的脚为轴翻下去，竖着拉、被横纹切，跟着同一张位移场扭
for m, base in [(cm, b) for cm, b in CARS] + PEOPLE:
    ysrc = np.clip(np.rint(2 * base - yf), 0, H - 1).astype(np.int32)
    fl = m[ysrc, xs_] * S(base - 1, base + 2, yf)
    fl = wc.blur2(fl.astype(np.float32), 5, 1.2) * np.clip(1 - (yf - base) / (1.4 * scale_at(base) + 10), 0, 1) * (1 - 0.5 * cuts) * (0.5 + 0.5 * puddle)
    P.add(fl.astype(np.float32), (30, 28, 50), 0.65)

# 6. 亮的横笔：积水边上、光路里，几笔被纸纹咬碎的亮（暗底子上的乱只能用亮的）
fleck = S(0.84, 0.87, wc.noise(4, 34, octaves=3)) * ground * np.clip(wc.blur(core_tot, 8) * 4, 0, 1) * OCC
P.lift((fleck * S(0.3, 0.55, P.grain) * 0.55).astype(np.float32), 1.0)
# ================= 最后：雨丝、几点甩出去的亮 =================
random.seed(SEED + 8)
rain = np.zeros((H, W), np.float32)
from PIL import ImageDraw
im = Image.new("L", (W, H), 0); dr = ImageDraw.Draw(im)
for _ in range(260):
    x0, y0 = random.uniform(-40, W), random.uniform(0, 1000)
    Lr = random.uniform(14, 44)
    dr.line([(x0, y0), (x0 - Lr * 0.2, y0 + Lr)], fill=random.randint(90, 255), width=1)
rain = wc.blur(np.asarray(im, np.float32) / 255.0, 0.5)
lit = np.clip(sum(np.exp(-(np.hypot(xf - lx, yf - ly_) / (6 * r + 20)) ** 2) * bright for lx, ly_, base, r, col, bright, lenk in LIGHTS if col != RED), 0, 1)
P.lift((rain * S(0.3, 0.55, P.grain) * (0.12 + 0.55 * lit)).astype(np.float32), 1.0)

def brush(pts, w, color=(30, 22, 36), strength=1.4, entry=0.08, dry_from=0.65, n=80):
    return wc.brush(P, pts, w, color, strength, entry, dry_from, n)


def _calligraphy():
    """全画柔化完，在焦点（红伞那个人、那辆车）附近落几笔真正硬、快、带方向的深色（哥）。"""
    brush([(391, 590), (412, 594), (440, 592), (468, 586)], 3.2, (60, 16, 24), 1.3)          # 红伞底下一道暗
    brush([(414, 612), (410, 648), (407, 690), (409, 712)], 5.5, strength=1.5)                   # 大衣背后一刀，往下收
    brush([(446, 614), (450, 660), (452, 700)], 3.0, strength=1.3, dry_from=0.5)                # 另一边轻一笔
    brush([(588, 636), (630, 633), (676, 634)], 3.4, strength=1.3)                                # 车后窗下沿
    brush([(585, 646), (584, 668), (586, 692)], 5.5, strength=1.5, dry_from=0.55)               # 车左边一刀硬边：左边找得到，右边化掉
isolated(_calligraphy)


def _splatter():
    """甩点：白的是雨点（留白/擦出来的），深的是笔上甩下来的颜料；灯和窗附近多一点。"""
    from PIL import ImageDraw
    random.seed(SEED + 40)
    aa = 2
    imw = Image.new("L", (W * aa, H * aa), 0); dw = ImageDraw.Draw(imw)
    imd = Image.new("L", (W * aa, H * aa), 0); dd = ImageDraw.Draw(imd)
    for _ in range(300):
        if random.random() < 0.6 and LIGHTS:
            lx, ly_, base, r, col, bright, lenk = random.choice(LIGHTS)
            x0 = random.gauss(lx, 12 * r + 20); y0 = random.gauss(ly_ + 30, 25 * r + 40)
        else:
            x0, y0 = random.uniform(0, W), random.uniform(HZ, H)
        rr = random.choice([0.45, 0.55, 0.7, 0.9, 1.1, 1.5]) * aa
        dw.ellipse([x0 * aa - rr, y0 * aa - rr * random.uniform(0.7, 1.1), x0 * aa + rr, y0 * aa + rr], fill=random.randint(150, 255))
    for _ in range(90):
        x0, y0 = random.uniform(0, W), random.uniform(0, H)
        rr = random.choice([0.6, 0.9, 1.3, 2.0]) * aa
        dd.ellipse([x0 * aa - rr, y0 * aa - rr, x0 * aa + rr, y0 * aa + rr], fill=random.randint(120, 255))
    mw = np.asarray(imw.resize((W, H), Image.BILINEAR), np.float32) / 255.0
    md = np.asarray(imd.resize((W, H), Image.BILINEAR), np.float32) / 255.0
    P.lift((mw * 0.75).astype(np.float32), 1.0)
    P.add((md * 0.8).astype(np.float32), (40, 34, 60), 0.9)
isolated(_splatter)

grain = wc.grain_from_profile(os.path.join(SKILL_DIR, "pigment_profile.npy"), seed=SEED)
out = os.path.join(HERE, "rainnight.png" if SEED == 11 else f"rainnight_{SEED}.png")
P.render(pigment_tex=grain, tex_amount=0.045).save(out)
print("saved", out, round(time.time() - t0, 1), "s")
