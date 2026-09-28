"""雨夜街口：天是深蓝，楼更深，灯是留出来的纸白 + 一圈暖；湿马路把每一盏灯都拉成一条长长的竖亮。
第一遍：天（上深、街尽头一团城市的暖光）/ 两边楼（一整块深）/ 路（中深，远处反着暖光）。缩小看。
第二遍：店面的暖窗、楼上零星亮窗、一排路灯、远处一座钟楼剪影、车、打伞的人。
第三遍：倒影——灯的竖亮、车灯的长光、人和车的深影，被雨打出的横纹切碎；前景一截斑马线。
最后：一点雨丝。
从 skill 根目录跑：python scenes/rainnight_old_cutout.py [seed] [values]
"""
import sys, os, math, random, time
import numpy as np
from PIL import Image, ImageDraw
HERE = os.path.dirname(os.path.abspath(__file__))
SKILL_DIR = os.path.dirname(HERE)
sys.path.insert(0, SKILL_DIR)
import watercolor_lib as wc

W, H = wc.W, wc.H
t0 = time.time()
SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 6
STAGE = sys.argv[2] if len(sys.argv) > 2 else "full"
wc.set_seed(SEED)
P = wc.Paper()
yy, xx = np.mgrid[:H, :W]
yf, xf = yy.astype(np.float32), xx.astype(np.float32)
S = wc.smoothstep

VP = (470, 612)
NIGHT = (58, 66, 96)
NIGHT_DEEP = (30, 34, 50)
DEEP_WARM = (70, 52, 50)
GLOW = (236, 186, 112)
AMBER = (224, 150, 72)
RED = (196, 56, 46)
ROAD = (86, 92, 114)


def ly(x, near_y, side):
    f = (VP[0] - x) / VP[0] if side == "L" else (x - VP[0]) / (W - VP[0])
    return VP[1] + (near_y - VP[1]) * f


def skyline(segs, side):
    pts = []
    for xa, xb, T, orn in segs:
        pts.append((xa, ly(xa, T, side)))
        for kind, u, s in orn:
            x = xa + (xb - xa) * u
            y = ly(x, T, side)
            f = abs(VP[0] - x) / (VP[0] if side == "L" else W - VP[0])
            s = s * max(0.25, f)
            if kind == "chim":
                pts += [(x - s * 0.18, y), (x - s * 0.18, y - s * 0.6), (x + s * 0.18, y - s * 0.6), (x + s * 0.18, y)]
            elif kind == "gable":
                pts += [(x - s * 0.6, y), (x, y - s * 0.55), (x + s * 0.6, y)]
            elif kind == "mansard":
                pts += [(x - s * 0.7, y), (x - s * 0.5, y - s * 0.4), (x + s * 0.5, y - s * 0.4), (x + s * 0.7, y)]
        pts.append((xb, ly(xb, T, side)))
    return pts


cityglow = np.exp(-(((xf - VP[0]) / 260.0) ** 2 + ((yf - (VP[1] - 30)) / 170.0) ** 2))
dist = np.clip(1 - np.hypot(xf - VP[0], (yf - VP[1]) * 1.3) / 400.0, 0, 1)
ybL = ly(xf, 830, "L")
ybR = ly(xf, 820, "R")
yb = np.where(xf < VP[0], ybL, ybR)
road = S(yb - 2, yb + 3, yf)
wetm = np.clip(0.15 + 0.6 * dist, 0, 1).astype(np.float32)

# ================= 第一遍：只有大块 =================
# 1. 天：上面深蓝，往街尽头一团城市反上来的暖；湿接湿
skym = 1 - road
top = np.clip(1 - yf / 640.0, 0, 1)
P.add((skym * (0.35 + 0.65 * top) * (1 - 0.8 * cityglow)).astype(np.float32), NIGHT, 0.95, granulate=0.03)
wc.wet(P, (skym * top * (0.5 + 0.5 * wc.noise(160, 260, octaves=2))).astype(np.float32), NIGHT_DEEP, strength=0.35, spread=50)
wc.wet(P, (skym * cityglow).astype(np.float32), (200, 150, 110), strength=0.35, spread=30)
# 2. 两边的楼：一整块深，远处被城市的光和雨雾吃浅
L_segs = [(-40, 90, 110, [("chim", 0.3, 50), ("mansard", 0.7, 90)]), (90, 200, 190, [("gable", 0.5, 80)]),
          (200, 290, 260, [("chim", 0.6, 40)]), (290, 360, 300, [("mansard", 0.5, 60)]), (360, 420, 360, []), (420, 466, 420, [])]
R_segs = [(474, 520, 420, []), (520, 580, 360, [("chim", 0.5, 30)]), (580, 660, 300, [("mansard", 0.5, 60)]),
          (660, 760, 220, [("gable", 0.5, 80)]), (760, 940, 120, [("chim", 0.3, 50), ("mansard", 0.7, 100)])]
left = skyline(L_segs, "L") + [(466, float(ly(466, 830, "L"))), (-40, float(ly(-40, 830, "L")))]
right = skyline(R_segs, "R") + [(940, float(ly(940, 820, "R"))), (474, float(ly(474, 820, "R")))]
fadeL = (1 - S(ybL - 4, ybL + 8, yf)).astype(np.float32)
fadeR = (1 - S(ybR - 4, ybR + 8, yf)).astype(np.float32)
covL = wc.wash(P, left, NIGHT_DEEP, strength=1.0, var=[0.012] * (len(left) - 2) + [0.03] * 2, layers=26, edge=0.45, granulate=0.04,
               fade=fadeL, wet_map=wetm * 0.4, wet_r=6)
covR = wc.wash(P, right, NIGHT_DEEP, strength=1.0, var=[0.012] * (len(right) - 2) + [0.03] * 2, layers=26, edge=0.45, granulate=0.04,
               fade=fadeR, wet_map=wetm * 0.4, wet_r=6)
covB = np.maximum(covL, covR)
wc.wet(P, (covB * S(0.5, 0.72, wc.noise(80, octaves=3))).astype(np.float32), DEEP_WARM, strength=0.45, spread=8)
for segs, cov, side in ((L_segs[-3:], covL, "L"), (R_segs[:3], covR, "R")):
    amts = (0.25, 0.42, 0.6) if side == "L" else (0.6, 0.42, 0.25)
    for (xa, xb, T, _), amt in zip(segs, amts):
        blk = (S(xa - 2, xa + 2, xf) * (1 - S(xb - 2, xb + 2, xf))).astype(np.float32)   # 不重叠：重叠处会被擦两次，成一道亮线
        P.lift((cov * blk * amt).astype(np.float32), 1.0)
        P.add((cov * blk * amt * 0.6).astype(np.float32), (150, 120, 110), 0.3)     # 远楼被城市的暖光染一点
# 3. 路：中深的冷灰，远处反着那团暖光（湿路是镜子）
near = np.clip((yf - VP[1]) / (H - VP[1]), 0, 1)
P.add((road * (0.45 + 0.55 * near)).astype(np.float32), (58, 64, 88), 1.0, granulate=0.08)
wc.wet(P, (road * near * S(0.5, 0.72, wc.noise(30, 160, octaves=3))).astype(np.float32), NIGHT_DEEP, strength=0.3, spread=4)
wc.wet(P, (road * (1 - near) * np.exp(-(((xf - VP[0]) / 140.0) ** 2))).astype(np.float32), (200, 150, 110), strength=0.3, spread=20)
P.lift((road * np.exp(-(((xf - VP[0]) / 60.0) ** 2 + ((yf - VP[1] - 20) / 70.0) ** 2)) * 0.6).astype(np.float32), 1.0)

if STAGE == "values":
    img = P.render()
    out = os.path.join(HERE, "rainnight_values.png")
    img.save(out)
    img.convert("L").resize((64, 82), Image.LANCZOS).resize((256, 328), Image.NEAREST).save(out.replace(".png", "_thumb.png"))
    print("values", out, round(time.time() - t0, 1)); sys.exit()

# ---- 地上不规矩的那一层（跟例图的树影同一个路子）：积水的形从噪声里长出来，硬边；湿的地方是镜子，干一点的地方倒影弱 ----
dens = 0.62 * wc.noise(16, 120, octaves=4, persistence=0.55) + 0.38 * wc.noise(60, 260, octaves=2)
thr_m = 0.5 + 0.06 * (1 - near)                                  # 远处积水少一点，近处连成片
mirror = (S(thr_m - 0.012, thr_m + 0.012, dens) * road).astype(np.float32)
hd = wc.noise(3.5, 240, octaves=2, persistence=0.45)


def swipe(x0, y0, length, w, dry0, dry1, d=-1):
    """一笔横扫：笔心实、笔边碎、越往后越干（例图地面扫笔的同一招）。d=-1 往左扫。"""
    x1 = x0 + d * length
    xa_, xb_ = min(x0, x1), max(x0, x1)
    xs = np.array([xa_, (xa_ + xb_) / 2, xb_]); ys = np.array([y0, y0 + random.uniform(-3, 3), y0 + random.uniform(-2, 2)])
    ws = np.array([w * random.uniform(0.5, 0.9), w, w * random.uniform(0.3, 0.8)])
    xc = np.clip(xf[0], xa_, xb_)
    yc = np.interp(xc, xs, ys)[None, :]
    wv = np.interp(xc, xs, ws)[None, :] * (0.75 + 0.5 * wc.noise(8, 50, octaves=3))
    tpos = (np.clip(xf, xa_, xb_) - xa_) / (xb_ - xa_ + 1e-6)
    if d < 0:
        tpos = 1 - tpos
    r = np.abs(yf - yc) / wv
    inside = ((r < 1) & (xf >= xa_) & (xf <= xb_)).astype(np.float32)
    dry = dry0 + (dry1 - dry0) * tpos + 0.3 * r ** 2
    return inside * S(dry - 0.015, dry + 0.015, 0.75 * hd + 0.25 * P.grain)


# 干一点的路面上压几笔深的横扫；积水里湿的深一块、暖一点（湿接湿，边自己跑）
random.seed(SEED + 30)
sw_dark = np.zeros((H, W), np.float32)
for _ in range(14):
    y0 = random.uniform(700, 1120)
    x0 = random.uniform(-40, 940)
    sw_dark = np.maximum(sw_dark, swipe(x0, y0, random.uniform(160, 520), random.uniform(5, 20) * (0.4 + near[int(y0), 0]), 0.18, random.uniform(0.45, 0.7),
                                        d=random.choice([-1, 1])))
# （深色横扫在暗路面上像扫描线，不用。）积水是镜子：比路面亮一截，反着天上的光；硬边、不规则
P.lift((mirror * (0.2 + 0.22 * (1 - near))).astype(np.float32), 1.0)
P.add((mirror * (1 - near) * np.exp(-(((xf - VP[0]) / 220.0) ** 2))).astype(np.float32), (200, 150, 110), 0.2)
wc.wet(P, (mirror * S(0.55, 0.75, wc.noise(40, 90, octaves=3))).astype(np.float32), (48, 52, 80), strength=0.3, spread=5, bloom=0.6)
wc.wet(P, (mirror * S(0.6, 0.8, wc.noise(50, 70, octaves=3))).astype(np.float32), (120, 90, 80), strength=0.15, spread=5)
# 天上几处水渍回流：颜料没干透时掉进去的水
wc.wet(P, (skym * top * S(0.55, 0.75, wc.noise(70, 110, octaves=3))).astype(np.float32), NIGHT, strength=0.25, spread=12, bloom=0.9)

LIGHTS = []   # 每个光源：(mask, 贴地的 y, 颜色, 倒影长度) —— 第三遍用它拉竖亮

# ================= 破形：远处一座钟楼，剪影，钟面一点暖 =================
TX = 452
tower = [(TX - 13, 560), (TX - 13, 420), (TX - 15, 418), (TX - 15, 400), (TX - 9, 392), (TX, 360), (TX + 9, 392), (TX + 15, 400), (TX + 15, 418), (TX + 13, 420), (TX + 13, 560)]
wc.wash(P, tower, (70, 70, 90), strength=0.7, var=0.006, layers=8, edge=0.3, fade=(1 - 0.8 * S(500, 560, yf)).astype(np.float32))
clock = np.clip(1 - S(4.5, 6.5, np.hypot(xf - TX, yf - 432)), 0, 1).astype(np.float32)
P.lift(clock, 1.0); P.add(clock * 0.8, GLOW, 0.5)

# ================= 第二遍：灯和东西 =================
def glow_at(m, radius, color, amt):
    """光往暗里渗：边不是圆的渐变，是湿纸上颜料自己跑出去的形，外沿一圈沉积。"""
    g = wc.blur(m.astype(np.float32), radius)
    g = g / (g.max() + 1e-6)
    rag = np.clip(g * (0.4 + 1.0 * wc.noise(max(4.0, radius * 1.6), octaves=3)), 0, 1)
    P.lift((rag * amt * 0.5).astype(np.float32), 1.0)
    wc.wet(P, (rag * amt).astype(np.float32), color, strength=0.35, spread=max(1.5, radius * 0.25), bloom=0.5)

# 左边一楼的店：一格格暖窗（留纸 + 里面一层暖 + 往外渗一圈光），窗框是深的
random.seed(SEED + 1)
x = -10.0
while x < 430:
    f = (VP[0] - x) / VP[0]
    if f < 0.08:
        break
    ww = (70 * f + 8) * random.uniform(0.8, 1.2)
    if random.random() < 0.72:
        top_n, bot_n = 680, 800
        q = [(x, ly(x, top_n, "L")), (x + ww, ly(x + ww, top_n, "L")), (x + ww, ly(x + ww, bot_n, "L")), (x, ly(x, bot_n, "L"))]
        m = wc.poly_mask(q)
        P.lift((m * 0.95).astype(np.float32), 1.0)
        warm = random.choice([GLOW, GLOW, AMBER, (230, 206, 150)])
        wc.wash(P, q, warm, strength=0.45, var=0.02, layers=5, edge=0.3)
        # 窗里几个剪影 / 窗格竖线
        for k in range(random.randint(1, 3)):
            xk = x + ww * random.uniform(0.2, 0.85)
            P.add(wc.stroke_mask([(xk, ly(xk, top_n, "L") + 2), (xk, ly(xk, bot_n, "L") - 2)], 1.6 * f + 0.6, 1.6 * f + 0.6, taper=False) * S(0.3, 0.5, P.grain), (60, 44, 40), 0.6)
        glow_at(m, 10 * f + 3, warm, 0.6)
        LIGHTS.append((m, float(ly(x + ww / 2, 830, "L")), warm, 120 * f + 30))
    x += ww * random.uniform(1.25, 1.6)
# 左边店面上方一条招牌：一段暖红，字不画
sg = [(40, ly(40, 660, "L")), (230, ly(230, 660, "L")), (230, ly(230, 675, "L")), (40, ly(40, 675, "L"))]
m = wc.poly_mask(sg); P.lift(m * 0.8, 1.0); wc.wash(P, sg, RED, strength=0.6, var=0.02, layers=4, edge=0.4); glow_at(m, 8, RED, 0.5)
LIGHTS.append((m * 0.6, float(ly(135, 830, "L")), RED, 90))

# 右边：咖啡馆的遮阳棚底下一排暖光，棚子本身深红，棚下有两桌人
aw_top, aw_bot = 700, 745
awn = [(640, ly(640, aw_top, "R")), (900, ly(900, aw_top, "R")), (905, ly(905, aw_bot, "R")), (636, ly(636, aw_bot, "R"))]
wc.wash(P, awn, (120, 44, 44), strength=0.85, var=0.015, layers=6, edge=0.45)
under = [(640, ly(640, aw_bot, "R")), (900, ly(900, aw_bot, "R")), (900, ly(900, 818, "R")), (640, ly(640, 818, "R"))]
um = wc.poly_mask(under)
P.lift((um * 0.9).astype(np.float32), 1.0)
wc.wash(P, under, GLOW, strength=0.5, var=0.02, layers=5, edge=0.3)
glow_at(um, 12, GLOW, 0.5)
LIGHTS.append((um, float(ly(770, 820, "R")), GLOW, 170))
for sx0 in (690, 740, 810, 860):                                            # 棚下坐着的人：剪影
    f = (sx0 - VP[0]) / (W - VP[0]); h = 60 * f + 10
    yb0 = ly(sx0, 812, "R")
    wc.dab(P, sx0, yb0 - h * 0.8, h * 0.1, (70, 50, 46), 1.1)
    wc.wash(P, [(sx0 - h * 0.14, yb0 - h * 0.68), (sx0 + h * 0.14, yb0 - h * 0.68), (sx0 + h * 0.18, yb0 - h * 0.25), (sx0 - h * 0.18, yb0 - h * 0.25)],
            (44, 40, 50), strength=0.9, var=0.03, layers=3, edge=0.4)

# 楼上零星亮窗：暖的小方块，稀，远的更小更淡
random.seed(SEED + 2)
for segs, side, base in ((L_segs, "L", 830), (R_segs, "R", 820)):
    for xa, xb, T, _ in segs:
        for row in range(4):
            yn = T + 60 + row * 110
            if yn > 640:
                continue
            x = xa + 10
            while x < xb - 8:
                f = (VP[0] - x) / VP[0] if side == "L" else (x - VP[0]) / (W - VP[0])
                if f < 0.1:
                    break
                wy = ly(x, yn, side); ww_, wh_ = 11 * f + 2, 30 * f + 4
                r = random.random()
                q = [(x, wy), (x + ww_, wy + (ww_ * 0.15 if side == "R" else -ww_ * 0.15)), (x + ww_, wy + wh_), (x, wy + wh_)]
                if r < 0.16:
                    m = wc.poly_mask(q); P.lift(m * 0.85, 1.0); wc.wash(P, q, random.choice([GLOW, AMBER]), strength=0.45, var=0.02, layers=3, edge=0.3)
                    glow_at(m, 4, GLOW, 0.25)
                elif r < 0.45:
                    wc.wash(P, q, (22, 24, 34), strength=0.4, var=0.02, layers=3, edge=0.4)
                x += ww_ * random.uniform(2.2, 3.2)

# 路灯：左边人行道边一排，往远处退；灯头一团光晕，灯杆上半截硬下半截化进暗里
for t in (0.0, 0.33, 0.58, 0.76, 0.88):
    x = 150 + (VP[0] - 150) * t * 1.0
    f = (VP[0] - x) / VP[0]
    base = ly(x, 860, "L")
    hgt = 470 * f + 30
    top_ = base - hgt
    P.add(wc.stroke_mask([(x, base), (x, top_)], 3.2 * f + 1, 2.2 * f + 0.8, taper=False, rough=0.1) * (1 - 0.5 * S(base - hgt * 0.4, base, yf)),
          (26, 28, 36), 0.95)
    P.add(wc.stroke_mask([(x, top_), (x + 26 * f + 3, top_ + 4 * f)], 2.4 * f + 0.8, 1.8 * f + 0.6, taper=False), (26, 28, 36), 0.9)
    lx, ly_ = x + 26 * f + 3, top_ + 8 * f + 2
    m = np.clip(1 - S(3 * f + 1.5, 4.5 * f + 2.5, np.hypot(xf - lx, (yf - ly_) * 1.2)), 0, 1).astype(np.float32)
    P.lift(m, 1.0)
    halo = np.exp(-(np.hypot(xf - lx, yf - ly_) / (50 * f + 12)) ** 2)
    P.lift((halo * 0.45).astype(np.float32), 1.0); P.add((halo * 0.8).astype(np.float32), GLOW, 0.3)
    LIGHTS.append((m, float(base), GLOW, hgt * 0.9))

# 楼的深倒影：在车和人之前做（先铺背景的倒影，车人画在上面，不然车灯会被倒影压灰）
rip = wc.noise(1.8, 70, octaves=3, persistence=0.5)
cuts = S(0.6, 0.66, rip) * (0.4 + 0.6 * near)
Dsrc = P.D.copy()
axis = np.where(xf[0] < VP[0], ly(xf[0], 830, "L"), ly(xf[0], 820, "R")).astype(np.float32)
src_y = np.clip((2 * axis[None, :] - yf).astype(np.int32), 0, H - 1)
dark_ref = 1 - np.exp(-Dsrc.mean(axis=2))
dref = dark_ref[src_y, xx] * S(axis[None, :] + 1, axis[None, :] + 4, yf) * np.clip(1 - (yf - axis[None, :]) / 380.0, 0, 1)
dref = wc.blur2(dref.astype(np.float32), 10, 2) * (1 - 0.7 * cuts) * road * (0.4 + 0.6 * mirror)
P.add(dref.astype(np.float32), (40, 44, 62), 0.45)

# 车：迎面一辆（两盏大灯）、往里开一辆（两点红尾灯），身子深，下沿化进湿地
def car(x, y, w, h, facing_us=True):
    body = [(x, y), (x, y - h * 0.55), (x + w * 0.16, y - h), (x + w * 0.84, y - h), (x + w, y - h * 0.55), (x + w, y)]
    wc.wash(P, body, (34, 36, 48), strength=1.0, var=0.015, layers=8, edge=0.45, wet_map=S(y - h * 0.3, y, yf).astype(np.float32), wet_r=4)
    wc.wash(P, [(x + w * 0.2, y - h * 0.94), (x + w * 0.8, y - h * 0.94), (x + w * 0.9, y - h * 0.56), (x + w * 0.1, y - h * 0.56)], (20, 22, 30),
            strength=0.8, var=0.02, layers=4)
    P.lift(wc.stroke_mask([(x + w * 0.18, y - h + 1), (x + w * 0.82, y - h + 1)], 1.6, 1.4, taper=False) * 0.6, 1.0)
    for lx in (x + w * 0.16, x + w * 0.84):
        ly_ = y - h * 0.32
        r = w * 0.07
        m = np.clip(1 - S(r * 0.7, r * 1.2, np.hypot(xf - lx, (yf - ly_) * 1.4)), 0, 1).astype(np.float32)
        if facing_us:
            P.lift(m, 1.0)
            halo = np.exp(-(np.hypot(xf - lx, yf - ly_) / (w * 0.35)) ** 2)
            P.lift((halo * 0.4).astype(np.float32), 1.0); P.add((halo * 0.6).astype(np.float32), (240, 226, 190), 0.25)
            LIGHTS.append((m, float(y), (240, 226, 190), h * 3.2))
        else:
            P.lift(m * 0.7, 1.0); P.add(m, RED, 0.9); glow_at(m, r * 1.5, RED, 0.5)
            LIGHTS.append((m * 0.8, float(y), RED, h * 2.4))

car(512, 690, 70, 40, facing_us=True)
car(396, 664, 44, 26, facing_us=False)
car(560, 652, 30, 18, facing_us=True)

# 打伞的人：伞是一块半圆的深（边上吃一线店里的光），腿是往下拖开的两笔
PEOPLE = []
def umbrella_person(x, y, scale=1.0, ucol=(30, 32, 42), coat=(40, 40, 52)):
    h = 0.4 * (y - VP[1]) * scale
    wc.wash(P, [(x - h * 0.13, y - h * 0.68), (x + h * 0.13, y - h * 0.68), (x + h * 0.1, y - h * 0.3), (x - h * 0.1, y - h * 0.3)], coat,
            strength=1.0, var=0.03, layers=3, edge=0.4)
    legs = np.zeros((H, W), np.float32)
    for dx in (-0.04, 0.05):
        legs = np.maximum(legs, wc.stroke_mask([(x + dx * h, y - h * 0.32), (x + dx * h * 1.4, y)], h * 0.07, h * 0.03, taper=False, rough=0.2))
    legs = legs * S(0.3, 0.55, 0.5 * P.vstreak + 0.5 * P.grain)
    P.add(legs, coat, 1.0)
    ux, uy, ur = x + h * 0.02, y - h * 0.8, h * 0.36
    dome = [(ux + ur * math.cos(a), uy - ur * 0.55 * math.sin(a)) for a in np.linspace(0, math.pi, 20)]
    scal = [(ux - ur + k * ur / 3, uy + (3 if k % 2 else 0) * scale) for k in range(7)]
    um = wc.poly_mask(dome + scal[::-1])
    wc.wash(P, dome + scal[::-1], ucol, strength=1.0, var=0.02, layers=4, edge=0.45)
    rim = np.clip(um - wc.blur(um, 1.2), 0, 1) * (yf < uy - ur * 0.2) * S(0.3, 0.55, P.grain)
    P.lift((rim * 1.4).astype(np.float32), 0.7)
    P.add(wc.stroke_mask([(ux, uy), (ux, uy - ur * 0.62)], 1.0, 1.0, taper=False), (30, 30, 36), 0.6)
    PEOPLE.append(np.maximum(um, legs))

random.seed(SEED + 4)
umbrella_person(250, 842, 1.0)
umbrella_person(290, 836, 0.95, ucol=(150, 40, 40))                       # 全画最暖最艳的一块：一把红伞
umbrella_person(612, 760, 1.0)
umbrella_person(372, 735, 1.0)
umbrella_person(404, 730, 0.92, ucol=(40, 50, 80))
umbrella_person(135, 930, 1.05, coat=(50, 44, 50))

# ================= 第三遍：倒影 =================
# 店门口、路灯底下地上的光斑：不是圆的渐变，是湿地上一摊一摊不规则的暖——夜里地面的"树影"
pool_n = wc.noise(10, 40, octaves=4, persistence=0.6)
pools = np.zeros((H, W), np.float32)
for m, gy, col, L in LIGHTS:
    xs_ = np.nonzero(m.max(axis=0) > 0.3)[0]
    if len(xs_) == 0 or col == RED:
        continue
    cx, wdt = xs_.mean(), max(4.0, float(xs_.max() - xs_.min()))
    rx, ry = wdt * 0.9 + 20, 10 + 0.12 * (gy - VP[1])
    e = np.exp(-(((xf - cx) / rx) ** 2 + ((yf - gy - ry * 0.7) / ry) ** 2)) * S(gy - 2, gy + 2, yf)
    pools = np.maximum(pools, e)
pool = S(0.34, 0.36, pools * (0.55 + 0.9 * pool_n)) * road                 # 硬边、被咬碎
pool_soft = wc.blur(pools, 3) * road
P.lift((pool * 0.45 + pool_soft * 0.15).astype(np.float32), 1.0)
wc.wet(P, (pool * 0.9).astype(np.float32), GLOW, strength=0.3, spread=2, bloom=0.5)
# 灯的倒影：从贴地那条线往下拉成长竖亮，边虚，被雨打出的横纹一截截切开
rip = wc.noise(1.8, 70, octaves=3, persistence=0.5)
cuts = S(0.6, 0.66, rip) * (0.4 + 0.6 * near)
light_ref = np.zeros((H, W), np.float32)
col_acc = np.zeros((H, W, 3), np.float32)
for m, gy, col, L in LIGHTS:
    xs_ = np.nonzero(m.max(axis=0) > 0.3)[0]
    if len(xs_) == 0:
        continue
    cx, wdt = xs_.mean(), max(2.0, (xs_.max() - xs_.min()) * 0.8)
    wob = (wc.noise(12, 3, octaves=2) - 0.5) * (wdt * 0.6 + 3)                     # 拖下来的笔，宽窄和位置都在晃
    streak = np.exp(-(((xf - cx - wob) / (wdt * 0.5 + 1.5) / (0.7 + 0.6 * wc.noise(20, 4, octaves=2))) ** 2)) * S(gy - 1, gy + 3, yf) * np.clip(1 - (yf - gy) / (L * (0.8 + 0.5 * mirror) + 1), 0, 1) ** 0.8
    streak = wc.blur2(streak.astype(np.float32), 2, 1.5) * (1 - 0.8 * cuts) * (0.55 + 0.45 * S(0.3, 0.6, 0.6 * P.vstreak + 0.4 * P.grain))
    light_ref = np.maximum(light_ref, streak)
    col_acc += streak[..., None] * np.array(col, np.float32)[None, None, :]
light_ref *= road * (0.3 + 0.7 * mirror)                                    # 积水里亮，干一点的路面上只剩一点
P.lift((np.clip(light_ref * 1.2, 0, 1)).astype(np.float32), 1.0)
tint = np.clip(col_acc / (light_ref[..., None] + 1e-3), 0, 255)
for c_, col in ((0, GLOW), (1, RED)):
    pass
P.add((light_ref * 0.8).astype(np.float32), GLOW, 0.45)
# 人的深倒影：以各自的脚为轴往下翻，竖着拉、被横纹切
for pm in PEOPLE:
    ys_ = np.nonzero(pm.max(axis=1) > 0.3)[0]
    if len(ys_) == 0:
        continue
    gy = ys_.max()
    fl = np.zeros_like(pm)
    for y in range(gy, min(H, gy + 120)):
        sy = 2 * gy - y
        if 0 <= sy < H:
            fl[y] = pm[sy]
    fl = wc.blur2(fl, 6, 1.5) * np.clip(1 - (yf - gy) / 120.0, 0, 1) * (1 - 0.6 * cuts) * (0.35 + 0.65 * mirror)
    P.add(fl.astype(np.float32), (30, 32, 44), 0.55)

# 前景一截斑马线：几条宽的浅（湿、被倒影切断），透视往里收
for k in range(6):
    xa_ = -60 + k * 120
    stripe = [(xa_, 1010), (xa_ + 70, 1010), (xa_ + 70 + (xa_ + 35 - VP[0]) * 0.18, 1060), (xa_ + (xa_ + 35 - VP[0]) * 0.18, 1060)]
    sm = wc.poly_mask(stripe) * S(0.3, 0.55, 0.6 * P.hstreak + 0.4 * P.grain) * (1 - 0.6 * light_ref)
    P.lift((sm * 0.5).astype(np.float32), 1.0)

# ================= 最后：一点雨丝 =================
random.seed(SEED + 8)
rain = np.zeros((H, W), np.float32)
for _ in range(90):
    x0, y0 = random.uniform(0, W), random.uniform(0, 900)
    L = random.uniform(18, 50)
    rain = np.maximum(rain, wc.stroke_mask([(x0, y0), (x0 - L * 0.18, y0 + L)], 0.8, 0.6, taper=True, rough=0.1))
P.lift((rain * S(0.3, 0.55, P.grain) * 0.35 * (0.4 + 0.6 * (1 - road))).astype(np.float32), 1.0)

grain = wc.grain_from_profile(os.path.join(SKILL_DIR, "pigment_profile.npy"), seed=SEED)
out = os.path.join(HERE, "rainnight_old_cutout.png" if SEED == 6 else f"rainnight_old_cutout_{SEED}.png")
P.render(pigment_tex=grain, tex_amount=0.06).save(out)   # 深色里颗粒显得特别重，减一点
print("saved", out, round(time.time() - t0, 1), "s")
