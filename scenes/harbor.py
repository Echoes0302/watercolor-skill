"""渔港清晨：太阳刚离开海面，一整条光路铺到脚下；渔船全是逆光剪影，桅杆是细线。
第一遍：天（金→粉→冷灰）/ 远岸一条淡 / 水（天的镜子，两边暗、中间光路）。缩小看。
第二遍：远到近的船（越远越淡越冷、越近越深越实），左边一道木栈桥。
第三遍：倒影（每条船以自己的水线为轴）、光路里的碎金。
最后：桅杆、索具、几只鸟。
从 skill 根目录跑：python scenes/harbor.py [seed] [values]
"""
import sys, os, math, random, time
import numpy as np
from PIL import Image, ImageDraw
HERE = os.path.dirname(os.path.abspath(__file__))
SKILL_DIR = os.path.dirname(HERE)
sys.path.insert(0, SKILL_DIR)
import watercolor_lib as wc

wc.set_size(1200, 850)
W, H = wc.W, wc.H
t0 = time.time()
SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 4
STAGE = sys.argv[2] if len(sys.argv) > 2 else "full"
wc.set_seed(SEED)
P = wc.Paper()
yy, xx = np.mgrid[:H, :W]
yf, xf = yy.astype(np.float32), xx.astype(np.float32)
S = wc.smoothstep

HZ = 470                      # 地平线
SUN = (770, 438)
GOLD = (222, 170, 96)
PEACH = (224, 168, 140)
ROSE = (196, 150, 150)
COOL = (140, 150, 168)
MIST = (170, 172, 180)
DEEP = (38, 42, 54)
DEEP_WARM = (74, 56, 50)

sea = S(HZ - 0.5, HZ + 1.5, yf)
skym = 1 - sea
sd = np.hypot(xf - SUN[0], (yf - SUN[1]) * 1.6)
sunglow = np.exp(-(sd / 170.0) ** 2)
halo = np.exp(-(sd / 420.0) ** 2)
# 光路：太阳正下方，往近处变宽
pw = 24 + (yf - HZ).clip(0) * 0.42
path = np.exp(-(((xf - SUN[0] - (yf - HZ) * 0.04) / pw) ** 2)) * sea
wetm = np.clip(0.2 + 0.6 * halo, 0, 1).astype(np.float32)

# ================= 第一遍：只有大块 =================
# 天：上冷下暖，太阳附近几乎是纸
top = np.clip(1 - yf / HZ, 0, 1)
# 清晨的天是顺的：一层层平铺的渐变，不是乌云那种团块。只留很轻的一点水痕
P.add((skym * (0.15 + 0.85 * top ** 1.3) * (1 - 0.92 * sunglow)).astype(np.float32), COOL, 0.42, granulate=0.05)
wc.wet(P, (skym * top * (1 - sunglow)).astype(np.float32), COOL, strength=0.1, spread=70)
P.add((skym * halo * (1 - 0.85 * sunglow) * (1 - 0.6 * top)).astype(np.float32), PEACH, 0.4, granulate=0.05)
P.add((skym * np.exp(-(sd / 240.0) ** 2) * (1 - 0.97 * np.exp(-(sd / 55.0) ** 2))).astype(np.float32), GOLD, 0.5)
# 几道横的云：干笔横扫，压着太阳过去，越靠近太阳边越亮
random.seed(SEED + 3)
clouds = np.zeros((H, W), np.float32)
for cy, cx0, cx1, w in [(300, 380, 1120, 14), (345, 180, 700, 9), (395, 560, 1050, 7), (250, -20, 520, 18), (418, 820, 1200, 5)]:
    xs = np.array([cx0, (cx0 + cx1) / 2, cx1], float)
    ys = np.array([cy, cy + random.uniform(-6, 6), cy + random.uniform(-4, 4)])
    yc = np.interp(np.clip(xf[0], cx0, cx1), xs, ys)[None, :]
    wv = w * (0.6 + 0.8 * wc.noise(6, 60, octaves=3))
    inside = (np.abs(yf - yc) < wv) & (xf >= cx0) & (xf <= cx1)
    t = np.clip((xf - cx0) / (cx1 - cx0), 0, 1)
    ends = np.minimum(t, 1 - t)
    clouds = np.maximum(clouds, inside * S(0.0, 0.25, ends))
cl = clouds * S(0.35, 0.5, 0.7 * P.hstreak + 0.3 * P.grain) * skym
wc.wet(P, (cl * (1 - 0.6 * sunglow)).astype(np.float32), (132, 118, 130), strength=0.36, spread=2, granulate=0.2)
P.lift((cl * np.exp(-(sd / 150.0) ** 2) * 0.6).astype(np.float32), 1.0)   # 靠太阳那截被照透
# 远岸：左边一条低低的山，右边一个很远的小镇 + 一根教堂尖（破形），全淡、冷、贴着地平线
hill = []
for x in np.linspace(-20, 560, 40):
    hill.append((x, HZ - 26 * math.exp(-((x - 140) / 180) ** 2) - 12 * math.exp(-((x - 420) / 90) ** 2) - 4))
hill += [(560, HZ + 1), (-20, HZ + 1)]
wc.wash(P, hill, MIST, strength=0.5, var=0.01, layers=12, edge=0.3, wet_map=np.full((H, W), 0.6, np.float32), wet_r=4)
town = [(930, HZ + 1), (930, HZ - 8), (960, HZ - 8), (962, HZ - 14), (990, HZ - 14), (992, HZ - 9), (1030, HZ - 9), (1032, HZ - 16),
        (1044, HZ - 16), (1046, HZ - 50), (1049, HZ - 60), (1052, HZ - 50), (1054, HZ - 16), (1080, HZ - 12), (1120, HZ - 12), (1122, HZ - 6),
        (1200, HZ - 6), (1200, HZ + 1)]
wc.wash(P, town, MIST, strength=0.45, var=0.006, layers=10, edge=0.3)
# 水：天的镜子——上面（远）亮暖，近处冷深；光路留白
near = np.clip((yf - HZ) / (H - HZ), 0, 1)
P.add((sea * (0.2 + 0.8 * near ** 0.8) * (1 - 0.9 * path)).astype(np.float32), COOL, 0.62, granulate=0.2)
P.add((sea * halo * (1 - near) * (1 - 0.8 * path)).astype(np.float32), PEACH, 0.35)
P.add((sea * np.exp(-(((xf - SUN[0]) / 200.0) ** 2)) * (1 - near) * 0.8).astype(np.float32), GOLD, 0.3)       # 光路两边的金
swellb = S(0.42, 0.74, wc.noise(9, 260, octaves=3)) * S(0.0, 0.25, near)                                                              # 水是一道道横的起伏
P.add((wc.blur(sea * swellb, 1.5) * (0.3 + 0.7 * near) * (1 - 0.75 * path)).astype(np.float32), (62, 72, 94), 0.3, granulate=0.2)
# 太阳：一个擦白的圆 + 贴着它的金
sun_m = np.clip(1 - S(13, 16, np.hypot(xf - SUN[0], yf - SUN[1])), 0, 1)
P.lift(sun_m.astype(np.float32), 1.0)

if STAGE == "values":
    img = P.render()
    out = os.path.join(HERE, "harbor_values.png")
    img.save(out)
    img.convert("L").resize((96, 68), Image.LANCZOS).resize((384, 272), Image.NEAREST).save(out.replace(".png", "_thumb.png"))
    print("values", out, round(time.time() - t0, 1)); sys.exit()

# ================= 第二遍：船和栈桥 =================
BOATS = []   # (mask, waterline_y, x_mast_list, mast_tops)


def boat(x, y, L, facing=1, hh=None, dark=1.0, cabin=True, masts=1, tone=(48, 42, 52), lean=0.0):
    """一条渔船的逆光剪影。x=船中，y=水线，L=船长。越远 dark 越小（被晨雾吃掉，偏冷偏浅）。
    船身下沿不收边：直接接进倒影里，中间只留一线水光。"""
    hh = hh or L * 0.15
    xa, xb = x - L / 2, x + L / 2
    bow = xb if facing > 0 else xa
    stern = xa if facing > 0 else xb
    sheer = []
    for u in np.linspace(0, 1, 26):                          # 舷弧：船尾低平，往船头翘起来
        sheer.append((stern + (bow - stern) * u, y - hh * (0.85 + 0.75 * u ** 2.6)))
    hull = sheer + [(bow - facing * L * 0.05, y - hh * 0.2), (bow - facing * L * 0.1, y + 2),
                    (stern + facing * L * 0.05, y + 2), (stern, y - hh * 0.3)]
    col = tuple(int(c * dark + m * (1 - dark)) for c, m in zip(tone, MIST))
    cov = wc.wash(P, hull, col, strength=0.6 + 0.55 * dark, var=0.01, layers=10, edge=0.45, granulate=0.12)
    wc.wet(P, (cov * S(y - hh * 0.9, y, yf)).astype(np.float32), DEEP_WARM, strength=0.25 * dark, spread=2)   # 船身下半截更深一点
    mask = cov.copy()
    if cabin:
        cx = stern + facing * L * 0.18
        cw, ch = L * 0.26, hh * 1.25
        yb_ = y - hh * 0.9
        cab = [(cx, yb_), (cx, yb_ - ch), (cx - facing * cw * 0.08, yb_ - ch - 3), (cx + facing * cw * 1.02, yb_ - ch - 3),
               (cx + facing * cw * 0.95, yb_ - ch), (cx + facing * cw * 1.08, yb_ - ch * 0.35), (cx + facing * cw * 1.08, yb_)]
        c2 = wc.wash(P, cab, col, strength=0.55 + 0.5 * dark, var=0.012, layers=6, edge=0.45)
        mask = np.maximum(mask, c2)
        if dark > 0.55:   # 驾驶室前窗透一点对面的天
            wx = cx + facing * cw * 0.72
            P.lift((wc.poly_mask([(wx, yb_ - ch * 0.45), (wx + facing * cw * 0.26, yb_ - ch * 0.38),
                                  (wx + facing * cw * 0.26, yb_ - ch * 0.78), (wx, yb_ - ch * 0.82)]) * S(0.3, 0.6, P.grain) * 0.4).astype(np.float32), 1.0)
    # 舷上一线亮：逆光，只有靠近光的那几条才有
    im = Image.new("L", (W, H), 0)
    ImageDraw.Draw(im).line([(int(a), int(b)) for a, b in sheer], fill=255, width=2)
    rim = wc.blur(np.asarray(im, np.float32) / 255.0, 0.7) * S(0.3, 0.55, P.grain) * np.clip(1.3 - np.abs(xf - SUN[0]) / 420, 0, 1)
    P.lift((rim * 0.65 * dark).astype(np.float32), 1.0)
    mxs, mts = [], []
    for k in range(masts):
        mx = x + facing * L * (0.12 - 0.3 * k)
        mt = y - hh - L * (0.5 - 0.1 * k) * random.uniform(0.9, 1.1)
        mxs.append(mx); mts.append(mt)
    BOATS.append(dict(mask=mask, y=y, x=x, L=L, facing=facing, hh=hh, dark=dark, col=col, mxs=mxs, mts=mts, bow=bow, stern=stern, lean=lean))


random.seed(SEED + 11)
# 远：一排小的，淡、冷，几乎化进雾里；高低错开，不排队
for x, dy, L, f in [(470, 12, 44, 1), (548, 16, 36, -1), (628, 10, 50, 1), (905, 14, 42, -1), (975, 20, 30, 1), (1098, 13, 38, -1)]:
    boat(x, HZ + dy, L, f, dark=0.28, cabin=random.random() < 0.6)
# 中：光路两边、光路里，前后错开，有一条叠在另一条后面
for x, y, L, f, m, d in [(906, 506, 64, -1, 1, 0.5), (706, 516, 104, 1, 1, 0.75), (812, 534, 92, -1, 1, 0.8),
                         (566, 556, 88, -1, 2, 0.78), (1030, 572, 140, 1, 1, 0.85)]:
    boat(x, y, L, f, dark=d, masts=m)
# 近：左下一条大的，一半出画，最深
boat(190, 712, 420, 1, hh=52, dark=1.0, masts=1, tone=(40, 36, 44))

# 左边的木栈桥：一排桩子往远处退，上面一条桥面；尽头站两个钓鱼的
PV = (560, HZ)                  # 栈桥的消失点
def on_pier(t):                 # t=0 近端（画面左下外），t=1 远端
    x = -60 + (PV[0] - (-60)) * t * 0.62
    return x
deck_near, deck_far = (-60, 520), (340, 486)
pier_top = [deck_near, deck_far, (deck_far[0], deck_far[1] + 4), (deck_near[0], deck_near[1] + 14)]
covP = wc.wash(P, pier_top, (58, 52, 50), strength=0.9, var=0.01, layers=8, edge=0.45)
for k in range(9):
    t = k / 8
    px_ = deck_near[0] + (deck_far[0] - deck_near[0]) * (t ** 0.8)
    py_ = deck_near[1] + (deck_far[1] - deck_near[1]) * (t ** 0.8)
    s_ = 1 - 0.8 * t
    wl = HZ + (py_ - HZ) * 1.9                    # 桩脚进水的地方
    P.add(wc.stroke_mask([(px_, py_ + 4), (px_ + 1, wl)], 6 * s_ + 1.2, 6.5 * s_ + 1.4, taper=False, rough=0.2), (48, 44, 44), 0.9)
    BOATS.append(dict(pile=True, x=px_, y=wl, w=6 * s_ + 1.2, top=py_))
# 栈桥栏杆：一条细线 + 几根竖
P.add(wc.stroke_mask([(deck_near[0], deck_near[1] - 26), (deck_far[0], deck_far[1] - 7)], 2.2, 1.0, taper=False, rough=0.15), (50, 46, 46), 0.8)
for k in range(7):
    t = (k + 0.5) / 7
    px_ = deck_near[0] + (deck_far[0] - deck_near[0]) * t
    py_ = deck_near[1] + (deck_far[1] - deck_near[1]) * t
    P.add(wc.stroke_mask([(px_, py_), (px_, py_ - 26 + 19 * t)], 1.8 - t, 1.6 - t, taper=False), (50, 46, 46), 0.8)
# 两个钓鱼的：一站一坐，鱼竿一根细弧
def fisher(x, y, h, sit=False):
    wc.dab(P, x, y - h * 0.9, h * 0.08, (96, 74, 66), 1.1)
    if sit:
        body = [(x - h * 0.11, y - h * 0.8), (x + h * 0.1, y - h * 0.8), (x + h * 0.14, y - h * 0.45), (x - h * 0.12, y - h * 0.45)]
    else:
        body = [(x - h * 0.1, y - h * 0.82), (x + h * 0.1, y - h * 0.82), (x + h * 0.06, y - h * 0.36), (x - h * 0.06, y - h * 0.36)]
    wc.wash(P, body, (40, 40, 48), strength=1.0, var=0.03, layers=3, edge=0.4)
    if not sit:
        for dx in (-0.03, 0.04):
            P.add(wc.stroke_mask([(x + dx * h, y - h * 0.38), (x + dx * h * 1.3, y)], h * 0.07, h * 0.03, taper=False), (40, 40, 48), 0.95)
    rod = [(x + h * 0.08, y - h * 0.7), (x + h * 0.9, y - h * 1.35), (x + h * 1.6, y - h * 1.5)]
    P.add(wc.stroke_mask(rod, 1.2, 0.6, taper=False, rough=0.05), (50, 50, 56), 0.7)
    P.add(wc.stroke_mask([rod[-1], (rod[-1][0] + 4, HZ + 40)], 0.6, 0.5, taper=False) * S(0.4, 0.6, P.grain), (70, 70, 76), 0.35)

fisher(236, 497, 34)
fisher(262, 497, 26, sit=True)

# ================= 第三遍：倒影和碎金 =================
ripple = wc.noise(2.4, 80, octaves=3, persistence=0.5)
patch = S(0.4, 0.6, wc.noise(40, 220, octaves=2))
brk = S(0.62 - 0.1 * near, 0.65 - 0.1 * near, ripple) * (0.35 + 0.65 * patch)
# 水在动：倒影按位移场取，一横条一横条各自错开，越近越大（竖的桅杆、船身倒下来成锯齿）
dxw = (wc.noise(2.2, 40, octaves=3, persistence=0.55) - 0.5) * 2 * (2 + 18 * near)
dyw = (wc.noise(5, 60, octaves=2) - 0.5) * 2 * (1 + 4 * near)
refl = np.zeros((H, W), np.float32)
for b in BOATS:
    if b.get("pile"):
        pts = [(b["x"] + 1.2 * math.sin(k * 1.7 + b["x"]), b["y"] + k * 5) for k in range(7)]
        m = wc.stroke_mask(pts, b["w"], b["w"] * 0.7, taper=False, rough=0.3) * S(0.3, 0.5, P.vstreak)
        refl = np.maximum(refl, m * 0.7)
        continue
    y0 = b["y"]
    sy_ = np.clip((2 * y0 - yf + dyw).astype(np.int32), 0, H - 1)
    sx_ = np.clip((xx + dxw).astype(np.int32), 0, W - 1)
    fl = b["mask"][sy_, sx_] * (yf > y0 + 1) * np.clip(1 - (yf - y0) / (b["hh"] * 3 + 30), 0, 1)
    fl = wc.blur2(fl.astype(np.float32), 1.5, 1.0)
    refl = np.maximum(refl, fl * (0.45 + 0.55 * b["dark"]))
refl *= (1 - 0.5 * brk) * sea
# 海面不规矩的那一层：反着天光的亮块，噪声长出来，横着压扁、硬边；光路两边多、远处碎
densw = 0.6 * wc.noise(5, 130, octaves=4, persistence=0.55) + 0.4 * wc.noise(28, 280, octaves=2)
thr_s = 0.58 - 0.08 * near
glints = S(thr_s - 0.012, thr_s + 0.012, densw) * sea * (1 - 0.6 * path)
refl *= (1 - 0.7 * glints)
P.lift((glints * (0.3 + 0.2 * (1 - near))).astype(np.float32), 1.0)
P.add((glints * (1 - near) * np.exp(-(((xf - SUN[0]) / 380.0) ** 2))).astype(np.float32), PEACH, 0.2)
wc.wet(P, (sea * (1 - glints) * S(0.6, 0.78, wc.noise(30, 110, octaves=3)) * near).astype(np.float32), (56, 64, 88), strength=0.3, spread=4, bloom=0.6)
P.add(refl.astype(np.float32), (44, 42, 56), 0.95)
P.lift((refl * S(0.5, 0.62, wc.noise(1.2, 40, octaves=2)) * S(0.2, 0.5, refl) * 0.5).astype(np.float32), 1.0)   # 水线那儿一线断续的水光
# 光路里的碎金：横的短亮，靠中线最密，近处大
spark = S(0.62 - 0.14 * path, 0.64 - 0.14 * path, 0.7 * ripple + 0.3 * wc.noise(1.6, 18, octaves=2)) * path
P.lift((spark * 0.95).astype(np.float32), 1.0)
P.add((spark * 0.4 * (1 - near)).astype(np.float32), GOLD, 0.25)
P.lift((path * (1 - near) * 0.35).astype(np.float32), 1.0)                     # 远处光路本身更亮
# 太阳贴着海面那一截：海面上一小片刺眼的亮
P.lift((np.exp(-(((xf - SUN[0]) / 60.0) ** 2 + ((yf - HZ - 6) / 8.0) ** 2)) * sea * 0.8).astype(np.float32), 1.0)

# ================= 最后：桅杆、索具、鸟 =================
for b in BOATS:
    if b.get("pile"):
        continue
    for mx, mt in zip(b["mxs"], b["mts"]):
        base = b["y"] - b["hh"] * 1.0
        w0 = 1.0 + 1.6 * b["dark"] * (b["L"] / 300)
        P.add((wc.stroke_mask([(mx, base), (mx + b["lean"], mt)], w0 + 0.6, w0 * 0.5, taper=False, rough=0.1)).astype(np.float32), b["col"], 0.9)
        if b["dark"] > 0.5:   # 索具：桅顶到船头、船尾两根细线，断续
            for end in (b["bow"], b["stern"]):
                m = wc.stroke_mask([(mx, mt + 2), (end, b["y"] - b["hh"] * 1.1)], 0.9, 0.7, taper=False, rough=0.1)
                P.add((m * S(0.35, 0.55, P.grain)).astype(np.float32), b["col"], 0.55)
        # 桅杆的倒影：扭着的细线
        L_ = (b["y"] - mt) * 0.45
        pts = [(mx + float(dxw[min(H - 1, int(b['y'] + 4 + L_ * k / 14)), int(min(W - 1, max(0, mx)))]), b["y"] + 4 + L_ * k / 14) for k in range(15)]
        rm = wc.stroke_mask(pts, w0 * 0.9, w0 * 0.3, taper=False, rough=0.35) * (1 - 0.8 * brk) * S(0.35, 0.5, P.grain)
        P.add(rm.astype(np.float32), b["col"], 0.5 * b["dark"] + 0.2)
# 鸟：几只海鸥，逆光的剪影，远的更淡更小
random.seed(SEED + 21)
for bx_, by_, s, a in [(640, 230, 11, 0.8), (680, 250, 8, 0.7), (720, 214, 7, 0.6), (900, 300, 6, 0.5), (420, 380, 9, 0.7), (1010, 190, 5, 0.45)]:
    lift = random.uniform(0.2, 0.45)
    P.add(wc.stroke_mask([(bx_ - s, by_ + s * 0.1), (bx_ - s * 0.45, by_ - s * lift), (bx_, by_)], 1.5, 1.1), (54, 54, 64), a)
    P.add(wc.stroke_mask([(bx_, by_), (bx_ + s * 0.45, by_ - s * lift * 1.1), (bx_ + s, by_ + s * 0.05)], 1.5, 1.1), (54, 54, 64), a)

grain = wc.grain_from_profile(os.path.join(SKILL_DIR, "pigment_profile.npy"), seed=SEED)
out = os.path.join(HERE, "harbor.png" if SEED == 4 else f"harbor_{SEED}.png")
P.render(pigment_tex=grain, tex_amount=0.1).save(out)
print("saved", out, round(time.time() - t0, 1), "s")
