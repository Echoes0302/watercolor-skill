"""威尼斯小运河，下午逆光。用 watercolor skill 的方法：先大块，再从块里长东西，再水面，最后少量线。
太阳在运河尽头偏左、压得很低：右岸立面朝左迎光（暖），左岸背光（一整块深）；桥在逆光里是剪影。
从 skill 根目录跑：python scenes/venice.py [seed] [values]
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
SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 5
STAGE = sys.argv[2] if len(sys.argv) > 2 else "full"
wc.set_seed(SEED)
P = wc.Paper()
yy, xx = np.mgrid[:H, :W]
yf, xf = yy.astype(np.float32), xx.astype(np.float32)
S = wc.smoothstep

VP = (430, 600)


def isolated(fn):
    """新加的细节自己用一段随机数，画完把两条随机数流还回去——后面的水面一丝不变。"""
    st, rs = random.getstate(), wc.rng.bit_generator.state
    fn()
    random.setstate(st); wc.rng.bit_generator.state = rs
KL, KR = 0.651, 0.50              # 水线：左岸 y-600 = KL*(430-x)，右岸 y-600 = KR*(x-430)（同一个水面，两岸离我们远近不同）
GLARE = (395, 565)                # 太阳在桥后面偏左，压得很低

WARM_LIGHT = (200, 186, 164)
COOL_AIR = (148, 158, 168)
DEEP = (44, 54, 64)
DEEP_WARM = (80, 62, 54)
OCHRE = (208, 162, 106)
TERRA = (188, 114, 86)
PINK = (210, 160, 148)
CREAM = (214, 196, 166)
TEAL = (104, 132, 128)
GOLD = (214, 166, 92)


def water_y(x):
    x = np.asarray(x, np.float32)
    return np.where(x < VP[0], VP[1] + KL * (VP[0] - x), VP[1] + KR * (x - VP[0]))


def ly(x, near_y, side):
    """过消失点的线：side L 在 x=0 处 y=near_y；side R 在 x=W 处 y=near_y。"""
    f = (VP[0] - x) / VP[0] if side == "L" else (x - VP[0]) / (W - VP[0])
    return VP[1] + (near_y - VP[1]) * f

def rec(x, y, x2):
    """墙上过 (x, y) 的水平线（朝消失点收）走到 x2 时的 y。窗的上下沿、拱顶、窗台、百叶都顺着它。
    9/29 她抓的：以前每格窗的斜度写死（右岸一律往右下、左岸一律往右上），视线以上的窗全往外翘。"""
    return VP[1] + (y - VP[1]) * (x2 - VP[0]) / (x - VP[0])



def skyline(segs, side):
    pts = []
    for xa, xb, T, orn in segs:
        pts.append((xa, ly(xa, T, side)))
        for kind, u, s in orn:
            x = xa + (xb - xa) * u
            y = ly(x, T, side)
            f = abs(VP[0] - x) / (VP[0] if side == "L" else W - VP[0])
            s = s * max(0.22, f)
            if kind == "funnel":          # 威尼斯的喇叭烟囱
                pts += [(x - s * 0.13, y), (x - s * 0.11, y - s * 0.55), (x - s * 0.3, y - s * 0.85), (x + s * 0.3, y - s * 0.85),
                        (x + s * 0.11, y - s * 0.55), (x + s * 0.13, y)]
            elif kind == "chim":
                pts += [(x - s * 0.18, y), (x - s * 0.18, y - s * 0.55), (x + s * 0.18, y - s * 0.55), (x + s * 0.18, y)]
            elif kind == "altana":        # 屋顶木平台：一个小方架子
                pts += [(x - s * 0.5, y), (x - s * 0.5, y - s * 0.35), (x + s * 0.5, y - s * 0.35), (x + s * 0.5, y)]
        pts.append((xb, ly(xb, T, side)))
    return pts


# ---------- 湿度图：远处、光里湿；焦点（贡多拉、桥）干 ----------
glare = np.exp(-(((xf - GLARE[0]) / 150.0) ** 2 + ((yf - GLARE[1]) / 110.0) ** 2))
dist = np.clip(1 - np.hypot(xf - VP[0], (yf - VP[1]) * 1.3) / 380.0, 0, 1)
focus = np.exp(-(((xf - 330) / 190.0) ** 2 + ((yf - 790) / 60.0) ** 2)) + np.exp(-(((xf - 440) / 90.0) ** 2 + ((yf - 615) / 40.0) ** 2))
wet = np.clip(0.1 + 0.5 * dist + 0.45 * glare - 0.9 * focus, 0, 1).astype(np.float32)
WL = water_y(xf[0])[None, :]                          # 每一列的水线
water = S(WL - 1, WL + 2, yf)

# ================= 第一遍：只有大块 =================
# 1. 天：一整片湿，上面冷一点，光那里透白
sky = (1 - water) * 1.0
top_dark = np.clip(1 - yf / 620.0, 0, 1) ** 0.8
wc.wet(P, (sky * (0.45 + 0.55 * top_dark) * (1 - 0.95 * glare)).astype(np.float32), WARM_LIGHT, strength=0.55, spread=45, granulate=0.05)
wc.wet(P, (sky * top_dark * (1 - glare) * (0.6 + 0.4 * wc.noise(200, 300, octaves=2))).astype(np.float32), COOL_AIR, strength=0.3, spread=60)

def _sky_more():
    # 参考（约瑟夫的威尼斯）：天很大、很亮，上面压着几团软的灰云，湿接湿、边全化开，越往地平线越亮；一点暖只在光那一圈
    corner = np.clip(1 - np.hypot(xf / 900.0, yf / 700.0), 0, 1) * 0.5 + np.clip(1 - np.hypot((W - xf) / 1100.0, yf / 600.0), 0, 1) * 0.3
    wc.wet(P, (sky * corner * (1 - glare)).astype(np.float32), (140, 142, 152), strength=0.35, spread=60, granulate=0.05)
    warm_r = np.exp(-(((xf - GLARE[0]) / 260.0) ** 2 + ((yf - GLARE[1]) / 200.0) ** 2)) * (1 - 0.9 * np.exp(-(((xf - GLARE[0]) / 90.0) ** 2 + ((yf - GLARE[1]) / 70.0) ** 2)))
    wc.wet(P, (sky * warm_r).astype(np.float32), (222, 180, 120), strength=0.25, spread=40)
    mass = 0.6 * wc.noise(55, 140, octaves=3) + 0.4 * wc.noise(140, 300, octaves=2)
    cl = S(0.48, 0.66, mass) * sky * (1 - S(120, 330, yf)) * (1 - 0.8 * glare)
    wc.wet(P, cl.astype(np.float32), (146, 148, 156), strength=0.32, spread=14, bloom=0.6)
    cl2 = S(0.58, 0.7, mass) * sky * (1 - S(60, 220, yf))
    wc.wet(P, cl2.astype(np.float32), (120, 122, 132), strength=0.18, spread=10, bloom=0.8)
    P.lift((sky * S(0.66, 0.74, mass) * (1 - S(80, 260, yf)) * 0.35).astype(np.float32), 1.0)      # 云团之间透出几块亮


isolated(_sky_more)

# 2. 左岸：背光，一整块深，顶上吃一圈亮边；远处化进光里
L_segs = [(-40, 70, 120, [("funnel", 0.3, 60), ("altana", 0.75, 60)]),
          (70, 150, 200, [("chim", 0.5, 40)]),
          (150, 230, 150, [("funnel", 0.6, 50)]),
          (230, 300, 250, [("altana", 0.4, 60)]),
          (300, 355, 210, [("funnel", 0.5, 50)]),
          (355, 400, 300, []),
          (400, 426, 330, [])]
left = skyline(L_segs, "L") + [(426, float(water_y(426))), (-40, float(water_y(-40)) + 4)]
jL = (wc.noise(12, 30, octaves=3) - 0.5) * 30
fadeL = (1 - S(WL - 10 + jL, WL + 6 + jL, yf)).astype(np.float32)
covL = wc.wash(P, left, DEEP, strength=0.92, var=[0.012] * (len(left) - 2) + [0.03] * 2, layers=30, edge=0.5,
               granulate=0.14, fade=fadeL, wet_map=wet * 0.35, wet_r=6)
wc.wet(P, covL * S(0.45, 0.7, wc.noise(90, octaves=3)), DEEP_WARM, strength=0.5, spread=8)   # 暗里掺暖，不死黑
# 远处按栋变浅（空气透视），楼顶那条边留着
for (xa, xb, T, _), amt in zip(L_segs[-3:], (0.25, 0.45, 0.65)):
    blk = (S(xa - 2, xa + 2, xf) * (1 - S(xb - 2, xb + 2, xf))).astype(np.float32)
    P.lift((covL * blk * amt).astype(np.float32), 1.0)

# 3. 右岸：迎光，一栋一栋不同的暖色（接缝是颜色和深浅的差，不是线）
R_segs = [(436, 470, 350, [], CREAM),
          (470, 520, 300, [("funnel", 0.5, 40)], PINK),
          (520, 590, 250, [("altana", 0.5, 50)], OCHRE),
          (590, 670, 300, [("funnel", 0.3, 50), ("chim", 0.8, 34)], CREAM),
          (670, 780, 180, [("funnel", 0.6, 60)], TERRA),
          (780, 940, 130, [("chim", 0.3, 50), ("funnel", 0.75, 70)], OCHRE)]
# 先一整块暖的中间调铺满右岸（一块，不是一条条），再在湿的时候给每栋掉一点自己的颜色，边自己化开
right = skyline([(a, b, T, o) for a, b, T, o, _ in R_segs], "R") + [(940, float(water_y(940))), (436, float(water_y(436)))]
jR = (wc.noise(10, 30, octaves=3) - 0.5) * 20
fadeR = (1 - S(WL - 8 + jR, WL + 6 + jR, yf)).astype(np.float32)
covR_all = wc.wash(P, right, CREAM, strength=0.7, var=[0.01] * (len(right) - 2) + [0.03] * 2, layers=26, edge=0.45,
                   granulate=0.25, fade=fadeR, wet_map=wet * 0.4, wet_r=5)
for xa, xb, T, orn, col in R_segs:
    f0 = (xa - VP[0]) / (W - VP[0])
    blk = S(xa - 3, xa + 3, xf) * (1 - S(xb - 3, xb + 3, xf))
    wc.wet(P, (covR_all * blk * (0.6 + 0.4 * wc.noise(60, 40, octaves=3))).astype(np.float32), col, strength=0.35 + 0.35 * f0, spread=4, granulate=0.2)
# 远处几栋往光里退：按栋分档变浅
for (xa, xb, T, _, _), amt in zip(R_segs[:3], (0.55, 0.38, 0.18)):
    blk = (S(xa - 2, xa + 2, xf) * (1 - S(xb - 2, xb + 2, xf))).astype(np.float32)
    P.lift((covR_all * blk * amt).astype(np.float32), 1.0)
# 迎光墙上只有冷暖变化：一块块掉进去的赭石和一点粉，不是从上往下的渐变
wc.wet(P, covR_all * S(0.52, 0.72, wc.noise(70, octaves=3)), (178, 128, 96), strength=0.25, spread=5)
wc.wet(P, covR_all * S(0.55, 0.75, wc.noise(50, 90, octaves=3)), COOL_AIR, strength=0.12, spread=6)

# 4. 水：一整片偏绿的灰，近处深，光路那里留白
path = np.exp(-(((xf - (GLARE[0] + 20) - (yf - 640) * 0.05) / (26 + (yf - 640).clip(0) * 0.28)) ** 2))
wtone = water * (0.45 + 0.55 * np.clip((yf - 620) / 520.0, 0, 1))
wc.wet(P, (wtone * (1 - 0.9 * path)).astype(np.float32), TEAL, strength=0.55, spread=12, granulate=0.2)

# 5. 桥：在逆光里，一块深的剪影；拱洞里透着桥后面的光
BY = 668.0
xL = VP[0] - (BY - VP[1]) / KL
xR = VP[0] + (BY - VP[1]) / KR
bx = np.linspace(xL - 8, xR + 8, 30)
deck = [(x, BY - 66 - 20 * math.sin(math.pi * (x - (xL - 8)) / (xR - xL + 16))) for x in bx]
arch = [(x, BY - 40 * math.sqrt(max(0.0, 1 - ((x - (xL + xR) / 2) / ((xR - xL) / 2 - 30)) ** 2)))
        for x in np.linspace(xR - 30, xL + 30, 28)]
bridge = deck + [(xR + 8, BY + 1), (xR - 30, BY + 1)] + arch + [(xL + 30, BY + 1), (xL - 8, BY + 1)]
covB = wc.wash(P, bridge, (70, 72, 80), strength=0.9, var=0.01, layers=14, edge=0.5, granulate=0.1)
wc.wet(P, covB * S(0.5, 0.7, wc.noise(20, octaves=2)), DEEP_WARM, strength=0.35, spread=3)
# 桥上沿吃一线亮
im = Image.new("L", (W, H), 0)
ImageDraw.Draw(im).line([(int(x), int(y) + 1) for x, y in deck], fill=255, width=2)
P.lift((wc.blur(np.asarray(im, np.float32) / 255.0, 0.8) * S(0.35, 0.55, P.grain)).astype(np.float32), 0.8)

if STAGE == "values":
    img = P.render()
    out = os.path.join(HERE, "venice_values.png")
    img.save(out)
    img.convert("L").resize((64, 82), Image.LANCZOS).resize((256, 328), Image.NEAREST).save(out.replace(".png", "_thumb.png"))
    print("values", out, round(time.time() - t0, 1)); sys.exit()

# ================= 破形：远处一座钟楼，很淡，脚藏在右岸楼后面 =================
TX, tb = 515, 566
behind = (1 - wc.blur(covR_all, 1.5)).astype(np.float32)                 # 脚藏在右岸远楼后面
bw = 15                                                                  # 半宽 15：再细就成了灯塔/电视塔
wc.wash(P, [(TX - bw, tb), (TX - bw, 430), (TX + bw, 430), (TX + bw, tb)], (150, 156, 166), strength=0.36, var=0.006, layers=8, edge=0.3,
        fade=((1 - 0.6 * S(480, 560, yf)) * behind).astype(np.float32))
wc.wash(P, [(TX - bw - 1, 432), (TX - bw - 1, 398), (TX + bw + 1, 398), (TX + bw + 1, 432)], (140, 146, 156), strength=0.4, var=0.006, layers=6, edge=0.3)  # 钟室
for ox in (-9, 1):
    P.lift(wc.poly_mask([(TX + ox, 427), (TX + ox, 404), (TX + ox + 8, 404), (TX + ox + 8, 427)]) * 0.4, 1.0)
wc.wash(P, [(TX - bw + 2, 398), (TX - bw + 2, 388), (TX + bw - 2, 388), (TX + bw - 2, 398)], (140, 146, 156), strength=0.42, var=0.006, layers=5, edge=0.3)  # 小方顶
wc.wash(P, [(TX - bw + 1, 389), (TX, 352), (TX + bw - 1, 389)], (136, 144, 156), strength=0.42, var=0.006, layers=6, edge=0.3)                 # 四棱尖顶：宽、矮
P.add(wc.stroke_mask([(TX, 353), (TX, 344)], 1.0, 0.8, taper=False), (130, 136, 146), 0.6)

# ================= 第二遍：块里长东西 =================
# 右岸：拱窗一排排（深，偶尔一格百叶窗是灰绿的），每层一条浅石线，水线上一圈深绿的水渍，水门
random.seed(SEED + 1)
NOB = (1 - wc.blur(covB, 1.0)).astype(np.float32)          # 桥挡着的地方不画
for xa, xb, T, orn, col in R_segs:
    f0 = (xa - VP[0]) / (W - VP[0])
    if f0 < 0.05:
        continue
    for k, off in enumerate((40, 170, 300, 430)):         # 石线：浅、断续
        yn = T + off
        if ly(xa, yn, "R") > float(water_y(xa)) - 20 * f0:
            continue
        m = wc.stroke_mask([(xa + 3, ly(xa + 3, yn, "R")), (xb - 3, ly(xb - 3, yn, "R"))], 2.4 * f0 + 0.8, 1.6 * f0 + 0.6, taper=False, rough=0.4)
        P.lift((m * S(0.45, 0.6, 0.6 * P.hstreak + 0.4 * P.grain) * 0.45).astype(np.float32), 1.0)
    for row in range(5):
        yn = T + 75 + row * 128 + random.uniform(-8, 8)
        if ly(xb, yn + 60, "R") > float(water_y(xb)) - 30 * f0:
            continue
        x = xa + 10 * f0 + 4
        while x < xb - 6:
            f = (x - VP[0]) / (W - VP[0])
            wy = ly(x, yn, "R")
            ww, wh = (13 * f + 2) * random.uniform(0.8, 1.15), (46 * f + 5) * random.uniform(0.8, 1.05)
            q = [(x, wy + wh * 0.18), (x + ww * 0.5, rec(x, wy - wh * 0.02, x + ww * 0.5)), (x + ww, rec(x, wy + wh * 0.18, x + ww)),
                 (x + ww, rec(x, wy + wh, x + ww)), (x, wy + wh)]
            r = random.random()
            if r < 0.5:
                wc.wash(P, q, (40, 38, 42), strength=random.uniform(0.25, 0.65), var=0.03, layers=3, edge=0.4, fade=NOB)
                def _depth(x=x, wy=wy, ww=ww, wh=wh, f=f, col=col):
                    # 窗是个洞：洞口上沿（过梁的影）和靠里那一侧更深；窗台是亮的一横，底下一线影；窗台下往下淌一道
                    top_ = [(x, wy + wh * 0.18), (x + ww * 0.5, rec(x, wy - wh * 0.02, x + ww * 0.5)), (x + ww, rec(x, wy + wh * 0.18, x + ww)),
                            (x + ww, rec(x, wy + wh * 0.42, x + ww)), (x, wy + wh * 0.4)]
                    wc.wash(P, top_, (26, 24, 28), strength=0.45, var=0.02, layers=3, edge=0.4, fade=NOB)
                    side = [(x, wy + wh * 0.2), (x + ww * 0.28, rec(x, wy + wh * 0.1, x + ww * 0.28)), (x + ww * 0.28, rec(x, wy + wh, x + ww * 0.28)), (x, wy + wh)]
                    wc.wash(P, side, (26, 24, 28), strength=0.35, var=0.02, layers=3, edge=0.4, fade=NOB)
                    ys = wy + wh + 1.5
                    sill = wc.stroke_mask([(x - ww * 0.18, rec(x, ys, x - ww * 0.18)), (x + ww * 1.18, rec(x, ys, x + ww * 1.18))], 1.3 * f + 0.9, 1.3 * f + 0.9, taper=False, rough=0.2)
                    P.lift((sill * NOB * S(0.3, 0.55, P.grain) * 0.45).astype(np.float32), 1.0)       # 窗台亮，但被纸纹咬碎，不是一条下划线
                    P.add((np.roll(sill, int(1.4 * f + 1.5), axis=0) * NOB).astype(np.float32), (110, 80, 68), 0.35)
                    u = random.uniform(0.2, 0.8)
                    L = random.uniform(8, 44) * f + 4
                    drip = wc.stroke_mask([(x + ww * u, wy + wh + 3), (x + ww * u + random.uniform(-1, 1), wy + wh + 3 + L)], ww * 0.3, ww * 0.06, taper=False, rough=0.3)
                    P.add((drip * S(0.3, 0.6, P.vstreak) * NOB).astype(np.float32), tuple(int(c * 0.72) for c in col), 0.35)
                isolated(_depth)
                if random.random() < 0.2 and f > 0.25:        # 旁边一扇灰绿的百叶
                    q2 = [(a + ww * 1.05, rec(a, b, a + ww * 1.05)) for a, b in q]
                    wc.wash(P, q2, (96, 114, 96), strength=0.4, var=0.03, layers=3, edge=0.4, fade=NOB)
            elif r < 0.62:
                P.lift(wc.poly_mask(q) * 0.3, 1.0)
            x += ww * random.uniform(1.9, 2.8)
    # 水线上的水渍：深绿褐，上沿被干笔扫碎
    wb = float(water_y(xb)); wa = float(water_y(xa))
    band = [(xa, wa - 34 * f0 - 4), (xb, wb - 34 * f0 - 4), (xb, wb + 1), (xa, wa + 1)]
    wc.wash(P, band, (70, 76, 60), strength=0.5, var=[0.03, 0.03, 0.01, 0.01], layers=6, edge=0.3,
            fade=(S(0.35, 0.55, 0.6 * P.vstreak + 0.4 * P.grain) * NOB).astype(np.float32))
    # 水门：一个深拱，挨着水
    if f0 > 0.2 and random.random() < 0.8:
        gx = xa + (xb - xa) * random.uniform(0.3, 0.6)
        gw, gh = 26 * f0 + 4, 70 * f0 + 8
        gy = float(water_y(gx))
        gate = [(gx, gy), (gx, gy - gh * 0.7), (gx + gw * 0.5, gy - gh), (gx + gw, gy - gh * 0.7 + gw * 0.12), (gx + gw, gy + gw * 0.12)]
        wc.wash(P, gate, (34, 36, 40), strength=0.8, var=0.015, layers=5, edge=0.5)
def _right_volume():
    # 檐口挑出来：顶边一线亮（檐口吃光），底下墙上压一条深的檐影，下沿往下淌几道
    for i, (xa, xb, T, orn, col) in enumerate(R_segs):
        f0 = (xa - VP[0]) / (W - VP[0])
        if f0 < 0.08:
            continue
        ya, yb_ = ly(xa, T, "R"), ly(xb, T, "R")
        e = 10 * f0 + 3
        lip = wc.stroke_mask([(xa + 1, ya + 1), (xb - 1, yb_ + 1)], 2.2 * f0 + 1, 2.2 * f0 + 1, taper=False, rough=0.2)
        P.lift((lip * 0.55 * NOB).astype(np.float32), 1.0)
        band = [(xa, ya + 2.5), (xb, yb_ + 2.5), (xb, yb_ + 2.5 + e), (xa, ya + 2.5 + e)]
        wc.wash(P, band, (112, 76, 66), strength=0.68, var=[0.01, 0.01, 0.05, 0.05], layers=5, edge=0.45, fade=NOB)
        for k in range(int(4 + 8 * f0)):
            dx_ = xa + (xb - xa) * random.uniform(0.05, 0.95)
            y0_ = ly(dx_, T, "R") + 2 + e
            L = random.uniform(20, 110) * f0 + 6
            m = wc.stroke_mask([(dx_, y0_ - 2), (dx_ + random.uniform(-1, 1), y0_ + L)], e * 0.5, e * 0.1, taper=False, rough=0.3)
            P.add((m * S(0.3, 0.6, P.vstreak) * NOB).astype(np.float32), (120, 84, 72), 0.45)
    # 楼不齐平：远的那栋往运河里凸出一截，转角处露出一面正对我们的山墙（和画面平行），在背光里。
    # 顶和立面屋檐在楼角那一点接上：立面的檐口线跑到楼角，拐个直角变成正面这面墙的水平上沿（她改的图）。
    # 它投到旁边立面上的影，贴着山墙直接接上，同一个暖暗色系只浅一点——墙和影是一整块暗，中间不留亮缝。
    for bnd, far_i in ((590, 2), (780, 4)):
        xa, xb, T, orn, col = R_segs[far_i]
        f = (bnd - VP[0]) / (W - VP[0])
        w = 16 * f + 5
        xc = bnd - w                                      # 楼角（凸出去的那条竖边）
        yc = ly(xc, T, "R")                               # 楼角处的檐口高度：和立面屋顶线接上
        rise = 0.0                                        # 她画的：顶是平的（正对我们的面，上沿水平），但高度正好接在立面檐口的楼角上
        bot = float(water_y(bnd))
        # 楼到楼角就拐走了：楼角右边、这面墙上沿以上，原来画着的那截立面和檐影要擦掉换回天（她：「红砖到阴影那边就已经转角了，上面不应该接到黄砖」）
        for cx_ in range(int(xc), int(bnd) + 3):
            yt = int(ly(cx_, T, "R")) - 6
            d_top = P.D[max(0, yt - 10):max(1, yt - 3), cx_].mean(axis=0)
            y_end = int(yc) + 1
            if y_end > yt:
                P.D[yt:y_end, cx_] = d_top[None, :]
        endw = [(xc, yc), (bnd + 1.5, yc - rise), (bnd + 1.5, bot), (xc, bot)]
        dark = tuple(int(0.7 * c * 0.55 + 0.3 * d) for c, d in zip(col, DEEP))      # 一遍铺：两遍各自变形会在边上多出一圈灰
        wc.wash(P, endw, dark, strength=1.0, var=[0.006, 0.006, 0.02, 0.02], layers=8, edge=0.4, fade=NOB)
        em = wc.poly_mask(endw)
        P.lift((em * S(bot - 90 * f, bot, yf) * 0.3).astype(np.float32), 1.0)                         # 贴水一截：水面反上来的光
        P.add((em * S(bot - 90 * f, bot, yf)).astype(np.float32), (170, 140, 110), 0.15)
        eave = [(xc - 1, yc + 1), (bnd + 1.5, yc - rise + 1), (bnd + 1.5, yc - rise + 1 + 9 * f + 2), (xc - 1, yc + 1 + 9 * f + 2)]
        wc.wash(P, eave, (34, 30, 34), strength=0.45, var=0.02, layers=3, edge=0.4)                   # 檐口顺着坡转过来的影
        for k in range(3):                                                                               # 正对着的墙上几个正的窗洞
            wy_ = yc + (60 + k * 125) * f + 6
            if wy_ > bot - 40 * f:
                break
            q = [(xc + w * 0.3, wy_), (xc + w * 0.7, wy_), (xc + w * 0.7, wy_ + 30 * f + 4), (xc + w * 0.3, wy_ + 30 * f + 4)]
            wc.wash(P, q, (22, 20, 26), strength=0.55, var=0.015, layers=3, edge=0.4)
        # 投到右边（近的那栋）立面上的影：影落在谁身上就是谁的颜色压暗——落在黄墙上就是暗黄，不是红砖色
        # （上一版用了红砖色、顶边还往上斜，看着像红楼一路接到了黄楼）。顶边从楼角上沿往右下斜：太阳在前上方
        near_col = R_segs[far_i + 1][4]
        sw_ = w * 1.3
        sh = [(bnd - 2, yc + 3), (bnd + sw_, yc + 3 + sw_ * 1.1), (bnd + sw_ * 0.9, bot - 4), (bnd - 2, bot)]
        shm = wc.poly_mask(sh) * S(0.25, 0.5, 0.5 * P.vstreak + 0.5 * P.grain + 0.4 * S(bnd + sw_ * 0.6, bnd, xf))
        P.add((shm * NOB).astype(np.float32), tuple(int(c * 0.6) for c in near_col), 0.55, granulate=0.2)
        P.add((shm * NOB).astype(np.float32), (70, 72, 96), 0.12)


def _left_volume():
    # 背光那边：檐口底下只有一线反光（天光和水光），窗台被水面反上来的光擦亮一点
    for xa, xb, T, _ in L_segs[:5]:
        f0 = (VP[0] - xb) / VP[0]
        ya, yb_ = ly(xa, T, "L"), ly(xb, T, "L")
        e = 9 * f0 + 3
        lip = wc.stroke_mask([(xa + 1, ya + e + 3), (xb - 1, yb_ + e + 3)], 1.8 * f0 + 0.8, 1.8 * f0 + 0.8, taper=False, rough=0.2)
        P.lift((lip * S(0.35, 0.6, P.grain) * 0.35).astype(np.float32), 1.0)
        band = [(xa, ya + 1), (xb, yb_ + 1), (xb, yb_ + 1 + e), (xa, ya + 1 + e)]
        wc.wash(P, band, (20, 22, 28), strength=0.35, var=[0.01, 0.01, 0.03, 0.03], layers=4, edge=0.3)
    # 背光面的立面朝右，吃对岸暖墙反过来的光：整面一层很淡的暖（一块块掉进去的），正对我们的山墙吃不到——两个面就分开了
    bnc = (covL * (1 - dist) * (0.5 + 0.5 * S(0.4, 0.7, wc.noise(60, 40, octaves=3)))).astype(np.float32)
    P.lift(bnc * 0.22, 1.0)                                                                     # 暗面上要暖得先擦浅一点，光往上加颜色只会更暗
    P.add(bnc, (176, 126, 88), 0.22)
    # 楼不齐平：远的那栋往运河里凸出一截，转角露出正对我们的面。上沿水平，高度卡在它自己檐口跑到楼角那一点；
    # 楼角以外、上沿以上擦回天。这面墙在背光里，比立面更深更冷（吃不到对岸的反光），贴水一截吃一点水光
    for bnd, far_i in ((150, 2), (300, 4)):
        xa, xb, T, _ = L_segs[far_i]
        f = (VP[0] - bnd) / VP[0]
        w = 22 * f + 6
        xc = bnd + w
        yc = ly(xc, T, "L")
        for cx_ in range(int(bnd) - 2, int(xc) + 1):
            yt = int(ly(cx_, T, "L")) - 6
            d_top = P.D[max(0, yt - 26):max(1, yt - 16), cx_].mean(axis=0)          # 取得高一点：屋顶那圈擦亮的边不能被复制下来
            if int(yc) + 1 > yt:
                P.D[yt:int(yc) + 1, cx_] = d_top[None, :]
        bot = float(water_y(bnd))
        endw = [(bnd - 1.5, yc), (xc, yc), (xc, bot), (bnd - 1.5, bot)]
        em = wc.poly_mask(endw)
        P.lift((em * 0.55).astype(np.float32), 1.0)                                    # 先擦掉一半，再铺它自己的冷暗：和立面差在冷暖，不是差成一根黑柱子
        wc.wash(P, endw, (62, 70, 88), strength=0.75, var=[0.006, 0.006, 0.02, 0.02], layers=8, edge=0.4)
        P.lift((em * S(bot - 80 * f, bot, yf) * 0.22).astype(np.float32), 1.0)
        P.add((em * S(bot - 80 * f, bot, yf)).astype(np.float32), (150, 130, 110), 0.12)
        wc.wash(P, [(bnd - 1.5, yc + 1), (xc + 1, yc + 1), (xc + 1, yc + 1 + 8 * f + 2), (bnd - 1.5, yc + 1 + 8 * f + 2)], (12, 14, 20),
                strength=0.4, var=0.02, layers=3, edge=0.4)                                              # 檐口转过来
        P.lift((wc.stroke_mask([(bnd - 1, yc + 0.5), (xc, yc + 0.5)], 1.6 * f + 0.8, 1.6 * f + 0.8, taper=False) * S(0.35, 0.6, P.grain) * 0.3).astype(np.float32), 1.0)
        for k in range(3):
            wy_ = yc + (70 + k * 125) * f + 6
            if wy_ > bot - 40 * f:
                break
            q = [(bnd + w * 0.3, wy_), (bnd + w * 0.7, wy_), (bnd + w * 0.7, wy_ + 30 * f + 4), (bnd + w * 0.3, wy_ + 30 * f + 4)]
            wc.wash(P, q, (10, 12, 18), strength=0.5, var=0.015, layers=3, edge=0.4)


isolated(_right_volume)
isolated(_left_volume)

# 右岸一个小阳台：一条深的石台 + 一盆红花（栏杆不画，画了像挂着一把梳子）
bxa, byy = 700, ly(700, 330, "R")
bal = [(bxa, byy), (bxa + 58, ly(bxa + 58, 330, "R")), (bxa + 58, ly(bxa + 58, 330, "R") + 7), (bxa, byy + 7)]
wc.wash(P, bal, (70, 62, 60), strength=0.6, var=0.015, layers=4, edge=0.5)
wc.wash(P, [(bxa + 2, byy + 7), (bxa + 56, ly(bxa + 56, 330, "R") + 7), (bxa + 56, ly(bxa + 56, 330, "R") + 16), (bxa + 2, byy + 14)],
        (120, 96, 84), strength=0.3, var=0.02, layers=3, edge=0.3)                                   # 台子底下一点影
for dx, dy, r_, c_ in [(14, -6, 5, (96, 110, 70)), (20, -9, 3.5, (176, 60, 52)), (28, -7, 3, (190, 72, 58)), (44, -5, 4, (96, 110, 70))]:
    wc.dab(P, bxa + dx, ly(bxa + dx, 330, "R") + dy, r_, c_, 0.9, soft=0.6)

# 左岸（背光）：窗更深、稀；贴水线一截吃水面反上来的光（暖、浮动）
random.seed(SEED + 2)
for xa, xb, T, _ in L_segs:
    f0 = (VP[0] - xb) / VP[0]
    if f0 < 0.15:
        continue
    for row in range(4):
        yn = T + 80 + row * 130
        x = xa + 8
        while x < xb - 6:
            f = (VP[0] - x) / VP[0]
            wy = ly(x, yn, "L")
            if wy > float(water_y(x)) - 40 * f:
                break
            ww, wh = (11 * f + 2) * random.uniform(0.8, 1.1), (40 * f + 5) * random.uniform(0.8, 1.05)
            if random.random() < 0.45:
                q = [(x, wy + wh * 0.18), (x + ww * 0.5, rec(x, wy, x + ww * 0.5)), (x + ww, rec(x, wy + wh * 0.18, x + ww)),
                     (x + ww, rec(x, wy + wh, x + ww)), (x, wy + wh)]
                wc.wash(P, q, (22, 26, 32), strength=random.uniform(0.35, 0.6), var=0.02, layers=3, edge=0.45)
                def _sill(x=x, wy=wy, ww=ww, wh=wh, f=f):
                    # 背光面的窗也是洞：窗台被对岸和水面反上来的光擦亮一点，洞口上沿再深一点
                    ys = wy + wh + 1.5
                    sill = wc.stroke_mask([(x - ww * 0.15, rec(x, ys, x - ww * 0.15)), (x + ww * 1.15, rec(x, ys, x + ww * 1.15))], 1.2 * f + 0.8, 1.2 * f + 0.8, taper=False, rough=0.2)
                    P.lift((sill * S(0.3, 0.55, P.grain) * 0.35).astype(np.float32), 1.0)
                    top_ = [(x, wy + wh * 0.18), (x + ww * 0.5, rec(x, wy, x + ww * 0.5)), (x + ww, rec(x, wy + wh * 0.18, x + ww)),
                            (x + ww, rec(x, wy + wh * 0.4, x + ww)), (x, wy + wh * 0.4)]
                    wc.wash(P, top_, (14, 16, 22), strength=0.35, var=0.02, layers=3, edge=0.4)
                isolated(_sill)
            x += ww * random.uniform(2.0, 3.0)
bounce = covL * S(WL - 130, WL - 10, yf) * (1 - S(WL - 6, WL + 2, yf)) * (0.5 + 0.5 * S(0.4, 0.7, wc.noise(8, 60, octaves=3)))
P.lift((bounce * 0.35 * (1 - dist)).astype(np.float32), 1.0)
P.add((bounce * (1 - dist)).astype(np.float32), (170, 140, 100), 0.18)
# 左岸楼与楼之间：一面墙比另一面深一点，不画线
for xa, xb, T, _ in L_segs[1:5]:
    f0 = (VP[0] - xa) / VP[0]
    sw_ = 8 * f0 + 3
    plane = S(xa, xa + sw_ * 0.7, xf) * (1 - S(xa + sw_ * 0.7, xa + sw_, xf)) * S(ly(xa, T, "L") + 6, ly(xa, T, "L") + 40, yf) * (yf < WL - 10)
    P.add((plane * (0.4 + 0.6 * S(0.35, 0.65, P.vstreak))).astype(np.float32), DEEP, 0.2, granulate=0.3)
# 左岸屋顶亮边：只在靠近光的那一段
im = Image.new("L", (W, H), 0)
ImageDraw.Draw(im).line([(int(x), int(y) + 2) for x, y in left[:-2]], fill=255, width=3)
rim = wc.blur(np.asarray(im, np.float32) / 255.0, 1.0) * np.clip(1.25 - np.hypot(xf - GLARE[0], yf - GLARE[1]) / 420, 0, 1)
P.lift((rim * S(0.3, 0.6, P.grain)).astype(np.float32), 0.8)

# 桥上几个小人：背光剪影，成团不排队
def tiny(x, y, h, col=(40, 40, 46)):
    wc.dab(P, x, y - h * 0.9, h * 0.08, (110, 84, 72), 1.1)
    body = [(x - h * 0.1, y - h * 0.8), (x + h * 0.1, y - h * 0.8), (x + h * 0.06, y - h * 0.35), (x - h * 0.06, y - h * 0.35)]
    wc.wash(P, body, col, strength=1.0, var=0.03, layers=3, edge=0.4)
    P.add(wc.stroke_mask([(x - h * 0.03, y - h * 0.38), (x - h * 0.04, y)], h * 0.07, h * 0.03, taper=False) * S(0.3, 0.5, P.grain), col, 0.9)
    P.add(wc.stroke_mask([(x + h * 0.03, y - h * 0.38), (x + h * 0.05, y)], h * 0.07, h * 0.03, taper=False) * S(0.3, 0.5, P.grain), col, 0.9)

for x, h in [(400, 30), (412, 29), (486, 28)]:
    yb = BY - 66 - 20 * math.sin(math.pi * (x - (xL - 8)) / (xR - xL + 16))
    tiny(x, yb, h)

# ================= 第三遍：水面 =================
# 哥（9/28 深夜）：「约瑟夫式水面往往不是把水画满，而是一两片大洗色定基调，几组深色横笔切开，再靠纸白闪光。」
# 上一版满河都是同一种噪声切出来的浅色碎片 + 到处一样的横向错位——最明显的代码痕迹。现在：
#   倒影是湿的、竖着化开的一大片（慢慢弯，不锯齿）；只有三四片水在动；其余整片安静。
axis = np.where((xf[0] > xL) & (xf[0] < xR), BY, water_y(xf[0])).astype(np.float32)
near0 = np.clip((yf - 640) / 480.0, 0, 1)
near = near0
# 水是这张的戏（她：「详的东西放在水上」），但不是一把刷子铺满：每片水纹有自己的手——(cx, cy, rx, ry, 纹的竖尺度, 横尺度, 阈值)
ZONES = [(250, 852, 300, 64, 5, 55, 0.6, "mix"),        # 船和尾浪：短、碎，亮暗都有
         (770, 1000, 260, 150, 11, 160, 0.6, "dark"),    # 右前：长、宽，暖倒影被深的横纹切开
         (160, 1070, 270, 110, 13, 120, 0.58, "light"),  # 左下角：宽的一片片天光
         (462, 712, 150, 46, 3.5, 90, 0.62, "light"),    # 桥洞底下：细、密的亮
         (675, 782, 170, 56, 6, 70, 0.62, "mix"),        # 桩子一带
         (470, 1110, 190, 60, 8, 100, 0.66, "dark")]     # 正前方：稀的深纹
# 安静的地方只留两三块：船前面偏右那片、左边中段、右边暗柱的倒影里
def _zone_map():
    z = np.zeros((H, W), np.float32)
    for cx_, cy_, rx, ry, *_ in ZONES:
        z = np.maximum(z, np.exp(-(((xf - cx_) / rx) ** 2 + ((yf - cy_) / ry) ** 2)).astype(np.float32))
    return np.clip(z * (0.6 + 0.6 * wc.noise(20, 60, octaves=2)), 0, 1).astype(np.float32)
_h = []
isolated(lambda: _h.append(_zone_map()))
ACTIVE = _h[0]
slow = (wc.noise(16, 200, octaves=2) - 0.5) * 2 * (2 + 9 * near0)            # 慢的弯：竖的东西倒下来轻轻扭一下
fast = (wc.noise(2.4, 34, octaves=3) - 0.5) * 2 * (2 + 20 * near0) * ACTIVE   # 快的碎：只在那几片
dx = slow + fast
dy = (wc.noise(5, 60, octaves=2) - 0.5) * 2 * (1 + 5 * near0) * ACTIVE
src_y = np.clip((2 * axis[None, :] - yf + dy).astype(np.int32), 0, H - 1)
src_x = np.clip((xx + dx).astype(np.int32), 0, W - 1)
Dsrc = P.D.copy()
Dref = Dsrc[src_y, src_x]
below = S(axis[None, :] + 0.5, axis[None, :] + 3, yf) * water
Dref = np.stack([wc.blur2(Dref[..., c], 9, 1.6) for c in range(3)], axis=-1)       # 湿的：竖着化开，不是一张镜子
keep = below * (0.84 - 0.2 * near) * (1 - 0.25 * S(0.3, 0.9, near) * (1 - ACTIVE))     # 近处静水里倒影淡下去，交给大洗色
P.D += Dref * keep[..., None] * 0.9
# 碎片只留在那几片动的水里（原来满河都是）：横着压扁、硬边、近处大远处碎
big = wc.noise(34, 240, octaves=2)
breaks = np.zeros((H, W), np.float32)
darks = np.zeros((H, W), np.float32)
for cx_, cy_, rx, ry, sy_, sx_, thr, kind in ZONES:
    zm = np.exp(-(((xf - cx_) / rx) ** 2 + ((yf - cy_) / ry) ** 2)) * (0.7 + 0.5 * wc.noise(max(rx, ry) * 0.3, octaves=2))
    nz = wc.noise(sy_, sx_, octaves=4, persistence=0.55)
    dens = 0.62 * nz + 0.38 * big
    t_ = thr - 0.05 * near + 0.12 * (1 - np.clip(zm, 0, 1))          # 片的中心密、往外越来越稀，不是一刀切的边
    zz = S(0.12, 0.4, zm)
    if kind in ("light", "mix"):
        breaks = np.maximum(breaks, (S(t_ - 0.012, t_ + 0.012, dens) * zz).astype(np.float32))
    if kind in ("dark", "mix"):
        dd = 0.62 * (1 - nz) + 0.38 * big                              # 深纹长在亮纹的缝里：同一片水的波谷
        td = t_ + (0.04 if kind == "mix" else 0.0)
        darks = np.maximum(darks, (S(td - 0.012, td + 0.012, dd) * zz).astype(np.float32))
breaks = (breaks * below).astype(np.float32)
darks = (darks * below * (1 - breaks)).astype(np.float32)
P.D *= (1 - breaks * 0.62)[..., None]
P.lift((breaks * 0.12).astype(np.float32), 1.0)
P.D *= (1 + darks * 0.7)[..., None]                                  # 波谷：就是那一处倒影自己的颜色压深
# 一两片大洗色：整片深的蓝绿压下去定基调，湿接湿，只在两处开一点花
wc.wet(P, (water * (0.35 + 0.65 * near)).astype(np.float32), (46, 86, 88), strength=0.3, spread=20, granulate=0.2)
wc.wet(P, (below * S(0.62, 0.8, wc.noise(70, 160, octaves=2)) * near).astype(np.float32), (40, 58, 62), strength=0.25, spread=10, bloom=0.35)
# 光道：桥洞底下一片软的亮，跟着慢的弯扭；不再是一格一格的梯子
path_w = np.exp(-(((xf + 1.3 * dx - (GLARE[0] + 20) - (yf - 640) * 0.05) / (26 + (yf - 640).clip(0) * 0.28)) ** 2))
dpath = (path_w - path) * wtone * water
P.lift((np.clip(dpath, 0, 1) * 0.5).astype(np.float32), 1.0)
P.add((np.clip(-dpath, 0, 1)).astype(np.float32), TEAL, 0.5, granulate=0.2)
path = path_w
# 光道是这张水的主角：一条亮的天光从桥洞底下一路铺到脚下——亮是一片片碎的，远处细密、近处宽大，形状各不一样，不是一格一格的梯子
def _path_glitter():
    pw = np.clip(path_w * 1.25, 0, 1) * water
    P.lift((pw * (0.28 + 0.22 * wc.noise(40, 120, octaves=2))).astype(np.float32), 1.0)
    n_far = wc.noise(2.2, 60, octaves=3, persistence=0.55)
    n_near = 0.6 * wc.noise(9, 150, octaves=4, persistence=0.55) + 0.4 * wc.noise(26, 260, octaves=2)
    n_ = (1 - near) * n_far + near * n_near
    thr = 0.56 - 0.05 * near + 0.18 * (1 - pw)                     # 光道中间亮片多，往两边越来越稀
    gl = S(thr - 0.012, thr + 0.012, n_) * S(0.08, 0.3, pw)
    P.lift((gl * (0.55 + 0.25 * near)).astype(np.float32), 1.0)
    P.add((gl * 0.6).astype(np.float32), (214, 206, 186), 0.1)
    P.add((pw * (1 - gl) * S(0.0, 0.5, near)).astype(np.float32), (40, 78, 82), 0.28, granulate=0.2)   # 亮片之间是深的水
isolated(_path_glitter)

# 横笔：一笔一笔的，聚成几组，每组长短不一；中间大片不碰
hd = wc.noise(3.5, 240, octaves=2, persistence=0.45)
def swipe(x0, y0, length, w, dry0, dry1, d=-1):
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


def _strokes():
    rs = random.Random(SEED + 60)
    # 1) 纸白闪光：光道里和桥洞底下，一小撮横的干笔（亮、碎），越近越长；不是白条
    lite = np.zeros((H, W), np.float32)
    for y0, n in ((684, 4), (706, 3), (740, 3), (800, 2), (905, 2), (1050, 2)):
        for _ in range(n):
            yy0 = y0 + rs.uniform(-14, 14)
            nr = near0[int(yy0), 0]
            cx_ = GLARE[0] + 20 + (yy0 - 640) * 0.05 + rs.uniform(-34, 34) * (0.4 + nr)
            L_ = rs.uniform(18, 70) * (0.5 + 1.5 * nr)
            lite = np.maximum(lite, swipe(cx_ + L_ / 2, yy0, L_, rs.uniform(1.2, 3.0) * (0.6 + nr), 0.42, rs.uniform(0.62, 0.85), d=-1))
    P.lift((lite * water * 0.5).astype(np.float32), 1.0)
    P.add((lite * water).astype(np.float32), (214, 206, 186), 0.1)
    # 2) 几组深色横笔切开：颜色是那一处倒影自己的颜色压深（暖墙底下是暗赭、别处是暗蓝绿），不是黑；
    #    笔是扁的、有厚度、微微弯，有的一刀硬边，有的落在还湿的水上化开一边
    groups = [((200, 838), 5, (40, 130), 1.0), ((440, 846), 3, (30, 80), 0.8), ((668, 746), 2, (18, 40), 0.6), ((770, 992), 5, (50, 170), 1.25)]
    img_D = P.D.copy()
    for (gx, gy_), n, (lmin, lmax), sc in groups:
        for k in range(n):
            y0 = gy_ + rs.uniform(-10, 10) * sc + k * rs.uniform(4, 10) * sc
            x0 = gx + rs.uniform(-70, 70) * sc
            nr = near0[int(min(H - 1, y0)), 0]
            L_ = rs.uniform(lmin, lmax)
            d_ = rs.choice([-1, 1])
            sag = rs.uniform(-3, 3) * sc
            pts = [(x0, y0), (x0 + d_ * L_ * 0.35, y0 + sag), (x0 + d_ * L_ * 0.7, y0 + sag * 0.6), (x0 + d_ * L_, y0 + rs.uniform(-2, 2))]
            xi, yi_ = int(np.clip(x0 + d_ * L_ * 0.4, 0, W - 1)), int(np.clip(y0, 0, H - 1))
            dloc = img_D[max(0, yi_ - 6):yi_ + 6, max(0, xi - 10):xi + 10].reshape(-1, 3).mean(axis=0)
            col = tuple(int(255 * c) for c in wc.PAPER * np.exp(-dloc * 1.35 - 0.12))
            w_ = rs.uniform(2.6, 6.0) * sc * (0.55 + nr)
            m = wc.brush(P, pts, w_, None, entry=0.07, dry_from=rs.uniform(0.45, 0.75))
            if rs.random() < 0.45:                                            # 落在湿的上面：一边化开
                m = np.clip(m * 0.55 + wc.blur2(m, 2.5, 1.2) * 1.1, 0, 1)
            P.add((m * water).astype(np.float32), col, rs.uniform(0.9, 1.25))
isolated(_strokes)

# 系船桩：右岸前面几根，蓝白条，顶一个小帽；倒影是扭着的
random.seed(SEED + 5)
for px_, hgt in [(628, 150), (646, 160), (700, 205)]:
    py = float(water_y(px_)) + 10
    top = py - hgt
    pw = 3.2 * (px_ - VP[0]) / 200 + 1.6
    pole = wc.stroke_mask([(px_, top), (px_ + 1, py)], pw, pw * 1.1, taper=False, rough=0.15)
    P.add(pole.astype(np.float32), (70, 64, 60), 0.7)
    for k in range(4):
        yb0 = top + 14 + k * 26
        m = pole * S(yb0, yb0 + 1, yf) * (1 - S(yb0 + 12, yb0 + 13, yf))
        P.lift((m * 0.6).astype(np.float32), 1.0)
        P.add(m.astype(np.float32), (70, 96, 150), 0.5)
    wc.dab(P, px_, top - 2, pw * 0.9, (60, 56, 56), 0.9, soft=0.5)
    # 倒影：一条扭的线，断断续续
    pts = [(px_ + 3 * math.sin(k * 1.3 + px_), py + k * 12) for k in range(10)]
    rm = wc.stroke_mask(pts, pw * 0.9, pw * 0.4, taper=False, rough=0.4) * S(0.4, 0.55, 0.6 * P.vstreak + 0.4 * P.grain)
    P.add((rm * water).astype(np.float32), (60, 70, 90), 0.5)

# 贡多拉：一条长的深弧，船头的铁梳子翘起来，船尾站着船夫，桨斜插进水里
gx0, gx1, gy = 150.0, 560.0, 808.0
hull_top = [(x, gy - 22 * ((x - (gx0 + gx1) / 2) / ((gx1 - gx0) / 2)) ** 4 - 6 * ((x - gx0) / (gx1 - gx0))) for x in np.linspace(gx0, gx1, 40)]
hull_bot = [(x, gy + 11 - 13 * ((x - (gx0 + gx1) / 2) / ((gx1 - gx0) / 2)) ** 4) for x in np.linspace(gx1 - 14, gx0 + 10, 40)]
hull = hull_top + hull_bot
covG = wc.wash(P, hull, (22, 24, 30), strength=1.1, var=0.008, layers=12, edge=0.5, granulate=0.08,
               fade=(1 - 0.75 * S(gx1 - 150, gx1 - 20, xf) * S(gy - 16, gy + 4, yf)).astype(np.float32))      # 船头下半截化进水里（哥：别画完整轮廓）
P.lift((wc.stroke_mask([(x, y + 1.5) for x, y in hull_top[6:34]], 1.6, 1.6, taper=True) * S(0.35, 0.55, P.grain)).astype(np.float32), 0.7)  # 船舷一线光
# 船头的铁（ferro）：竖起来的一片，带几齿
fx, fy = gx1 - 4, gy - 20
ferro = [(fx - 4, fy + 6), (fx - 2, fy - 20), (fx + 7, fy - 28), (fx + 9, fy - 22), (fx + 4, fy + 4)]
wc.wash(P, ferro, (26, 28, 34), strength=1.0, var=0.01, layers=5, edge=0.5)
for k in range(5):
    P.add(wc.stroke_mask([(fx - 1, fy - 16 + k * 4.5), (fx - 8, fy - 15 + k * 4.5)], 1.6, 1.0), (26, 28, 34), 0.9)
# 船中间：两个坐着的人（一小团暖色的头 + 一块身子）
for sx0 in (330, 358):
    wc.dab(P, sx0, gy - 36, 5.5, (150, 104, 84), 1.1)
    wc.wash(P, [(sx0 - 9, gy - 30), (sx0 + 9, gy - 30), (sx0 + 12, gy - 8), (sx0 - 12, gy - 8)], (62, 70, 96) if sx0 == 330 else (150, 70, 60),
            strength=0.9, var=0.03, layers=4, edge=0.4)
# 船夫：船尾站着，草帽一线亮，身子深，一条桨从手里斜插进水
kx, ky, kh = 196, gy - 18, 92
wc.dab(P, kx + 2, ky - kh * 0.92, kh * 0.06, (140, 100, 80), 1.1)
P.add(wc.stroke_mask([(kx - 12, ky - kh * 0.96), (kx + 16, ky - kh * 0.97)], 2.6, 2.6, taper=True), (190, 170, 120), 0.8)   # 草帽檐
wc.wash(P, [(kx - 9, ky - kh * 0.84), (kx + 10, ky - kh * 0.84), (kx + 6, ky - kh * 0.42), (kx - 6, ky - kh * 0.42)], (30, 32, 40),
        strength=1.0, var=0.02, layers=4, edge=0.4)
for k in range(3):   # 横条纹衫：几道淡的
    yk = ky - kh * (0.78 - k * 0.1)
    P.lift(wc.stroke_mask([(kx - 8, yk), (kx + 8, yk)], 1.6, 1.6, taper=False) * 0.35, 1.0)
P.add(wc.stroke_mask([(kx - 3, ky - kh * 0.44), (kx - 6, ky)], kh * 0.07, kh * 0.04, taper=False), (30, 32, 40), 1.0)
P.add(wc.stroke_mask([(kx + 3, ky - kh * 0.44), (kx + 7, ky)], kh * 0.07, kh * 0.04, taper=False), (30, 32, 40), 1.0)
P.add(wc.stroke_mask([(kx + 10, ky - kh * 0.7), (kx + 64, gy + 30)], 2.2, 1.6, taper=False, rough=0.1), (40, 38, 40), 0.9)   # 桨
# 贡多拉的倒影：深，竖着拉，被波纹打断
gy_i = int(gy)
sy_ = np.clip((2 * gy - yf + 0.5 * dy).astype(np.int32), 0, H - 1)
sx_ = np.clip((xx + 1.2 * dx).astype(np.int32), 0, W - 1)                      # 船的倒影跟着水一起碎
gsrc = np.maximum(covG, wc.blur(((yf < gy) & (yf > gy - 110) & (np.abs(xf - 200) < 30)).astype(np.float32) * 0, 1))
gref = covG[sy_, sx_] * (yf > gy + 2) * np.clip(1 - (yf - gy) / 70.0, 0, 1) * (1 - 0.6 * breaks)
kref = np.zeros((H, W), np.float32)                                            # 船夫的倒影：一截竖的深，碎
kref += np.exp(-(((xf - 196 - 1.4 * dx) / 7.0) ** 2)) * S(gy + 4, gy + 12, yf) * (1 - S(gy + 60, gy + 95, yf))
P.add((np.clip(gref + kref * 0.7, 0, 1) * water).astype(np.float32), (30, 34, 44), 0.8)

# ================= 景深：远处化开 =================
def _depth():
    # 约瑟夫最看重的远近：离光（街尽头）越近越远，边全化开、颜色被一层暖的空气吃浅；近处的保持干、硬、深。
    # 桥是中景的焦点，只化一点；桥上的小人也留着。
    dvp = np.hypot(xf - VP[0], (yf - VP[1]) * 1.25)
    far = np.clip(1 - dvp / 330.0, 0, 1) ** 1.1
    # 桥在近地，不在雾里（她：「桥不用化成那样，没有那么浓的雾」）：化开和空气都绕开它，最后原样贴回去；只比近处的楼浅一点
    keep_b = 1 - wc.blur(np.clip(covB * 1.5, 0, 1), 2.0)
    bm = np.clip(wc.blur(np.clip(covB * 1.5, 0, 1), 0.8) * 1.2, 0, 1).astype(np.float32)
    D0 = P.D.copy()
    m1 = (S(0.08, 0.45, far) * keep_b).astype(np.float32)
    m2 = (S(0.45, 0.85, far) * keep_b).astype(np.float32)
    D1 = np.stack([wc.blur(P.D[..., c], 3.0) for c in range(3)], axis=-1)
    D2 = np.stack([wc.blur(P.D[..., c], 7.0) for c in range(3)], axis=-1)
    P.D = P.D * (1 - m1[..., None]) + D1 * m1[..., None]
    P.D = P.D * (1 - m2[..., None]) + D2 * m2[..., None]
    haze = (S(0.15, 0.8, far) * keep_b * (1 - water * 0.5)).astype(np.float32)
    P.lift(haze * 0.32, 1.0)
    P.add(haze * (0.6 + 0.4 * wc.noise(60, octaves=2)), (222, 196, 160), 0.14)
    P.D = P.D * (1 - bm[..., None]) + D0 * bm[..., None]                    # 桥原样贴回：边是干的
    P.lift(bm * 0.15, 1.0)                                                   # 只浅一点点


isolated(_depth)

# ================= 最后：少量线 =================
# 横过运河的晾衣绳：一条下垂的细线，几件衣服（小色块）
lx0, lx1 = 180.0, 755.0                                              # 同一个进深：两头都拴在墙上
ly0, ly1 = 465.0, 445.0
line_pts = [(x, ly0 + (ly1 - ly0) * (x - lx0) / (lx1 - lx0) + 26 * (1 - ((x - (lx0 + lx1) / 2) / ((lx1 - lx0) / 2)) ** 2)) for x in np.linspace(lx0, lx1, 40)]
P.add((wc.stroke_mask(line_pts, 1.1, 0.8, taper=False, rough=0.1) * S(0.3, 0.5, P.grain)).astype(np.float32), (60, 58, 60), 0.6)
random.seed(SEED + 9)
for x, col in [(300, (236, 232, 222)), (318, (178, 70, 60)), (344, (236, 232, 222)), (560, (92, 110, 150)), (584, (220, 200, 150)), (640, (236, 232, 222))]:
    y = np.interp(x, [p[0] for p in line_pts], [p[1] for p in line_pts])
    w_, h_ = random.uniform(7, 11), random.uniform(10, 16)
    q = [(x - w_ / 2, y + 1), (x + w_ / 2, y + 1.5), (x + w_ / 2 - 1, y + h_), (x - w_ / 2 + 1, y + h_ - 1)]
    if col[0] > 230:
        P.lift(wc.poly_mask(q) * 0.7, 1.0)
        wc.wash(P, q, (200, 196, 190), strength=0.25, var=0.02, layers=3, edge=0.5)
    else:
        wc.wash(P, q, col, strength=0.7, var=0.02, layers=3, edge=0.5)
def _sky_accents():
    # 屋顶上的天线（只放近处的楼：远的早化进空气里了）
    for x0, T, side, h in [(40, 120, "L", 70), (118, 200, "L", 46), (850, 130, "R", 80), (720, 180, "R", 50)]:
        y0 = ly(x0, T, side)
        P.add(wc.stroke_mask([(x0, y0), (x0 + 1, y0 - h)], 1.6, 1.0, taper=False, rough=0.1) * S(0.3, 0.5, P.grain), (50, 48, 54), 0.8)
        for k, (yy0, ww0) in enumerate([(0.85, 16), (0.7, 11), (0.55, 7)]):
            yk = y0 - h * yy0
            P.add(wc.stroke_mask([(x0 - ww0 * 0.5, yk + 1), (x0 + ww0 * 0.5, yk - 1)], 1.2, 1.0, taper=False) * S(0.3, 0.5, P.grain), (50, 48, 54), 0.7)
    # 右边近楼挑出来一根旗杆，一面红金的小旗，被风吹得往左飘
    fx, fy = 860, ly(860, 330, "R")
    P.add(wc.stroke_mask([(fx, fy), (fx - 70, fy - 44)], 2.0, 1.4, taper=False, rough=0.1), (60, 50, 46), 0.9)
    flag = [(fx - 70, fy - 44), (fx - 104, fy - 40), (fx - 99, fy - 30), (fx - 108, fy - 22), (fx - 72, fy - 24)]
    wc.wash(P, flag, (176, 50, 40), strength=0.85, var=0.04, layers=4, edge=0.45)
    wc.dab(P, fx - 86, fy - 32, 3.5, (214, 170, 80), 0.9, soft=0.6)
    # 一小群鸟：大小不一、散的，有两只落单
    for bx_, by_, s_ in [(600, 150, 9), (626, 168, 6), (648, 142, 7), (668, 176, 5), (612, 198, 4.5), (700, 128, 6), (720, 160, 4),
                         (250, 118, 7), (282, 140, 5), (180, 210, 4), (760, 230, 3.5), (560, 250, 3.5)]:
        lift_ = random.uniform(0.25, 0.5)
        a_ = 0.5 + 0.4 * min(1.0, s_ / 8)
        P.add(wc.stroke_mask([(bx_ - s_, by_ + s_ * 0.1), (bx_ - s_ * 0.45, by_ - s_ * lift_), (bx_, by_)], 1.1 + s_ * 0.06, 0.9), (54, 54, 64), a_)
        P.add(wc.stroke_mask([(bx_, by_), (bx_ + s_ * 0.45, by_ - s_ * lift_ * 1.1), (bx_ + s_, by_ + s_ * 0.05)], 1.1 + s_ * 0.06, 0.9), (54, 54, 64), a_)


isolated(_sky_accents)
# 鸟：两三只海鸥
for bx_, by_, s in [(610, 150, 9), (640, 170, 6), (250, 120, 7)]:
    P.add(wc.stroke_mask([(bx_ - s, by_), (bx_ - s * 0.3, by_ - s * 0.35), (bx_, by_)], 1.2, 1.2), (60, 62, 70), 0.7)
    P.add(wc.stroke_mask([(bx_, by_), (bx_ + s * 0.35, by_ - s * 0.4), (bx_ + s, by_ - s * 0.1)], 1.2, 1.2), (60, 62, 70), 0.7)

grain = wc.grain_from_profile(os.path.join(SKILL_DIR, "pigment_profile.npy"), seed=SEED)
out = os.path.join(HERE, "venice.png" if SEED == 5 else f"venice_{SEED}.png")
P.render(pigment_tex=grain, tex_amount=0.1).save(out)
print("saved", out, round(time.time() - t0, 1), "s")
