"""范例：雨刚停的电车街（2026-09-28 和乌桕一起磨了十七版的那张）。

顺序就是方法：
  第一遍 只有大块（天 / 左边暖灰中间调 / 右边一整片背光的深，连成一块）→ 缩到 64px 先看亮暗
  破形   远处一座很淡的尖塔
  第二遍 暗块里冒出东西（楼缝、檐口的浅线、窗、店里的暖灯）；左楼窗、遮阳棚；电车；车；人
  第三遍 地上：树影（压扁的光斑 + 横扫干笔）、湿倒影、车灯光路、半干水迹
  最后   少量线：电线、灯杆、红灯、两团鸟；前景：从左边框伸进来的细枝和三层淡叶
跑：python example.py [seed] [values|narrow]
"""
import sys, time, random, math
import numpy as np
from PIL import Image, ImageDraw
import os
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import watercolor_lib as wc
W, H = wc.W, wc.H

t0 = time.time()
SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 3
STAGE = sys.argv[2] if len(sys.argv) > 2 else "full"          # "values" 只出三档小稿
NARROW = "narrow" in sys.argv[2:]                             # 窄街：对面楼的影爬上迎光墙
wc.set_seed(SEED)
P = wc.Paper()
yy, xx = np.mgrid[:H, :W]
yf, xf = yy.astype(np.float32), xx.astype(np.float32)
S = wc.smoothstep


def isolated(fn):
    """新加的细节自己用一段随机数，画完把两条随机数流还回去。"""
    st, rs = random.getstate(), wc.rng.bit_generator.state
    fn()
    random.setstate(st); wc.rng.bit_generator.state = rs

VP = (330, 640)
GLARE = (360, 600)            # 太阳在街尽头偏右、压得很低
SUNFOOT = (1250, 640)          # 太阳在右边楼后面、画外；影子横过马路朝左下

WARM_LIGHT = (196, 183, 163)
COOL_AIR = (146, 157, 165)
EARTH = (146, 122, 103)
DEEP = (48, 57, 64)
DEEP_WARM = (74, 60, 54)
RUST = (168, 94, 66)
GOLD = (193, 139, 61)
LAMP = (238, 206, 132)

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
            if kind == "spike":
                pts += [(x - s * 0.18, y), (x, y - s * 1.3), (x + s * 0.18, y)]
            elif kind == "chim":
                pts += [(x - s * 0.2, y), (x - s * 0.2, y - s * 0.6), (x + s * 0.2, y - s * 0.6), (x + s * 0.2, y)]
            elif kind == "gable":
                pts += [(x - s * 0.6, y), (x, y - s * 0.55), (x + s * 0.6, y)]
            elif kind == "mast":
                pts += [(x - 1.3, y), (x - 1.3, y - s * 1.6), (x + 1.3, y - s * 1.6), (x + 1.3, y)]
            elif kind == "dome":
                pts += [(x - s * 0.5, y), (x - s * 0.5, y - s * 0.4), (x - s * 0.3, y - s * 0.8), (x, y - s * 0.95),
                        (x + s * 0.3, y - s * 0.8), (x + s * 0.5, y - s * 0.4), (x + s * 0.5, y)]
        pts.append((xb, ly(xb, T, side)))
    return pts

# ---------- 湿度图：远处、光里、楼脚 = 湿；焦点 = 干 ----------
focus = np.exp(-(((xf - 420) / 150.0) ** 2 + ((yf - 700) / 110.0) ** 2))      # 电车 + 右边那撮人
glare = np.exp(-(((xf - GLARE[0]) / 170.0) ** 2 + ((yf - GLARE[1]) / 120.0) ** 2))
dist = np.clip(1 - np.hypot(xf - VP[0], (yf - VP[1]) * 1.3) / 420.0, 0, 1)    # 离消失点近 = 远处
wet = np.clip(0.1 + 0.5 * dist + 0.45 * glare - 0.9 * focus, 0, 1).astype(np.float32)

# ================= 第一遍：只有大块 =================
# 1. 天：一整片湿，暖灰压上面，光那里透白
sky = (yf < 690).astype(np.float32)
top_dark = np.clip(1 - yf / 650.0, 0, 1) ** 0.8
wc.wet(P, sky * (0.45 + 0.55 * top_dark) * (1 - 0.95 * glare), WARM_LIGHT, strength=0.55, spread=45, granulate=0.05)
wc.wet(P, sky * top_dark * (1 - glare) * (0.6 + 0.4 * wc.noise(200, 300, octaves=2)), COOL_AIR, strength=0.3, spread=60)

# 2. 左边：一整块暖灰的中间调，远处的顶化进天里
L_segs = [(-40, 60, 300, [("chim", 0.4, 30)]),
          (60, 150, 250, [("gable", 0.5, 50), ("spike", 0.9, 30)]),
          (150, 215, 330, [("chim", 0.3, 30), ("chim", 0.7, 30)]),
          (215, 262, 200, [("dome", 0.5, 70), ("mast", 0.5, 110)]),
          (262, 305, 310, []),
          (305, 334, 280, [])]
left = skyline(L_segs, "L") + [(334, 652), (320, 690), (200, 740), (60, 792), (-40, 820)]
jL = (wc.noise(12, 30, octaves=3) - 0.5) * 50
ybL = ly(xf, 800, "L")
zoneL = S(ybL - 46 + jL, ybL - 6 + jL, yf)                              # 楼脚那一带
dryL = S(0.44, 0.56, 0.6 * P.hstreak + 0.4 * P.grain)                  # 干笔扫过的碎
fadeL = ((1 - S(ybL - 6 + jL, ybL + 14 + jL, yf)) * (1 - zoneL * (1 - dryL) * 0.85)).astype(np.float32)
covL = wc.wash(P, left, WARM_LIGHT, strength=1.0, var=[0.012] * (len(left) - 5) + [0.04] * 5, layers=26, edge=0.4,
               granulate=0.2, fade=fadeL, wet_map=wet * 0.4, wet_r=5)
# 墙上的暖：一块一块掉进去的，不是从上往下的渐变
wc.wet(P, covL * S(0.52, 0.7, wc.noise(70, octaves=3)), EARTH, strength=0.3, spread=5)   # 只是颜色的冷暖变化，不是影

# 3. 右边：大、深、背光。楼 + 雨廊 + 车 + 人 一整块
R_segs = [(338, 380, 420, []),
          (380, 440, 360, [("chim", 0.5, 34)]),
          (440, 540, 250, [("spike", 0.2, 40), ("chim", 0.7, 40)]),
          (540, 640, 300, [("gable", 0.5, 70)]),
          (640, 780, 140, [("chim", 0.25, 50), ("dome", 0.7, 90)]),
          (780, 940, 200, [("spike", 0.4, 60), ("chim", 0.8, 44)])]
right = skyline(R_segs, "R") + [(940, 860), (820, 836), (700, 800), (560, 750), (440, 700), (360, 672), (338, 660)]
jR = (wc.noise(30, 18, octaves=3) - 0.5) * 50
fadeR = (1 - S(ly(xf, 860, "R") - 150 + jR, ly(xf, 860, "R") + 10 + jR, yf)).astype(np.float32)
covR = wc.wash(P, right, DEEP, strength=0.95, var=[0.012] * (len(right) - 7) + [0.04] * 7, layers=30, edge=0.5,
               granulate=0.12, fade=fadeR, wet_map=wet * 0.35, wet_r=6)
wc.wet(P, covR * S(0.45, 0.7, wc.noise(90, octaves=3)), DEEP_WARM, strength=0.55, spread=8)   # 暗里掺暖，不死黑
# 远处：按栋分档变浅（空气透视），楼顶那条边保住
for (xa, xb, T, _), amt in zip(R_segs[:3], (0.62, 0.42, 0.18)):
    blk = ((xf >= xa - 1) & (xf < xb + 1)).astype(np.float32)
    P.lift((covR * blk * amt).astype(np.float32), 1.0)
for (xa, xb, T, _), amt in zip(L_segs[-3:], (0.14, 0.3, 0.5)):
    blk = ((xf >= xa - 1) & (xf < xb + 1)).astype(np.float32)
    P.lift((covL * blk * amt).astype(np.float32), 1.0)
# 远楼上的竖向节奏：越远越细越淡，给眼睛一点东西抓，不靠糊
_rs = random.getstate(); random.seed(SEED + 31)
for side, xs_ in (("L", [248, 266, 282, 296, 307, 316, 323, 328]), ("R", [346, 356, 368, 382, 398, 416, 436])):
    for x in xs_:
        f = abs(x - VP[0]) / (VP[0] if side == "L" else W - VP[0])
        top = ly(x, 330 if side == "L" else 380, side); bot = ly(x, 790, side)
        m = wc.stroke_mask([(x, top), (x + random.uniform(-0.5, 0.5), bot)], 2.2 * f + 0.8, 1.6 * f + 0.6, taper=False, rough=0.3)
        m *= S(0.4, 0.55, 0.6 * P.vstreak + 0.4 * P.grain)
        P.add(m.astype(np.float32), (120, 124, 132), 0.35 + 0.4 * f)
random.setstate(_rs)


# 4. 路：留白为主；光路从街尽头拉到脚下
road = S(655, 700, yf)
path = np.exp(-(((xf - (VP[0] + 30) - (yf - 650) * 0.35) / (40 + (yf - 650) * 0.45).clip(20)) ** 2))
wc.wet(P, road * (1 - 0.95 * path) * (0.35 + 0.65 * np.clip((yf - 650) / 500.0, 0, 1)), COOL_AIR,
       strength=0.55, spread=14)

if STAGE == "values":
    img = P.render()
    out = os.path.join(HERE, "example_values.png")
    img.save(out)
    img.convert("L").resize((64, 82), Image.LANCZOS).resize((256, 328), Image.NEAREST).save(out.replace(".png", "_thumb.png"))
    print("values", out); sys.exit()

# ---- 破形：街尽头远处一座尖塔，很淡，只让天际线跳一下 ----
TX = 462                                  # 远处：街尽头偏右，楼后面
base_y = 560                              # 塔脚藏在近楼后面
bw = 15
s1y = 392
spire_tip = 248
tower = [(TX - bw, base_y), (TX - bw, s1y), (TX - bw + 3, s1y - 5), (TX - bw * 0.45, s1y - 22), (TX, spire_tip),
         (TX + bw * 0.45, s1y - 22), (TX + bw - 3, s1y - 5), (TX + bw, s1y), (TX + bw, base_y)]
# 远：很淡的灰，往下被街尽头的光吃掉；上边清楚一点，下边化开
covT = wc.wash(P, tower, (150, 154, 162), strength=0.6, var=[0.02] + [0.004] * 7 + [0.02], layers=16, edge=0.35,
               granulate=0.2, fade=(1 - S(470, 560, yf)).astype(np.float32))
# 钟楼口两道，比塔身深一点点
for ox in (-4, 3):
    P.add(wc.stroke_mask([(TX + ox, s1y + 12), (TX + ox, s1y + 32)], 2.4, 2.4, taper=False, rough=0.2) * 0.8, (120, 124, 134), 0.5)
P.add(wc.stroke_mask([(TX, spire_tip + 1), (TX, spire_tip - 9)], 1.0, 0.9, taper=False), (130, 134, 142), 0.6)

# ================= 第二遍：暗部里冒出东西 =================
# 背光楼：顶上吃一圈亮边（只在靠近光的那一段）
im = Image.new("L", (W, H), 0)
ImageDraw.Draw(im).line([(int(x), int(y) + 2) for x, y in right[:-7]], fill=255, width=3)
rim = wc.blur(np.asarray(im, np.float32) / 255.0, 1.0) * np.clip(1.2 - np.hypot(xf - GLARE[0], yf - GLARE[1]) / 500, 0, 1)
P.lift((rim * S(0.3, 0.6, P.grain)).astype(np.float32), 0.85)
# ---- 右边暗楼里的楼：檐口吃天光（浅线）、窗一排排（更深，偶尔一格反天光）、楼与楼之间一道缝 ----
random.seed(SEED + 1)
PRES_R = wc.noise(150, 110, octaves=2)          # 窗不到处盖印章：只有这张低频图高的一两处有窗（哥：整组消失）
dens_now = 1 - np.exp(-P.D.mean(axis=2))
for xa, xb, T, _ in R_segs:
    f0 = (xa - VP[0]) / (W - VP[0])
    if f0 < 0.08:
        continue
    # 楼与楼之间：不是一道钢笔线，是一面墙和下一面墙的深浅差——缝左边一条竖着的淡深色（一边软一边硬、上下断断续续），
    # 线本身只剩几截若有若无的。（stroke_mask 这一笔照旧调用，吃掉同样的随机数，后面的地面才不会变）
    seam = wc.stroke_mask([(xa, ly(xa, T, "R") + 4), (xa, ly(xa, 780, "R"))], 3 * f0 + 1, 2 * f0 + 1, taper=False, rough=0.3)
    sw_ = 9 * f0 + 3
    plane = (S(xa - sw_, xa - sw_ * 0.25, xf) * (1 - S(xa - 0.6, xa + 0.6, xf))
             * S(ly(xa, T, "R") + 4, ly(xa, T, "R") + 30, yf) * (1 - S(ly(xa, 740, "R"), ly(xa, 790, "R"), yf)))
    plane = plane * (0.45 + 0.55 * S(0.35, 0.65, P.vstreak))
    P.add((plane * (1 - wet * 0.8)).astype(np.float32), DEEP, 0.22, granulate=0.3)
    P.add((seam * S(0.58, 0.72, 0.7 * P.vstreak + 0.3 * P.grain) * (1 - wet * 0.8)).astype(np.float32), (24, 28, 34), 0.22)
    # 檐口和楼层线：浅，断断续续
    for k, off in enumerate((22, 150, 280, 410)):
        yn = T + off
        if ly(xa, yn, "R") > ly(xa, 760, "R"):
            continue
        m = wc.stroke_mask([(xa + 4, ly(xa + 4, yn, "R")), (xb - 4, ly(xb - 4, yn, "R"))], 3.2 * f0 + 1, 2 * f0 + 1, taper=False, rough=0.4)
        P.lift((m * S(0.45, 0.6, 0.6 * P.hstreak + 0.4 * P.grain) * (0.55 if k == 0 else 0.3)).astype(np.float32), 1.0)
    # 窗
    for row in range(5):
        if random.random() < 0.3:
            continue
        yn = T + 60 + row * 125 + random.uniform(-10, 10)
        if ly(xa, yn, "R") > ly(xa, 700, "R"):
            continue
        x = xa + 14 * f0 + 4
        while x < xb - 8:
            f = (x - VP[0]) / (W - VP[0])
            wy = ly(x, yn, "R")
            ww, wh = (12 * f + 2) * random.uniform(0.7, 1.2), (40 * f + 4) * random.uniform(0.6, 1.1)
            wy += random.uniform(-4, 4) * f
            r = random.random()
            q = [(x, wy), (x + ww, wy + ww * 0.15), (x + ww, wy + wh), (x, wy + wh)]
            pres = PRES_R[int(min(H - 1, max(0, wy))), int(min(W - 1, max(0, x)))]
            if r < 0.5 and pres < 0.4:
                pass
            elif r < 0.45:
                wc.wash(P, q, (22, 26, 32), strength=random.uniform(0.3, 0.55), var=0.03, layers=3, edge=0.45)
                def _hole(x=x, wy=wy, ww=ww, wh=wh, f=f):
                    # 背光面的窗也是洞：洞口上沿深一截，窗台被街上反上来的光擦亮一点
                    top_ = [(x, wy), (x + ww, wy + ww * 0.15), (x + ww, wy + wh * 0.35), (x, wy + wh * 0.35)]
                    wc.wash(P, top_, (14, 16, 22), strength=0.35, var=0.02, layers=3, edge=0.4)
                    sill = wc.stroke_mask([(x - ww * 0.15, wy + wh + 1.5), (x + ww * 1.15, wy + wh + 1.5 + ww * 0.15)], 1.2 * f + 0.8, 1.2 * f + 0.8, taper=False, rough=0.2)
                    P.lift((sill * S(0.3, 0.55, P.grain) * 0.35).astype(np.float32), 1.0)
                isolated(_hole)
            elif r < 0.5:
                P.lift(wc.poly_mask(q) * 0.4, 1.0)                        # 一格窗反着天光
            x += ww * random.uniform(2.0, 3.2)
# 右边一楼：店面里的暖灯，几点
for x in (480, 530, 700, 760, 860):
    y = ly(x, 800, "R") - 34 * (x - VP[0]) / (W - VP[0]) - 6
    wc.dab(P, x, y, 3 + 6 * (x - VP[0]) / (W - VP[0]), GOLD, 0.9, soft=0.9, squash=0.6)

# ---- 左边暖楼：一排排窗、檐口的影、墙面竖擦，远的化掉 ----
wallv = wc.noise(90, 3.5, octaves=3, persistence=0.55)
P.add((covL * S(0.56, 0.63, 0.65 * wallv + 0.35 * P.grain) * (1 - dist) * np.clip((yf - 300) / 300, 0, 1)).astype(np.float32),
      EARTH, 0.35)
random.seed(SEED + 2)
PRES_L = wc.noise(140, 100, octaves=2)
for xa, xb, T, _ in L_segs:
    f0 = (VP[0] - xb) / VP[0]
    for k, off in enumerate((20, 160)):
        yn = T + off
        m = wc.stroke_mask([(xa + 3, ly(xa + 3, yn, "L")), (xb - 3, ly(xb - 3, yn, "L"))], 4 * f0 + 1.2, 2 * f0 + 1, taper=False, rough=0.3)
        wc.dry(P, m * (1 - wet * 0.7), (140, 120, 106), strength=0.4, thresh=0.45, streak="h")
    for row in range(4):
        yn = T + 60 + row * 110
        if yn > 700:
            continue
        x = xa + 10
        while x < xb - 6:
            f = (VP[0] - x) / VP[0]
            if f < 0.1:
                break
            wy = ly(x, yn, "L")
            ww, wh = (9 * f + 2) * random.uniform(0.8, 1.2), (30 * f + 4) * random.uniform(0.7, 1.1)
            wy += random.uniform(-3, 3) * f
            if random.random() < 0.42 and PRES_L[int(min(H - 1, max(0, wy))), int(max(0, min(W - 1, x)))] > 0.45:
                q = [(x, wy), (x + ww, wy - ww * 0.2), (x + ww, wy + wh), (x, wy + wh)]
                dfar = float(dist[int(min(H - 1, wy)), int(max(0, min(W - 1, x)))])
                wc.wash(P, q, (86, 70, 66), strength=random.uniform(0.4, 0.75) * (1 - 0.7 * dfar),
                        var=0.02, layers=3, edge=0.5)
                if f > 0.3:
                    def _hole(x=x, wy=wy, ww=ww, wh=wh, f=f, dfar=dfar):
                        # 窗是个洞：过梁的影（上沿）和里侧更深；窗台亮一横但被纸纹咬碎；窗台下一线影、往下淌一道
                        top_ = [(x, wy), (x + ww, wy - ww * 0.2), (x + ww, wy + wh * 0.3 - ww * 0.2), (x, wy + wh * 0.3)]
                        wc.wash(P, top_, (54, 40, 38), strength=0.45 * (1 - 0.7 * dfar), var=0.02, layers=3, edge=0.4)
                        side = [(x, wy), (x + ww * 0.3, wy - ww * 0.06), (x + ww * 0.3, wy + wh), (x, wy + wh)]
                        wc.wash(P, side, (54, 40, 38), strength=0.3 * (1 - 0.7 * dfar), var=0.02, layers=3, edge=0.4)
                        sill = wc.stroke_mask([(x - ww * 0.2, wy + wh + 1.2), (x + ww * 1.2, wy + wh + 1.2 - ww * 0.22)], 1.2 * f + 0.8, 1.2 * f + 0.8, taper=False, rough=0.2)
                        P.lift((sill * S(0.3, 0.55, P.grain) * 0.4).astype(np.float32), 1.0)
                        P.add(np.roll(sill, int(1.2 * f + 1.5), axis=0).astype(np.float32), (120, 92, 78), 0.3)
                        L_ = random.uniform(6, 30) * f + 3
                        u = random.uniform(0.2, 0.8)
                        drip = wc.stroke_mask([(x + ww * u, wy + wh + 3), (x + ww * u, wy + wh + 3 + L_)], ww * 0.3, ww * 0.06, taper=False, rough=0.3)
                        P.add((drip * S(0.3, 0.6, P.vstreak)).astype(np.float32), (140, 112, 96), 0.3)
                    isolated(_hole)
            x += ww * 2.3
# 左边一楼：店面一条深，顶边硬（招牌/檐口那条线），底边被人和车咬碎；越往远越浅越简
shop = [(-40, ly(-40, 704, "L")), (326, ly(326, 704, "L")), (326, ly(326, 812, "L")), (-40, ly(-40, 812, "L"))]
depth_fade = np.clip((VP[0] - xf) / VP[0], 0, 1) ** 0.6
covS = wc.wash(P, shop, (118, 92, 76), strength=0.55, var=[0.006, 0.006, 0.03, 0.03], layers=14, edge=0.45, granulate=0.3,
               fade=(fadeL * (0.35 + 0.65 * depth_fade)).astype(np.float32))
_rs = random.getstate(); random.seed(SEED + 21)
x = -20.0
while x < 300:
    f = (VP[0] - x) / VP[0]
    wwid = (26 * f + 5) * random.uniform(0.8, 1.2)
    top_, bot_ = ly(x, 718, "L"), ly(x, 782, "L")
    r = random.random()
    q = [(x, ly(x, 718, "L")), (x + wwid, ly(x + wwid, 718, "L")), (x + wwid, ly(x + wwid, 782, "L")), (x, ly(x, 782, "L"))]
    if r < 0.5:                                           # 橱窗：亮一块，里面一点暖
        P.lift((wc.poly_mask(q) * random.uniform(0.35, 0.6) * f).astype(np.float32), 1.0)
        if random.random() < 0.5:
            wc.dab(P, x + wwid * 0.5, (top_ + bot_) / 2, wwid * 0.3, GOLD, 0.5, soft=0.9, squash=1.4)
    elif r < 0.7:                                         # 门：更深
        wc.wash(P, q, (40, 34, 32), strength=0.6 * f + 0.2, var=0.02, layers=3, edge=0.4)
    P.add((wc.stroke_mask([(x - 2, ly(x, 706, "L")), (x - 2, ly(x, 796, "L"))], 2.2 * f + 0.8, 1.8 * f + 0.6, taper=False, rough=0.3)
           * S(0.35, 0.5, P.grain)).astype(np.float32), (44, 38, 36), 0.7)
    x += wwid * random.uniform(1.4, 2.0)
random.setstate(_rs)
# 左边一楼：一面锈红的遮阳棚（整张画最暖的一块）
aw = [(96, 694), (200, 712), (212, 734), (100, 722)]
wc.wash(P, aw, RUST, strength=0.9, var=0.015, layers=8, edge=0.55)
for i in range(6):
    x0 = 100 + i * 18
    P.lift(wc.stroke_mask([(x0, 697 + i * 3.1), (x0 + 3, 722 + i * 2.2)], 3.5, 3, taper=False) * 0.45, 1.0)
lowAw = np.exp(-(((xf - 156) / 56.0) ** 2 + ((yf - 742) / 12.0) ** 2))     # 只剩遮阳棚自己底下一窄条
wc.wet(P, lowAw.astype(np.float32), DEEP_WARM, strength=0.4, spread=3)


def _volume():
    """楼的体积（威尼斯那张学的）：转折有厚度，不只是一块颜色接着另一块颜色。"""
    # 左边迎光：檐口顶边一线亮，紧贴着一条檐影，往下淌几道
    for xa, xb, T, _ in L_segs[:4]:
        f0 = (VP[0] - xb) / VP[0]
        if f0 < 0.12:
            continue
        ya, yb_ = ly(xa, T, "L"), ly(xb, T, "L")
        e = 9 * f0 + 3
        lip = wc.stroke_mask([(xa + 1, ya + 1), (xb - 1, yb_ + 1)], 2.0 * f0 + 1, 2.0 * f0 + 1, taper=False, rough=0.2)
        P.lift((lip * 0.45 * (1 - wet * 0.6)).astype(np.float32), 1.0)
        band = [(xa, ya + 2.5), (xb, yb_ + 2.5), (xb, yb_ + 2.5 + e), (xa, ya + 2.5 + e)]
        wc.wash(P, band, (112, 86, 72), strength=0.55, var=[0.01, 0.01, 0.05, 0.05], layers=5, edge=0.45,
                fade=(1 - wet * 0.6).astype(np.float32))
        for k in range(int(3 + 6 * f0)):
            dx_ = xa + (xb - xa) * random.uniform(0.05, 0.95)
            y0_ = ly(dx_, T, "L") + 2 + e
            L_ = random.uniform(16, 80) * f0 + 5
            m = wc.stroke_mask([(dx_, y0_ - 2), (dx_ + random.uniform(-1, 1), y0_ + L_)], e * 0.5, e * 0.1, taper=False, rough=0.3)
            P.add((m * S(0.3, 0.6, P.vstreak) * (1 - wet * 0.6)).astype(np.float32), (126, 100, 84), 0.4)
    # 左边：远一点那栋往街上凸出来一截，楼角露出正对我们的一面墙（太阳在前方偏右，这面墙在影里）
    #   上沿水平、高度卡在它自己檐口跑到楼角那一点；楼角以外、上沿以上擦回天；影落在近的那栋立面上 = 那面墙的颜色压暗
    for bnd, far_i in ((150, 2),):
        xa, xb, T, _ = L_segs[far_i]
        f = (VP[0] - bnd) / VP[0]
        w = 20 * f + 5
        xc = bnd + w
        yc = ly(xc, T, "L")
        for cx_ in range(int(bnd) - 1, int(xc) + 1):
            yt = int(ly(cx_, T, "L")) - 6
            d_top = P.D[max(0, yt - 26):max(1, yt - 16), cx_].mean(axis=0)
            if int(yc) + 1 > yt:
                P.D[yt:int(yc) + 1, cx_] = d_top[None, :]
        bot = ly(bnd, 800, "L") - 10
        endw = [(bnd - 1.5, yc), (xc, yc), (xc, bot), (bnd - 1.5, bot)]
        wc.wash(P, endw, (120, 100, 88), strength=0.8, var=[0.006, 0.006, 0.02, 0.02], layers=8, edge=0.4)
        wc.wash(P, [(bnd - 1.5, yc + 1), (xc + 1, yc + 1), (xc + 1, yc + 1 + 8 * f + 2), (bnd - 1.5, yc + 1 + 8 * f + 2)], (70, 56, 50),
                strength=0.4, var=0.02, layers=3, edge=0.4)                                              # 檐口转过来
        for k in range(3):                                                                               # 正对着的墙上几个正的窗洞
            wy_ = yc + (60 + k * 110) * f + 6
            if wy_ > bot - 40 * f:
                break
            q = [(bnd + w * 0.3, wy_), (bnd + w * 0.7, wy_), (bnd + w * 0.7, wy_ + 28 * f + 4), (bnd + w * 0.3, wy_ + 28 * f + 4)]
            wc.wash(P, q, (60, 46, 44), strength=0.5, var=0.015, layers=3, edge=0.4)
        sw_ = w * 1.4
        sh = [(bnd + 2, yc + 3), (bnd - sw_, yc + 3 + sw_ * 1.1), (bnd - sw_ * 0.9, bot - 4), (bnd + 2, bot)]
        shm = wc.poly_mask(sh) * S(0.25, 0.5, 0.5 * P.vstreak + 0.5 * P.grain + 0.4 * S(bnd - sw_ * 0.6, bnd, xf))
        P.add(shm.astype(np.float32), (118, 100, 88), 0.5, granulate=0.2)
    # 右边背光：立面朝左，吃对面亮墙反过来的光——整面一层很淡的暖（先擦浅再加色，往深色上直接加暖只会更暗）
    bnc = (covR * (1 - dist) * (0.5 + 0.5 * S(0.4, 0.7, wc.noise(60, 40, octaves=3))) * (1 - fadeR * 0 )).astype(np.float32)
    bnc *= (yf < ly(xf, 760, "R")).astype(np.float32)
    P.lift(bnc * 0.2, 1.0)
    P.add(bnc, (170, 128, 96), 0.2)
    # 右边檐口底下一线反光
    for xa, xb, T, _ in R_segs[2:]:
        f0 = (xa - VP[0]) / (W - VP[0])
        e = 8 * f0 + 3
        lip = wc.stroke_mask([(xa + 2, ly(xa + 2, T, "R") + e + 3), (xb - 2, ly(xb - 2, T, "R") + e + 3)], 1.8 * f0 + 0.8, 1.8 * f0 + 0.8, taper=False, rough=0.2)
        P.lift((lip * S(0.35, 0.6, P.grain) * 0.3).astype(np.float32), 1.0)
    # 右边：远的那栋（540–640）往街上凸出来，楼角露出正对我们的山墙。背光，但吃不到对面的反光 → 冷；先擦掉一半再铺冷灰
    bnd = 540
    xa, xb, T, _ = R_segs[2]
    f = (bnd - VP[0]) / (W - VP[0])
    w = 18 * f + 6
    xc = bnd - w
    yc = ly(xc, T, "R")
    bot = ly(bnd, 800, "R") - 20
    endw = [(xc, yc), (bnd + 1.5, yc), (bnd + 1.5, bot), (xc, bot)]
    em = wc.poly_mask(endw) * S(0.02, 0.3, wc.blur(covR, 2.0))
    P.lift((em * 0.3).astype(np.float32), 1.0)
    wc.wash(P, endw, (70, 80, 96), strength=0.7, var=[0.006, 0.006, 0.02, 0.02], layers=8, edge=0.4,
            fade=(1 - S(bot - 60, bot, yf)).astype(np.float32))
    wc.wash(P, [(xc - 1, yc + 1), (bnd + 1.5, yc + 1), (bnd + 1.5, yc + 1 + 8 * f + 2), (xc - 1, yc + 1 + 8 * f + 2)], (18, 22, 28),
            strength=0.45, var=0.02, layers=3, edge=0.4)
    P.lift((wc.stroke_mask([(xc, yc + 0.5), (bnd + 1, yc + 0.5)], 1.6 * f + 0.8, 1.6 * f + 0.8, taper=False) * S(0.35, 0.6, P.grain) * 0.4).astype(np.float32), 1.0)
    for k in range(3):
        wy_ = yc + (70 + k * 120) * f + 6
        if wy_ > bot - 50 * f:
            break
        q = [(xc + w * 0.3, wy_), (xc + w * 0.7, wy_), (xc + w * 0.7, wy_ + 30 * f + 4), (xc + w * 0.3, wy_ + 30 * f + 4)]
        wc.wash(P, q, (16, 18, 26), strength=0.5, var=0.015, layers=3, edge=0.4)
random.seed(SEED + 61)
isolated(_volume)


def _end_walls():
    """高低错落的楼：远的那栋比近的高，高出去的那一截是有厚度的（她 9/28 画给我看的）。
    街两边的楼，进深方向（垂直于街）和画面平行 → 在画上是水平线。所以远楼高出来的那块，露出的是它正对我们的那面端墙：
    上沿从它屋顶跑到楼角那一点起，水平地往外（右边的楼往右、左边的楼往左）走，直到撞上近楼的屋顶线；下沿就是近楼的屋顶线。
    近的比远的高时，端墙朝着消失点那边，看不见，只是一个台阶。
    端墙在背光里（太阳在前方偏右、压得很低）：左边迎光的楼，端墙比立面深一截、偏冷；右边的楼本来就暗，端墙再冷一点，顶边吃一线逆光。"""
    def seg_top(segs, side, x):
        for xa, xb, T, _ in segs:
            if xa <= x <= xb:
                return ly(x, T, side)
        return None
    sky_now = (1 - S(0.02, 0.3, wc.blur(np.clip(covR + covL, 0, 1), 1.0))).astype(np.float32)
    for segs, side in ((R_segs, "R"), (L_segs, "L")):
        for i in range(len(segs) - 1):
            a_, b_ = segs[i], segs[i + 1]                    # 右边：a 远 b 近；左边：a 近 b 远
            far, nearseg = (a_, b_) if side == "R" else (b_, a_)
            xc = a_[1]                                        # 两栋的交界
            y_top = ly(xc, far[2], side)
            y_nc = ly(xc, nearseg[2], side)
            if y_top >= y_nc - 3:                             # 近的比远的高（或差不多）：端墙朝里，看不见
                continue
            # 近楼屋顶线在哪一点升到 y_top：ly(x, T_near) = y_top
            T_n = nearseg[2]
            fx = (y_top - VP[1]) / (T_n - VP[1])
            xi = VP[0] + fx * (W - VP[0]) if side == "R" else VP[0] - fx * VP[0]
            xi = min(xi, nearseg[1]) if side == "R" else max(xi, nearseg[0])
            wall = [(xc, y_top), (xi, y_top), (xi, ly(xi, T_n, side)), (xc, y_nc + 2)]
            m = wc.poly_mask(wall)
            m = (m * np.clip(sky_now + 0.0, 0, 1)).astype(np.float32)
            f = abs(xc - VP[0]) / (VP[0] if side == "L" else W - VP[0])
            # 颜色 = 这栋楼自己立面靠楼角那一大块的中位数，再压深一点点、偏冷一点点：同一栋楼转了个面，不是另一块深色贴片
            if side == "R":
                box = P.D[int(y_top + 8):int(y_top + 8 + 90 * f + 20), int(xc - 40 * f - 8):int(xc - 3)]
            else:
                box = P.D[int(y_top + 8):int(y_top + 8 + 90 * f + 20), int(xc + 3):int(xc + 40 * f + 8)]
            dmed = np.median(box.reshape(-1, 3), axis=0)
            dt = dmed * (1.08 if side == "R" else 1.25) + np.array([0.03, 0.0, -0.04])
            mm = wc.poly_mask([(xc - (1.5 if side == "R" else -1.5), y_top), (xi, y_top), (xi, ly(xi, T_n, side) + 2), (xc - (1.5 if side == "R" else -1.5), y_nc + 2)])
            mm = np.clip(wc.blur(mm, 0.7) * (0.9 + 0.2 * wc.noise(10, octaves=2)), 0, 1) * S(0.02, 0.6, sky_now + wc.poly_mask(wall)) * (1 - 0.6 * dist)
            noise_d = (0.9 + 0.2 * wc.noise(20, 8, octaves=3))[..., None]
            P.D = P.D * (1 - mm[..., None]) + (dt[None, None, :] * noise_d) * mm[..., None]
            # 顶边：一线逆光（被纸纹咬碎）；贴着底下一窄条檐影
            top_line = wc.stroke_mask([(xc, y_top + 1), (xi, y_top + 1)], 1.6 * f + 1.0, 1.4 * f + 0.8, taper=False, rough=0.2)
            P.lift((top_line * S(0.3, 0.6, P.grain) * 0.5 * (1 - 0.5 * dist)).astype(np.float32), 1.0)
            eave = [(xc, y_top + 2.5), (xi, y_top + 2.5), (xi, y_top + 2.5 + 7 * f + 2), (xc, y_top + 2.5 + 7 * f + 2)]
            wc.wash(P, eave, (22, 24, 30) if side == "R" else (84, 70, 64), strength=0.18, var=0.02, layers=3, edge=0.4,
                    fade=wc.poly_mask(wall).astype(np.float32))
            # 端墙上一两扇正的窗（这面墙和画面平行 → 窗是正的矩形）
            if abs(xi - xc) > 24:
                for k in range(2):
                    wx = xc + (xi - xc) * (0.3 + 0.35 * k)
                    wy = y_top + (16 + 34 * k) * f + 8
                    ww, wh = 7 * f + 3, 20 * f + 6
                    q = [(wx - ww / 2, wy), (wx + ww / 2, wy), (wx + ww / 2, wy + wh), (wx - ww / 2, wy + wh)]
                    wc.wash(P, q, (18, 20, 26) if side == "R" else (70, 56, 52), strength=0.55, var=0.015, layers=3, edge=0.4,
                            fade=wc.poly_mask(wall).astype(np.float32))
random.seed(SEED + 62)
isolated(_end_walls)



LIT = np.ones((H, W), np.float32)     # 有直射光的地方。影子只能落在光里：影子里再投不出影子
if NARROW:
  # ---- 窄街才用：太阳在右边楼后面压得很低，右边一整排楼的影是【一整块】——
  #      从右楼脚铺过整条街、再爬上左墙。影能上墙，地上就一定全被盖住：地上没有分界线，树影人影都被吞掉。
  #      左墙上的影边 = 右边屋顶轮廓的投影。屋顶线平行于街道 → 影边也收向同一个消失点；一段一段台阶 + 烟囱小方凸。
  #      光只能从真的口子进来：右边楼开一条横街（口子里亮），光带斜着横过马路，树影挪进这道光里。
  _rs_np = wc.rng.bit_generator.state; _rs_py = random.getstate(); random.seed(SEED + 55)
  # 1) 墙上的影边：每一段都是一条过消失点的线（ly 的 near_y 不同），段与段之间是竖直的台阶
  xs_edge = np.arange(-40, VP[0] + 1, 1.0)
  steps = []; seg_x = -40.0
  while seg_x < VP[0]:
      seg_w = random.uniform(50, 110) * max(0.25, (VP[0] - seg_x) / VP[0])
      steps.append((seg_x, seg_x + seg_w, 470 + random.uniform(-40, 40)))
      seg_x += seg_w
  edge_y = np.array([ly(x, next((T for a, b_, T in steps if a <= x < b_), steps[-1][2]), "L") for x in xs_edge])
  for a, b_, T in steps:
      if random.random() < 0.45:
          cx = (a + b_) / 2; f = max(0.25, (VP[0] - cx) / VP[0])
          edge_y[(xs_edge > cx - 7 * f) & (xs_edge < cx + 7 * f)] -= 18 * f
  edge_map = np.interp(xf[0], xs_edge, edge_y)[None, :]
  jag = (wc.noise(3, 20, octaves=2) - 0.5) * 3                 # 只一点点毛：影边是硬的
  wall_part = covL * S(edge_map - 1.2 + jag, edge_map + 1.2 + jag, yf) * (xf < VP[0])
  # 2) 地：两边楼脚之间整条街都在影里，和墙上那块是同一块，不分开画
  yb = np.where(xf < VP[0], ybL, ly(xf, 860, "R"))
  floor_strict = S(yb - 10, yb, yf)
  floor = np.where(xf < VP[0], floor_strict, S(yb - 120, yb - 20, yf))   # 右边楼脚本来化在雾里，影从那里就接上，不留一条亮缝
  # 3) 光的口子：右边楼在 GAP 这一段断开——一条横街。站在街上看过去，口子里不是一条缝：
  #    远处那栋楼的【山墙正对着我们】（和画面平行 → 上下边都是水平的，窗是正的矩形，顶上是往里升的坡），
  #    它和临街立面在楼角那条竖线上转折——体积就在这个转折上。山墙在背光里，只有贴地一截吃横街地面反上来的暖。
  #    山墙顶上面才露天；山墙脚前面是被太阳照亮的横街地面，光就从这里铺出来。
  GAP = (585, 675)
  sx = S(GAP[0] - 0.8, GAP[0] + 0.8, xf) * (1 - S(GAP[1] - 0.8, GAP[1] + 0.8, xf))
  hole = (sx * S(0.02, 0.3, wc.blur(covR, 2.0)) * (yf < ly(xf, 862, "R"))).astype(np.float32)
  # 先把口子整个换成天：每一竖列从楼顶正上方那点天往下接，越往地平线越亮（不留旧屋顶的鬼影）
  src = P.D.copy()
  for x in range(GAP[0] - 2, GAP[1] + 3):
      col_h = hole[:, x]
      if col_h.max() < 0.05:
          continue
      yt = int(np.argmax(col_h > 0.05))
      d_top = P.D[max(0, yt - 12):max(1, yt - 4), x].mean(axis=0)
      t = np.clip((np.arange(H) - yt) / max(1.0, VP[1] - yt), 0, 1)[:, None]
      src[:, x] = d_top[None, :] * (1 - 0.55 * t)
  P.D = P.D * (1 - hole[..., None]) + src * hole[..., None]
  # 山墙：楼角在 GAP[0]，檐口高度 = 远楼临街立面在楼角处的顶；往右（往楼里）坡顶升上去
  e_top, e_bot = ly(GAP[0], 300, "R"), ly(GAP[0], 860, "R")
  gable = [(GAP[0], e_top), (GAP[1] + 3, e_top - 30), (GAP[1] + 3, e_bot), (GAP[0], e_bot)]
  wc.wash(P, gable, (58, 66, 76), strength=0.85, var=0.008, layers=12, edge=0.5, granulate=0.15)
  gm = wc.poly_mask(gable) * hole
  P.add((gm * S(e_top + 40, e_top + 6, yf) * 0.5).astype(np.float32), DEEP, 0.35)        # 坡顶底下一条檐影
  P.add((gm * S(e_bot - 90, e_bot, yf)).astype(np.float32), (160, 126, 98), 0.3)        # 贴地一截：横街地面反上来的暖
  _rs2 = random.getstate()
  for row, yy0 in enumerate((e_top + 34, e_top + 96, e_top + 158)):
      for cx0 in (GAP[0] + 16, GAP[0] + 46):
          if random.random() < 0.25:
              continue
          q = [(cx0, yy0), (cx0 + 11, yy0), (cx0 + 11, yy0 + 26), (cx0, yy0 + 26)]      # 正的矩形：这面墙和画面平行
          wc.wash(P, q, (26, 30, 36), strength=0.5, var=0.01, layers=3, edge=0.45)
  random.setstate(_rs2)
  # 楼角：不画线——立面深、山墙浅，转折就靠这两面的深浅差（压一条深在楼角上，又会变回钢笔线）
  # 横街地面：山墙脚（水平线）和主街楼脚线之间那个三角，被太阳照着
  lit_floor = wc.poly_mask([(GAP[0], e_bot + 1), (GAP[1] + 2, e_bot + 1), (GAP[1] + 2, ly(GAP[1] + 2, 862, "R"))]) * sx
  P.lift((lit_floor * 0.9).astype(np.float32), 1.0)
  P.add((lit_floor * 0.6).astype(np.float32), (214, 176, 124), 0.22)
  am = hole
  # 4) 光带：两条边过 SUNFOOT（和人影同一个放射点），只从口子往外走；离口子越远，边越软（半影）
  g1, g2 = (GAP[0], ly(GAP[0], 860, "R")), (GAP[1], ly(GAP[1], 860, "R"))
  slope = lambda px, py: (py - SUNFOOT[1]) / (SUNFOOT[0] - px)
  s_pix = (yf - SUNFOOT[1]) / np.maximum(SUNFOOT[0] - xf, 1.0)
  # 光和影的交界不是尺子划的：离口子越远越碎——边上被咬出缺口（中尺度噪声），交界带里干笔挂在纸纹上，
  # 影里留几点亮、光里落几点影。换成像素距离再加扰动，两条边各自独立
  rr = np.maximum(SUNFOOT[0] - xf, 1.0)
  d_in = np.minimum((s_pix - slope(*g1)) * rr, (slope(*g2) - s_pix) * rr)          # 离光带边的像素距离，里面为正
  far_ = np.clip((g1[0] - xf) / 600.0, 0, 1)
  bite = (wc.noise(14, 40, octaves=3) - 0.5) * (4 + 16 * far_)                     # 缺口、凸出
  dryb = (0.65 * P.hstreak + 0.35 * P.grain - 0.5) * (6 + 26 * far_)               # 干笔：沿横纹断开
  band = S(-0.8, 0.8, d_in + bite + dryb) * floor_strict
  SH = np.clip(np.maximum(wall_part, floor) - band, 0, 1)
  # 5) 大影：一层冷蓝一次铺下去；远处被空气吃掉，近处压深一点框住画面；里面横着压扁的深块，不死平
  grad = 1 - floor * (1 - S(780, 1150, yf)) * 0.25                                   # 地上的影：离光近的浅，近处深，一整片渐变
  shade = SH * (1 - 0.8 * dist) * (1 - 0.25 * path * floor) * grad
  shade *= 1 - 0.3 * S(0.55, 0.7, P.grain)
  P.add(shade.astype(np.float32), (108, 116, 142), 0.56, granulate=0.3)
  P.lift((floor * S(0.635, 0.65, wc.noise(40, 120, octaves=3)) * S(860, 980, yf) * 0.4).astype(np.float32), 1.0)  # 近处几洼水倒着天：比影子亮、硬边。深色斑块会被读成影子里又有树影
  wc.wet(P, (shade * floor * S(0.6, 0.8, wc.noise(70, 60, octaves=3))).astype(np.float32), (150, 120, 100), strength=0.12, spread=4)
  P.add((wall_part * (1 - band) * S(ly(xf, 700, "L"), ly(xf, 800, "L"), yf) * 0.6).astype(np.float32), (160, 128, 100), 0.12)  # 墙脚一点反光
  # 6) 光带里：暖一点 + 横街里的树投过来的斑驳影（叶团压扁、硬边）+ 一道树干
  P.add((band * 0.5).astype(np.float32), (206, 168, 120), 0.14)
  dens = 0.62 * wc.noise(12, 80, octaves=4, persistence=0.55) + 0.38 * wc.noise(45, 200, octaves=2)
  dap = S(0.535, 0.56, dens) * band
  gc = np.array([(g1[0] + g2[0]) / 2, (g1[1] + g2[1]) / 2]); dv = gc - np.array(SUNFOOT, float); dv /= np.linalg.norm(dv)
  trunk = wc.stroke_mask([tuple(gc + dv * 40), tuple(gc + dv * 330), tuple(gc + dv * 760)], 5, 11, taper=False, rough=0.5)
  dap = np.maximum(dap, trunk * band * S(0.4, 0.55, 0.6 * P.hstreak + 0.4 * P.grain))
  P.add((dap * (0.8 + 0.4 * wc.noise(90, 220, octaves=2))).astype(np.float32), (98, 106, 132), 0.62, granulate=0.3)
  # 7) 湿地是镜子：口子在地上倒出一条竖的、碎的亮
  col = S(GAP[0] + 30, GAP[0] + 44, xf) * (1 - S(GAP[1] - 12, GAP[1] - 2, xf)) * S(764, 780, yf) * np.clip(1 - (yf - 780) / 110.0, 0, 1) ** 1.3
  col = wc.blur2(col.astype(np.float32), 6, 3) * S(0.35, 0.6, 0.6 * P.vstreak + 0.4 * P.grain)
  P.lift((col * 0.7).astype(np.float32), 1.0)
  P.add((col * 0.5).astype(np.float32), (214, 176, 124), 0.2)
  LIT = np.clip(1 - SH, 0, 1).astype(np.float32)
  wc.rng.bit_generator.state = _rs_np; random.setstate(_rs_py)

# 电车：下沿化进湿地，只留车窗、一盏灯、顶上一丝亮
tram = [(356, 604), (424, 601), (428, 690), (352, 693)]
wet_tram = np.clip(S(655, 700, yf) * 0.9, 0, 1).astype(np.float32)
wc.wash(P, tram, (70, 82, 96), strength=1.0, var=0.01, layers=16, edge=0.5, wet_map=wet_tram, wet_r=6)
wc.wash(P, [(360, 610), (420, 608), (421, 634), (360, 636)], DEEP, strength=1.0, var=0.01, layers=8, edge=0.5)
P.lift(wc.stroke_mask([(358, 603), (422, 600)], 2.2, 2.2, taper=False) * 0.85, 1.0)
wc.dab(P, 372, 612, 3, GOLD, 1.0, soft=0.6)          # 路牌一点暖
P.add(wc.stroke_mask([(392, 602), (388, 560), (384, 530)], 1.6, 1.0, taper=False), DEEP, 0.9)
# 两盏小灯，一大一小、不在正中——一只圆圆的大亮灯在正中像「发光独眼」（哥）
for lx_, ly_, r_, a_ in ((374, 677, 4.2, 1.0), (405, 678, 3.2, 0.75)):
    d2 = (xf - lx_) ** 2 + ((yf - ly_) * 1.5) ** 2
    P.lift(np.clip(np.exp(-d2 / (2 * r_ ** 2)) * 1.8 * a_, 0, 1).astype(np.float32), 1.0)
    P.add((np.exp(-d2 / (2 * (r_ * 2.6) ** 2)) * 0.3 * a_).astype(np.float32), LAMP, 0.9)
P.lift((wc.stroke_mask([(364, 648), (416, 646)], 3.0, 2.4, taper=False) * S(0.3, 0.55, P.grain) * 0.35).astype(np.float32), 1.0)   # 车头下一截反着天光

# 车：只给一块车身 + 深车窗 + 车顶一线亮，下沿化开
def car(x, y, w, h, tone, lights=RUST):
    wc.wash(P, [(x, y + h * 0.4), (x + w * 0.2, y), (x + w * 0.8, y), (x + w, y + h * 0.4), (x + w, y + h), (x, y + h)],
            tone, strength=0.9, var=0.02, layers=8, edge=0.45, wet_map=S(y + h * 0.6, y + h, yf).astype(np.float32), wet_r=5)
    wc.wash(P, [(x + w * 0.24, y + h * 0.1), (x + w * 0.76, y + h * 0.1), (x + w * 0.86, y + h * 0.42), (x + w * 0.14, y + h * 0.42)],
            DEEP, strength=0.9, var=0.02, layers=5)
    P.lift(wc.stroke_mask([(x + w * 0.22, y + 1.5), (x + w * 0.78, y + 1.5)], max(1.6, h * 0.07), None, taper=False) * 0.9, 1.0)
    for tx in (x + w * 0.12, x + w * 0.88):
        wc.dab(P, tx, y + h * 0.6, max(2, w * 0.04), lights, 1.3)

car(470, 690, 70, 34, (86, 94, 104))
car(560, 716, 96, 46, (60, 68, 78))
car(250, 712, 70, 34, EARTH)

# 人：色块。暖色的头、一笔身子、一笔拖开的腿，融进暗里；成团但不排队
random.seed(SEED + 4)

def cast_shadow(x, y, h, stride):
    """人影：两条腿的影从两只脚出发、往外并拢成身子的影、头一小团；贴脚最深，越远越淡，边被纸纹咬碎。"""
    dd = np.array([x - SUNFOOT[0], y - SUNFOOT[1]], float); dd /= np.linalg.norm(dd)
    nn = np.array([-dd[1], dd[0]])
    L = h * random.uniform(1.0, 1.45)
    def pt(t, s):
        return (x + dd[0] * L * t + nn[0] * s, y + dd[1] * L * t + nn[1] * s)
    lw, sp = h * 0.045, h * (0.05 + stride * 0.6)
    im = Image.new("L", (W * 2, H * 2), 0); d = ImageDraw.Draw(im)
    def poly(pts, v):
        d.polygon([(a * 2, b * 2) for a, b in pts], fill=int(255 * v))
    poly([pt(0, -sp - lw), pt(0, -sp + lw), pt(0.46, lw * 0.8), pt(0.46, -lw * 1.6)], 1.0)
    poly([pt(0, sp - lw), pt(0, sp + lw), pt(0.46, lw * 1.6), pt(0.46, -lw * 0.8)], 1.0)
    poly([pt(0.44, -h * 0.07), pt(0.44, h * 0.07), pt(0.62, h * 0.11), pt(0.86, h * 0.07), pt(0.88, -h * 0.07), pt(0.62, -h * 0.11)], 1.0)
    cx, cy = pt(0.95, 0); r = h * 0.065
    d.ellipse([(cx - r) * 2, (cy - r * 0.7) * 2, (cx + r) * 2, (cy + r * 0.7) * 2], fill=255)
    m = np.asarray(im.resize((W, H), Image.BILINEAR), np.float32) / 255.0
    along = np.clip(((xf - x) * dd[0] + (yf - y) * dd[1]) / L, 0, 1)
    m = m * (1.0 - 0.55 * along)
    m = m * LIT                                                # 站在影子里的人没有影
    m = m * (1 - np.clip(m - wc.blur(m, 1.5), 0, 1) * S(0.5, 0.6, P.grain) * 1.5)
    P.add(np.clip(m, 0, 1).astype(np.float32), (84, 92, 116), 0.9, edge=0.3, edge_r=1.0)

def person(x, y, coat, walking=True, face=0, bag=False, scale=1.0):
    """h 跟着透视走：离地平线越远越大。face: -1 往左走 / 1 往右 / 0 朝我们或背对。"""
    h = 0.44 * (y - VP[1]) * scale * random.uniform(0.9, 1.08)
    lean = 0.04 * face if walking else 0.0
    hx = x + lean * h
    wc.dab(P, hx + random.uniform(-1, 1), y - h * 0.93, h * 0.058, (122, 92, 78), 1.2, squash=1.15)
    sw = random.uniform(0.1, 0.13)                                  # 肩宽别太大，不然像举手
    body = [(hx - h * sw * 0.8, y - h * 0.85), (hx - h * sw, y - h * 0.78), (hx + h * sw, y - h * 0.78), (hx + h * sw * 0.8, y - h * 0.85),
            (x + h * 0.06, y - h * 0.37), (x - h * 0.07, y - h * 0.39)]
    wc.wash(P, body, coat, strength=1.2, var=0.04, layers=4, edge=0.45)
    if bag:
        side = random.choice([-1, 1])
        wc.wash(P, [(x + side * h * 0.12, y - h * 0.5), (x + side * h * 0.2, y - h * 0.5), (x + side * h * 0.2, y - h * 0.36),
                    (x + side * h * 0.12, y - h * 0.36)], (70, 60, 56), strength=1.0, var=0.03, layers=3, edge=0.4)
    legs = np.zeros((H, W), np.float32)
    st = random.uniform(0.05, 0.12) if walking else 0.012
    for dx, k in ((-h * 0.035, -1), (h * 0.04, 1)):
        legs = np.maximum(legs, wc.stroke_mask([(x + dx, y - h * 0.4), (x + dx + k * h * st, y - random.uniform(0, h * 0.03))],
                                               h * 0.085, h * 0.028, taper=False, rough=0.2))
    if walking:
        sm = np.zeros_like(legs); sd = -face if face else random.choice([-1, 1])
        for i in range(12):
            sm = np.maximum(sm, np.roll(legs, (int(i * 1.5), int(sd * i * 1.2)), axis=(0, 1)) * (1 - i / 13) ** 1.3)
        legs = wc.blur(sm, 1.4) * S(0.35, 0.55, 0.5 * P.vstreak + 0.5 * P.grain + 0.3 * sm)
    P.add(legs, DEEP, 1.05)
    cast_shadow(x, y, h, st)

# 右边那撮：有走有停，前后错开，有一对挨着，一个落单往这边来
for x, y, coat, wk, face, bag in [
        (447, 736, (58, 60, 70), True, -1, False),        # 往左走的，远一点
        (515, 741, (54, 58, 70), False, 0, False),        # 站着等，更远
        (478, 751, (60, 64, 76), True, 1, False),         # 一对，挨着
        (491, 755, RUST, True, 1, True),
        (540, 778, (70, 78, 96), True, -1, False),        # 近一点，往左
        (604, 812, (64, 62, 72), True, -1, True)]:        # 横穿马路的，最近
    person(x, y, coat, wk, face, bag)
# 左边遮阳棚底下：两个挨着说话，一个往里走
for x, y, coat, wk, face, bag in [
        (206, 794, (66, 64, 74), False, 0, False),
        (224, 789, (90, 96, 118), False, 0, True),
        (272, 769, (70, 66, 72), True, 1, False)]:
    person(x, y, coat, wk, face, bag)
# 近处一对：一前一后
person(118, 858, (72, 70, 80), True, 1, False, 0.95)
person(160, 880, GOLD, True, 1, True, 0.95)
# 远处零星两个，化在光里
person(298, 688, (110, 110, 118), True, 1)
person(318, 694, (110, 110, 118), True, -1)

# ================= 第三遍：影子、倒影 =================
# 随机数存档：第三遍开头把两条随机数流存成文件，以后前面（楼、窗、车、人）怎么改，地上的树影、倒影一丝不变（「局部修改不许带歪别处」第 3 招）
import pickle
_ST = os.path.join(HERE, f"example_state_pass3_{SEED}.pkl")
if os.path.exists(_ST):
    _s = pickle.load(open(_ST, "rb")); wc.rng.bit_generator.state = _s["np"]; random.setstate(_s["py"])
else:
    pickle.dump({"np": wc.rng.bit_generator.state, "py": random.getstate()}, open(_ST, "wb"))
# 楼影：两三笔，从右边的暗块伸出来，朝左下；近根部硬，远了淡
hdry = wc.noise(4, 260, octaves=2, persistence=0.45)
def sdir(p):
    d = np.array([p[0] - SUNFOOT[0], p[1] - SUNFOOT[1]], float); return d / np.linalg.norm(d)

# ---- 树影：画面外右边几棵行道树，影子横着铺过马路 ----
# 1) 叶团：地面上被压扁的光斑图（横向拉长），硬边
leaf = wc.noise(18, 110, octaves=4, persistence=0.55)
lumps = wc.noise(70, 260, octaves=2)
env = np.zeros((H, W), np.float32)
for cx, cy, rx, ry in [(980, 900, 700, 95), (1050, 1080, 820, 120), (820, 800, 380, 40)]:
    env = np.maximum(env, np.exp(-(((xf - cx) / rx) ** 2 + ((yf - cy) / ry) ** 2) ** 1.6))
env *= S(770, 830, yf)
dens = 0.62 * leaf + 0.38 * lumps
thr = 0.62 - 0.32 * env                                     # 包络中心密，边缘散成光斑
canopy = S(thr - 0.012, thr + 0.012, dens) * S(0.08, 0.3, env)
# 2) 扫笔：一条条横着的干笔，长短不一，沿影子方向微斜
random.seed(SEED + 7)
sweep = np.zeros((H, W), np.float32)
hd = wc.noise(3.5, 240, octaves=2, persistence=0.45)
def swipe(x0, y0, length, w, dry0, dry1):
    d = sdir((x0, y0))
    xs = np.array([x0, x0 + d[0] * length * 0.5, x0 + d[0] * length])
    ys = np.array([y0, y0 + d[1] * length * 0.5 + random.uniform(-4, 4), y0 + d[1] * length])
    order = np.argsort(xs); xs, ys = xs[order], ys[order]
    ws = np.array([w * random.uniform(0.5, 0.9), w, w * random.uniform(0.3, 0.8)])
    xc = np.clip(xf[0], xs[0], xs[-1])
    yc = np.interp(xc, xs, ys)[None, :]
    wv = np.interp(xc, xs, ws)[None, :] * (0.75 + 0.5 * wc.noise(8, 50, octaves=3))
    tpos = (np.clip(xf, xs[0], xs[-1]) - xs[0]) / (xs[-1] - xs[0] + 1e-6)
    if d[0] < 0:
        tpos = 1 - tpos                                       # 笔从右往左扫，越往左越干
    r = np.abs(yf - yc) / wv
    inside = ((r < 1) & (xf >= xs[0]) & (xf <= xs[-1])).astype(np.float32)
    dry = dry0 + (dry1 - dry0) * tpos + 0.3 * r ** 2
    return inside * S(dry - 0.015, dry + 0.015, 0.75 * hd + 0.25 * P.grain)
for _ in range(16):
    y0 = random.uniform(800, 1160)
    x0 = random.uniform(820, 980)
    sweep = np.maximum(sweep, swipe(x0, y0, random.uniform(220, 900), random.uniform(6, 26), 0.2, random.uniform(0.45, 0.7)))
# 3) 树干：两道长条横穿马路
for y0, w in [(868, 13), (1034, 18)]:
    sweep = np.maximum(sweep, swipe(990, y0, 1200, w, 0.18, 0.5))
tree_sh = np.clip(np.maximum(canopy, sweep), 0, 1) * S(760, 800, yf)
tree_sh = tree_sh * (1 - 0.5 * path)                              # 车灯那条光里淡一点
tree_sh = tree_sh * LIT * (0.0 if NARROW else 1.0)                # 窄街：树也在楼影里，地上只剩那道光带
P.add((tree_sh * (0.8 + 0.4 * wc.noise(90, 220, octaves=2))).astype(np.float32), (98, 106, 132), 0.62, granulate=0.3)
# 影子里湿的地方深一块、暖一点
wc.wet(P, (tree_sh * S(0.55, 0.8, wc.noise(60, 150, octaves=3))).astype(np.float32), (80, 86, 112), strength=0.4, spread=2.5)
P.add((tree_sh * S(0.6, 0.8, wc.noise(80, 40, octaves=3))).astype(np.float32), (140, 112, 100), 0.22)
# 远处一小块楼影，也是扫出来的
far_sh = np.zeros((H, W), np.float32)
for _ in range(5):
    far_sh = np.maximum(far_sh, swipe(random.uniform(560, 640), random.uniform(752, 790), random.uniform(160, 320),
                                      random.uniform(4, 10), 0.25, 0.6))
P.add((far_sh * S(740, 760, yf) * LIT * (0.0 if NARROW else 1.0)).astype(np.float32), (104, 112, 136), 0.5)   # 窄街：远处那块楼影本来就在大影里

# 倒影：把暗块、车灯、人往下翻，竖着拉长、横着只糊一点、打碎
src = np.zeros((H, W), np.float32)
dark_now = 1 - np.exp(-P.D.mean(axis=2))
band = (yf > 560) & (yf < 700)
src[band] = dark_now[band]
flip = np.zeros_like(src)
axis_y = 700
for y in range(axis_y, H):
    sy = 2 * axis_y - y
    if 0 <= sy < H:
        flip[y] = src[sy]
refl = wc.blur2(flip, 16, 3) * np.clip(1 - (yf - axis_y) / 260.0, 0, 1) ** 1.4
refl *= 0.55 + 0.45 * S(0.3, 0.7, 0.5 * P.vstreak + 0.5 * wc.noise(40, 10, octaves=3))
P.add((refl * (1 - 0.8 * path)).astype(np.float32), (70, 74, 88), 0.4)
# 车头灯的光在湿地上：一条宽的、碎的暖光通道
lp = wc.blur2(wc.stroke_mask([(390, 700), (400, 820), (420, 980)], 30, 60, rough=0.4), 20, 5)
P.lift((lp * S(0.35, 0.6, 0.6 * P.vstreak + 0.4 * P.grain)).astype(np.float32), 0.9)
P.add((lp * 0.5).astype(np.float32), LAMP, 0.35)
for tx, ty in [(478, 711), (632, 744), (565, 744), (258, 732)]:
    r = wc.blur2(wc.stroke_mask([(tx, ty + 8), (tx, ty + 70)], 6, 3), 8, 1.5)
    P.add((r * S(0.3, 0.6, P.vstreak)).astype(np.float32), RUST, 0.6)

# 路面的水渍：半干时留下的——边硬、边上一圈沉积、里面比周围浅或深
wn = wc.noise(170, 240, octaves=4, persistence=0.5)
fg = S(780, 920, yf) * (1 - 0.6 * path)
half = S(0.4, 0.6, wc.noise(60, octaves=3))                 # 沉积边只在一部分边上
for lo, hi, amt in [(0.62, 0.635, 0.3), (0.72, 0.735, 0.25)]:
    m = (S(lo, hi, wn) * fg).astype(np.float32)
    ringm = wc.blur(np.clip(m - wc.blur(m, 2.5), 0, 1) * 2.0, 0.7) * half
    P.lift((m * amt * (0.45 if NARROW else 1.0)).astype(np.float32), 1.0)   # 窄街：在影里提白太跳，像云
    P.add(ringm.astype(np.float32), (96, 104, 124), 0.28)

# ================= 最后：少量线 =================
# 电线：三根，粗细渐变、断、越远越淡，穿过暗块时自己没了
dark_mask = S(0.45, 0.7, 1 - np.exp(-P.D.mean(axis=2)))
# 电线：原来五根从画框边扎向同一个点，像放射状辅助线（哥）。现在：一根横过马路的跨线，在两根灯杆之间松松地垂着；
# 顺着街的只画近处一段，在两个挂点之间各自往下垂，往里走就断了、没了——不再往一个点收
def sag(p0, p1, dip, n=24):
    return [(p0[0] + (p1[0] - p0[0]) * t, p0[1] + (p1[1] - p0[1]) * t + dip * 4 * t * (1 - t)) for t in np.linspace(0, 1, n)]
wires = [(sag((-20, 196), (212, 334), 16), 2.0, 1.4),          # 顺街：画框边 → 近灯杆
         (sag((212, 334), (292, 438), 6), 1.4, 0.7),           # 再往里一段，淡了
         (sag((212, 331), (609, 378), 30), 1.5, 1.0),          # 跨线：两根灯杆之间横过马路
         (sag((940, 262), (609, 380), 22), 1.8, 1.2)]          # 右边：画框边 → 右灯杆
for pts, w0, w1 in wires:
    m = wc.stroke_mask(pts, w0, w1, taper=False, rough=0.4)
    far = np.clip(1 - np.hypot(xf - VP[0], yf - 520) / 500.0, 0, 1)
    m = m * (1 - 0.8 * far) * (1 - dark_mask) * S(0.35, 0.45, wc.noise(30, octaves=2) * 0.6 + 0.4 * P.grain)
    P.add(m.astype(np.float32), (58, 62, 68), 1.0)
# 灯杆两根：上半截硬，下半截化进人群
# 灯杆：不再是三根一样的纯黑直尺（哥：太完整、太直、太黑）。近的那根画完整、落地（她：近的明确的东西要完整），
# 但颜色是暖的深灰、手画的微弯、上细下粗；中间那根走到暗楼前面就和楼同值、自己没了；远的那根进了雾
for x, yb, yt, arm, tone, stg, lost_dark in [(214, 770, 330, 80, (80, 74, 74), 0.95, 0.0), (610, 740, 380, -64, (52, 58, 66), 1.0, 0.85),
                                             (455, 690, 540, -34, (110, 114, 122), 0.7, 0.0)]:
    pts = [(x, yb), (x + 1.5, (yb + yt) / 2 + 20), (x + 0.5, yt + 30), (x + 2, yt)]
    m = wc.stroke_mask(pts, 4.6, 2.2, taper=False, rough=0.2)
    m *= np.clip((yb - 8 - yf) / 60.0, 0, 1) ** 0.7
    m *= 1 - lost_dark * dark_mask
    m *= 0.8 + 0.2 * S(0.3, 0.6, P.vstreak)
    P.add(m.astype(np.float32), tone, stg)
    P.add(wc.stroke_mask([(x + 2, yt), (x + arm * 0.45, yt - 16), (x + arm, yt - 8)], 2.4, 1.3, taper=False), tone, stg * 0.95)
    wc.dab(P, x + arm, yt - 6, 5.5, tone, stg, squash=0.45)
# 红绿灯：两盏，一点红
for x, y in [(300, 628), (446, 622), (318, 640)]:
    P.add(wc.stroke_mask([(x, y + 8), (x, y + 50)], 2.0, 2.0, taper=False) * (1 - dark_mask), DEEP, 0.9)
    wc.dab(P, x, y, 3.4, RUST, 1.6)
    wc.dab(P, x, y, 1.6, (230, 80, 50), 1.2)
# 鸟：两小团
random.seed(SEED + 5)
for cx, cy, n in [(170, 170, 4), (520, 110, 3)]:
    for _ in range(n):
        bx, by = cx + random.gauss(0, 30), cy + random.gauss(0, 18)
        s = random.uniform(3, 6)
        P.add(wc.stroke_mask([(bx - s, by - s * 0.4), (bx, by), (bx + s, by - s * 0.5)], 1.6, 1.0), DEEP, 1.0)
wc.splatter(P, 12, (380, 700, 700, 820), DEEP, 0.6, 1.8, 1.1)

# ================= 前景：从左边框伸进来的细枝 + 三层淡叶，包住左上角；点缀，不抢眼 =================
random.seed(SEED + 9)
AA = 2
OLIVE = (122, 126, 102)
BLUEG = (108, 120, 132)
OCHRE = (172, 140, 92)

def mask_from(im, sigma=0.0):
    m = np.asarray(im.resize((W, H), Image.BILINEAR), np.float32) / 255.0
    return wc.blur(m, sigma) if sigma else m

# 1) 枝：手定几根从左边框伸进来的细枝，再让小枝自己长；越高越平
segs = []          # (pts, width)
tips = []
def curve(p0, p1, bend, n=10):
    (x0, y0), (x1, y1) = p0, p1
    mx, my = (x0 + x1) / 2 + bend[0], (y0 + y1) / 2 + bend[1]
    out = []
    for k in range(n + 1):
        u = k / n
        x = (1 - u) ** 2 * x0 + 2 * (1 - u) * u * mx + u * u * x1
        y = (1 - u) ** 2 * y0 + 2 * (1 - u) * u * my + u * u * y1
        out.append((x + random.uniform(-1.5, 1.5), y + random.uniform(-1.5, 1.5)))
    return out

def twig(x, y, ang, L, w, depth):
    pts = [(x, y)]
    for k in range(4):
        ang += random.uniform(-0.18, 0.18)
        x += math.cos(ang) * L / 4; y += math.sin(ang) * L / 4
        pts.append((x, y))
    segs.append((pts, w))
    if depth >= 3 or w < 0.9:
        tips.append((x, y)); return
    if depth >= 1:
        tips.append((x, y))
    height = max(0.0, min(1.0, (520 - y) / 520.0))
    for side in (-1, 1):
        if random.random() < 0.7:
            a2 = ang + side * random.uniform(0.3, 0.75)
            target = -0.2 if math.cos(a2) >= 0 else -2.95          # 高处往水平拐
            a2 = a2 * (1 - 0.5 * height) + target * 0.5 * height
            twig(x, y, a2, L * random.uniform(0.6, 0.8), w * 0.62, depth + 1)

# 没有大树干（她：树干拆成细枝）；从左边框伸进来，不从地上长
stems = [
    (curve((-12, 700), (50, 470), (-6, 0)) + curve((50, 470), (140, 270), (-16, 10))[1:] + curve((140, 270), (310, 170), (0, -14))[1:], 4.2),
    (curve((-12, 560), (30, 380), (4, 0)) + curve((30, 380), (26, 120), (14, 0))[1:], 3.4),
    (curve((-12, 470), (110, 340), (0, 10)) + curve((110, 340), (250, 300), (0, -8))[1:], 3.0),
    (curve((-12, 780), (60, 640), (0, 10)) + curve((60, 640), (180, 520), (-6, 16))[1:], 2.6),
    # 往左上角去的几根，把角包住
    (curve((-12, 420), (24, 220), (6, 0)) + curve((24, 220), (70, -20), (-10, 0))[1:], 3.0),
    (curve((-12, 330), (90, 150), (-8, 12)) + curve((90, 150), (200, -20), (-6, 0))[1:], 2.6),
    (curve((-12, 250), (60, 90), (0, 8)) + curve((60, 90), (-10, -20), (14, 0))[1:], 2.2),
]
for pts, w in stems:
    n = len(pts)
    for j, (a, b) in enumerate([(0, n // 3), (n // 3, 2 * n // 3), (2 * n // 3, n)]):
        segs.append((pts[a:b + 1], w * (1 - 0.28 * j)))
    for k in range(n // 3, n, 3):
        x, y = pts[k]
        dx, dy = pts[min(n - 1, k + 1)][0] - x, pts[min(n - 1, k + 1)][1] - y
        base = math.atan2(dy, dx)
        for side in (-1, 1):
            if random.random() < 0.55:
                twig(x, y, base + side * random.uniform(0.4, 1.0) - 0.2, random.uniform(40, 90), w * 0.4, 1)
    tips.append(pts[-1])

# 2) 叶团：挂在枝梢；远层淡蓝灰、中层灰绿、近层（贴着我们）深而少
def blob(cx, cy, r, k=15):
    pts = [(cx + r * math.cos(2 * math.pi * i / k) * random.uniform(0.55, 1.3),
            cy + r * 0.75 * math.sin(2 * math.pi * i / k) * random.uniform(0.55, 1.3)) for i in range(k)]
    return pts

def wash_many(polys, layers=9, var=0.06):
    acc = np.zeros((H, W), np.float32)
    for li in range(layers):
        im = Image.new("L", (W, H), 0); d = ImageDraw.Draw(im)
        for poly in polys:
            p, _ = wc.deform(poly, [var] * len(poly), 3)
            d.polygon([(int(a), int(b)) for a, b in p], fill=255)
        acc += np.asarray(im, np.float32) / 255.0
    return np.clip(acc / layers * 1.3, 0, 1)

far_polys, mid_polys, near_polys = [], [], []
TREE_KEEP = random.Random(SEED + 90)                              # 叶团减三成（哥：树像另一套素材盖在画上）
for _ in range(14):                                               # 左上角：包住那个角
    cx, cy = random.uniform(-40, 230), random.uniform(-40, 200)
    if TREE_KEEP.random() < 0.15:
        continue
    if cx / 230 + cy / 200 > 1.25:
        continue
    far_polys.append(blob(cx, cy, random.uniform(30, 55)))
    if random.random() < 0.6:
        mid_polys.append(blob(cx + random.gauss(0, 15), cy + random.gauss(0, 12), random.uniform(12, 24)))
    if random.random() < 0.25:
        near_polys.append(blob(cx + random.gauss(0, 10), cy + random.gauss(0, 10), random.uniform(8, 14)))
for tx, ty in tips:
    if ty > 500 or tx > 500:
        continue
    if TREE_KEEP.random() < 0.15:
        continue
    if random.random() < 0.55:
        far_polys.append(blob(tx + random.gauss(0, 30), ty + random.gauss(-10, 22), random.uniform(22, 42)))
    if random.random() < 0.45:
        mid_polys.append(blob(tx + random.gauss(0, 18), ty + random.gauss(0, 14), random.uniform(10, 22)))
    if random.random() < 0.12:
        near_polys.append(blob(tx + random.gauss(0, 12), ty + random.gauss(8, 10), random.uniform(6, 12)))

far = wash_many(far_polys, var=0.08)
far *= S(0.3, 0.55, 0.6 * wc.noise(22, octaves=3) + 0.4 * wc.noise(60, octaves=2))       # 边碎、漏天
lostT = (0.7 * S(0.5, 0.72, wc.noise(80, octaves=2))).astype(np.float32)                       # 树冠有几处轮廓化进天里：和天用同一种湿
far = (far * (1 - lostT) + wc.blur(far, 8) * lostT * 0.75).astype(np.float32)
P.add(far, BLUEG, 0.2, edge=0.4, edge_r=2.0, granulate=0.3)
wc.wet(P, (far * S(0.55, 0.75, wc.noise(50, octaves=3))).astype(np.float32), OCHRE, strength=0.25, spread=3, bloom=0.5)   # 湿接湿掉进一点赭石

mid = wash_many(mid_polys, var=0.07)
mid *= S(0.35, 0.55, 0.55 * wc.noise(14, octaves=3) + 0.45 * P.grain)
mid = (mid * (1 - 0.6 * lostT) + wc.blur(mid, 4) * lostT * 0.5).astype(np.float32)
P.add(mid, OLIVE, 0.3, edge=0.5, edge_r=1.5, granulate=0.35)

near = wash_many(near_polys, var=0.06)
P.add(near, (96, 102, 96), 0.4, edge=0.55, edge_r=1.2)

# 3) 叶点：团的边上甩出去的小叶，硬边
flick = Image.new("L", (W * AA, H * AA), 0); df = ImageDraw.Draw(flick)
for tx, ty in tips:
    if ty > 520 or tx > 520:
        continue
    for _ in range(random.randint(1, 4)):                        # 甩出去的小叶少一点：多了像撒的纸屑
        cx, cy = tx + random.gauss(0, 34), ty + random.gauss(0, 26)
        L = random.uniform(4, 9); ang = random.uniform(0, 6.28)
        pts = []
        for i in range(7):
            u = i / 6; pts.append((u * L, L * 0.32 * math.sin(math.pi * u)))
        pts += [(x_, -y_) for x_, y_ in pts[::-1]]
        c, s = math.cos(ang), math.sin(ang)
        df.polygon([((cx + x_ * c - y_ * s) * AA, (cy + x_ * s + y_ * c) * AA) for x_, y_ in pts],
                   fill=int(255 * random.uniform(0.5, 1.0)))
P.add(mask_from(flick), (104, 110, 104), 0.45, edge=0.4, edge_r=1.0)

# 4) 枝：最后上，淡；一部分藏进叶子里
bm = Image.new("L", (W * AA, H * AA), 0); db = ImageDraw.Draw(bm)
for pts, w in segs:
    db.line([(x_ * AA, y_ * AA) for x_, y_ in pts], fill=255, width=max(1, int(w * AA)), joint="curve")
bmask = mask_from(bm, 0.4)
leafcov = np.clip(mid + near, 0, 1)
bmask = bmask * (1 - 0.6 * leafcov)
bmask = bmask * (0.55 + 0.45 * S(0.3, 0.6, wc.noise(46, octaves=2))) * (1 - 0.4 * lostT)       # 枝也有几截没了，和叶一样化进天
trunk_tex = S(0.3, 0.5, 0.6 * P.vstreak + 0.4 * P.grain)
P.add((bmask * (0.65 + 0.35 * trunk_tex)).astype(np.float32), (104, 100, 96), 0.55, edge=0.3, edge_r=1.2)

grain = wc.grain_from_profile(os.path.join(HERE, "pigment_profile.npy"), seed=SEED)
img = P.render(pigment_tex=grain, tex_amount=0.1)
out = os.path.join(HERE, ("example" + ("_narrow" if NARROW else "")) + (".png" if SEED == 3 else f"_{SEED}.png"))
img.save(out)
img.convert("L").resize((64, 82), Image.LANCZOS).resize((256, 328), Image.NEAREST).save(out.replace(".png", "_thumb.png"))
print("saved", out, round(time.time() - t0, 1), "s")
