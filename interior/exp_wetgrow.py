"""茶馆一角（10/1 和乌桕磨的第二张）：老茶馆下午，有人刚起身走了，竹椅斜着没推回去，盖碗盖子歪着，茶还在冒气，
窗里的光正好落在那张桌子上。

她给的方法（看步骤图）：
  - 景要小。只画一个角落：一扇窗、一张桌、一把椅子、那缕热气。力气全花在窗下那一小块。
  - 先把白留出来（窗、桌面、椅面、地上的光和窗的倒影、梁的底面），再一大遍暗湿着铺进去把它们围住，颜色在暗里互相渗。
  - 每个面一个明暗，折角是硬的；块的两个面明暗不一样。
  - 家具不是一个个盒子（c1 的错：桌子是黑方块、椅子是纸板）——是几笔：桌面留白、前沿一道深、腿是几刀往下收的书法笔。
  - 前景两团深的切进来压住画框。人不重要——这张没有人，只有人刚走的痕迹。
跑：python corner.py [seed] [values] [tag]
"""
import sys, os, math, random
import numpy as np
from PIL import Image
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))          # skill 根目录（装在 ~/.claude 或 ~/.codex 都行）
import watercolor_lib as wc
from scene import *

SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 5
STAGE = sys.argv[2] if len(sys.argv) > 2 else "full"
TAG = sys.argv[3] if len(sys.argv) > 3 else ""
wc.set_size(W, H); wc.set_seed(SEED); random.seed(SEED)
P = wc.Paper()
yy, xx = np.mgrid[:H, :W]; yf, xf = yy.astype(np.float32), xx.astype(np.float32)
Ss = wc.smoothstep
T, K, D = depth_map()
floorm = (K == 1).astype(np.float32); bwm = (K == 2).astype(np.float32); rwm = (K == 3).astype(np.float32); ceilm = (K == 4).astype(np.float32)

def M(pts3):
    p = clip_poly(pts3)
    return wc.poly_mask(p) if len(p) >= 3 else np.zeros((H, W), np.float32)

def hullM(pts3):
    h = hull2([pr(p) for p in pts3 if cam(p)[2] > NEAR])
    return wc.poly_mask(h) if len(h) >= 3 else np.zeros((H, W), np.float32)

def isolated(fn):
    st, rs = random.getstate(), wc.rng.bit_generator.state
    fn(); random.setstate(st); wc.rng.bit_generator.state = rs

def px(m, p):
    return m * F / max(depth_of(p), 0.3)

def post(a3, b3, w_m, color, strength=1.0, dry=0.3, rough=0.12):
    """直的东西（桌腿、椅腿、窗棂）：一笔粗细几乎不变，边有点抖，尾巴（贴地/贴框那头）干掉挂纸纹。"""
    a, b = pr(a3), pr(b3)
    w = max(1.4, px(w_m, ((a3[0] + b3[0]) / 2, (a3[1] + b3[1]) / 2, (a3[2] + b3[2]) / 2)))
    m = wc.stroke_mask([a, b], w, w * 0.85, taper=False, rough=rough)
    L = math.hypot(b[0] - a[0], b[1] - a[1]) + 1e-6
    tpos = np.clip(((xf - a[0]) * (b[0] - a[0]) + (yf - a[1]) * (b[1] - a[1])) / (L * L), 0, 1)
    thr = np.clip((tpos - (1 - dry)) / max(dry, 1e-3), 0, 1) * 0.7
    m = m * Ss(thr - 0.06, thr + 0.06, P.grain)
    if color is None:
        return m.astype(np.float32)
    P.add(m.astype(np.float32), color, strength)
    return m

def bstroke(a3, b3, w_m, color, strength=1.0, dry_from=0.7, bend=0.0, entry=0.1):
    """一笔书法：两头从 3D 投下来（透视对），宽度按深度给，起笔快粗、收笔尖、尾巴干。bend 让它微弯（手不是尺子）。"""
    a, b = pr(a3), pr(b3)
    mid = ((a[0] + b[0]) / 2 + bend * (b[1] - a[1]) * 0.08, (a[1] + b[1]) / 2 - bend * (b[0] - a[0]) * 0.08)
    w = px(w_m, ((a3[0] + b3[0]) / 2, (a3[1] + b3[1]) / 2, (a3[2] + b3[2]) / 2))
    return wc.brush(P, [a, mid, b], max(1.5, w), color=color, strength=strength, entry=entry, dry_from=dry_from)

# ---------- 颜色 ----------
UMBER = (98, 72, 58)
VIOLET = (108, 66, 118)
INDIGO = (58, 70, 122)
SIENNA = (182, 102, 52)
GOLD = (238, 210, 156)
DEEP = (40, 30, 30)
BAMBOO = (150, 120, 76)

# ---------- 东西摆在哪 ----------
TAB = dict(cx=2.5, cz=3.4, rot=4, s=0.92)
CHAIR = dict(cx=1.72, cz=2.62, rot=28)
GAIWAN = (2.25, 0.8, 3.15)
FLASK = (2.8, 0.8, 3.6)
FG_R = dict(cx=3.25, cz=1.3, rot=18)
FG_L = dict(cx=0.25, cz=1.45, rot=-30)

def box_c(cx, cz, w, d, y, rot):
    return box(cx, cz, w, d, y, y, rot)[:4]       # 一圈角，(−,−)(+,−)(+,+)(−,+)

def chair_geo(cx, cz, rot):
    seat = box(cx, cz, 0.44, 0.44, 0.38, 0.44, rot)
    c, s = math.cos(math.radians(rot)), math.sin(math.radians(rot))
    bk = (cx + 0.2 * s, cz - 0.2 * c)
    back = box(bk[0], bk[1], 0.44, 0.05, 0.44, 0.88, rot)
    legs = [(x + (cx - x) * 0.1, z + (cz - z) * 0.1) for (x, y, z) in seat[:4]]
    return seat, back, legs

TOPC = box_c(TAB["cx"], TAB["cz"], TAB["s"], TAB["s"], 0.8, TAB["rot"])
LEGS_T = [(x + (TAB["cx"] - x) * 0.09, z + (TAB["cz"] - z) * 0.09) for (x, y, z) in TOPC]
SEAT, BACK, LEGS_C = chair_geo(**CHAIR)

# ---------- 光 ----------
WINM = M([(WIN[0], WIN[2], BZ), (WIN[0], WIN[3], BZ), (WIN[1], WIN[3], BZ), (WIN[1], WIN[2], BZ)])
occ = [box(TAB["cx"], TAB["cz"], TAB["s"], TAB["s"], 0.62, 0.8, TAB["rot"])] + [box(x, z, 0.07, 0.07, 0, 0.66) for x, z in LEGS_T] + [SEAT, BACK] + [box(x, z, 0.04, 0.04, 0, 0.4) for x, z in LEGS_C]
OCC = np.zeros((H, W), np.float32)
for o in occ:
    OCC = np.maximum(OCC, M(shadow_poly(o)))
FL = light_on_plane(0.0, D) * floorm * (1 - wc.blur(OCC, 0.8))
TOPM = wc.poly_mask(clip_poly(TOPC))
TL = light_on_plane(0.8, D) * TOPM
SEATM = wc.poly_mask(clip_poly(SEAT[4:]))
SL = light_on_plane(0.44, D) * SEATM
bite = Ss(0.42, 0.58, 0.55 * wc.noise(6, 20, octaves=3) + 0.45 * P.grain)
band = np.clip(wc.blur(FL, 3) * (1 - wc.blur(FL, 3)) * 4, 0, 1)
FLb = np.clip(FL * (1 - 0.6 * band * (1 - bite)) + 0.3 * band * bite * wc.blur(FL, 4), 0, 1)
# 老石板地反窗：一条竖的冷亮，竖着拉长、横着断
wr = M([(WIN[0] + 0.05, -WIN[2], BZ), (WIN[0] + 0.05, -WIN[3] * 1.5, BZ), (WIN[1] - 0.05, -WIN[3] * 1.5, BZ), (WIN[1] - 0.05, -WIN[2], BZ)]) * floorm
wr = wc.blur2(wr, 26, 7) * (0.45 + 0.8 * wc.noise(50, 30, octaves=3))
wrpos = pr(((WIN[0] + WIN[1]) / 2, 0, BZ))
wr *= np.exp(-np.clip(yf - wrpos[1], 0, None) / 330.0)

def wet_keep(cov, color, strength, spread, bloom=0.0, keep=None):
    """wc.wet 会把覆盖图整个糊开——糊完再乘留白，不然颜色渗进留白里（窗影、倒影被盖掉）。"""
    c = wc.blur(cov, spread) * (0.55 + 0.9 * wc.noise(90, octaves=3))
    if bloom:
        b = wc.noise(55, octaves=4)
        ring = np.exp(-((b - 0.62) / 0.035) ** 2) * 0.9
        c = c * (1 - bloom * Ss(0.62, 0.75, b)) + bloom * ring * c
    if keep is not None:
        c = c * keep
    P.add(np.clip(c, 0, 1.4).astype(np.float32), color, strength, granulate=0.1)

# ================= 第一步：白留出来 =================
R = np.clip(np.maximum.reduce([WINM, FLb * 0.95, TL * 0.95, SL * 0.9, np.clip(wr * 0.9, 0, 0.7)]), 0, 1)

# ================= 第二步：一大遍暗，湿着铺进去，把白围住 =================
wc_ = pr(((WIN[0] + WIN[1]) / 2, (WIN[2] + WIN[3]) / 2, BZ))
glow = np.exp(-(((xf - wc_[0]) / 380.0) ** 2 + ((yf - wc_[1]) / 430.0) ** 2))      # 窗边被光烧成一大片（哥的示意图）
edge_dark = np.clip(np.hypot((xf - 330) / 560, (yf - 600) / 620), 0, 1.4)
hard = (1 - wc.blur(R, 0.7)).astype(np.float32)          # 暗贴着白的边停住：边硬
cover = np.clip(0.55 + 0.6 * edge_dark - 0.75 * glow, 0.08, 1.2) * (1 - 0.55 * ceilm)
# 暗不是一张平滑的渐变（那是数码的灰）：每个面一大笔 wash，自己带斑驳、带边上那圈沉积；面和面交界就是两笔的边
PLN = {"back": [(LX, 0, BZ), (LX, CY, BZ), (RX, CY, BZ), (RX, 0, BZ)],
       "right": [(RX, 0, BZ), (RX, CY, BZ), (RX, CY, 0.2), (RX, 0, 0.2)],
       "floor": [(LX, 0, 0.2), (LX, 0, BZ), (RX, 0, BZ), (RX, 0, 0.2)],
       "ceil": [(LX, CY, 0.2), (LX, CY, BZ), (RX, CY, BZ), (RX, CY, 0.2)]}
fade_all = (np.clip(cover, 0, 1.25) / 1.25 * hard).astype(np.float32)
# 调子（10/1 拿哥的示意图量过：我整张中位亮度 52、他 110；暗压成了死黑）：墙是暖赭、地上影子冷紫、暗最深别到底
for key, col, st in (("ceil", (196, 164, 124), 0.8), ("back", (174, 136, 98), 1.0), ("right", (74, 58, 84), 1.15), ("floor", (100, 92, 126), 1.05)):
    wc.wash(P, clip_poly(PLN[key]), col, strength=st * 0.62, var=0.004, layers=16, edge=0.3, granulate=0.15, mottle=0.25,
            fade=fade_all, wet_map=(0.5 * (1 - glow)).astype(np.float32), wet_r=8)
# ================= 湿笔：一笔湿画进另一笔湿色里（哥 10/1：不是先平涂再撒斑驳，是一片颜色自己长出来） =================
# 一笔有面积、有方向、有含水量；落下去看纸有多湿——湿就晕开、边上开花，干就留硬边、颜料沉在边上。
# 每落一笔纸干一点：先下的几笔融成一片，后下的留边。虚实就从这些色块的相遇里出来。
WET = np.zeros((H, W), np.float32)
NZF = wc.noise(11, octaves=3)
def wetstroke(pts3, width_m, color, strength, water, plane, keep=None, dry_tail=0.85, entry=0.2):
    global WET
    p2 = [pr(p) for p in pts3]
    w = max(3.0, px(width_m, pts3[len(pts3) // 2]))
    m = wc.brush(P, p2, w, color=None, entry=entry, dry_from=dry_tail, n=60) * plane
    if keep is not None:
        m = m * keep
    wl = WET
    s_ = np.clip(wl * water, 0, 1)
    b1, b2, b3 = wc.blur(m, 2.5), wc.blur(m, 9), wc.blur(m, 22)
    spread = np.where(s_ < 0.5, b1 + (b2 - b1) * (2 * s_), b2 + (b3 - b2) * (2 * s_ - 1))
    oy, ox = random.randint(0, 300), random.randint(0, 300)
    n = np.roll(NZF, (oy, ox), (0, 1))
    flower = Ss(0.12, 0.42, spread + 0.35 * s_ * (n - 0.5))                  # 湿的地方边不是一条顺的线，是开花
    cov = (spread * (1 - 0.6 * s_) + flower * spread.clip(0, 1) ** 0.3 * 0.6 * s_) * plane
    if keep is not None:
        cov = cov * keep
    rim = np.clip(cov - wc.blur(cov, 1.8), 0, 1) * (1 - 0.75 * wl)            # 停在干纸上的边：颜料沉一圈
    P.add(cov.astype(np.float32), color, strength, granulate=0.28)
    P.add((rim * 1.6).astype(np.float32), color, strength)
    WET = np.clip(WET * 0.93 + cov * water * 0.5, 0, 1)
    return cov

def wall_pt(X, Y):
    return (X, Y, BZ - 0.001)

def _grow():
    global WET
    kp = hard
    # --- 后墙：先一层浅赭湿底，纸是湿的 ---
    WET = np.maximum(WET, bwm * 0.95)
    P.add((bwm * kp * (0.5 + 0.3 * wc.noise(140, octaves=2))).astype(np.float32), (214, 184, 140), 0.45)
    R_ = random.Random(SEED * 7 + 1)
    pal_b = [((206, 160, 90), 0.4), ((200, 150, 96), 0.35), ((140, 124, 150), 0.35), ((186, 112, 74), 0.3),
             ((192, 128, 124), 0.25), ((150, 160, 120), 0.25), ((124, 104, 120), 0.35)]
    for i in range(22):
        col, st = R_.choice(pal_b)
        X0, Y0 = R_.uniform(-0.8, 3.8), R_.uniform(0.1, 2.6)
        ang = R_.choice([R_.uniform(-0.5, 0.5), R_.uniform(1.2, 1.9)])        # 横扫或竖拖
        L = R_.uniform(0.4, 1.3)
        pts = [wall_pt(X0, Y0), wall_pt(X0 + L * 0.5 * math.cos(ang) + R_.uniform(-0.1, 0.1), Y0 + L * 0.5 * math.sin(ang) + R_.uniform(-0.1, 0.1)),
               wall_pt(X0 + L * math.cos(ang), Y0 + L * math.sin(ang))]
        water = R_.uniform(0.5, 1.0) if i < 14 else R_.uniform(0.1, 0.4)        # 后下的几笔纸干了：留边
        wetstroke(pts, R_.uniform(0.18, 0.45), col, st * (0.7 + 0.6 * (1 - glow.mean())), water, bwm, kp)
    # 靠窗那片被光烧亮：一两笔偏黄、很湿
    for _ in range(3):
        X0, Y0 = R_.uniform(0.6, 2.8), R_.uniform(0.6, 2.4)
        wetstroke([wall_pt(X0, Y0), wall_pt(X0 + 0.6, Y0 + R_.uniform(-0.2, 0.2))], 0.5, (228, 186, 110), 0.35, 1.0, bwm, kp)
    # --- 右墙：深的，竖着拖，靛蓝、紫、熟褐在里面相遇 ---
    WET = np.maximum(WET, rwm * 0.9)
    pal_r = [((70, 66, 110), 0.55), ((96, 66, 104), 0.5), ((90, 70, 60), 0.45), ((60, 80, 96), 0.4)]
    for i in range(12):
        col, st = R_.choice(pal_r)
        Z0, Y0 = R_.uniform(0.6, 3.9), R_.uniform(0.0, 2.6)
        pts = [(RX - 0.001, Y0, Z0), (RX - 0.001, Y0 - R_.uniform(0.4, 1.0), Z0 + R_.uniform(-0.15, 0.15))]
        wetstroke(pts, R_.uniform(0.25, 0.5), col, st, R_.uniform(0.4, 1.0) if i < 8 else 0.25, rwm, kp)
    # --- 天花板：顺着板的方向扫，灰赭、淡紫灰、一点玫瑰 ---
    WET = np.maximum(WET, ceilm * 0.85)
    pal_c = [((180, 150, 112), 0.3), ((150, 136, 150), 0.28), ((176, 128, 110), 0.22), ((140, 118, 96), 0.3)]
    for i in range(14):
        col, st = R_.choice(pal_c)
        X0, Z0 = R_.uniform(-0.8, 3.6), R_.uniform(2.0, 3.95)
        pts = [(X0, CY - 0.001, Z0), (X0 + R_.uniform(0.6, 1.5), CY - 0.001, Z0 + R_.uniform(-0.1, 0.1))]
        wetstroke(pts, R_.uniform(0.15, 0.35), col, st, R_.uniform(0.5, 1.0) if i < 9 else 0.2, ceilm, kp)
    # --- 地：大的横笔，群青、紫、熟褐，光斑边上掉一点暖 ---
    WET = np.maximum(WET, floorm * 0.95)
    pal_f = [((84, 92, 150), 0.55), ((118, 92, 140), 0.45), ((110, 88, 78), 0.4), ((70, 78, 120), 0.5)]
    for i in range(18):
        col, st = R_.choice(pal_f)
        X0, Z0 = R_.uniform(-0.8, 3.8), R_.uniform(0.6, 3.9)
        pts = [(X0, 0.001, Z0), (X0 + R_.uniform(0.5, 1.4), 0.001, Z0 + R_.uniform(-0.25, 0.25))]
        wetstroke(pts, R_.uniform(0.25, 0.6), col, st, R_.uniform(0.5, 1.0) if i < 12 else 0.25, floorm, kp)
    nearp = np.clip(wc.blur(FL, 30) * 4, 0, 1)
    for _ in range(4):
        ys_, xs_ = np.nonzero(nearp * floorm > 0.4)
        if len(ys_) == 0:
            break
        k = R_.randrange(len(ys_)); X0 = Xmap[ys_[k], xs_[k]]; Z0 = Zmap[ys_[k], xs_[k]]
        wetstroke([(X0, 0.001, Z0), (X0 + 0.5, 0.001, Z0 - 0.1)], 0.3, (196, 140, 92), 0.3, 1.0, floorm, kp)
Xmap, Ymap, Zmap = CAM[0] + T * D[0], CAM[1] + T * D[1], CAM[2] + T * D[2]
isolated(_grow)
wet_keep(glow * (1 - ceilm) * 0.85, SIENNA, 0.3, 24, 0.0, hard)
wet_keep(np.clip(glow * 1.3, 0, 1), (232, 192, 120), 0.35, 30, 0.0, hard)
P.lift((np.exp(-(((xf - wc_[0]) / 210.0) ** 2 + ((yf - wc_[1]) / 260.0) ** 2)) * 0.35 * hard).astype(np.float32), 1.0)   # 窗边最亮那圈往奶白去       # 窗光烧出来的那片暖黄，墙和天花板都染
P.add((wc.blur(R, 14) * (1 - R) * 0.8 * (1 - ceilm)).astype(np.float32), SIENNA, 0.28)
# 折角：右墙比后墙深一层（背着窗），交界那条竖棱是硬的；地上影里压一层冷
P.add((rwm * hard).astype(np.float32), (66, 52, 64), 0.35)
P.add((floorm * (1 - wc.blur(FL + wr, 6)).clip(0, 1) * hard * 0.6).astype(np.float32), (86, 84, 100), 0.4)

# ================= 第二步半：老墙皮——不规矩的地方 =================
Xmap, Ymap, Zmap = CAM[0] + T * D[0], CAM[1] + T * D[1], CAM[2] + T * D[2]
def spot(p, rx, ry):
    a = pr(p)
    return np.exp(-(((xf - a[0]) / rx) ** 2 + ((yf - a[1]) / ry) ** 2))

def _wall_skin():
    bw = (bwm * hard).astype(np.float32)
    walls = (np.clip(bwm + rwm, 0, 1) * hard).astype(np.float32)
    # 1. 剥落：噪声长出来的硬边形（跟街上树影一个手法），里面露出砖
    env = np.maximum(spot((0.45, 0.6, BZ), 130, 120), 0.8 * spot((3.15, 2.4, BZ), 70, 40))     # 哥：破损集中在少数位置
    dens = 0.55 * wc.noise(16, 30, octaves=4) + 0.45 * wc.noise(60, octaves=2)
    thr = 0.72 - 0.3 * env
    peel = Ss(thr - 0.012, thr + 0.012, dens) * Ss(0.08, 0.35, env) * bw
    P.lift(peel * 0.3, 1.0)
    P.add(peel, (150, 84, 60), 0.55, granulate=0.3)
    br = np.zeros((H, W), np.float32)
    for yb in np.arange(0.03, 2.6, 0.066):
        br = np.maximum(br, wc.stroke_mask([pr((LX + 2.5, yb, BZ - 0.005)), pr((RX, yb, BZ - 0.005))], 1.0, 1.0, taper=False, rough=0.3))
    row = np.floor(Ymap / 0.066)
    vx = ((Xmap + (row % 2) * 0.12) / 0.24) % 1
    br = np.maximum(br, (np.abs(vx - 0.5) > 0.47) * 0.8)
    P.add((br * peel * Ss(0.35, 0.55, P.grain)).astype(np.float32), (88, 56, 46), 0.55)
    sh_edge = np.clip(peel - np.roll(peel, (-2, -2), (0, 1)), 0, 1)     # 墙皮断口：上沿一道深（墙皮的厚度投下的影）
    P.add(wc.blur(sh_edge, 0.6).astype(np.float32), (52, 40, 36), 0.7)
    lit_edge = np.clip(peel - np.roll(peel, (2, 2), (0, 1)), 0, 1)
    P.lift((wc.blur(lit_edge, 0.6) * 0.5).astype(np.float32), 1.0)
    # 2. 水渍：不是一排整齐的竖条（10/1 她：那样像窗棂的影子，水痕不会这么工整）。
    #    雨从窗台两头漫下来，一大片、往下越来越窄、长短不一，边上一圈圈潮水线（跟天花板的水渍一个画法），颜色很淡
    for (xs, w_m, L_m) in ((1.18, 0.22, 0.75), (2.12, 0.15, 0.45)):
        top_y = WIN[2] - 0.1
        cy_ = top_y - L_m * 0.5
        a = pr((xs, cy_, BZ)); rx, ry = px(w_m, (xs, cy_, BZ)), px(L_m * 0.6, (xs, cy_, BZ))
        env = np.exp(-(((xf - a[0]) / rx) ** 2 * (1 + 0.8 * Ss(a[1] - ry, a[1] + ry, yf)) + ((yf - a[1]) / ry) ** 2))
        env *= Ss(pr((xs, top_y + 0.02, BZ))[1] - 2, pr((xs, top_y, BZ))[1] + 4, yf)       # 从窗台底沿开始
        dens = 0.45 * wc.blur2(wc.noise(30, 14, octaves=3), 4, 1) + 0.55 * env
        for i, t in enumerate((0.4, 0.5, 0.6)):
            stn = Ss(t - 0.01, t + 0.01, dens) * Ss(0.08, 0.3, env) * bw
            P.add(stn.astype(np.float32), (140, 108, 80), 0.08 + 0.03 * i, edge=2.0, edge_r=1.4)
    # 窗台：被雨泡过的那截颜色深一点、边角磕掉几块
    sill_top = M([(WIN[0] - 0.12, WIN[2] - 0.07, BZ - 0.02), (WIN[1] + 0.12, WIN[2] - 0.07, BZ - 0.02), (WIN[1] + 0.12, WIN[2] - 0.1, BZ - 0.02), (WIN[0] - 0.12, WIN[2] - 0.1, BZ - 0.02)])
    P.add((sill_top * Ss(0.5, 0.7, wc.noise(6, 20, octaves=3))).astype(np.float32), (90, 70, 58), 0.4)
    # 3. 墙根返潮：一圈深，顶边不齐，顶边一道潮水线
    top = 0.32 + 0.16 * wc.noise(30, 120, octaves=3)
    damp = (Ymap < top) * walls
    damp = wc.blur(damp.astype(np.float32), 1.0)
    P.add((damp * (0.6 + 0.4 * wc.noise(25, 60, octaves=2))).astype(np.float32), (70, 60, 66), 0.5)
    line = np.clip(damp - np.roll(damp, 2, 0), 0, 1)
    P.add((wc.blur(line, 0.8) * 1.5).astype(np.float32), (60, 46, 42), 0.5)
    # 4. 竖着拖几笔干笔（墙是一刷子一刷子刷出来的）
    wc.dry(P, (walls * Ss(0.62, 0.78, wc.noise(160, 80, octaves=2)) * Ss(0.6, 0.0, np.clip(spot((2.0, 1.2, BZ), 300, 300), 0, 1))).astype(np.float32), (84, 64, 54), strength=0.22, thresh=0.7, streak="v")
    # 5. 暗里再掉两种颜色：一点群青、一点玫瑰，湿的
    wet_keep(walls * Ss(0.6, 0.8, wc.noise(70, octaves=3)) * (1 - glow), (80, 90, 130), 0.35, 14, 0.4, hard)
    wet_keep(walls * Ss(0.62, 0.82, wc.noise(60, octaves=3)) * glow, (170, 96, 96), 0.25, 14, 0.0, hard)
isolated(_wall_skin)

# ================= 藏色（她 9/30 教的）：同一个亮度上转色相，一块一块的，不是大渐变 =================
# 窗外的日光是暖的、天光是冷的、竹子反进来一点绿、地上的光斑往墙根反橙、红对联红热水瓶旁边染一点玫瑰。
# 做法：只改色相不改亮度——往 P.D 里加一个「减掉亮度分量」的吸光向量；形状用噪声阈值切成块（跟树影一个手法），
# 推饱和度一照应该是一块块不同的颜色，而不是一条从暖到冷的渐变（9/30 第一版就是栽在渐变上）。
LUMW = np.array([0.30, 0.59, 0.11], np.float32)
HIDE = 0.22     # 藏：推到 1.0 就是迷彩（c6 翻过）
def hue_shift(mask, color, amt):
    k = wc.Paper.K(color)
    k0 = k - (k @ LUMW) / LUMW.sum()
    P.D = np.maximum(P.D + (mask * amt * HIDE)[..., None] * k0[None, None, :], 0)

def patches(sc_y, sc_x, thr, soft=0.09):
    return Ss(thr - soft, thr + soft, wc.noise(sc_y, sc_x, octaves=3)).astype(np.float32)

def _hide_walls():
    walls = (np.clip(bwm + rwm, 0, 1) * hard).astype(np.float32)
    near_win = np.exp(-(((xf - wc_[0]) / 330.0) ** 2 + ((yf - wc_[1]) / 380.0) ** 2))
    low = Ss(620, 820, yf)                                                  # 墙根：地上光斑往上反
    near_patch = np.clip(wc.blur(FL, 60) * 6, 0, 1)
    hue_shift(walls * near_win * patches(50, 40, 0.55), (222, 170, 92), 0.9)                       # 日光的暖
    hue_shift(walls * (1 - near_win * 0.7) * patches(60, 50, 0.52), (84, 96, 140), 0.85)           # 天光的冷
    hue_shift(walls * patches(45, 35, 0.66) * (0.4 + 0.6 * near_win), (128, 172, 118), 0.8)        # 竹子反进来的绿
    hue_shift(walls * low * near_patch * patches(30, 50, 0.5), (226, 140, 78), 1.0)                # 地上的光往墙根反橙
    red_near = np.maximum(spot((0.9, 1.6, BZ), 70, 160), spot(FLASK, 70, 80))
    hue_shift(walls * red_near * patches(25, 25, 0.45), (196, 108, 118), 0.8)                     # 红对联、红热水瓶边上染玫瑰
    hue_shift(rwm * hard * patches(70, 40, 0.5), (126, 94, 160), 0.8)                              # 右墙紫里转蓝转绿
    hue_shift(rwm * hard * patches(50, 60, 0.6), (96, 140, 130), 0.6)
    fl = (floorm * hard).astype(np.float32)
    hue_shift(fl * patches(40, 90, 0.5) * (1 - near_patch * 0.5), (80, 100, 150), 0.9)            # 地上反天光：蓝
    hue_shift(fl * patches(35, 80, 0.6), (130, 96, 150), 0.7)                                      # 紫
    hue_shift(fl * near_patch * patches(25, 60, 0.5) * (1 - FL), (220, 150, 90), 0.8)              # 光斑边上反暖
isolated(_hide_walls)

# ================= 水彩自己的纹理：回流花、颜料沉在水边（哥示意图里满墙那种），每个面一个尺度 =================
def backrun(mask, sc, amount, levels=(0.55, 0.66)):
    n = wc.noise(sc, octaves=4)
    mod = np.zeros((H, W), np.float32)
    for t in levels:
        inside = Ss(t - 0.008, t + 0.008, n)
        ring = np.clip(inside - wc.blur(inside, 1.6), 0, 1)
        mod += -0.35 * inside + 1.3 * ring
    P.D *= (1 + amount * mask * mod)[..., None]

def _texture():
    keep = (1 - wc.blur(np.clip(WINM + FLb + TL + SL, 0, 1), 1.0)).astype(np.float32)
    backrun((bwm * keep).astype(np.float32), 26, 0.35)
    backrun((bwm * keep).astype(np.float32), 9, 0.22, levels=(0.6,))
    backrun((ceilm * keep).astype(np.float32), 40, 0.3)
    backrun((rwm * keep).astype(np.float32), 30, 0.3)
    backrun((floorm * keep).astype(np.float32), 34, 0.35)
    backrun((floorm * keep).astype(np.float32), 12, 0.2, levels=(0.62,))
# isolated(_texture)   # 10/1：被湿笔取代——纹理不该是最后撒上去的

if STAGE == "values":
    img = P.render(); out = os.path.join(HERE, f"corner_values{TAG}.png"); img.save(out)
    img.convert("L").resize((64, 82), Image.LANCZOS).resize((256, 328), Image.NEAREST).save(out.replace(".png", "_thumb.png"))
    print("values", out); sys.exit()

# ================= 第三步：天花板——浅的板、几根梁，松 =================
def _ceiling():
    P.add((ceilm * (0.5 + 0.5 * wc.noise(60, 160, octaves=3))).astype(np.float32), (160, 136, 112), 0.5, granulate=0.1)
    for Z in np.arange(0.5, BZ, 0.34):
        m = wc.stroke_mask([pr((-1.0, CY, Z)), pr((RX, CY, Z))], 1.3, 1.3, taper=False, rough=0.4)
        P.add((m * ceilm * Ss(0.5, 0.66, 0.6 * P.hstreak + 0.4 * P.grain) * Ss(0.4, 0.65, wc.noise(30, 120, octaves=2))).astype(np.float32), (84, 66, 56), 0.4)
    a, b = pr((-1.0, CY, BZ)), pr((RX, CY, BZ))
    P.add((wc.stroke_mask([a, b], 3, 3, taper=False, rough=0.3) * Ss(0.3, 0.5, P.grain + 0.1)).astype(np.float32), (58, 44, 40), 0.55)
isolated(_ceiling)

def _hide_ceiling():
    cm = ceilm.astype(np.float32)
    front = Ss(250, 0, yf)                                                                          # 近处天花板吃地上的暖
    hue_shift(cm * front * patches(40, 90, 0.52), (222, 172, 104), 0.8)
    hue_shift(cm * (1 - front * 0.6) * patches(40, 90, 0.55), (110, 116, 150), 0.7)
    hue_shift(cm * patches(30, 70, 0.66), (180, 120, 130), 0.5)
isolated(_hide_ceiling)

# ================= 天花板不能空：木纹、毛刺、节疤、水渍 =================
def _ceiling_detail():
    cm = ceilm.astype(np.float32)
    # 木纹：每块板顺着长边几道细的、弯弯的线，断断续续；偶尔一个节疤，纹路绕着它走
    knots = [(random.uniform(-0.5, 3.6), random.uniform(0.8, 3.8)) for _ in range(5)]
    grain_m = np.zeros((H, W), np.float32)
    for z0 in np.arange(0.5, BZ, 0.34):
        if random.random() < 0.55:
            continue                                    # 不是每块板都有纹：留大块安静的
        for _ in range(random.randint(2, 3)):
            zc = z0 + random.uniform(0.04, 0.3); ph = random.uniform(0, 6); amp = random.uniform(0.004, 0.012)
            xs_ = np.linspace(LX + 2.4, RX, 26)
            pts = []
            for X in xs_:
                dz = amp * math.sin(X * random.uniform(2.5, 4.0) + ph)
                for kx, kz in knots:
                    d = math.hypot(X - kx, (zc - kz) * 3)
                    if d < 0.25:
                        dz += 0.03 * (1 - d / 0.25) * (1 if zc > kz else -1)
                pts.append(pr((X, CY, zc + dz)))
            grain_m = np.maximum(grain_m, wc.stroke_mask(pts, 1.0, 0.8, taper=False, rough=0.4) * random.uniform(0.4, 1.0))
    grain_m *= cm * Ss(0.4, 0.58, 0.55 * P.hstreak + 0.45 * P.grain) * Ss(0.35, 0.6, wc.noise(40, 160, octaves=2))
    P.add(grain_m.astype(np.float32), (96, 74, 58), 0.45)
    for kx, kz in knots:
        a = pr((kx, CY, kz)); r = px(0.035, (kx, CY, kz))
        for rr in (1.0, 1.7, 2.5):
            ring = wc.dab(P, a[0], a[1], r * rr, (0, 0, 0), 0.0, squash=0.45)
            P.add((np.clip(ring - wc.dab(P, a[0], a[1], r * rr * 0.8, (0, 0, 0), 0.0, squash=0.45), 0, 1) * cm).astype(np.float32), (90, 68, 52), 0.5)
        wc.dab(P, a[0], a[1], r * 0.7, (80, 58, 44), 0.7, squash=0.45)
    # 毛刺：板缝边上一撮撮短的刺，顺着木纹方向
    for z0 in np.arange(0.5, BZ, 0.34):
        for _ in range(random.randint(4, 8)):
            X = random.uniform(LX + 2.4, RX); zz = z0 + random.choice((0.01, -0.01)) * random.uniform(0.5, 2)
            a, b = pr((X, CY, zz)), pr((X + random.uniform(0.04, 0.12), CY, zz + random.uniform(-0.006, 0.006)))
            P.add((wc.stroke_mask([a, b], 1.1, 0.4, taper=False, rough=0.3) * cm).astype(np.float32), (80, 60, 48), 0.55)
    # 水渍：漏过雨的地方，一圈圈潮水线——颜料干在边上（watercolor 自己最会画这个）
    for (sx, sz, rad) in ((3.2, 3.8, 0.28),):          # 哥：顶上的大水痕在抢注意力——只留靠墙一小块
        env = spot((sx, CY, sz), px(rad, (sx, CY, sz)) * 1.4, px(rad, (sx, CY, sz)) * 0.55)
        dens = 0.45 * wc.noise(40, 80, octaves=3) + 0.55 * env          # 低频：一大片、几圈潮水线，不是碎岛
        for i, t in enumerate((0.42, 0.52, 0.62)):
            st = Ss(t - 0.008, t + 0.008, dens) * Ss(0.1, 0.3, env) * cm
            P.add(st.astype(np.float32), (150, 112, 76), 0.1 + 0.04 * i, edge=2.2, edge_r=1.5)
        P.add((Ss(0.6, 0.75, dens) * Ss(0.1, 0.3, env) * cm).astype(np.float32), (120, 96, 80), 0.12)
isolated(_ceiling_detail)

# 梁最后压上去（10/1 她抓的：水渍木纹最后盖上去，把梁底那条亮也染了，梁扁成一根黑线——体积丢了）
def _beams():
    for X in (0.3, 1.2, 2.1, 3.0):
        w, h = 0.16, 0.2
        bot = clip_poly([(X - w / 2, CY - h, 0.6), (X + w / 2, CY - h, 0.6), (X + w / 2, CY - h, BZ), (X - w / 2, CY - h, BZ)])
        sx = X - w / 2 if X > CAM[0] else X + w / 2
        side = clip_poly([(sx, CY - h, 0.6), (sx, CY, 0.6), (sx, CY, BZ), (sx, CY - h, BZ)])
        bm, sm_ = wc.poly_mask(bot), wc.poly_mask(side)
        # 沿着梁长的「在场度」：哥——梁仍然可以笔直，但某一段清楚、某一段只剩色面、另一段隐进天花板
        e0, e1 = pr((sx, CY - h, 0.7)), pr((sx, CY - h, BZ))
        L2 = (e1[0] - e0[0]) ** 2 + (e1[1] - e0[1]) ** 2 + 1e-6
        tpos = np.clip(((xf - e0[0]) * (e1[0] - e0[0]) + (yf - e0[1]) * (e1[1] - e0[1])) / L2, 0, 1)
        prof = np.interp(np.linspace(0, 1, 9), np.linspace(0, 1, 9), [random.uniform(0.2, 1.0) for _ in range(9)])
        pres = (0.55 + 0.45 * Ss(0.35, 0.75, np.interp(tpos, np.linspace(0, 1, 9), prof))).astype(np.float32)
        P.lift(wc.blur(np.clip(bm + sm_, 0, 1), 0.6) * 0.7 * pres, 1.0)
        wc.wash(P, side, (62, 46, 40), strength=1.1, var=0.003, layers=6, edge=0.3, fade=pres)
        P.add((bm * pres * (0.55 + 0.45 * wc.noise(30, 90, octaves=2))).astype(np.float32), (150, 122, 96), 0.75)
        hue_shift(bm * patches(30, 80, 0.5), (222, 172, 104), 0.7)
        P.add((bm * Ss(0.0, 1.0, (yf - pr((X, CY - h, BZ))[1]) / 300.0 + 0.3) * 0.0).astype(np.float32), (0, 0, 0), 0.0)
        # 底面和侧面交界那条棱：一线亮，棱是硬的
        e0, e1 = pr((sx, CY - h, 0.7)), pr((sx, CY - h, BZ))
        P.lift((wc.stroke_mask([e0, e1], 1.4, 1.0, taper=False, rough=0.2) * Ss(0.3, 0.5, P.grain) * 0.5 * Ss(0.6, 0.9, pres)).astype(np.float32), 1.0)
        # 侧面顺着长度几道纹、一两道裂；底面几道淡纹
        for k in range(3):
            yy_ = CY - h * random.uniform(0.15, 0.85)
            pts = [pr((sx, yy_ + 0.008 * math.sin(zz * 3 + k), zz)) for zz in np.linspace(0.7, BZ, 14)]
            P.add((wc.stroke_mask(pts, 1.0, 0.7, taper=False, rough=0.4) * Ss(0.4, 0.6, P.hstreak * 0.5 + P.grain * 0.5)).astype(np.float32), (36, 28, 26), 0.5)
        for k in range(2):
            xo = X + random.uniform(-w * 0.35, w * 0.35)
            pts = [pr((xo + 0.004 * math.sin(zz * 4 + k), CY - h, zz)) for zz in np.linspace(0.7, BZ, 14)]
            P.add((wc.stroke_mask(pts, 0.9, 0.6, taper=False, rough=0.4) * bm * Ss(0.45, 0.6, P.grain)).astype(np.float32), (120, 96, 76), 0.45)
        za = random.uniform(1.0, 2.5)
        crack = [pr((sx, CY - h * 0.5 + 0.01 * math.sin(zz * 9), zz)) for zz in np.linspace(za, za + random.uniform(0.6, 1.2), 10)]
        P.add(wc.stroke_mask(crack, 1.6, 0.6, taper=True, rough=0.4), (30, 22, 22), 0.8)
isolated(_beams)

# ================= 第三步半：墙上的东西 =================
def _deco():
    z = BZ - 0.012
    # 窗左边一条褪色的红对联：纸皱、颜色不匀、一角翘起来；字是几个说不清的深点
    cx0, cx1, cy0, cy1 = 0.82, 0.98, 0.95, 2.2
    cp = clip_poly([(cx0, cy0, z), (cx0, cy1, z), (cx1, cy1, z), (cx1 + 0.01, cy0 + 0.02, z)])
    cm = wc.poly_mask(cp)
    P.lift(wc.blur(cm, 0.6) * 0.55, 1.0)
    wc.wash(P, cp, (176, 72, 58), strength=0.85, var=0.01, layers=8, edge=0.4, granulate=0.3, mottle=0.5)
    P.add((cm * Ss(0.55, 0.75, wc.noise(8, 30, octaves=3))).astype(np.float32), (120, 52, 46), 0.4)
    def glyph(x, y, size, col, st):
        # 一个看不清但像字的字：横、竖、撇、捺、点挤在一个方框里，书法笔（起笔粗收笔尖）
        h = size / 2
        kinds = random.sample(["h", "h", "h", "v", "v", "pie", "na", "dot", "dot"], random.randint(4, 6))
        for kd in kinds:
            if kd == "h":
                yy_ = y + random.uniform(-h * 0.8, h * 0.8); x0_ = x - h * random.uniform(0.5, 0.95); x1_ = x + h * random.uniform(0.4, 0.95)
                pts = [(x0_, yy_), (x1_, yy_ - h * 0.05)]
            elif kd == "v":
                xx_ = x + random.uniform(-h * 0.5, h * 0.5)
                pts = [(xx_, y - h * random.uniform(0.6, 0.95)), (xx_, y + h * random.uniform(0.5, 0.95))]
            elif kd == "pie":
                pts = [(x + h * 0.1, y - h * 0.6), (x - h * 0.7, y + h * 0.8)]
            elif kd == "na":
                pts = [(x - h * 0.1, y - h * 0.2), (x + h * 0.8, y + h * 0.8)]
            else:
                px_, py_ = x + random.uniform(-h * 0.7, h * 0.7), y + random.uniform(-h * 0.8, h * 0.6)
                pts = [(px_, py_), (px_ + h * 0.15, py_ + h * 0.2)]
            wc.brush(P, pts, max(1.4, size * 0.13), color=col, strength=st, entry=0.15, dry_from=0.9, n=30)
    for k in range(7):
        yk = cy1 - 0.14 - k * 0.17
        a = pr(((cx0 + cx1) / 2, yk, z))
        glyph(a[0], a[1], px(0.11, (cx0, yk, z)), (40, 30, 30), 0.8)
    # 价目牌：一块木框、里面发黄的纸，竖着几行看不清的字，一个红戳
    bx0, bx1, by0, by1 = 3.0, 3.52, 1.45, 2.02
    fr = clip_poly([(bx0, by0, z), (bx0, by1, z), (bx1, by1, z), (bx1, by0, z)])
    P.add(wc.poly_mask(fr), (70, 52, 42), 1.0)
    inner = clip_poly([(bx0 + 0.04, by0 + 0.04, z), (bx0 + 0.04, by1 - 0.04, z), (bx1 - 0.04, by1 - 0.04, z), (bx1 - 0.04, by0 + 0.04, z)])
    im = wc.poly_mask(inner)
    P.lift(wc.blur(im, 0.6) * 0.8, 1.0)
    wc.wash(P, inner, (196, 178, 140), strength=0.6, var=0.004, layers=6, edge=0.4, mottle=0.4)
    for col in range(5):
        xk = bx1 - 0.09 - col * 0.085
        for k in range(random.randint(4, 7)):
            yk = by1 - 0.09 - k * 0.055
            a = pr((xk, yk, z))
            glyph(a[0], a[1], px(0.035, (xk, yk, z)), (50, 40, 40), 0.6)
    a = pr((bx0 + 0.1, by0 + 0.1, z)); wc.dab(P, a[0], a[1], px(0.028, (bx0, by0, z)), (170, 50, 44), 0.8)
    # 一根老电线：沿着梁底走过来，到灯那里垂下去；另一根顺墙往下到窗边的开关，松松地垂
    pts = [pr((2.1, CY - 0.21, z0)) for z0 in np.linspace(0.6, 3.2, 8)]
    P.add(wc.stroke_mask(pts, 1.3, 1.1, taper=False, rough=0.2), (36, 30, 30), 0.8)
    w0 = pr((2.62, CY - 0.01, z)); w1 = pr((2.62, 1.42, z))
    sag = [(w0[0] + (w1[0] - w0[0]) * t + 4 * math.sin(math.pi * t), w0[1] + (w1[1] - w0[1]) * t) for t in np.linspace(0, 1, 9)]
    P.add(wc.stroke_mask(sag, 1.2, 1.1, taper=False, rough=0.2), (36, 30, 30), 0.75)
    sw = clip_poly([(2.57, 1.32, z), (2.57, 1.42, z), (2.67, 1.42, z), (2.67, 1.32, z)])
    P.lift(wc.poly_mask(sw) * 0.5, 1.0); wc.wash(P, sw, (190, 180, 160), strength=0.5, var=0.004, layers=3, edge=0.5)
    # 右墙一层木板架子，一排热水瓶：暗里的几块颜色，逆光边一线亮
    xw = RX - 0.01
    shelf = clip_poly([(xw, 1.22, 2.0), (xw, 1.28, 2.0), (xw, 1.28, 3.9), (xw, 1.22, 3.9)])
    P.add(wc.poly_mask(shelf), (52, 40, 36), 1.0)
    edge_ = clip_poly([(xw - 0.24, 1.28, 2.0), (xw, 1.28, 2.0), (xw, 1.28, 3.9), (xw - 0.24, 1.28, 3.9)])
    P.add(wc.poly_mask(edge_), (120, 96, 80), 0.6)
    for zt, col in ((2.62, (110, 70, 66)), (3.2, (84, 100, 100))):        # 哥：柜里的瓶子只露一两处微弱色差
        p0 = (xw - 0.13, 1.28, zt); a = pr(p0); hh, ww = px(0.36, p0), px(0.11, p0)
        pts = [(a[0] - ww / 2, a[1]), (a[0] - ww / 2, a[1] - hh * 0.82), (a[0] - ww * 0.3, a[1] - hh), (a[0] + ww * 0.3, a[1] - hh), (a[0] + ww / 2, a[1] - hh * 0.82), (a[0] + ww / 2, a[1])]
        fm = wc.poly_mask(pts)
        P.lift(wc.blur(fm, 2.0) * 0.25, 1.0)
        wc.wash(P, pts, col, strength=0.6, var=0.008, layers=4, edge=0.1, wet_map=np.ones((H, W), np.float32) * 0.7, wet_r=4)
isolated(_deco)

# ================= 第四步：家具——几笔，不是盒子 =================
def _table():
    # 桌下一团软的暗，连进地上的暗里
    foot = M(box_c(TAB["cx"], TAB["cz"], 1.05, 1.05, 0, TAB["rot"]))
    wet_keep(foot * floorm, DEEP, 0.9, 10, 0.0, hard)
    # 桌面：没吃光的那截是深的老漆（湿着铺，边软一点），吃光的那截留白上一层淡暖
    unlit = TOPM * (1 - wc.blur(TL, 1.5))
    P.add((wc.blur(unlit, 0.8) * 0.95).astype(np.float32), (76, 56, 46), 1.0)
    P.add(wc.blur(TL, 0.8).astype(np.float32), GOLD, 0.3)
    # 前沿厚度 + 牙板：一道深的横笔（两段：朝我们那面深、侧面稍浅），棱硬
    c0, c1, c2, c3 = TOPC
    lo = lambda p, dy: (p[0], p[1] - dy, p[2])
    face_f = clip_poly([c0, c1, lo(c1, 0.16), lo(c0, 0.16)])
    face_s = clip_poly([c1, c2, lo(c2, 0.16), lo(c1, 0.16)]) if cam(c2)[0] > cam(c1)[0] else []
    P.lift(wc.blur(wc.poly_mask(face_f), 0.6) * 0.5, 1.0)
    pa, pb = pr(c0), pr(c1)
    fade_r = (1 - 0.55 * Ss(pa[0] + (pb[0] - pa[0]) * 0.4, pb[0] + 10, xf)).astype(np.float32)     # 靠光那头实，往右化进暗
    wc.wash(P, face_f, (44, 34, 32), strength=1.25, var=0.004, layers=6, edge=0.35, fade=fade_r)
    if face_s:
        wc.wash(P, face_s, (90, 70, 58), strength=0.7, var=0.004, layers=6, edge=0.2)
    # 腿：几刀往下收的书法笔，后腿淡、前腿深；贴地那截干掉
    for i, (lx, lz) in enumerate(LEGS_T):
        far = i >= 2
        m = post((lx, 0.65, lz), (lx + random.uniform(-0.01, 0.01), 0.0, lz), 0.06, None, dry=0.25)
        y_hi, y_lo = pr((lx, 0.4 if not far else 0.6, lz))[1], pr((lx, 0.0, lz))[1]
        m = m * (1 - 0.95 * Ss(y_hi, y_lo, yf))                                   # 腿不描到脚：下半截进桌下的暗
        P.add(m.astype(np.float32), (40, 30, 28) if not far else (70, 56, 50), 1.2 if not far else 0.45)
    # 桌面前沿一线亮（棱吃光）
    a, b = pr(c0), pr(c1)
    P.lift((wc.stroke_mask([a, b], max(1.5, px(0.012, c0)), None, taper=False, rough=0.3) * Ss(0.3, 0.55, P.grain)).astype(np.float32), 0.7)
isolated(_table)

def _chair():
    cx, cz, rot = CHAIR["cx"], CHAIR["cz"], CHAIR["rot"]
    foot = M(box_c(cx, cz, 0.55, 0.55, 0, rot))
    wet_keep(foot * floorm, DEEP, 0.6, 8, 0.0, hard)
    # 川西竹靠椅：一根根竹管弯出来的——座框、座面竹片、椅背弯梁顺下来成扶手、四条腿、贴地一圈横撑。
    c, s_ = math.cos(math.radians(rot)), math.sin(math.radians(rot))
    def W_(u, v, y):
        return (cx + u * c - v * s_, y, cz + u * s_ + v * c)
    def tube(pts_uvy, w_m, col, st, dry=0.0, lit_rim=False, lose=None, gaps=0.0):
        """lose=(y_lo, y_hi, k)：世界高度 y_hi 往下到 y_lo 越来越淡（融进地上的暗）；gaps：沿着这根竹子随机几段只剩色面"""
        pts3 = [W_(*q) for q in pts_uvy]
        p2 = [pr(q) for q in pts3]
        mid = pts3[len(pts3) // 2]
        w = max(1.4, px(w_m, mid))
        m = wc.stroke_mask(p2, w, w * 0.9, taper=False, rough=0.12)
        if lose is not None:
            ylo, yhi, k = lose
            u_, v_ = pts_uvy[len(pts_uvy) // 2][0], pts_uvy[len(pts_uvy) // 2][1]
            s_lo, s_hi = pr(W_(u_, v_, ylo))[1], pr(W_(u_, v_, yhi))[1]
            m = m * (1 - k * Ss(s_hi, s_lo, yf))
        if gaps:
            a2, b2 = p2[0], p2[-1]
            L = math.hypot(b2[0] - a2[0], b2[1] - a2[1]) + 1e-6
            tpos = np.clip(((xf - a2[0]) * (b2[0] - a2[0]) + (yf - a2[1]) * (b2[1] - a2[1])) / (L * L), 0, 1)
            prof = [random.uniform(0, 1) for _ in range(6)]
            m = m * (1 - gaps * (1 - Ss(0.3, 0.6, np.interp(tpos, np.linspace(0, 1, 6), prof))))
        if dry:
            a2, b2 = p2[0], p2[-1]
            L = math.hypot(b2[0] - a2[0], b2[1] - a2[1]) + 1e-6
            tpos = np.clip(((xf - a2[0]) * (b2[0] - a2[0]) + (yf - a2[1]) * (b2[1] - a2[1])) / (L * L), 0, 1)
            thr = np.clip((tpos - (1 - dry)) / dry, 0, 1) * 0.7
            m = m * Ss(thr - 0.06, thr + 0.06, P.grain)
        P.add(m.astype(np.float32), col, st)
        if lit_rim and light_at(mid) > 0.05:
            top = np.clip(m - np.roll(m, max(1, int(w * 0.45)), axis=0), 0, 1)
            P.lift((top * Ss(0.3, 0.5, P.grain) * 0.75).astype(np.float32), 1.0)
            P.add(top.astype(np.float32), (232, 196, 124), 0.3)
        return m
    SH_ = (92, 72, 46); FAR_ = (128, 104, 70)
    def arc(p, q, bulge_v, n=7):
        out = []
        for i in range(n):
            t = i / (n - 1)
            out.append((p[0] + (q[0] - p[0]) * t, p[1] + (q[1] - p[1]) * t + bulge_v * math.sin(math.pi * t), p[2] + (q[2] - p[2]) * t))
        return out
    # 远的（后腿、后横撑）先，淡
    # 远的（后腿、后横撑）：几乎被暗吞了，只剩一点点
    tube([(-0.2, -0.2, 0.0), (-0.2, -0.2, 0.4)], 0.035, FAR_, 0.45, lose=(0.0, 0.35, 0.95))
    tube([(0.2, -0.2, 0.0), (0.2, -0.2, 0.4)], 0.035, FAR_, 0.4, lose=(0.0, 0.35, 0.95))
    # 椅背：两根柱往后仰，顶上一道往后鼓的弯梁，中间三根细竹
    # 椅背：上半截靠着受光的弯梁，清楚；往下化进座面底下的暗
    tube([(-0.2, -0.2, 0.4), (-0.22, -0.25, 0.86)], 0.036, SH_, 1.05, lose=(0.42, 0.62, 0.7))
    tube([(0.2, -0.2, 0.4), (0.22, -0.25, 0.86)], 0.036, SH_, 1.05, lose=(0.42, 0.62, 0.8), gaps=0.5)
    tube(arc((-0.22, -0.25, 0.86), (0.22, -0.25, 0.86), -0.07), 0.045, SH_, 1.1, lit_rim=True)          # 最实的一笔
    for u in (-0.09, 0.0, 0.09):
        tube([(u, -0.24 - 0.04 * (1 - abs(u) * 5), 0.84), (u * 1.05, -0.2, 0.44)], 0.016, FAR_, 0.45, lose=(0.44, 0.7, 0.9), gaps=0.6)
    # 座面：竹片（吃光的地方亮），座框一圈
    P.add((wc.blur(SEATM * (1 - wc.blur(SL, 1.0)), 0.8) * 0.9).astype(np.float32), BAMBOO, 0.8)
    P.add(wc.blur(SL, 0.8).astype(np.float32), (232, 196, 124), 0.4)
    for v in (-0.12, -0.04, 0.04, 0.12):
        tube([(-0.2, v, 0.43), (0.2, v, 0.43)], 0.008, (110, 86, 56), 0.4, gaps=0.7)
    tube([(-0.21, 0.21, 0.42), (0.21, 0.21, 0.42)], 0.03, SH_, 1.05, lit_rim=True)                        # 座框前沿：实
    tube([(0.21, 0.21, 0.42), (0.21, -0.21, 0.42)], 0.03, SH_, 0.8, gaps=0.6)
    tube([(-0.21, -0.21, 0.42), (-0.21, 0.21, 0.42)], 0.03, SH_, 0.6, gaps=0.7)
    # 扶手：从椅背柱顺下来，往前弯进前腿
    for sgn, st_, gp in ((-1, 1.05, 0.0), (1, 0.7, 0.6)):     # 一边扶手实、一边化掉一半
        tube([(sgn * 0.21, -0.23, 0.64), (sgn * 0.24, -0.02, 0.65), (sgn * 0.23, 0.15, 0.6), (sgn * 0.2, 0.2, 0.42)], 0.03, SH_, st_, lit_rim=(sgn < 0), gaps=gp)
    # 近的前腿：上半截在，下半截融进地上的暗；横撑只剩一截
    tube([(-0.2, 0.2, 0.42), (-0.2, 0.2, 0.0)], 0.036, (76, 58, 38), 1.1, lose=(0.0, 0.3, 0.92))
    tube([(0.2, 0.2, 0.42), (0.2, 0.2, 0.0)], 0.036, (76, 58, 38), 1.0, lose=(0.0, 0.34, 0.95))
    tube([(-0.2, 0.2, 0.12), (0.2, 0.2, 0.12)], 0.025, (76, 58, 38), 0.4, gaps=0.8)
isolated(_chair)

# ================= 第五步：窗、桌上的东西、那缕热气 =================
def _window():
    x0, x1, y0, y1 = WIN
    eat = np.exp(-(((xf - wc_[0]) / 95.0) ** 2 + ((yf - wc_[1]) / 125.0) ** 2))
    t = 0.07
    # 窗框四条边各自一条「在场度」：几段实的、几段被光吃掉跟墙接上——不是一圈完整等粗的框（哥：保留几段关键框线就够）
    sides = [([(x0 - t, y0 - t), (x0, y0 - t), (x0, y1 + t), (x0 - t, y1 + t)], (x0 - t / 2, y0), (x0 - t / 2, y1)),
             ([(x1, y0 - t), (x1 + t, y0 - t), (x1 + t, y1 + t), (x1, y1 + t)], (x1 + t / 2, y0), (x1 + t / 2, y1)),
             ([(x0 - t, y1), (x1 + t, y1), (x1 + t, y1 + t), (x0 - t, y1 + t)], (x0, y1 + t / 2), (x1, y1 + t / 2)),
             ([(x0 - t, y0 - t), (x1 + t, y0 - t), (x1 + t, y0), (x0 - t, y0)], (x0, y0 - t / 2), (x1, y0 - t / 2))]
    for q, pa_, pb_ in sides:
        m = M([(a_, b_, BZ - 0.01) for a_, b_ in q])
        e0, e1 = pr((pa_[0], pa_[1], BZ)), pr((pb_[0], pb_[1], BZ))
        L2 = (e1[0] - e0[0]) ** 2 + (e1[1] - e0[1]) ** 2 + 1e-6
        tpos = np.clip(((xf - e0[0]) * (e1[0] - e0[0]) + (yf - e0[1]) * (e1[1] - e0[1])) / L2, 0, 1)
        prof = [random.choice([1.0, 1.0, random.uniform(0.0, 0.5)]) for _ in range(7)]
        pres = 0.12 + 0.88 * Ss(0.3, 0.7, np.interp(tpos, np.linspace(0, 1, 7), prof))
        P.add((wc.blur(m, 0.6) * pres * (1 - 0.5 * eat)).astype(np.float32), (64, 50, 44), 1.0)
    # 窗棂：几笔竖、一笔横，被光吃细吃断——不是尺子画的格子
    for bx in np.linspace(x0, x1, 5)[1:-1]:
        m = post((bx, y1, BZ), (bx + 0.004, y0, BZ), 0.028, None, dry=0.15, rough=0.2)
        P.add((m * (1 - 0.9 * eat)).astype(np.float32), (70, 56, 48), 0.85)
    ym = (y0 + y1) * 0.52
    m = post((x0, ym, BZ), (x1, ym + 0.004, BZ), 0.028, None, dry=0.15, rough=0.2)
    P.add((m * (1 - 0.9 * eat)).astype(np.float32), (70, 56, 48), 0.8)
    # 窗台：亮的一横，被纸纹咬
    sill = M([(x0 - 0.12, y0 - 0.07, BZ - 0.02), (x1 + 0.12, y0 - 0.07, BZ - 0.02), (x1 + 0.12, y0 - 0.1, BZ - 0.02), (x0 - 0.12, y0 - 0.1, BZ - 0.02)])
    P.lift((sill * Ss(0.35, 0.55, P.grain)).astype(np.float32), 0.6)
    # 外面：竹子的一点点淡绿，只在窗边
    P.add((WINM * (1 - eat) * Ss(0.55, 0.78, wc.noise(14, 9, octaves=3)) * 0.6).astype(np.float32), (160, 176, 128), 0.35)
isolated(_window)

def _things():
    a = pr(GAIWAN); r = px(0.07, GAIWAN)
    bowl = wc.poly_mask([(a[0] - r, a[1] - r * 0.7), (a[0] + r, a[1] - r * 0.7), (a[0] + r * 0.55, a[1]), (a[0] - r * 0.55, a[1])])
    P.lift(wc.blur(bowl, 0.6), 1.0)
    P.add((bowl * Ss(0.0, 1.0, (xf - a[0] + r) / (2 * r))).astype(np.float32), (150, 152, 166), 0.5)
    lid = wc.dab(P, a[0] + r * 0.55, a[1] - r * 0.95, r * 0.75, (0, 0, 0), 0.0, squash=0.38, rot=-0.35)
    P.lift(lid * 0.9, 1.0); P.add(lid, (180, 182, 192), 0.25)
    sh = wc.dab(P, a[0] + r * 0.3, a[1] + r * 0.2, r * 1.15, (0, 0, 0), 0.0, squash=0.25)
    P.add((sh * (1 - bowl)).astype(np.float32), DEEP, 0.55)
    b = pr(FLASK); hh, ww = px(0.36, FLASK), px(0.12, FLASK)
    pts = [(b[0] - ww / 2, b[1]), (b[0] - ww / 2, b[1] - hh * 0.82), (b[0] - ww * 0.32, b[1] - hh), (b[0] + ww * 0.32, b[1] - hh), (b[0] + ww / 2, b[1] - hh * 0.82), (b[0] + ww / 2, b[1])]
    fm = wc.poly_mask(pts)
    P.lift(wc.blur(fm, 0.6) * 0.6, 1.0)
    wc.wash(P, pts, (150, 58, 50), strength=1.0, var=0.006, layers=5, edge=0.4)
    P.add((fm * Ss(0.3, 1.0, (xf - b[0] + ww / 2) / ww)).astype(np.float32), (60, 30, 30), 0.6)
    P.lift((wc.stroke_mask([(b[0] - ww * 0.42, b[1] - hh * 0.9), (b[0] - ww * 0.42, b[1] - hh * 0.1)], max(1.2, ww * 0.12), None, taper=False) * 0.7).astype(np.float32), 1.0)
    # 热气：从碗里升起来，穿过窗的光
    st = np.zeros((H, W), np.float32)
    x0, y0 = a[0], a[1] - r * 0.8
    for i in range(95):
        x0 += random.uniform(-1.2, 1.2) + 1.3 * math.sin(i * 0.17) * (0.3 + i / 95); y0 -= 2.7
        st = np.maximum(st, wc.dab(P, x0, y0, 1.2 + i * 0.14, (0, 0, 0), 0.0) * (1 - i / 95) ** 0.9)
    st = wc.blur(st, 2.0) * (0.4 + 0.9 * wc.noise(8, octaves=3))
    P.lift(np.clip(st * 0.85, 0, 0.8).astype(np.float32), 1.0)
isolated(_things)

def _more_things():
    # 第二碗盖碗（两个人刚聊完）、一碟瓜子、一份叠着的报纸
    for G in ((2.78, 0.8, 3.22),):
        a = pr(G); r = px(0.065, G)
        bowl = wc.poly_mask([(a[0] - r, a[1] - r * 0.7), (a[0] + r, a[1] - r * 0.7), (a[0] + r * 0.55, a[1]), (a[0] - r * 0.55, a[1])])
        P.lift(wc.blur(bowl, 0.6), 1.0)
        P.add((bowl * Ss(0.0, 1.0, (xf - a[0] + r) / (2 * r))).astype(np.float32), (140, 144, 164), 0.55)
        lid = wc.dab(P, a[0], a[1] - r * 0.78, r * 0.95, (0, 0, 0), 0.0, squash=0.3)
        P.lift(lid * 0.85, 1.0); P.add(lid, (170, 174, 190), 0.3)
        wc.dab(P, a[0] + r * 0.25, a[1] + r * 0.15, r * 1.1, DEEP, 0.4, squash=0.22)
    d = (2.48, 0.8, 3.32); a = pr(d); r = px(0.1, d)
    dish = wc.dab(P, a[0], a[1], r, (0, 0, 0), 0.0, squash=0.32)
    P.lift(dish * 0.8, 1.0); P.add(dish, (190, 186, 176), 0.3)
    for k in range(14):
        wc.dab(P, a[0] + random.uniform(-r * 0.7, r * 0.7), a[1] + random.uniform(-r * 0.2, r * 0.15), random.uniform(0.8, 1.6), (50, 40, 36), 0.8, squash=0.5, rot=random.uniform(0, 3))
    np_ = [(2.1, 0.801, 2.95), (2.42, 0.801, 2.9), (2.46, 0.801, 3.12), (2.13, 0.801, 3.17)]
    pp = clip_poly(np_); pm = wc.poly_mask(pp)
    P.lift(wc.blur(pm, 0.6) * 0.75, 1.0)
    wc.wash(P, pp, (200, 194, 176), strength=0.45, var=0.006, layers=4, edge=0.4)
    for t in np.linspace(0.2, 0.8, 6):
        q0 = (np_[0][0] + (np_[3][0] - np_[0][0]) * t, 0.802, np_[0][2] + (np_[3][2] - np_[0][2]) * t)
        q1 = (np_[1][0] + (np_[2][0] - np_[1][0]) * t, 0.802, np_[1][2] + (np_[2][2] - np_[1][2]) * t)
        P.add((wc.stroke_mask([pr(q0), pr(q1)], 0.9, 0.9, taper=False, rough=0.5) * Ss(0.45, 0.6, P.grain) * pm).astype(np.float32), (90, 86, 84), 0.5)
    # 光斑里几粒瓜子壳
    ys_, xs_ = np.nonzero(FL > 0.7)
    for k in range(18):
        i = random.randrange(len(ys_))
        wc.dab(P, xs_[i] + random.uniform(-2, 2), ys_[i], random.uniform(1.2, 2.4), (60, 48, 42), 0.85, squash=0.45, rot=random.uniform(0, 3))
    # 石板缝：几道往里收、几道横的，断断续续，在光里看得清、在暗里没了
    sm = np.zeros((H, W), np.float32)
    for X in (0.4, 1.2, 2.0, 2.8, 3.6):
        sm = np.maximum(sm, wc.stroke_mask([pr((X, 0, 0.6)), pr((X, 0, BZ))], 1.8, 1.0, taper=False, rough=0.45))
    for Zs in (0.9, 1.6, 2.35, 3.2):
        sm = np.maximum(sm, wc.stroke_mask([pr((LX + 2.5, 0, Zs)), pr((RX, 0, Zs))], 1.6, 1.6, taper=False, rough=0.45))
    sm *= floorm * Ss(0.45, 0.62, wc.noise(30, 70, octaves=3)) * Ss(0.35, 0.55, P.grain)
    P.add((sm * (0.4 + 0.6 * wc.blur(FL, 4))).astype(np.float32), (60, 48, 46), 0.6)
    # 窗外竹叶：淡的灰绿长叶子，一簇一簇，窗中间被光吃掉
    eat = np.exp(-(((xf - wc_[0]) / 110.0) ** 2 + ((yf - wc_[1]) / 140.0) ** 2))
    lv = np.zeros((H, W), np.float32)
    for k in range(26):
        u0 = random.uniform(WIN[0], WIN[1]); v0 = random.uniform(WIN[2] + 0.4, WIN[3] + 0.1)
        ang = random.uniform(-0.9, -0.2) * (1 if random.random() < 0.6 else -1)
        L = random.uniform(0.12, 0.22)
        a = pr((u0, v0, BZ + 0.3)); b = pr((u0 + L * math.cos(ang), v0 + L * math.sin(ang), BZ + 0.3))
        lv = np.maximum(lv, wc.brush(P, [a, b], px(0.04, (u0, v0, BZ)), color=None, entry=0.3, dry_from=0.95))
    P.add((lv * WINM * (1 - 0.85 * eat)).astype(np.float32), (150, 166, 130), 0.5)
    # 甩点：暗里几十个小颜料点
    wc.splatter(P, 40, (0, 700, W, H), (70, 56, 60), rmin=0.6, rmax=2.4, strength=0.8)
    wc.splatter(P, 18, (550, 250, W, 800), (80, 60, 90), rmin=0.6, rmax=2.0, strength=0.7)
isolated(_more_things)

def _lamp():
    top = pr((2.55, CY - 0.2, 3.2)); bot = pr((2.55, 1.85, 3.2)); w = px(0.17, (2.55, 1.85, 3.2))
    P.add(wc.stroke_mask([top, bot], 1.3, 1.1, taper=False), (40, 32, 30), 0.9)
    shade = [(bot[0] - w * 0.3, bot[1]), (bot[0] + w * 0.3, bot[1]), (bot[0] + w, bot[1] + w * 0.55), (bot[0] - w, bot[1] + w * 0.55)]
    wc.wash(P, shade, (46, 40, 40), strength=1.2, var=0.004, layers=5, edge=0.4)
    P.lift(wc.stroke_mask([(bot[0] - w * 0.5, bot[1] + w * 0.2), (bot[0] - w * 0.9, bot[1] + w * 0.5)], max(1, w * 0.06), None, taper=False) * 0.6, 1.0)
isolated(_lamp)

# ================= 第六步：前景两团深的切进来 =================
def _fg():
    # 右下：一块老地毯（她把原来那条长凳看成了地毯——那就认了，用笔触织出来，不是一张纸片）
    from PIL import ImageDraw
    cx, cz, rot, Lx, Lz = 3.05, 1.95, 9, 1.55, 0.95
    c, s_ = math.cos(math.radians(rot)), math.sin(math.radians(rot))
    def RW_(u, v):
        return (cx + u * c - v * s_, 0.004, cz + u * s_ + v * c)
    corners = [RW_(-Lx / 2, -Lz / 2), RW_(Lx / 2, -Lz / 2), RW_(Lx / 2, Lz / 2), RW_(-Lx / 2, Lz / 2)]
    poly = clip_poly(corners); rm = wc.poly_mask(poly)
    U = (Xmap - cx) * c + (Zmap - cz) * s_; V = -(Xmap - cx) * s_ + (Zmap - cz) * c
    P.lift(wc.blur(rm, 0.8) * 0.75, 1.0)
    # 底：茜红湿着铺，颜色不匀，暗处掺紫
    wc.wash(P, poly, (156, 62, 50), strength=0.95, var=0.008, layers=14, edge=0.35, granulate=0.35, mottle=0.5)
    P.add((rm * Ss(1.0, 0.0, (xf - 450) / 450)).astype(np.float32), (70, 46, 70), 0.0)
    wet_keep(rm * Ss(0.4, 0.7, wc.noise(40, octaves=3)), (92, 40, 56), 0.5, 8, 0.4, None)
    # 花边：一圈赭黄带，里外两道细线
    au, av = np.abs(U), np.abs(V)
    inside = (au < Lx / 2 - 0.05) & (av < Lz / 2 - 0.05)
    band = inside & ((au > Lx / 2 - 0.17) | (av > Lz / 2 - 0.17))
    band = wc.blur(band.astype(np.float32), 0.7) * rm
    P.add((band * Ss(0.3, 0.55, P.grain + 0.15)).astype(np.float32), (176, 126, 66), 0.7)
    for off in (0.05, 0.17, 0.21):
        ln = (np.abs(np.maximum(au - (Lx / 2 - off), av - (Lz / 2 - off))) < 0.008).astype(np.float32)
        P.add((wc.blur(ln, 0.5) * rm * Ss(0.3, 0.5, P.grain + 0.1)).astype(np.float32), (60, 34, 40), 0.7)
    # 中间三个菱形纹样：一圈靛蓝、一圈米白，里面一点芯
    for u0 in (-0.42, 0.0, 0.42):
        d = np.abs(U - u0) / 0.2 + np.abs(V) / 0.24
        ring1 = (np.abs(d - 0.85) < 0.12).astype(np.float32); ring2 = (np.abs(d - 0.55) < 0.09).astype(np.float32); core = (d < 0.22).astype(np.float32)
        P.add((wc.blur(ring1, 0.6) * rm * Ss(0.25, 0.5, P.grain + 0.1)).astype(np.float32), (58, 66, 108), 0.8)
        P.lift((wc.blur(ring2, 0.6) * rm * Ss(0.3, 0.55, P.grain + 0.1) * 0.45).astype(np.float32), 1.0)
        P.add((wc.blur(ring2, 0.6) * rm).astype(np.float32), (206, 180, 140), 0.3)
        P.add((wc.blur(core, 0.6) * rm).astype(np.float32), (176, 126, 66), 0.6)
    # 笔触：顺着经线一笔笔扫，深的浅的交错，干笔挂纸纹——织物的毛是笔触给的
    for col, st, n, wpx in (((76, 30, 38), 0.7, 320, 2.4), ((210, 130, 100), 0.0, 220, 2.0)):
        im = Image.new("L", (W * 2, H * 2), 0); dr = ImageDraw.Draw(im)
        for _ in range(n):
            u0 = random.uniform(-Lx / 2, Lx / 2); v0 = random.uniform(-Lz / 2, Lz / 2)
            L = random.uniform(0.08, 0.24)
            a2, b2 = pr(RW_(u0, v0)), pr(RW_(u0 + L, v0 + random.uniform(-0.01, 0.01)))
            dr.line([(a2[0] * 2, a2[1] * 2), (b2[0] * 2, b2[1] * 2)], fill=int(255 * random.uniform(0.5, 1.0)), width=int(wpx * 2 * random.uniform(0.7, 1.3)))
        m = np.asarray(im.resize((W, H), Image.BILINEAR), np.float32) / 255.0
        m = m * rm * Ss(0.35, 0.6, P.hstreak * 0.5 + P.grain * 0.5)
        if st:
            P.add(m.astype(np.float32), col, st)
        else:
            P.lift((m * 0.45).astype(np.float32), 1.0)
            P.add(m.astype(np.float32), (214, 140, 116), 0.25)
    # 两头流苏：一根根短的、米色，近的那头看得见
    im = Image.new("L", (W * 2, H * 2), 0); dr = ImageDraw.Draw(im)
    for uu in (-Lx / 2, Lx / 2):
        sg = 1 if uu > 0 else -1
        for v0 in np.linspace(-Lz / 2 + 0.03, Lz / 2 - 0.03, 22):
            a2 = pr(RW_(uu, v0)); b2 = pr(RW_(uu + sg * random.uniform(0.05, 0.09), v0 + random.uniform(-0.015, 0.015)))
            dr.line([(a2[0] * 2, a2[1] * 2), (b2[0] * 2, b2[1] * 2)], fill=220, width=3)
    fr = np.asarray(im.resize((W, H), Image.BILINEAR), np.float32) / 255.0
    P.lift((fr * 0.5).astype(np.float32), 1.0); P.add(fr.astype(np.float32), (196, 172, 130), 0.5)
    # 窗的光落到地毯上的那一截：亮、暖（地毯是平的，光照样落）
    lit = rm * FLb
    P.lift((lit * 0.6).astype(np.float32), 1.0); P.add(lit.astype(np.float32), (236, 196, 140), 0.35)
    # 地毯有厚度：近的那条边一线深的影
    e0, e1 = pr(corners[0]), pr(corners[1])
    P.add((wc.stroke_mask([e0, e1], 3, 3, taper=False, rough=0.3) * Ss(0.3, 0.5, P.grain + 0.1)).astype(np.float32), (36, 26, 28), 0.7)
isolated(_fg)


# ================= 一整块连通的暗（哥：程序要允许两件物体共用一片暗色，而不只是每件分别填色描边贴纹理） =================
def _shared_dark():
    # 哥 10/1：暗部合并要真的缩小两边的色差——只叠一层暗，桌腿和地板一起变深，差还在，边就还在。
    # 所以在选中的暗区里，把桌腿、椅腿、地面一起往同一个蓝紫暗色收拢（混合，不是叠加），差没了，边自己丢。
    cl = [(x + (CHAIR["cx"] - x) * 0.1, z + (CHAIR["cz"] - z) * 0.1) for (x, y, z) in SEAT[:4]]
    u = np.maximum.reduce([spot((2.55, 0.25, 3.35), 230, 110), spot((1.75, 0.12, 2.62), 100, 60),
                           0.7 * np.clip((xf - 620) / 260, 0, 1) * np.clip((yf - 760) / 160, 0, 1) * Ss(950, 880, yf)])
    zone = Ss(0.42, 0.58, 0.7 * u + 0.3 * wc.noise(30, octaves=3))
    keep = (1 - wc.blur(np.clip(FLb + TL + SL + WINM, 0, 1), 1.0)).astype(np.float32)
    zone = wc.blur((zone * keep).astype(np.float32), 3.0)          # 边界只在小范围里混
    Kt = wc.Paper.K((66, 58, 92))
    # 收拢：每个像素往目标暗色走一段，暗区中心走得多（差几乎没了），边上走得少
    m = np.clip(zone * 0.45, 0, 0.45)[..., None]                    # 只收一半：差变小（边丢），暗里的东西和笔触还在
    var_ = (0.85 + 0.3 * wc.noise(60, octaves=3))[..., None]
    P.D = P.D * (1 - m) + (Kt[None, None, :] * var_) * m
isolated(_shared_dark)

# ================= 最后才把「最值钱的几笔」补回来：椅背上沿、扶手转角、座框前沿、桌角、几个接地点 =================
def _key_strokes():
    cx, cz, rot = CHAIR["cx"], CHAIR["cz"], CHAIR["rot"]
    c, s_ = math.cos(math.radians(rot)), math.sin(math.radians(rot))
    W_ = lambda u, v, y: (cx + u * c - v * s_, y, cz + u * s_ + v * c)
    sc = F / depth_of(W_(0, 0, 0.5))
    # 椅背上沿：一笔深 + 顶上一线亮
    rail = [pr(W_(-0.22 + 0.44 * t, -0.25 - 0.07 * math.sin(math.pi * t), 0.86)) for t in np.linspace(0, 1, 8)]
    P.add(wc.stroke_mask(rail, 0.04 * sc, 0.035 * sc, taper=False, rough=0.1), (70, 52, 36), 0.9)
    top = wc.stroke_mask([(p[0], p[1] - 0.018 * sc) for p in rail], 0.012 * sc, 0.01 * sc, taper=False, rough=0.2)
    P.lift((top * Ss(0.3, 0.5, P.grain) * 0.8).astype(np.float32), 1.0); P.add(top, (236, 200, 128), 0.3)
    # 左扶手转角
    arm = [pr(W_(-0.24, -0.02, 0.65)), pr(W_(-0.23, 0.15, 0.6)), pr(W_(-0.2, 0.2, 0.42))]
    P.add(wc.stroke_mask(arm, 0.032 * sc, 0.03 * sc, taper=False, rough=0.1), (70, 52, 36), 0.85)
    # 座框前沿
    P.add(wc.stroke_mask([pr(W_(-0.21, 0.21, 0.42)), pr(W_(0.21, 0.21, 0.42))], 0.03 * sc, 0.028 * sc, taper=False, rough=0.1), (60, 44, 32), 0.9)
    # 接地点：前腿一短截深的落点（不画整条腿，眼睛自己补）
    for u_ in (-0.2, 0.2):
        a = pr(W_(u_, 0.2, 0.0)); b = pr(W_(u_, 0.2, 0.07))
        P.add(wc.stroke_mask([b, a], 0.04 * sc, 0.045 * sc, taper=False, rough=0.2), (40, 32, 34), 0.9)
    # 桌角：靠光那个角一笔硬的深 + 桌沿一线亮；左前腿一个接地点
    c0, c1 = TOPC[0], TOPC[1]
    lo = (c0[0], c0[1] - 0.16, c0[2])
    P.add(wc.stroke_mask([pr(c0), pr(lo)], px(0.03, c0), px(0.03, c0), taper=False, rough=0.1), (36, 28, 26), 1.0)
    P.lift((wc.stroke_mask([pr(c0), pr((c0[0] + (c1[0] - c0[0]) * 0.45, 0.8, c0[2] + (c1[2] - c0[2]) * 0.45))], max(1.5, px(0.012, c0)), None, taper=False, rough=0.2) * Ss(0.3, 0.5, P.grain)).astype(np.float32), 0.8)
    lx, lz = LEGS_T[0]
    a = pr((lx, 0.0, lz)); b = pr((lx, 0.08, lz))
    P.add(wc.stroke_mask([b, a], px(0.06, (lx, 0, lz)), px(0.065, (lx, 0, lz)), taper=False, rough=0.2), (36, 30, 32), 0.9)
isolated(_key_strokes)


DMAX = 2.0                     # 最深 ≈ 亮度 30：水彩的暗是透明的深，不是死黑
P.D = DMAX * np.tanh(P.D / DMAX)
grain = wc.grain_from_profile(os.path.join(os.path.dirname(HERE), "pigment_profile.npy"), seed=SEED)
img = P.render(paper_strength=0.65, pigment_tex=grain, tex_amount=0.05)
out = os.path.join(HERE, f"corner{TAG}.png"); img.save(out)
img.convert("L").resize((64, 82), Image.LANCZOS).resize((256, 328), Image.NEAREST).save(out.replace(".png", "_thumb.png"))
print("saved", out)
