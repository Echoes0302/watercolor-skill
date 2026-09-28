# Watercolor Skill

一个用 Python 从零叠透明颜料的城市水彩 skill。它画的是纸上的光、湿度、颜料沉积、干笔与留白，不调用图像生成模型，也不是给照片套滤镜。

![雨后城市街景](final_0928.png)

## 它会什么

- 先用四五块亮、中、暗组织画面，再让楼、窗、人和车从色块里长出来。
- 模拟冷压纸起伏、透明颜料吸光、边缘沉积、颗粒、湿画法与干笔。
- 处理逆光、湿路倒影、透视街景、人物和树影。
- 附带窄街模式：一整块楼影铺过街面、爬上对面墙，再由横街漏进一道有出处的光。

![窄街光影](final_0928_narrow.png)

## 安装为 skill

Claude Code：

```bash
git clone https://github.com/Echoes0302/watercolor-skill.git ~/.claude/skills/watercolor
```

Codex：

```bash
git clone https://github.com/Echoes0302/watercolor-skill.git ~/.codex/skills/watercolor
```

安装后可以直接说“画一张雨后城市水彩”“用水彩画逆光街景”，或者让 agent 先运行范例熟悉画法。

## 自己运行范例

需要 Python 3.10+：

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python example.py
```

其他模式：

```bash
.venv/bin/python example.py 3 values  # 只画第一遍大块和 64px 亮暗缩略图
.venv/bin/python example.py 3 narrow  # 窄街、墙面投影和横街漏光
```

默认画布是 900×1150。完整范例通常十几秒生成，速度取决于机器。

## 文件

- `SKILL.md`：完整构图方法、光影判断、翻车记录和修改原则。
- `watercolor_lib.py`：纸张、颜料、湿画、干笔、留白、噪声和纹理引擎。
- `example.py`：一张完整城市街景从大块到细节的可运行范例。
- `pigment_profile.npy`：一维径向功率谱，用随机相位合成颜料颗粒；不包含参考画面。
- `wrong_0928_narrow_mask.png`：旧窄街投影的反例，供对照判断。

由 Clavis 和乌桕一起磨出来；Vesper 参与了画面组织、缩略验收、纸纹吸收和颗粒方法的讨论。

