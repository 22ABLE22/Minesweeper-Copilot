# -*- coding: utf-8 -*-
"""标定棋盘几何，并直接写回包内的 config.py。

config.py 是整个项目唯一的参数入口，本脚本只改其中的 BG_REGION / ROWS / COLS
三行，其余内容原样保留。autoplay 与 tools.collect 都按这三行切格子，
所以标定错了它们会一起错。

用法（在仓库根目录执行）:
    python -m minesweeper_copilot.tools.calibrate
    python -m minesweeper_copilot.tools.calibrate --no-preview      # 不弹网格预览
    python -m minesweeper_copilot.tools.calibrate --rows 30 --cols 64
    python -m minesweeper_copilot.tools.calibrate --snap            # 只存截图，自己量坐标

流程:
    1. 弹出一张全屏截图，点两个点：棋盘「最左上一格的左上角」、再点
       「最右下一格的右下角」，两点围出的矩形就是 BG_REGION；
    2. 程序在这个矩形里用自相关找出格子周期，自动推算行列数；
    3. 把推算出的网格画在截图上再弹一次，目视确认：
       网格线正好压在格子边界上就回车确认，错位就输入真实的行列数；
    4. 确认后把 BG_REGION / ROWS / COLS 三行写回 config.py。

点击过程中不要移动/缩放游戏窗口，也不要切走页面。
"""

import argparse
import os
import re

import numpy as np
import pyautogui

# 本文件在 <根>/minesweeper_copilot/tools/ 下，config.py 在它的上一级
CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'config.py')

# 单格边长的合理范围（像素）。1x1 的格子小于 8px 屏幕截图就分不清了，
# 400px 对应「超大格子」的极端情况，超过它通常是识别到了子网格。
MIN_CELL = 8
MAX_CELL = 400


def snap(out_dir='screenshots'):
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, 'calib.png')
    shot = pyautogui.screenshot()
    shot.save(out)
    print('[calib] 已保存全屏截图: %s (%dx%d)' % (out, shot.width, shot.height))
    print('[calib] 用画图打开它，读出棋盘左上角 (x1,y1) 和右下角 (x2,y2)，'
          'BG_REGION = (x1, y1, x2-x1, y2-y1)')
    return out


def pick(img):
    """在全屏截图里点两下，返回 (x1, y1, x2, y2)，屏幕像素坐标。"""
    import matplotlib.pyplot as plt

    print('[calib] 已截图，请在弹窗里点两下（左上角、右下角），然后关闭窗口')
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.imshow(img)
    ax.set_title('click top-left corner, then bottom-right corner')
    ax.set_axis_off()
    pts = plt.ginput(2, timeout=0)
    plt.close(fig)
    if len(pts) < 2:
        raise SystemExit('[calib] 没有取到两个点，退出（没有图形界面时请用 --snap）')
    (x1, y1), (x2, y2) = pts
    x1, x2 = sorted((int(round(x1)), int(round(x2))))
    y1, y2 = sorted((int(round(y1)), int(round(y2))))
    return x1, y1, x2, y2


# ---------------------------------------------------------------- 网格检测

def _edge_profile(gray, axis):
    """沿某个轴求平均梯度：横轴 axis=1 得到「每列有多像一条竖线」。"""
    diff = np.abs(np.diff(gray, axis=axis))
    return diff.mean(axis=0 if axis == 1 else 1)


def _autocorr(profile):
    """归一化自相关，ac[0] 恒为 1，ac[k] 表示间隔 k 像素的相似度。"""
    x = np.asarray(profile, np.float64)
    x = x - x.mean()
    n = x.size
    if n < 8:
        return None
    spec = np.fft.rfft(x, 2 * n)
    ac = np.fft.irfft(spec * np.conj(spec))[:n]
    return ac / (ac[0] + 1e-12)


def _period(profile, min_cell=MIN_CELL, max_cell=MAX_CELL, keep=0.90):
    """从梯度曲线里找格子周期（像素，含亚像素修正）。

    格子边缘会在自相关上每隔一个单格边长出现一次峰，而 2 倍、3 倍周期同样
    是峰（甚至更高）。所以不能直接取最大值，而是取「所有高于 keep*峰值 的
    局部极大里最小的那个 lag」——真正的单格周期一定是最小的那个谐波。
    """
    ac = _autocorr(profile)
    if ac is None:
        return None
    n = ac.size
    lo = max(3, int(min_cell))
    hi = int(min(max_cell, n // 2))
    if hi <= lo + 2:
        return None
    seg = ac[lo:hi]
    best = float(seg.max())
    if best <= 0.05:
        return None
    cands = []
    for i in range(1, seg.size - 1):
        if seg[i] >= keep * best and seg[i] >= seg[i - 1] and seg[i] >= seg[i + 1]:
            cands.append(lo + i)
    if not cands:
        cands = [lo + int(np.argmax(seg))]
    lag = min(cands)
    # 抛物线插值，把周期精确到亚像素，否则 60 多列时误差会累积到一格以上
    if 0 < lag < n - 1:
        y0, y1, y2 = ac[lag - 1], ac[lag], ac[lag + 1]
        den = y0 - 2 * y1 + y2
        off = 0.5 * (y0 - y2) / den if abs(den) > 1e-9 else 0.0
        off = max(-0.5, min(0.5, off))
    else:
        off = 0.0
    return lag + off, float(ac[lag])


def detect_grid(img, region, min_cell=MIN_CELL, max_cell=MAX_CELL):
    """在 region 里自动推算 (rows, cols, cell_w, cell_h)；失败返回 None。"""
    x, y, w, h = region
    if w < 2 * min_cell or h < 2 * min_cell:
        return None
    crop = np.asarray(img)[y:y + h, x:x + w]
    gray = crop.astype(np.float32)
    if gray.ndim == 3:
        gray = gray.mean(axis=2)

    px = _period(_edge_profile(gray, 1), min_cell, max_cell)
    py = _period(_edge_profile(gray, 0), min_cell, max_cell)
    if px is None or py is None:
        return None

    cols = int(round(w / px[0]))
    rows = int(round(h / py[0]))
    if rows < 2 or cols < 2:
        return None
    if not (2 <= cols <= 1000 and 2 <= rows <= 1000):
        return None
    # 用「边长 / 格数」反推周期：rect 是用户点的外框，本身就等于 格数*周期，
    # 这样得到的周期没有累计误差，顺带校验了自相关的结果没偏。
    cw, ch = w / float(cols), h / float(rows)
    if abs(cw - px[0]) / px[0] > 0.05 or abs(ch - py[0]) / py[0] > 0.05:
        return None
    # 扫雷的格子一定是正方形。横竖算出来的边长差太多，说明这块区域根本不是棋盘
    # （比如标定时点到了别的窗口），或者周期找成了某条花纹的周期，宁可退回手填。
    if abs(cw - ch) / max(cw, ch) > 0.06:
        return None
    return rows, cols, cw, ch


def grid_quality(img, region, rows, cols):
    """返回 (seam, center)：网格线上的平均梯度 / 格心的平均梯度。

    网格对齐时前者明显更大。只用来自检，不参与决策。
    """
    x, y, w, h = region
    crop = np.asarray(img)[y:y + h, x:x + w].astype(np.float32)
    if crop.ndim == 3:
        crop = crop.mean(axis=2)
    out = []
    for axis, n, size in ((1, cols, w), (0, rows, h)):
        prof = _edge_profile(crop, axis)
        if prof.size < 4:
            return None
        step = size / float(n)
        idx = np.arange(prof.size)
        seam = np.abs(((idx + 0.5) / step) - np.round((idx + 0.5) / step))
        on_seam = seam < 0.15
        on_center = np.abs(seam - 0.5) < 0.15
        if not on_seam.any() or not on_center.any():
            return None
        out.append(float(prof[on_seam].mean() / (prof[on_center].mean() + 1e-6)))
    return out[0], out[1]


def preview(img, region, rows, cols, title=None):
    """把网格画在截图上弹出来，给人目视确认。"""
    import matplotlib.pyplot as plt

    x, y, w, h = region
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.imshow(img)
    for i in range(cols + 1):
        gx = x + i * w / float(cols)
        ax.plot([gx, gx], [y, y + h], color='red', lw=0.6, alpha=0.85)
    for j in range(rows + 1):
        gy = y + j * h / float(rows)
        ax.plot([x, x + w], [gy, gy], color='red', lw=0.6, alpha=0.85)
    ax.add_patch(plt.Rectangle((x, y), w, h, fill=False, ec='yellow', lw=1.4))
    ax.set_xlim(max(0, x - w * 0.15), min(img.shape[1], x + w * 1.15))
    ax.set_ylim(min(img.shape[0], y + h * 1.15), max(0, y - h * 0.15))
    ax.set_title(title or ('%d rows x %d cols  --  grid on the cell borders?' % (rows, cols)))
    ax.set_axis_off()
    plt.tight_layout()
    plt.show()
    plt.close(fig)


# ---------------------------------------------------------------- 写回配置

def write_config(region, rows=None, cols=None, path=CONFIG_PATH):
    """把标定结果写回 config.py，返回被改写后的行，方便核对。"""
    with open(path, 'r', encoding='utf-8', newline='') as f:
        text = f.read()

    wx, wy, ww, wh = region
    new_text, changed = text, []

    repl = 'BG_REGION = (%d, %d, %d, %d)' % (wx, wy, ww, wh)
    new_text, n = re.subn(r'^BG_REGION\s*=\s*\([^)]*\)', repl, new_text, count=1, flags=re.M)
    if n:
        changed.append(repl)

    for name, value in (('ROWS', rows), ('COLS', cols)):
        if value is None:
            continue
        line = '%s = %d' % (name, value)
        new_text, n = re.subn(r'^%s\s*=\s*\d+' % name, line, new_text, count=1, flags=re.M)
        if n:
            changed.append(line)

    if not changed:
        print('[calib] 没有找到可改写的行，请手动粘贴到 config.py')
        return []

    with open(path, 'w', encoding='utf-8', newline='') as f:
        f.write(new_text)
    print('[calib] 已写入 %s：' % os.path.basename(path))
    for line in changed:
        print('    ' + line)
    return changed


def _report(region, rows, cols, quality=None):
    x, y, w, h = region
    print()
    print('BG_REGION = (%d, %d, %d, %d)' % region)
    print('ROWS = %d, COLS = %d' % (rows, cols))
    print('=> 单格 %.2f x %.2f px（两个数应接近，差太多说明点到别的地方了）'
          % (h / float(rows), w / float(cols)))
    if quality:
        print('   网格贴合度 %.2f / %.2f（网格线上的梯度 ÷ 格心的梯度，越大说明网格线'
              '越压在格子边界上；软件渲染的棋盘通常也在 1.2 以上）' % quality)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--rows', type=int, default=None)
    ap.add_argument('--cols', type=int, default=None)
    ap.add_argument('--snap', action='store_true')
    ap.add_argument('--no-write', action='store_true', help='只打印，不修改 config.py')
    ap.add_argument('--no-preview', action='store_true', help='不弹网格预览，直接采信推算结果')
    ap.add_argument('--min-cell', type=int, default=MIN_CELL)
    ap.add_argument('--max-cell', type=int, default=MAX_CELL)
    args = ap.parse_args()

    if args.snap:
        snap()
        return

    shot = pyautogui.screenshot()
    img = np.array(shot)
    print('[calib] 已截图 %dx%d' % (shot.width, shot.height))
    x1, y1, x2, y2 = pick(img)
    region = (x1, y1, x2 - x1, y2 - y1)
    if region[2] < 2 * args.min_cell or region[3] < 2 * args.min_cell:
        raise SystemExit('[calib] 两点围出的区域太小（%dx%d），请重新标定'
                         % (region[2], region[3]))

    rows, cols = args.rows, args.cols
    if not (rows and cols):
        guess = detect_grid(img, region, args.min_cell, args.max_cell)
        if guess:
            rows, cols = guess[0], guess[1]
            print('[calib] 自动推算：%d 行 x %d 列（单格 %.2f x %.2f px）'
                  % (rows, cols, guess[2], guess[3]))
        else:
            print('[calib] 没能自动推算出网格，请先目视数一下行列数再输入')
            rows, cols = 0, 0

    while True:
        if rows and cols:
            quality = grid_quality(img, region, rows, cols)
            _report(region, rows, cols, quality)
            if not args.no_preview:
                preview(img, region, rows, cols)
        else:
            print()
            print('BG_REGION = (%d, %d, %d, %d)' % region)

        try:
            ans = input('\n回车确认%s / 输入 "行 列" 修正 / r 重新取点 / q 退出: '
                        % ('（%d x %d）' % (rows, cols) if rows and cols else '')).strip()
        except EOFError:
            ans = 'q'

        low = ans.lower()
        if low == 'q':
            print('[calib] 已取消，config.py 没有改动')
            return
        if low == 'r':
            x1, y1, x2, y2 = pick(img)
            region = (x1, y1, x2 - x1, y2 - y1)
            guess = detect_grid(img, region, args.min_cell, args.max_cell)
            rows, cols = (guess[0], guess[1]) if guess else (0, 0)
            continue
        if not ans:
            if rows and cols:
                break
            print('[calib] 还不知道行列数，请输入两个数字，例如 30 64')
            continue
        parts = ans.replace(',', ' ').replace('x', ' ').split()
        if len(parts) == 2 and all(p.isdigit() for p in parts):
            rows, cols = int(parts[0]), int(parts[1])
            continue
        print('[calib] 看不懂 "%s"，请输入两个数字，例如 30 64' % ans)

    print()
    if args.no_write:
        print('[calib] --no-write：没有修改 config.py')
        return
    write_config(region, rows, cols)
    print('[calib] 完成。如需撤销，用 git checkout minesweeper_copilot/config.py')


if __name__ == '__main__':
    main()
