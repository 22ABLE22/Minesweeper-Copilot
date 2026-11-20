# -*- coding: utf-8 -*-
"""从真实棋盘采集训练样本，标签来自数字颜色，不依赖模型预测。

为什么用颜色打标签：游戏里数字的颜色是固定的（3 红、4 靛蓝、5 暗红、6 青），
所以「青色的 6」一定能拿到标签 6，不会出现「模型把它认成 4，于是样本被存进
4 的文件夹」这种把错误标签固化下来的问题。

只采集 COLLECT_CLASSES 里列出的类别（5、6 和爆炸的红雷 10）：
这三类在真实棋盘上稀有，靠自然出现攒不够；3、4 打几局就够 400 张，1、2 更多，
再收只是重复占用磁盘。7 与普通地雷同为纯黑，排色分不开，实战也几乎不出现，
类别表里本就没有它。空块、未翻开、旗子、普通地雷这些形状类继续用 imgs/ 的样本。

用法:
    python collect.py                       # 抓一次当前屏幕
    python collect.py --from-file board.png # 用已有截图（整块棋盘区域）
    python collect.py --sheet               # 顺便输出一张缩略图拼板供肉眼抽查
"""

import argparse
import os
from collections import Counter
from time import time

import numpy as np
from PIL import Image

from config import BG_REGION, COLS, ROWS, SAMPLES_PATH, cols_size, row_size

# 经典扫雷的数字配色
PALETTE = {
    1: (0, 0, 255),
    2: (0, 128, 0),
    3: (255, 0, 0),
    4: (0, 0, 128),
    5: (128, 0, 0),
    6: (0, 128, 128),
}

# 只采集这些类别：5、6 在真实棋盘上稀有且容易认错（6 曾被认成 4），
# 加上爆炸的红雷 10（出现频率极低，必须专门攒）。
# 3、4 打几局就有 400 张以上，再收只是重复；1、2 更多；
# 7 与地雷同为纯黑，排色区分不开，且实战几乎不出现。
COLLECT_CLASSES = (5, 6, 10)

# 笔画覆盖率的硬下限。旗子的红三角只占 0.31~0.35，是所有类里最低的，
# 而真正的数字都在 0.43 以上，所以这条线能把旗子挡在外面（旗本来会把红三角
# 送去 3 或 5）。0.40 相对旗子留了 14% 余量，相对 4 也留了 7%。
MIN_INK_COVER = 0.40

# 暗红底（棋盘的底色 / 5 的笔画）占比。5 的笔画本身就是暗红，覆盖率 0.41 以上；
# 旗子的底座加起来不到 0.05。这条用来再兜一次底。
MIN_DARK_RED = 0.20

# 爆炸的红雷：踩中的那颗雷是整格红底 + 黑色雷。红色像素占比要够大，
# 同时红 + 黑几乎铺满整格；红色的 3 只有笔画是红的，红+黑远达不到这个比例。
BOOM_LABEL = 10
BOOM_RED = 0.12
BOOM_COVER = 0.92


def grab_board(region=BG_REGION):
    """按 config 的几何抓取棋盘区域，返回 RGB ndarray。"""
    import pyautogui

    shot = pyautogui.screenshot(region=region)
    return np.array(shot.convert('RGB'))


def split_cells(region_img, rows=ROWS, cols=COLS):
    """把棋盘区域按 (行数, 列数) 切成格子，返回 rows x cols 的 PIL 图像。"""
    h, w = region_img.shape[:2]
    ch, cw = h / float(rows), w / float(cols)
    cells = []
    for r in range(rows):
        line = []
        for c in range(cols):
            u, l = int(round(r * ch)), int(round(c * cw))
            d, rr = int(round(u + ch)), int(round(l + cw))
            line.append(Image.fromarray(region_img[u:d, l:rr]))
        cells.append(line)
    return cells


def label_by_color(cell, max_dist=48.0, margin=22.0, min_ink=14):
    """按颜色判断格子类别，只返回 COLLECT_CLASSES 里的值。其余一律 None。"""
    a = np.asarray(cell, dtype=np.int16)
    h, w = a.shape[:2]
    # 去掉四周的可能残留的格子边框
    m = max(2, int(round(min(h, w) * 0.18)))
    b = a[m:h - m, m:w - m]
    if b.size == 0:
        return None
    red = (b[:, :, 0] > 140) & (b[:, :, 1] < 110) & (b[:, :, 2] < 110)
    dark = b.sum(axis=2) / 3.0 < 70
    if red.mean() > BOOM_RED and (red | dark).mean() > BOOM_COVER:
        return BOOM_LABEL
    sat = b.max(axis=2) - b.min(axis=2)
    lum = b.sum(axis=2) / 3.0
    ink = (sat > 90) | (lum < 60)
    if ink.sum() < min_ink or ink.mean() < MIN_INK_COVER:
        return None
    px = b[ink]
    col = np.median(px, axis=0)
    dist = {k: float(np.linalg.norm(col - np.array(v))) for k, v in PALETTE.items()}
    order = sorted(dist.items(), key=lambda kv: kv[1])
    best, dbest = order[0]
    dsecond = order[1][1]
    if dbest > max_dist or dsecond - dbest < margin:
        return None
    if best == 5:
        dark_red = (b[:, :, 0] > 40) & (b[:, :, 0] < 130) & (b[:, :, 1] < 60) & (b[:, :, 2] < 60)
        if dark_red.mean() < MIN_DARK_RED:
            return None
    return best if best in COLLECT_CLASSES else None


def next_index(folder, digit):
    """返回该类别目录里下一个可用编号（已有 1 编号则从 2 起）。"""
    best = 0
    if os.path.isdir(folder):
        prefix = '%d_' % digit
        for name in os.listdir(folder):
            if name.startswith(prefix) and name.lower().endswith('.png'):
                num = name[len(prefix):-4]
                if num.isdigit():
                    best = max(best, int(num))
    return best + 1


def collect_board(region_img=None, region=BG_REGION, out=SAMPLES_PATH,
                  save=True, verbose=True, sheet=False, min_cells=8):
    """切格 -> 颜色判标签 -> 存图。返回 Counter({标签: 张数})。"""
    if region_img is None:
        region_img = grab_board(region)
    cells = split_cells(region_img)
    stamp = int(time())
    found = []
    for r in range(ROWS):
        for c in range(COLS):
            cell = cells[r][c]
            digit = label_by_color(cell)
            if digit is not None:
                found.append((r, c, digit, cell))

    counts = Counter(item[2] for item in found)
    # 棋盘还没展开、或者游戏窗口不在前台时，只会零星命中几个颜色块，
    # 这种「不像棋盘」的画面一律不存，避免把浏览器界面的杂色当样本。
    if len(found) < min_cells:
        if verbose:
            print('[collect] 只找到 %d 个可标注格子（少於 %d），看起来不是展开的棋盘，不保存'
                  % (len(found), min_cells))
        return counts

    if save:
        counters = {}
        for r, c, digit, cell in found:
            d = os.path.join(out, str(digit))
            os.makedirs(d, exist_ok=True)
            # 文件按 <类别>_<序号>.png 连续编号，从目录里已有的最大编号往后接，
            # 这样人工筛选、重排之后也不会撞名
            if digit not in counters:
                counters[digit] = next_index(d, digit)
            counters[digit] += 1
            cell.save(os.path.join(d, '%d_%04d.png' % (digit, counters[digit])))
    samples = [(digit, cell) for r, c, digit, cell in found[:60]]
    if verbose:
        print('[collect] 命中 %d 张，分布 %s' % (
            sum(counts.values()), dict(sorted(counts.items()))))
        if save:
            print('[collect] 已保存到 %s/<数字>/' % out)
    if sheet and samples:
        from PIL import ImageDraw

        cols = 10
        cw = 44
        rows_n = (len(samples) + cols - 1) // cols
        canvas = Image.new('RGB', (cols * cw, rows_n * (cw + 14)), 'white')
        draw = ImageDraw.Draw(canvas)
        for i, (digit, cell) in enumerate(samples):
            x, y = (i % cols) * cw, (i // cols) * (cw + 14)
            canvas.paste(cell.resize((cw - 4, cw - 4), Image.NEAREST), (x + 2, y + 2))
            draw.text((x + 2, y + cw - 2), str(digit), fill='black')
        path = os.path.join(os.path.dirname(out), 'collect_sheet.png')
        canvas.save(path)
        print('[collect] 抽查图: %s' % path)
    return counts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--from-file', default=None, help='用已有截图代替实时抓屏')
    ap.add_argument('--out', default=SAMPLES_PATH)
    ap.add_argument('--sheet', action='store_true', help='输出一张缩略图拼板')
    ap.add_argument('--dry-run', action='store_true', help='只统计不存图')
    ap.add_argument('--min-cells', type=int, default=8, help='少于这么多数字格就当作不是棋盘，不保存')
    args = ap.parse_args()

    if args.from_file:
        region_img = np.array(Image.open(args.from_file).convert('RGB'))
        print('[collect] 读入 %s %dx%d' % (args.from_file, region_img.shape[1], region_img.shape[0]))
        expected = (BG_REGION[3], BG_REGION[2])
        if region_img.shape[:2] != expected:
            print('[collect] 注意：图片尺寸 %s 与 config 的棋盘区域 %s 不一致，'
                  '整屏截图请先裁成棋盘区域' % (region_img.shape[:2], expected))
        counts = collect_board(region_img, save=not args.dry_run, sheet=args.sheet,
                               min_cells=args.min_cells)
    else:
        print('[collect] 棋盘区域 %s，%d 行 x %d 列，单格 %.2f x %.2f px'
              % (BG_REGION, ROWS, COLS, cols_size, row_size))
        counts = collect_board(save=not args.dry_run, sheet=args.sheet, min_cells=args.min_cells)
    print('[collect] 合计 %d 张' % sum(counts.values()))


if __name__ == '__main__':
    main()
