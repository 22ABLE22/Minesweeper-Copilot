# -*- coding: utf-8 -*-
"""从真实棋盘采集训练样本，标签来自数字颜色，不依赖模型预测。

为什么用颜色打标签：游戏里数字的颜色是固定的（1 蓝、2 绿、3 红、4 靛蓝、
5 暗红、6 青、7 黑），所以「青色的 6」一定能拿到标签 6，不会出现
「模型把它认成 4，于是样本被存进 4 的文件夹」这种把错误标签固化下来的问题。

只采集数字格（1..7）：空块、未翻开、旗子和地雷用颜色区分不可靠，
这些类别继续用 imgs/ 里的样本。

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
    7: (0, 0, 0),
}


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
    """按笔画颜色判断数字。无法确定时返回 None（宁可不收，也不要错标签）。"""
    a = np.asarray(cell, dtype=np.int16)
    h, w = a.shape[:2]
    # 去掉四周的可能残留的格子边框
    m = max(2, int(round(min(h, w) * 0.18)))
    b = a[m:h - m, m:w - m]
    if b.size == 0:
        return None
    sat = b.max(axis=2) - b.min(axis=2)
    lum = b.sum(axis=2) / 3.0
    ink = (sat > 90) | (lum < 60)
    if ink.sum() < min_ink:
        return None
    px = b[ink]
    col = np.median(px, axis=0)
    dist = {k: float(np.linalg.norm(col - np.array(v))) for k, v in PALETTE.items()}
    order = sorted(dist.items(), key=lambda kv: kv[1])
    best, dbest = order[0]
    dsecond = order[1][1]
    if dbest > max_dist or dsecond - dbest < margin:
        return None
    return best


def collect_board(region_img=None, region=BG_REGION, out=SAMPLES_PATH,
                  save=True, verbose=True, sheet=False, min_cells=8):
    """切格 -> 颜色判标签 -> 存图。返回 Counter({数字: 张数})。"""
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
            print('[collect] 只找到 %d 个数字格（少於 %d），看起来不是展开的棋盘，不保存'
                  % (len(found), min_cells))
        return counts

    if save:
        for r, c, digit, cell in found:
            d = os.path.join(out, str(digit))
            os.makedirs(d, exist_ok=True)
            cell.save(os.path.join(d, '%d_%d_%d_%d.png' % (digit, stamp, r, c)))
    samples = [(digit, cell) for r, c, digit, cell in found[:60]]
    if verbose:
        print('[collect] 数字格 %d 张，分布 %s' % (
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
