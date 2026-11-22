# -*- coding: utf-8 -*-
"""标定棋盘几何，并直接写回包内的 config.py。

config.py 是整个项目唯一的参数入口，本脚本只改其中的 BG_REGION / ROWS / COLS
三行，其余内容原样保留。autoplay 与 tools.collect 都按这三行切格子，
所以标定错了它们会一起错。

用法（在仓库根目录执行）:
    python -m minesweeper_copilot.tools.calibrate --rows 30 --cols 64
    python -m minesweeper_copilot.tools.calibrate --rows 30 --cols 64 --no-write
    python -m minesweeper_copilot.tools.calibrate --snap   # 只存截图，自己量坐标

点击顺序: 先点棋盘「最左上一格的左上角」，再点「最右下一格的右下角」。
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


def snap(out_dir='screenshots'):
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, 'calib.png')
    shot = pyautogui.screenshot()
    shot.save(out)
    print('[calib] 已保存全屏截图: %s (%dx%d)' % (out, shot.width, shot.height))
    print('[calib] 用画图打开它，读出棋盘左上角 (x1,y1) 和右下角 (x2,y2)，'
          'BG_REGION = (x1, y1, x2-x1, y2-y1)')
    return out


def pick():
    import matplotlib.pyplot as plt

    shot = pyautogui.screenshot()
    img = np.array(shot)
    print('[calib] 已截图 %dx%d，请在弹窗里点两下（左上角、右下角）' % (shot.width, shot.height))
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.imshow(img)
    ax.set_title('click top-left corner, then bottom-right corner')
    ax.set_axis_off()
    pts = plt.ginput(2, timeout=0)
    plt.close(fig)
    if len(pts) < 2:
        raise SystemExit('[calib] 没有取到两个点，退出（没有图形界面时请用 --snap）')
    (x1, y1), (x2, y2) = pts
    return int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--rows', type=int, default=None)
    ap.add_argument('--cols', type=int, default=None)
    ap.add_argument('--snap', action='store_true')
    ap.add_argument('--no-write', action='store_true', help='只打印，不修改 config.py')
    args = ap.parse_args()

    if args.snap:
        snap()
        return

    x1, y1, x2, y2 = pick()
    w, h = x2 - x1, y2 - y1
    print()
    print('BG_REGION = (%d, %d, %d, %d)' % (x1, y1, w, h))
    if args.rows and args.cols:
        print('ROWS = %d, COLS = %d' % (args.rows, args.cols))
        print('=> 单格 %.2f x %.2f px（两个数应接近，差太多说明点到别的地方了）'
              % (h / args.rows, w / args.cols))
    else:
        print('（补 --rows / --cols 再跑一次，可校验格子尺寸是否接近正方形）')

    if args.no_write:
        print('[calib] --no-write：没有修改 config.py')
        return
    write_config((x1, y1, w, h), args.rows, args.cols)
    print('[calib] 完成。如需撤销，用 git checkout minesweeper_copilot/config.py')


if __name__ == '__main__':
    main()
