# -*- coding: utf-8 -*-
"""标定 config.py 里的 BG_REGION / ROWS / COLS。

用法:
    python calibrate.py                        # 拍全屏截图，在弹窗里点两下
    python calibrate.py --rows 25 --cols 53    # 顺便校验格子是不是正方形
    python calibrate.py --snap                 # 只存截图，自己用画图量坐标

点击顺序: 先点棋盘「最左上一格的左上角」，再点「最右下一格的右下角」。
点击过程中不要移动/缩放游戏窗口，也不要切走页面。
"""

import argparse
import os

import numpy as np
import pyautogui


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--rows', type=int, default=None)
    ap.add_argument('--cols', type=int, default=None)
    ap.add_argument('--snap', action='store_true')
    args = ap.parse_args()

    if args.snap:
        snap()
        return

    x1, y1, x2, y2 = pick()
    w, h = x2 - x1, y2 - y1
    print()
    print('把这行贴进 config.py：')
    print('    BG_REGION = (%d, %d, %d, %d)' % (x1, y1, w, h))
    if args.rows and args.cols:
        print('    ROWS = %d' % args.rows)
        print('    COLS = %d' % args.cols)
        print('=> 单格 %.2f x %.2f px（两个数应接近，差太多说明点到别的地方了）'
              % (h / args.rows, w / args.cols))
    else:
        print('（补 --rows / --cols 再跑一次，可校验格子尺寸是否接近正方形）')


if __name__ == '__main__':
    main()
