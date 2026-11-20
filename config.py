# -*- coding: utf-8 -*-
"""游戏配置参数，也是整个仓库唯一的参数入口。

其他脚本（saolei.py、collect.py、train_img.py、minesolver.py）都从这里读，
改参数只改这个文件，不要去改各脚本里的常量。

注意：下面所有数值都和「你的屏幕 + 浏览器窗口位置 + 页面缩放比例」强绑定。
换机器、移动/缩放窗口、改浏览器缩放之后，必须重新标定（见 calibrate.py 与 README）。
"""

import os

# ------------------------------------------------------------------ 棋盘区域
# BG_REGION: 游戏棋盘在屏幕上的像素区域 (x, y, w, h)
#   x, y : 棋盘左上角在屏幕上的坐标（要贴着第一格的左上角，不要把窗口边框算进去）
#   w, h : 棋盘整体的宽、高（只算棋盘，不含标题栏、工具栏、状态栏）
# 如果习惯用 (x1, y1, x2, y2) 表示，就写成 (x1, y1, x2 - x1, y2 - y1)
BG_REGION = (18, 176, 2504, 1175)

# 棋盘行数、列数（必须和游戏里设置的完全一致，否则格子会被切错）
ROWS = 30
COLS = 64

# 整盘雷数。只给 minesolver.py 算概率用：推不动时它会算出「点哪一格最安全」。
# 填你开局时设置的雷数；填 0 表示不知道，程序按当前已翻开的数字估算（会不准）。
TOTAL_MINES = 0

# 单个格子的像素尺寸（由上面三个值推导，只读，不要手改）
row_size = BG_REGION[3] / ROWS
cols_size = BG_REGION[2] / COLS

# 自检：横竖格子尺寸必须几乎相等，否则说明 BG_REGION / ROWS / COLS 有一个填错了
if abs(row_size - cols_size) > 0.5:
    print(
        '[config] 警告：格子尺寸不一致（行高 %.2fpx vs 列宽 %.2fpx），'
        '请重新检查 BG_REGION / ROWS / COLS' % (row_size, cols_size)
    )

# ------------------------------------------------------------------ 编码表
# 编码和解码神经网络的输出。
# -1 旗子 / -2 未翻开 / 9 翻开的地雷 / 10 爆炸的红雷（踩中的那颗）/ 0-6 周围八格的地雷数
# 不含 7、8：实战中 7 需要周围八格里有七颗雷、8 需要八颗全雷，出现概率约 1e-5 与 1e-10，
# 且 7 与普通地雷同为纯黑，排色分不开。真遇到时会被识别成相邻的数字。
net_encoder = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 9: 7, 10: 10, -1: 8, -2: 9}
net_decoder = {v: k for k, v in net_encoder.items()}

# ------------------------------------------------------------------ 路径
# 全部基于本文件所在目录推导，脚本从任何工作目录启动都可以
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 训练集：每个类别一个子文件夹，文件夹名就是 net_encoder 的键（-2、-1、0..9）
TRAIN_PATH = os.path.join(BASE_DIR, 'imgs')

# collect.py 采集新样本的目录（按数字颜色打标签，见 README）。
# 筛完并入 imgs/ 后再重训；samples/ 不入库。
SAMPLES_PATH = os.path.join(BASE_DIR, 'samples', 'auto')

# 按 R 键记录新样本的目录（同样按类别分子文件夹）
RECORD_PATH = os.path.join(BASE_DIR, 'dir')
