import json
import logging
import os
import sys

# TensorFlow 的启动噪音（"GPU support is not available"、oneDNN/AVX 编译提示）
# 都是无害的，而且必须在 import keras / tensorflow 之前生效：
#   TF_CPP_MIN_LOG_LEVEL=2        屏蔽 C++ 侧的 INFO / WARNING，保留 ERROR
#   logging ... setLevel(ERROR)   屏蔽 Python 侧的 Warning（GPU 那条就是它）
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
logging.getLogger('tensorflow').setLevel(logging.ERROR)

import matplotlib.pyplot as plt
import numpy as np
import pyautogui
from PIL import Image
from keras.models import load_model
from multiprocess import Pool
from pynput.keyboard import KeyCode, Listener

# P1 颜色复核用：label_by_color 只按笔画颜色判数字，与 CNN 完全是两条独立路径
from collect import PALETTE as DIGIT_COLORS, label_by_color, next_index
from config import *
from minesolver import build_constraints, hint as mine_hint, solve_prob as mine_prob
from time import time

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'recognize_new.keras')

# 爆炸的红雷：踩中的那颗雷在界面上是红底的，出现它说明本局已经结束
BOOM_CLASS = 10

model = load_model(MODEL_PATH)  # 已训练的CNN（Keras 3 的 .keras 格式）

# 模型必须与 config 的类别表配套。只比类数不够：删掉一个类又加回另一个，
# 类数相同但下标全错位，模型会静默地给出错误结果。train_img.py 会把训练时的
# 类别表写成同名的 .classes.json，这里逐项比对。
MODEL_CLASSES_PATH = os.path.join(BASE_DIR, 'recognize_new.classes.json')


def check_model_classes():
    """模型类别表与 config 不一致时返回一句提示，一致或无从判断时返回 None。"""
    if int(model.layers[-1].units) != len(net_encoder):
        return ('当前模型输出 %d 类，config 里有 %d 类' % (
            int(model.layers[-1].units), len(net_encoder)))
    if not os.path.exists(MODEL_CLASSES_PATH):
        # 没有类别表就无从核对。类数相同但下标错位是最危险的情况，
        # 因此这里不当作通过，而是提醒重训。
        return '模型旁没有 %s，无法确认类别对应关系' % os.path.basename(MODEL_CLASSES_PATH)
    try:
        with open(MODEL_CLASSES_PATH, encoding='utf-8') as f:
            trained = {int(k): int(v) for k, v in json.load(f).items()}
    except Exception:
        return None
    if trained != net_encoder:
        return ('模型的类别表与 config 不一致，模型训练时的类别表是 %s' % trained)
    return None


_mismatch = check_model_classes()
if _mismatch:
    print('[model] %s；请用 train_img.py 重训后再运行' % _mismatch)


# 获取截图：按 config 的几何抓一次屏，切成原分辨率的格子。
# 一次抓屏只切一遍，CNN 用它缩放到 32x32，颜色复核用它按原始像素判色——
# 两条识别路径看的是同一张图，所以复核的结论才真的独立可信。
def grab_cells(rows=ROWS, cols=COLS, region=BG_REGION):
    back_ground = np.array(pyautogui.screenshot(region=region))
    cells = []
    for j in range(rows):
        line = []
        for i in range(cols):
            l, u, r, d = [int(o) for o in
                          (i * cols_size, j * row_size, i * cols_size + cols_size, j * row_size + row_size)]
            line.append(Image.fromarray(back_ground[u:d + 1, l:r + 1]))
        cells.append(line)
    return cells


def screen_shoot(rows=ROWS, cols=COLS, region=BG_REGION):
    """把棋盘切成 32x32 的数组，供显示 / 训练样本记录使用。"""
    cells = grab_cells(rows, cols, region)
    return [[np.array(cell.resize((32, 32))) for cell in line] for line in cells]


# 使用cnn识别图像
def get_cnn_board(model, cells=None, rows=ROWS, cols=COLS, region=BG_REGION):
    # 将所有格子层叠排列，一起输入到cnn，比一个一个输入cnn计算速度快很多
    if cells is None:
        cells = grab_cells(rows, cols, region)
    sq_board = np.array([[np.array(cell.resize((32, 32))) for cell in line]
                         for line in cells]) / 255.

    sq_pred = np.argmax(np.asarray(model(sq_board, training=False)), axis=1)  # 直接调用比 model.predict 快一倍多
    sq_pred = np.array([net_decoder[i] for i in sq_pred])  # 解码cnn的预测值
    sq_pred = sq_pred.reshape((rows, cols))
    return sq_pred


# 显示截图结果
def show_screen_shoot(board=None):
    if board is None:
        board = screen_shoot()
    b = np.zeros((ROWS * 32, COLS * 32, 3))
    for i in range(ROWS):
        for j in range(COLS):
            b[i * 32:i * 32 + 32, j * 32:j * 32 + 32] = board[i][j]
    b = b / 255
    print(b)
    plt.imshow(b)
    plt.show()


# 显示识别结果
def show_recognized(board):
    plt.imshow(board)
    plt.show()


# 将识别出来的图片保存到对应文件夹，可以增加训练样本
def record_board(model, path=RECORD_PATH):
    resize_shape = (32, 32)
    sq_board = []
    back_ground = pyautogui.screenshot(region=BG_REGION)
    back_ground = np.array(back_ground)
    crops = []
    for j in range(ROWS):
        crops_r = []
        for i in range(COLS):
            l, u, r, d = [int(o) for o in
                          (i * cols_size, j * row_size, i * cols_size + cols_size, j * row_size + row_size)]
            crop = back_ground[u:d + 1, l:r + 1]
            crops_r.append(Image.fromarray(crop))

            crop = Image.fromarray(crop).resize(resize_shape)
            crop = np.array(crop)
            sq_board.append(crop)
        crops.append(crops_r)
    sq_board = np.array(sq_board) / 255.  # 与推理端一致：模型输入是 0~1 的浮点
    sq_pred = np.argmax(np.asarray(model(sq_board, training=False)), axis=1)
    sq_pred = np.array([net_decoder[i] for i in sq_pred])
    sq_pred = sq_pred.reshape((ROWS, COLS))
    for _cls in net_encoder:  # 首次记录时类别子目录可能还不存在，先建好
        os.makedirs(os.path.join(path, str(_cls)), exist_ok=True)
    for i in range(ROWS):
        for j in range(COLS):
            save_path = path + '\\' + str(sq_pred[i][j]) + '\\' + str(sq_pred[i][j]) + '_' + str(
                int(time() % 10000)) + '.png'
            crops[i][j].save(save_path)


# AI算法
def deal_board(board):
    to_flag = set()
    to_click = set()
    not_sure = dict()
    pad_board = np.pad(board, 1)
    row, col = board.shape
    while 1:
        action_nums = to_flag.__len__() + to_click.__len__()
        # 第一步，标记可以标记的旗子，把位置保存到to_flag
        for i in range(1, row + 1):
            for j in range(1, col + 1):
                if pad_board[i, j] > 0:
                    n9 = pad_board[i - 1:i + 2, j - 1:j + 2]
                    unknown_mask = (n9 == -2)
                    maybe_bomb = (n9 == -2) | (n9 == -10) | (n9 == -1)
                    if maybe_bomb.sum() == pad_board[i, j]:
                        n9[unknown_mask] = -1
                        for iii in range(3):
                            for jjj in range(3):
                                if unknown_mask[iii, jjj]:
                                    to_flag.add((iii + i - 1, jjj + j - 1))

        # 第二步消去旗子 用-10表示消去的旗子
        for i in range(1, row + 1):
            for j in range(1, col + 1):
                if pad_board[i, j] == -1:
                    pad_board[i, j] = -10
                    for ii in range(3):
                        for jj in range(3):
                            if pad_board[ii + i - 1, jj + j - 1] > 0:
                                pad_board[ii + i - 1, jj + j - 1] -= 1

        # 第三步，寻找可翻开的格子
        for i in range(1, row + 1):
            for j in range(1, col + 1):
                if pad_board[i, j] == 0:
                    for ii in range(3):
                        for jj in range(3):
                            if pad_board[ii + i - 1, jj + j - 1] == -2:
                                to_click.add((ii + i - 1, jj + j - 1))
                                pad_board[ii + i - 1, jj + j - 1] = -23

        # 第四步，正常办法解决不了，开启推理模式
        if action_nums == to_flag.__len__() + to_click.__len__():
            # 对每个没有翻开的格子a，如果周围有翻开的格子b1，那么将b1周围的所有未翻开的格子list1和b1的大小保存到一个列表，
            #                    如果周围有翻开的格子b2，那么将b2周围的所有未翻开的格子list2和b2的大小保存到一个列表，...
            # 最后得到一个关联关系  a的位置：与a有关的未翻开的格子的位置及隐藏地雷数量，保存到not_sure中
            for i in range(1, row + 1):
                for j in range(1, col + 1):
                    if pad_board[i, j] > 0:
                        not_sure_set = set()
                        for ii in range(3):
                            for jj in range(3):
                                if pad_board[ii + i - 1, jj + j - 1] == -2:
                                    not_sure_set.add((ii + i - 1, jj + j - 1))
                        for co_pos in not_sure_set:
                            not_sure[co_pos] = not_sure.get(co_pos, []) + ([[not_sure_set, pad_board[i, j]]])
            #        * * *             *   *  *   *   *
            #        * 1 ？            ？   4  ？  1   ？
            #        * 1 ？            ？   *  ？  *   ？
            #        * * ？            ？   *  *   *   ？
            # 用*代表墙壁，？代表未翻开的格子，上图左边可以推理出最下面的问号不是地雷，
            # 上图右边可以推理出左边三个问号是地雷，最右边三个问号不是地雷
            for k, v in not_sure.items():
                for i in range(len(v)):
                    for j in range(1, len(v)):
                        s1, n1 = v[i]
                        s2, n2 = v[j]
                        if s1 == s2: continue
                        if s1.issubset(s2):
                            minus_s = s2 - s1
                            if n2 - n1 == minus_s.__len__():
                                for p in minus_s:
                                    to_flag.add(p)
                            elif n2 == n1:
                                for p in minus_s:
                                    to_click.add(p)
                        if s2.issubset(s1):
                            minus_s = s1 - s2
                            if n1 - n2 == minus_s.__len__():
                                for p in minus_s:
                                    to_flag.add(p)
                            elif n2 == n1:
                                for p in minus_s:
                                    to_click.add(p)

                        # 推理模式增强版
                        mixed = s1 & s2
                        if n1 > n2 and s1.__len__() - mixed.__len__() == n1 - n2:
                            for p in s1 - mixed:
                                to_flag.add(p)
                            for p in s2 - mixed:
                                to_click.add(p)
                        if n1 < n2 and s2.__len__() - mixed.__len__() == n2 - n1:
                            for p in s2 - mixed:
                                to_flag.add(p)
                            for p in s1 - mixed:
                                to_click.add(p)

            if action_nums == to_flag.__len__() + to_click.__len__():
                break
    return to_click, to_flag


def click_cell(r, c, right=False, duration=0.01, interval=0.01):
    pos = (BG_REGION[0] + (c - 0.5) * cols_size, BG_REGION[1] + (r - 0.5) * row_size)
    if right:
        pyautogui.rightClick(pos, duration=duration)
    else:
        pyautogui.click(pos, duration=duration)
    pyautogui.sleep(interval)


def find_boom(board):
    """返回爆炸的红雷坐标列表（棋盘第几行第几列，0 基数）。"""
    hits = np.argwhere(board == BOOM_CLASS)
    return [(int(r), int(c)) for r, c in hits]


def print_safety_hint(board):
    """推不动了：算出所有未翻开格子的中雷概率，把最安全的几个列出来。

    只给建议，绝不自动点击——猜错了这局就结束了，决定权留给人。
    """
    entries, info = mine_hint(board, top=3)
    if info.get('error'):
        print('           算不出安全概率：%s' % info['error'])
        return
    if info['contradictions']:
        # 约束自相矛盾说明棋盘识别有误（常见于 6 被认成 4），概率不可信。
        print('           注意：有 %d 条约束互相矛盾，棋盘识别可能有误，下面的概率仅供参考。'
              % info['contradictions'])
    print('           全盘剩余 %d 颗雷，%d 格未翻开；最安全的几格：'
          % (info['total_mines'] - info['flags'], info['unknown']))
    for i, (r, c, prob) in enumerate(entries, 1):
        mark = '（等于随便点，棋盘已无信息可用）' if prob >= 0.999 else ''
        print('             %d. 第 %2d 行第 %2d 列   中雷概率 %5.1f%% %s'
              % (i, r + 1, c + 1, 100.0 * prob, mark))


# ---------------------------------------------------------------- 点击前的两道复核
# 推理本身从不点雷（模拟验证 3000+ 次点击零失误），炸雷全部来自「数字被认错」：
# 6 被读成 4，周围「还剩 2 颗雷」提前成立，于是把真雷判成安全格。所以能不能不炸雷，
# 取决于能不能在上鼠标之前发现识别错了。两道复核都只做一件事：发现不确定就停。
#
# 便宜的那道（P1）先跑：颜色只看笔画，和神经网络毫无关系，认错数字时几乎必然对不上。

PROB_SAFE = 0.0  # 只点「在所有一致解里都不是雷」的格子：概率必须严格为 0


def verify_by_color(board, cells):
    """P1：用数字笔画配色独立复核 CNN。返回冲突 [(行, 列, CNN 认成, 颜色认为)]。

    只复核 1~6 这类有固定配色的数字格；空块、未翻开、旗子、地雷没有配色，
    label_by_color 返回 None，不算冲突。行列是 0 基数（直接索引 board）。
    """
    if not VERIFY_BY_COLOR:
        return []
    conflicts = []
    rows, cols = board.shape
    for r in range(rows):
        for c in range(cols):
            v = int(board[r, c])
            if v not in DIGIT_COLORS:  # 只有 1~6 有配色，其余跳过
                continue
            color = label_by_color(cells[r][c])
            if color is not None and color != v:
                conflicts.append((r, c, v, color))
    return conflicts


def save_color_conflicts(conflicts, cells):
    """把颜色复核发现的不一致格子存进 samples/：颜色给的标签是准的，是白捡的难样本。"""
    saved = 0
    for r, c, cnn, color in conflicts:
        try:
            folder = os.path.join(SAMPLES_PATH, str(color))
            os.makedirs(folder, exist_ok=True)
            name = '%d_%04d.png' % (color, next_index(folder, color))
            cells[r][c].save(os.path.join(folder, name))
            saved += 1
        except Exception:
            pass
    return saved


def verify_by_prob(board, to_click):
    """P0：点击前算全盘概率，只放行概率严格为 0 的格子。

    返回 (放行的格子, 被拦下的 [(行, 列, 概率)], info)。
    info['reason'] 非空表示整盘都不该点：'contra' 约束自相矛盾、
    'too_hard' 局面太大算不出、'normal' 正常放行。
    """
    info = {'reason': 'normal', 'checked': VERIFY_BY_PROB}
    if not VERIFY_BY_PROB or not to_click:
        return list(to_click), [], info
    _, _, contradictions = build_constraints(board)
    if contradictions:
        # 约束互斥只可能是识别错或标旗错，此时任何「安全」结论都不可信。
        info['reason'] = 'contra'
        info['contradictions'] = contradictions
        return [], [], info
    p = mine_prob(board)
    if p is None:
        info['reason'] = 'too_hard'
        return [], [], info
    safe, risky = [], []
    for r, c in to_click:
        prob = float(p[r - 1, c - 1])  # deal_board 的坐标是 1 基数
        if prob <= PROB_SAFE:
            safe.append((r, c))
        else:
            risky.append((r, c, prob))
    return safe, risky, info


_LAST_ROUND = {'sig': None}  # 上一轮实际执行的 (点击, 标雷) 组合，用于兜底检测空转


def game_once(duration=0.01, interval=0.01, auto_flag=False):
    cells = grab_cells()  # 只抓一次屏：CNN 和颜色复核看同一张图
    board = get_cnn_board(model, cells)
    boom = find_boom(board)
    if boom:
        r, c = boom[0]
        print('[boom] 第 %d 行第 %d 列识出爆炸的红雷，本局已结束，程序退出' % (r + 1, c + 1))
        sys.exit(0)
    to_click, to_flag = deal_board(board)

    print('click_pos', to_click)
    print('flag_pos', to_flag, '(自动标雷：旗由游戏插)' if auto_flag else '')

    if to_click:  # 没有要点的格子时不必复核，直接走下面的死局分支
        # P1 颜色复核：CNN 与笔画配色不一致，说明至少有一格认错了。
        # 认错的那格可能正是被当成「安全」的雷，所以整轮放弃，一格都不点。
        conflicts = verify_by_color(board, cells)
        if conflicts:
            saved = save_color_conflicts(conflicts, cells)
            print('[verify] 颜色复核发现 %d 处识别冲突，本轮不点击：' % len(conflicts))
            for r, c, cnn, color in conflicts[:6]:
                print('           第 %2d 行第 %2d 列：CNN 认成 %d，笔画颜色说明是 %d'
                      % (r + 1, c + 1, cnn, color))
            if len(conflicts) > 6:
                print('           ……另有 %d 处' % (len(conflicts) - 6))
            if saved:
                print('           已把这 %d 格存进 samples/ 当难样本，筛完并入 imgs/ 可重训。' % saved)
            print('           请手动确认这几格，或检查标定 / 窗口是否被遮挡，然后按 B 继续。')
            return False

        # P0 概率复核：只放行「在所有一致解里都不是雷」的格子。
        safe, risky, pinfo = verify_by_prob(board, to_click)
        if pinfo['reason'] == 'contra':
            print('[verify] 棋盘约束自相矛盾 %d 处，说明有格子被认错，本轮不点击。'
                  % pinfo['contradictions'])
            print('         请手动核对棋盘后按 B 继续；若反复出现，检查标定与窗口缩放。')
            print_safety_hint(board)
            return False
        if pinfo['reason'] == 'too_hard':
            print('[verify] 局面太大算不出概率，无法确认安全，本轮不点击。')
            print('         可以直接按 B 跳过复核（改 config.VERIFY_BY_PROB = False），但会失去这层保险。')
            return False
        if risky:
            print('[verify] 概率复核拦下 %d 格（推理说是安全，全盘概率不是 0）：' % len(risky))
            for r, c, prob in risky[:6]:
                print('           第 %2d 行第 %2d 列   中雷概率 %5.1f%%'
                      % (r + 1, c + 1, 100.0 * prob))
            if safe:
                print('         其余 %d 格概率为 0，照常点击。' % len(safe))
            else:
                print('         没有一格概率为 0，本轮不点击。')
                print_safety_hint(board)
                return False
            to_click = safe

    for lc in sorted(to_click):
        click_cell(lc[0], lc[1], duration=duration, interval=interval)
    # 自动标雷模式下不发右键：网页的右键是「插旗 / 取消旗」的开关，
    # 游戏刚替你插好的旗再点一次就会被取消，下一轮又插回来。
    if not auto_flag:
        for lc in sorted(to_flag):
            click_cell(lc[0], lc[1], right=True, duration=duration, interval=interval)

    # 判断「这一轮到底有没有动手」。自动标雷模式下不发右键，所以旗子不算动作：
    # 如果只推得出旗、推不出可点的安全格，这一轮一次鼠标都不会动，棋盘不会有任何
    # 变化，下一轮必然得到完全相同的结果 —— 那就是死循环，必须在这里停住。
    sig = (tuple(sorted(to_click)), tuple(sorted(to_flag)) if not auto_flag else ())
    repeated = sig == _LAST_ROUND['sig']
    _LAST_ROUND['sig'] = sig

    actions = len(to_click) + (0 if auto_flag else len(to_flag))
    if actions == 0:
        if to_flag:
            print('[deadlock] 只推得出 %d 个雷，没有可以确定安全、可以点开的格子。' % len(to_flag))
            print('           自动标雷模式下旗子由游戏插，程序不发右键，所以这一轮无事可做。')
        else:
            print('[deadlock] 没有能确定推理的格子。')
        print_safety_hint(board)
        print('           请手动点开上面最安全的一格，然后按 B 继续，或按 Z 只点安全格。')
        print('           Z 模式需要先在游戏里打开「自动标雷」，否则旗子插不上，会一直停在这里。')
        return False
    # 兜底：动作一模一样地重复，说明点了也没生效（坐标标定不对、窗口被遮挡、
    # 或游戏已经结束）。再跑下去就是无限循环，停住让人看一眼。
    if repeated:
        print('[stuck] 连续两轮要执行的动作完全相同 %d 个，点了没有产生任何变化。' % actions)
        print('        常见原因：标定坐标与实际棋盘不符、游戏窗口被别的窗口挡住、本局已经结束。')
        print('        请先跑 calibrate.py 重新标定，或手动点开一格后按 B 继续。')
        return False
    return True


def game_all(auto_flag=False):
    _LAST_ROUND['sig'] = None  # 每次按 B/Z 都是一次新的连续执行，重复检测从零开始
    while game_once(auto_flag=auto_flag):
        pass
    print('[done] 已停止。猜完一步后按 B 继续（自动标雷模式按 Z）。')


def on_press(key):
    # print(key,str(key))
    if isinstance(key, KeyCode):
        if key.char == 'a':
            game_once()

        elif key.char == 'b':
            game_all(auto_flag=False)

        elif key.char == 'z':  # 自动标雷模式：只左键点安全格，旗由游戏自己插
            game_all(auto_flag=True)

        elif key.char == 'r':
            record_board(model, path=RECORD_PATH)

        elif key.char == 's':  # 显示识别结果
            pool = Pool(1)
            pool.apply_async(show_recognized, args=(get_cnn_board(model),))
            pool.close()
            pool.join()

        elif key.char == 'c':  # 显示截图结果
            pool = Pool(1)
            pool.apply_async(show_screen_shoot)
            pool.close()
            pool.join()

        elif key.char == 'q':
            return False


def start_listen():
    with Listener(on_press=on_press) as listener:
        listener.join()


if __name__ == '__main__':
    print('开始监听,按 A 键截图并执行脚本一次,\n'
          '按 B 键截图并执行脚本直到无法推理,\n'
          '按 Z 键同上，但用「自动标雷」模式（需先在游戏里打开该选项）,\n'
          '按 C 键截图并显示截图结果,\n'
          '按 S 键截图并显示预测结果,\n'
          '按 R 键截图并保存截图到文件夹,\n'
          '按 Q 键退出,\n'
          '注：1.如果鼠标失去控制，快速移动鼠标，将鼠标移出屏幕左上角会强制退出程序\n'
          '2.显示截图结果和预测结果与主线程冲突，需要开启新进程，响应会比较慢')
    start_listen()
