# -*- coding: utf-8 -*-
"""训练扫雷识别用的 CNN。

用法:
    python train_img.py                      # 默认 1000 张样本、4 个 epoch
    python train_img.py --samples 3000 --epochs 8 --no-plot

数据: config.TRAIN_PATH（默认 imgs/）下按类别分文件夹，文件夹名必须是
      config.net_encoder 的键（-2 未翻开、-1 旗子、0..8 数字、9 地雷）。
产出: 仓库根目录的 recognize_new.keras（Keras 3 格式），saolei.py 直接加载。

想补充样本：先把游戏摆好，运行 saolei.py 后按 R 键，脚本会把当前棋盘按
预测类别分别存到 dir/<类别>/ 下；确认无误后把文件挪进 imgs/<类别>/。
"""

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import logging

logging.getLogger('tensorflow').setLevel(logging.ERROR)

import argparse
import glob
import os
import random

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from cnn_model import INPUT_SHAPE, NUM_CLASSES, build_model
from config import TRAIN_PATH, net_encoder

MODEL_OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'recognize_new.keras')


def load_data(sample_num, root=TRAIN_PATH):
    """把 imgs/ 下所有类别目录里的图片读成 (x, y)，随机抽 sample_num 张。"""
    paths, labels = [], []
    for cls_name in sorted(os.listdir(root)):
        cls_dir = os.path.join(root, cls_name)
        if not os.path.isdir(cls_dir):
            continue
        try:
            label = net_encoder[int(cls_name)]
        except (ValueError, KeyError):
            print('[train] 跳过无法识别的类别目录: %s' % cls_name)
            continue
        imgs = glob.glob(os.path.join(cls_dir, '*.png'))
        print('[train] 类别 %-3s -> 标签 %2d，%d 张' % (cls_name, label, len(imgs)))
        paths += imgs
        labels += [label] * len(imgs)

    if not paths:
        raise SystemExit('[train] %s 下没有找到训练图片，样例数据见仓库的 imgs/' % root)

    n = min(sample_num, len(paths))
    idx = random.sample(range(len(paths)), n)
    x = np.zeros((n,) + tuple(INPUT_SHAPE), dtype=np.float32)
    y = np.zeros((n, NUM_CLASSES), dtype=np.float32)
    for i, j in enumerate(idx):
        img = Image.open(paths[j]).convert('RGB').resize(INPUT_SHAPE[:2])
        x[i] = np.asarray(img, dtype=np.float32) / 255.0
        y[i, labels[j]] = 1.0
    return x, y


def show_cnn_result(model, x_test, y_test):
    plt.figure()
    plt.plot(model.history.history['accuracy'], label='train acc')
    if 'val_accuracy' in model.history.history:
        plt.plot(model.history.history['val_accuracy'], label='val acc')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.legend()
    plt.title('cnn train result')
    plt.show()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--samples', type=int, default=1000, help='随机抽取的训练样本数')
    ap.add_argument('--epochs', type=int, default=4)
    ap.add_argument('--batch-size', type=int, default=32)
    ap.add_argument('--no-plot', action='store_true', help='训练完不弹曲线窗口')
    args = ap.parse_args()

    x, y = load_data(args.samples)
    print('[train] 实际样本 %d 张，输入 %s，类别数 %d' % (len(x), x.shape[1:], y.shape[1]))

    # 95% 训练 / 5% 验证；样本太少时不做验证集，避免 val_accuracy 缺失
    split = int(len(x) * 0.95)
    if split >= 50 and len(x) - split >= 5:
        train_x, train_y = x[:split], y[:split]
        val = (x[split:], y[split:])
    else:
        train_x, train_y, val = x, y, None

    model = build_model()
    model.fit(train_x, train_y, epochs=args.epochs, batch_size=args.batch_size,
              validation_data=val, verbose=2)
    model.save(MODEL_OUT)
    print('[train] 已保存 %s (%d bytes)' % (MODEL_OUT, os.path.getsize(MODEL_OUT)))

    if not args.no_plot:
        show_cnn_result(model, x, y)


if __name__ == '__main__':
    main()
