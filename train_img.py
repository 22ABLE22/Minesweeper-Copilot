# -*- coding: utf-8 -*-
"""训练扫雷识别用的 CNN。

用法:
    python train_img.py                          # 用 imgs/ 训练
    python train_img.py --epochs 20 --per-class 300
    python train_img.py --no-augment             # 不用增强，只训原始样本

数据来源:
    imgs/  全部训练样本（按类别分文件夹，文件夹名是 net_encoder 的键）
    仓库里只跟踪 imgs.tar.gz；imgs/ 不存在时会自动解包（等价于
    python pack_samples.py --extract），解出来的 imgs/ 不入库。
    collect.py 采到的新样本默认也写进 imgs/，人工筛完直接重训即可

训练集用数据增强（平移、缩放、亮度、对比度、轻微模糊）把每类补到 --per-class 张，
验证集只用未增强的原始图片，保证报告的是真实分布下的准确率。
产出: 仓库根目录的 recognize_new.keras（Keras 3 格式），saolei.py 直接加载。
"""

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import logging

logging.getLogger('tensorflow').setLevel(logging.ERROR)

import argparse
import glob
import json
import random

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

from cnn_model import INPUT_SHAPE, NUM_CLASSES, build_model
from config import TRAIN_PATH, net_decoder, net_encoder
from pack_samples import OUT as SAMPLES_TAR, extract as extract_samples

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_OUT = os.path.join(BASE_DIR, 'recognize_new.keras')
MODEL_CLASSES = os.path.join(BASE_DIR, 'recognize_new.classes.json')


def ensure_samples():
    """imgs/ 不存在但仓库带着 imgs.tar.gz 时，先自动解包再训练。"""
    if os.path.isdir(TRAIN_PATH):
        return
    if not os.path.isfile(SAMPLES_TAR):
        return          # 交给 scan_sources 打印跳过提示
    print('[train] 没有 %s，从 %s 解包' % (TRAIN_PATH, os.path.basename(SAMPLES_TAR)))
    extract_samples()


def scan_sources(sources):
    """扫描各数据源，返回 {标签: [图片路径]}。"""
    pools = {}
    for root in sources:
        if not os.path.isdir(root):
            print('[train] 跳过不存在的目录: %s' % root)
            continue
        for name in sorted(os.listdir(root)):
            cls_dir = os.path.join(root, name)
            if not os.path.isdir(cls_dir):
                continue
            try:
                label = net_encoder[int(name)]
            except (ValueError, KeyError):
                continue
            files = sorted(glob.glob(os.path.join(cls_dir, '*.png')))
            pools.setdefault(label, []).extend(files)
    for label in sorted(pools):
        print('[train] 标签 %2d (%-2s) %d 张' % (label, net_decoder[label], len(pools[label])))
    return pools


def load_image(path):
    with Image.open(path) as im:
        return im.convert('RGB').copy()


def augment(im):
    """在原始分辨率上做随机扰动，再交给训练时 resize。"""
    w, h = im.size
    bg = im.getpixel((2, 2))
    scale = random.uniform(0.88, 1.12)
    im2 = im.resize((max(4, int(w * scale)), max(4, int(h * scale))), Image.BILINEAR)
    canvas = Image.new('RGB', (w, h), bg)
    ox = (w - im2.width) // 2 + random.randint(-3, 3)
    oy = (h - im2.height) // 2 + random.randint(-3, 3)
    canvas.paste(im2, (ox, oy))
    if random.random() < 0.5:
        canvas = ImageEnhance.Brightness(canvas).enhance(random.uniform(0.90, 1.10))
    if random.random() < 0.4:
        canvas = ImageEnhance.Contrast(canvas).enhance(random.uniform(0.94, 1.08))
    if random.random() < 0.35:
        canvas = canvas.filter(ImageFilter.GaussianBlur(random.uniform(0.3, 0.8)))
    return canvas


def to_array(im, size=INPUT_SHAPE[:2]):
    return np.asarray(im.resize(size, Image.BILINEAR), dtype=np.float32) / 255.0


def build_sets(pools, per_class, val_split, use_aug):
    """按类别切出验证集，训练集用增强补到 per_class 张。"""
    train_x, train_y, val_x, val_y = [], [], [], []
    for label in sorted(pools):
        files = pools[label]
        random.shuffle(files)
        n_val = max(1, int(round(len(files) * val_split))) if len(files) >= 4 else 0
        val_files, train_files = files[:n_val], files[n_val:]
        if not train_files:
            continue
        for f in val_files:
            val_x.append(to_array(load_image(f)))
            val_y.append(label)
        imgs = [load_image(f) for f in train_files]
        target = max(per_class, len(imgs)) if use_aug else len(imgs)
        for i in range(target):
            im = imgs[i % len(imgs)]
            if use_aug:
                im = augment(im)
            train_x.append(to_array(im))
            train_y.append(label)
    x = np.asarray(train_x, dtype=np.float32)
    vx = np.asarray(val_x, dtype=np.float32)
    y = np.zeros((len(train_y), NUM_CLASSES), dtype=np.float32)
    vy = np.zeros((len(val_y), NUM_CLASSES), dtype=np.float32)
    for i, label in enumerate(train_y):
        y[i, label] = 1.0
    for i, label in enumerate(val_y):
        vy[i, label] = 1.0
    return x, y, vx, vy


def class_weights(labels):
    counts = np.bincount(labels, minlength=NUM_CLASSES).astype(np.float64)
    counts[counts == 0] = 1.0
    w = counts.sum() / (len(counts) * counts)
    return {i: float(w[i]) for i in range(NUM_CLASSES) if counts[i] > 1}


def report(vx, vy, model):
    pred = np.argmax(np.asarray(model(vx, training=False)), axis=1)
    true = np.argmax(vy, axis=1)
    acc = float((pred == true).mean())
    print('[eval] 验证集 %d 张，整体准确率 %.2f%%' % (len(true), 100.0 * acc))
    cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=int)
    for t, p in zip(true, pred):
        cm[t, p] += 1
    print('[eval] 混淆矩阵（行=真实，列=预测）：')
    head = '        ' + ''.join('%4s' % net_decoder[i] for i in range(NUM_CLASSES))
    print(head)
    for i in range(NUM_CLASSES):
        if cm[i].sum() == 0:
            continue
        print('  %3s |' % net_decoder[i] + ''.join('%4d' % v for v in cm[i])
              + '   n=%-3d acc=%.0f%%' % (cm[i].sum(), 100.0 * cm[i, i] / cm[i].sum()))
    worst = [(net_decoder[i], round(100.0 * cm[i, i] / cm[i].sum())) for i in range(NUM_CLASSES)
             if cm[i].sum() and cm[i, i] / cm[i].sum() < 0.95]
    if worst:
        print('[eval] 准确率低于 95% 的类别:', worst)
    return acc


def plot_history(history):
    import matplotlib.pyplot as plt

    plt.figure()
    plt.plot(history.history['accuracy'], label='train acc')
    if 'val_accuracy' in history.history:
        plt.plot(history.history['val_accuracy'], label='val acc')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.legend()
    plt.title('cnn train result')
    plt.show()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sources', default=None, help='逗号分隔的数据目录，默认只用 imgs')
    ap.add_argument('--per-class', type=int, default=500, help='训练集每类目标张数（增强补足）')
    ap.add_argument('--val-split', type=float, default=0.15, help='验证集比例（只用原始图片）')
    ap.add_argument('--epochs', type=int, default=12)
    ap.add_argument('--batch-size', type=int, default=64)
    ap.add_argument('--no-augment', action='store_true', help='不做数据增强')
    ap.add_argument('--no-weights', action='store_true', help='不按类别频率加权')
    ap.add_argument('--no-plot', action='store_true', help='训练完不弹曲线窗口')
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    ensure_samples()

    if args.sources:
        sources = [os.path.join(BASE_DIR, p) if not os.path.isabs(p) else p
                   for p in args.sources.split(',')]
    else:
        sources = [TRAIN_PATH]

    pools = scan_sources(sources)
    if not pools:
        raise SystemExit('[train] 没有找到训练图片。仓库自带 imgs.tar.gz，'
                         '跑 python pack_samples.py --extract 解包后再试')

    use_aug = not args.no_augment
    x, y, vx, vy = build_sets(pools, args.per_class, args.val_split, use_aug)
    print('[train] 训练集 %d 张（增强=%s），验证集 %d 张，输入 %s，类别数 %d'
          % (len(x), use_aug, len(vx), x.shape[1:], NUM_CLASSES))

    model = build_model()
    labels = [int(np.argmax(row)) for row in y]
    cw = None if args.no_weights else class_weights(labels)

    callbacks = []
    from keras.callbacks import EarlyStopping, ModelCheckpoint

    if len(vx):
        callbacks = [
            EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True),
            ModelCheckpoint(MODEL_OUT, monitor='val_accuracy', save_best_only=True, verbose=0),
        ]
    history = model.fit(x, y, epochs=args.epochs, batch_size=args.batch_size,
                        validation_data=(vx, vy) if len(vx) else None,
                        class_weight=cw, callbacks=callbacks, verbose=2)

    model.save(MODEL_OUT)
    print('[train] 已保存 %s (%d bytes)' % (MODEL_OUT, os.path.getsize(MODEL_OUT)))

    # 把训练时的类别表一起存下来，推理端据此判断模型与 config 是否配套。
    # 只比「类数」不够：改过类别表但类数没变（例如删 7 再加回来）会静默错位。
    with open(MODEL_CLASSES, 'w', encoding='utf-8') as f:
        json.dump(net_encoder, f, ensure_ascii=False, indent=1)
    print('[train] 类别表已写入 %s' % MODEL_CLASSES)

    if len(vx):
        report(vx, vy, model)
    if not args.no_plot:
        plot_history(history)


if __name__ == '__main__':
    main()
