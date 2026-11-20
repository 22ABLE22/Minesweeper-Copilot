# -*- coding: utf-8 -*-
"""把 SavedModel 目录（saved_model.pb + variables/）转成 Keras 3 的 .keras 文件。

Keras 3 的 load_model() 只接受 .keras（V3）与 legacy .h5，SavedModel 目录需要
用 tf-keras 读进来再重建，本脚本就是这条转换链路：

    1. tf_keras.models.load_model(src)          读出 SavedModel 的层与权重
    2. cnn_model.build_model()                  建同构网络
    3. 逐层 set_weights                         复制权重
    4. new.save(dst)                            写成 .keras

不直接另存为 .keras 的原因：SavedModel 里的层带有 batch_input_shape 参数，
Keras 3 读这种文件会报
    ValueError: Unrecognized keyword arguments passed to Conv2D: {'batch_input_shape': ...}

依赖: pip install tf-keras==2.21.0   （版本号要与 tensorflow 对应）

用法:
    python convert_legacy_model.py recognize_new recognize_new.keras
    python convert_legacy_model.py <src_dir> <dst.keras>
"""

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import logging

logging.getLogger('tensorflow').setLevel(logging.ERROR)

import argparse
import sys

import numpy as np

from cnn_model import build_model


def convert(src, dst):
    try:
        import tf_keras
    except ImportError:
        sys.exit('需要先安装 tf-keras: pip install tf-keras==2.21.0')

    print('[convert] 用 tf-keras 读取 SavedModel: %s' % src)
    old = tf_keras.models.load_model(src, compile=False)
    new = build_model()

    if len(new.layers) != len(old.layers):
        sys.exit('[convert] 层数不一致: 源 %d / 目标 %d' % (len(old.layers), len(new.layers)))

    for dst_layer, src_layer in zip(new.layers, old.layers):
        w_dst = dst_layer.get_weights()
        w_src = src_layer.get_weights()
        if len(w_dst) != len(w_src):
            sys.exit('[convert] 层 %s 权重个数不一致' % src_layer.name)
        for a, b in zip(w_dst, w_src):
            if a.shape != b.shape:
                sys.exit('[convert] 层 %s 权重形状不一致: %s vs %s' % (src_layer.name, a.shape, b.shape))
        dst_layer.set_weights([np.array(w) for w in w_src])
        print('[convert] 已复制 %-12s %s' % (src_layer.name, [w.shape for w in w_src]))

    new.save(dst)
    print('[convert] 已保存 %s (%d bytes)' % (dst, os.path.getsize(dst)))


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser()
    ap.add_argument('src', nargs='?', default=os.path.join(here, 'recognize_new'))
    ap.add_argument('dst', nargs='?', default=os.path.join(here, 'recognize_new.keras'))
    args = ap.parse_args()
    convert(args.src, args.dst)


if __name__ == '__main__':
    main()
