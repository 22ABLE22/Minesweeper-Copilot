# -*- coding: utf-8 -*-
"""把老版本 Keras 2 / TF 2.x 存出来的 SavedModel 目录转成 Keras 3 的 .keras 文件。

背景: model_and_data.zip 里的 recognize_new/ 是一个 SavedModel 目录，
      Keras 3 的 load_model() 已经不认识这个格式，会报
      ValueError: File format not supported: filepath=recognize_new.
      仓库里已经放好了转换结果 recognize_new.keras，正常不用跑本脚本。

依赖: pip install tf-keras==2.21.0   （版本号要和 tensorflow 对应）

用法:
    python convert_legacy_model.py                     # recognize_new -> recognize_new.keras
    python convert_legacy_model.py my_old_dir out.keras

为什么不能直接 tf_keras 存成 .keras: 旧模型是按 batch_input_shape 建的 Conv2D,
Keras 3 读它自己的 .keras 文件时会报
    ValueError: Unrecognized keyword arguments passed to Conv2D: {'batch_input_shape': ...}
所以这里用 cnn_model.build_model() 新建同构网络，再逐层 set_weights。
"""

import argparse
import os
import sys

import numpy as np

from cnn_model import build_model


def convert(src, dst):
    try:
        import tf_keras
    except ImportError:
        sys.exit('需要先安装旧版 Keras: pip install tf-keras==2.21.0')

    print('[convert] 用 tf-keras 读取旧模型: %s' % src)
    old = tf_keras.models.load_model(src, compile=False)
    new = build_model()

    if len(new.layers) != len(old.layers):
        sys.exit('[convert] 层数不一致: 旧 %d / 新 %d' % (len(old.layers), len(new.layers)))

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
