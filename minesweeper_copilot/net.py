# -*- coding: utf-8 -*-
"""扫雷 CNN 结构（Keras 3）。

只有训练脚本 tools/train.py 导入本模块来建网；主程序 autoplay.py 不导入它，
而是直接加载训练好的 models/recognize_new.keras。所以改了层结构就必须重训，
旧权重对不上新结构。

输入：32x32x3，RGB，取值 0~1
输出：softmax，类别数量由 config.net_encoder 决定（当前 11 类）
"""

import keras
from keras import layers

from .config import net_encoder

INPUT_SHAPE = (32, 32, 3)
NUM_CLASSES = len(net_encoder)


def build_model(input_shape=INPUT_SHAPE, num_classes=NUM_CLASSES):
    """构建 CNN。层序与权重形状固定，改了结构就无法复用已有权重。"""
    model = keras.Sequential(
        [
            keras.Input(shape=input_shape),
            layers.Conv2D(32, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.Flatten(),
            layers.Dense(64, activation='relu'),
            layers.Dense(num_classes, activation='softmax'),
        ]
    )
    model.compile(
        optimizer='adam',
        loss=keras.losses.CategoricalCrossentropy(),
        metrics=['accuracy'],
    )
    return model


if __name__ == '__main__':
    build_model().summary()
