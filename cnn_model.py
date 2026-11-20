# -*- coding: utf-8 -*-
"""扫雷 CNN 结构（Keras 3）。

训练脚本（train_img.py）与推理脚本（saolei.py）共用这一份定义，
保证两边永远是同一套层结构与预处理。

输入：32x32x3，RGB，取值 0~1
输出：softmax，类别数量由 config.net_encoder 决定（当前含爆炸的红雷共 12 类）
"""

import keras
from keras import layers

from config import net_encoder

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
