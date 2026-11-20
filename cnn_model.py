# -*- coding: utf-8 -*-
"""扫雷 CNN 结构（Keras 3）。

训练脚本（train_img.py）与推理脚本（saolei.py）共用这一份定义，
避免「训练时存成一种格式、推理时又是另一种格式」的历史遗留问题。

输入：32x32x3，RGB，取值 0~1
输出：11 类 softmax，类别含义见 config.net_encoder
"""

import keras
from keras import layers

INPUT_SHAPE = (32, 32, 3)
NUM_CLASSES = 11


def build_model(input_shape=INPUT_SHAPE, num_classes=NUM_CLASSES):
    """构建与原仓库结构完全一致的 CNN（层名、层序都不变，可无损继承旧权重）。"""
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
