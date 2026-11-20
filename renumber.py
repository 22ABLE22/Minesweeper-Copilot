# -*- coding: utf-8 -*-
"""把 samples/auto 下的样本按目录类别重新编号为 <类别>_<序号>.png。

分两阶段：先把所有文件改成 __tmp__ 前缀，再改到最终名，
避免新旧名字撞车（例如 3_0001.png 已存在时被覆盖）。
"""
import os
import sys

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'samples', 'auto')
DRY = '--dry-run' in sys.argv

def main():
    if not os.path.isdir(ROOT):
        sys.exit('没有目录: %s' % ROOT)
    total = 0
    for cls in sorted(os.listdir(ROOT), key=lambda s: (len(s), s)):
        d = os.path.join(ROOT, cls)
        if not os.path.isdir(d):
            continue
        files = sorted(f for f in os.listdir(d) if f.lower().endswith('.png'))
        if not files:
            print('%-4s 空目录，跳过' % cls)
            continue
        width = max(4, len(str(len(files))))
        tmp = []
        for i, f in enumerate(files, 1):
            src = os.path.join(d, f)
            t = os.path.join(d, '__tmp__%d.png' % i)
            if not DRY:
                os.rename(src, t)
            tmp.append(t)
        for i, t in enumerate(tmp, 1):
            dst = os.path.join(d, '%s_%0*d.png' % (cls, width, i))
            if not DRY:
                os.rename(t, dst)
        print('%-4s %3d 张 -> %s_%0*d.png ... %s_%0*d.png' % (cls, len(files), cls, width, 1, cls, width, len(files)))
        total += len(files)
    print('合计 %d 张' % total)

main()
