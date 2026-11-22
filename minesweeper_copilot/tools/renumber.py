# -*- coding: utf-8 -*-
"""把样本目录按类别重新编号为 <类别>_<序号>.png。

整理样本的最后一步：tools/collect 采到 samples/auto/、人工筛选并入 data/imgs/ 之后，
用本脚本让编号从 _0001 连续排下去。编号只求不重名，不代表任何含义，
tools/train.py 也不读它。

用法（在仓库根目录执行，路径相对仓库根）:
    python -m minesweeper_copilot.tools.renumber                # 处理 samples/auto
    python -m minesweeper_copilot.tools.renumber data/imgs/6    # 只处理一个类别目录
    python -m minesweeper_copilot.tools.renumber data/imgs      # 处理所有类别
    python -m minesweeper_copilot.tools.renumber data/imgs --dry-run

分两阶段：先把所有文件改成 __tmp__ 前缀，再改到最终名，
避免新旧名字撞车（例如 3_0001.png 已存在时被覆盖）。
"""
import os
import sys

# 仓库根目录（本文件在 <根>/minesweeper_copilot/tools/ 下）
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DRY = '--dry-run' in sys.argv
ARGS = [a for a in sys.argv[1:] if not a.startswith('-')]


def resolve_root():
    """没给目录时沿用 samples/auto；给了就相对脚本所在目录解析。"""
    if not ARGS:
        return os.path.join(BASE_DIR, 'samples', 'auto')
    arg = ARGS[0].replace('\\', '/')
    return arg if os.path.isabs(arg) else os.path.join(BASE_DIR, arg)


ROOT = resolve_root()


def main():
    if not os.path.isdir(ROOT):
        sys.exit('没有目录: %s' % ROOT)
    total = 0
    # 传的目录本身就是类别目录（如 data/imgs/6）时，直接按它自己的名字当类别
    self_class = os.path.basename(os.path.normpath(ROOT))
    has_subdir = any(os.path.isdir(os.path.join(ROOT, e)) for e in os.listdir(ROOT))
    if has_subdir:
        targets = [(cls, os.path.join(ROOT, cls))
                   for cls in sorted(os.listdir(ROOT), key=lambda s: (len(s), s))
                   if os.path.isdir(os.path.join(ROOT, cls))]
    else:
        targets = [(self_class, ROOT)]
    for cls, d in targets:
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
