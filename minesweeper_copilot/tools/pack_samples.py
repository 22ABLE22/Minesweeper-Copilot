# -*- coding: utf-8 -*-
"""把 data/imgs/ 打包成 data/imgs.tar.gz —— 仓库里只存这个压缩包，不再逐个跟踪 3000 多张 PNG。

用法（在仓库根目录执行）:
    python -m minesweeper_copilot.tools.pack_samples            # 打包
    python -m minesweeper_copilot.tools.pack_samples --extract  # 解包
    python -m minesweeper_copilot.tools.pack_samples --list     # 清点

为什么用 tar.gz 而不是 zip：
    PNG 本身已经 deflate 压缩过，再压一遍 zip 反而更大（实测 imgs 的 2054094 字节
    打成 deflate zip 是 2407311 字节）。tar.gz 能把 3000 多个 PNG 之间重复的
    zlib 头和索引块一起压掉，实测 1783781 字节；更关键的是只占 1 个 git
    对象，而不是 3000 多个。

头像/索引信息统一置零，同一批图重复打包得到逐字节相同的文件，不会每次都产生假改动。
"""
import gzip
import os
import sys
import tarfile

# 仓库根目录（本文件在 <根>/minesweeper_copilot/tools/ 下）
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC = os.path.join(BASE_DIR, 'data', 'imgs')
OUT = os.path.join(BASE_DIR, 'data', 'imgs.tar.gz')


def collect():
    """按路径排序收集 data/imgs/ 下所有 PNG，保证打包顺序稳定。"""
    if not os.path.isdir(SRC):
        sys.exit('没有目录: %s（先跑 python -m minesweeper_copilot.tools.pack_samples --extract）' % SRC)
    files = []
    for root, dirs, fs in os.walk(SRC):
        dirs.sort()
        for f in fs:
            if f.lower().endswith('.png'):
                files.append(os.path.join(root, f))
    files.sort()
    if not files:
        sys.exit('data/imgs/ 里没有 PNG')
    return files


def pack():
    files = collect()
    # GzipFile 的 mtime 默认写当前时间，置 0 才能得到逐字节可复现的包
    with gzip.GzipFile(OUT, 'wb', compresslevel=9, mtime=0) as gz:
        with tarfile.open(fileobj=gz, mode='w', format=tarfile.GNU_FORMAT) as tf:
            for path in files:
                arc = os.path.relpath(path, BASE_DIR).replace(os.sep, '/')
                info = tf.gettarinfo(path, arcname=arc)
                info.mtime = 0
                info.uid = info.gid = 0
                info.uname = info.gname = ''
                with open(path, 'rb') as fh:
                    tf.addfile(info, fh)
    total = sum(os.path.getsize(p) for p in files)
    print('%d 张 -> %s（原始 %d 字节，压缩后 %d 字节）'
          % (len(files), os.path.basename(OUT), total, os.path.getsize(OUT)))


def extract():
    if not os.path.isfile(OUT):
        sys.exit('没有文件: %s' % OUT)
    with tarfile.open(OUT, 'r:gz') as tf:
        try:
            tf.extractall(BASE_DIR, filter='data')
        except TypeError:      # Python < 3.12 没有 filter 参数
            tf.extractall(BASE_DIR)
    n = len(collect())
    print('已解出 %d 张到 %s' % (n, SRC))


def list_pack():
    if not os.path.isfile(OUT):
        sys.exit('没有文件: %s' % OUT)
    counts = {}
    with tarfile.open(OUT, 'r:gz') as tf:
        for m in tf.getmembers():
            if not m.isfile() or not m.name.lower().endswith('.png'):
                continue
            # 归档里的路径是 data/imgs/<类别>/<文件>.png
            parts = m.name.split('/')
            if len(parts) >= 4:
                counts[parts[2]] = counts.get(parts[2], 0) + 1
    for cls in sorted(counts, key=lambda s: (len(s), s)):
        print('%-3s %4d 张' % (cls, counts[cls]))
    print('合计 %d 张' % sum(counts.values()))


if __name__ == '__main__':
    if '--extract' in sys.argv:
        extract()
    elif '--list' in sys.argv:
        list_pack()
    else:
        pack()
