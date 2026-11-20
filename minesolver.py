# -*- coding: utf-8 -*-
"""全局面雷概率求解器：推不动的时候，算出每个未翻开格子是雷的概率。

给你「现在点哪一格最安全」用。它只算概率、不做决定，程序不会拿它自动点击。

思路
----
1. 每个已翻开的数字格给出一个约束：它周围未翻开的邻居里恰好有 need 颗雷。
2. 共享格子的约束并成一个连通分量，分量之间唯一的耦合是「整盘的雷数」。
3. 分量内枚举全部一致的解，统计「用掉 k 颗雷的解法数」和「格子 j 在 k 雷解里是雷的次数」。
4. 分量之间按总雷数做卷积（对数域，避免解数过大溢出）。
5. 不与任何数字相邻的孤立格，用组合数 C(n, k) 加权。
6. 归一化，得到每格是雷的概率：p=0 必然安全，p=1 必然是雷。

局限
----
- 需要知道整盘雷数。用 config.TOTAL_MINES 填写；填 0 时按当前棋盘估算。
- 棋盘识别有误（比如 6 被认成 4）时，约束之间会互相矛盾。这种情况
  solve_prob 仍然会给数，但结论不可信——check() 返回的矛盾数就是判断依据。
- 单个分量解数超过 CAP_SOLS 时放弃（返回 None），不会给出错的概率。
"""
import math

import numpy as np
from config import TOTAL_MINES

UNK, FLAG = -2, -1

# 单个分量的解数上限。超过就放弃这一盘，宁可不说也不说错。
CAP_SOLS = 2000000


def neighbours(r, c, rows, cols):
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr or dc:
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols:
                    yield nr, nc


def build_constraints(board):
    """提取约束。返回 (cons, unknown, contradictions)。

    cons: [(cells, need)]，cells 是该数字周围未翻开的格子
    contradictions: 无法自洽的约束条数（棋盘识别或标旗出错时大于 0）
    """
    rows, cols = board.shape
    cons = []
    contradictions = 0
    for r in range(rows):
        for c in range(cols):
            v = board[r, c]
            if v <= 0 or v > 8:
                continue
            cells, f = [], 0
            for nr, nc in neighbours(r, c, rows, cols):
                x = board[nr, nc]
                if x == UNK:
                    cells.append((nr, nc))
                elif x == FLAG:
                    f += 1
            need = int(v) - f
            if not cells:
                if need != 0:
                    contradictions += 1
                continue
            if need < 0 or need > len(cells):
                # 满足不了的约束通常是识别出错。夹到可行区间继续算，
                # 比整盘放弃更有用，但 contradictions 会记下来。
                contradictions += 1
                need = min(max(need, 0), len(cells))
            cons.append((cells, need))
    unknown = [(r, c) for r in range(rows) for c in range(cols) if board[r, c] == UNK]
    return cons, unknown, contradictions


def split_components(cons):
    """按共享格子把约束分成互不相干的连通分量。"""
    parent = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for cells, _ in cons:
        for x in cells:
            find(x)
        for x in cells[1:]:
            ra, rb = find(cells[0]), find(x)
            if ra != rb:
                parent[ra] = rb

    groups = {}
    for i, (cells, _) in enumerate(cons):
        groups.setdefault(find(cells[0]), []).append(i)

    comps = []
    for _, cids in groups.items():
        cellset = set()
        for ci in cids:
            cellset.update(cons[ci][0])
        comps.append((sorted(cellset), cids))
    comps.sort(key=lambda x: len(x[0]))
    return comps


def enum_component(cells, cons, cids):
    """枚举一个分量的全部一致解。

    返回 (dist, cellmine) 或 (None, None)。
    dist[k] = 用掉 k 颗雷的解法数；cellmine[j][k] = 格子 j 在 k 雷解里是雷的次数。
    """
    n = len(cells)
    idx = {cell: i for i, cell in enumerate(cells)}
    local = [([idx[x] for x in cons[ci][0]], cons[ci][1]) for ci in cids]
    cell_cons = [[] for _ in range(n)]
    for li, (cs, _) in enumerate(local):
        for i in cs:
            cell_cons[i].append(li)
    for i in range(n):
        if not cell_cons[i]:
            return None, None

    need = [nd for _, nd in local]
    cnt = [len(cs) for cs, _ in local]
    assign = [0] * n
    dist = {}
    cellmine = [dict() for _ in range(n)]
    state = {'sols': 0, 'overflow': False}

    def dfs(pos, used):
        if pos == n:
            state['sols'] += 1
            if state['sols'] > CAP_SOLS:
                state['overflow'] = True
                return
            dist[used] = dist.get(used, 0) + 1
            for i in range(n):
                if assign[i]:
                    d = cellmine[i]
                    d[used] = d.get(used, 0) + 1
            return
        for val in (0, 1):
            ok = True
            touched = []
            for li in cell_cons[pos]:
                need[li] -= val
                cnt[li] -= 1
                touched.append(li)
                if need[li] < 0 or need[li] > cnt[li]:
                    ok = False
                    break
            if ok:
                assign[pos] = val
                dfs(pos + 1, used + val)
                assign[pos] = 0
            for li in touched:
                need[li] += val
                cnt[li] += 1
            if state['overflow']:
                return

    dfs(0, 0)
    if state['overflow'] or not dist:
        return None, None
    return dist, cellmine


def _log_comb(n, k):
    if k < 0 or k > n:
        return None
    return math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)


def _to_log(dist):
    return {k: (0.0 if v == 1 else math.log(v)) for k, v in dist.items()}


def _convolve(a, b):
    out = {}
    for ka, la in a.items():
        for kb, lb in b.items():
            k, lv = ka + kb, la + lb
            out[k] = _logadd(out[k], lv) if k in out else lv
    return out


def _logadd(a, b):
    if a == -math.inf:
        return b
    if b == -math.inf:
        return a
    if a < b:
        a, b = b, a
    return a + math.log1p(math.exp(b - a))


def estimate_total_mines(board):
    """不知道雷数时按当前棋盘估一个：用已翻开数字的平均密度外推整盘。

    每个数字格周围有 n 颗雷，把全盘数字加起来除以 8 就是「被数字覆盖到的雷」，
    据此得到密度，再乘以总格数。粗糙但足够给概率求解器当输入。
    """
    nums = board[(board >= 0) & (board <= 8)]
    if nums.size == 0:
        return int((board == FLAG).sum())
    density = float(nums.sum()) / (8.0 * nums.size)
    density = min(max(density, 0.02), 0.40)
    total_cells = board.size
    return int(round(density * total_cells))


def solve_prob(board, total_mines=None):
    """返回每格为雷的概率矩阵；解不出（解太多/无解）返回 None。

    total_mines 省略时用 config.TOTAL_MINES，再省略就按棋盘估算。
    """
    cons, unknown, contradictions = build_constraints(board)
    if not unknown:
        return None
    if total_mines is None:
        total_mines = TOTAL_MINES if TOTAL_MINES and TOTAL_MINES > 0 else estimate_total_mines(board)

    flags = int((board == FLAG).sum())
    n_unk = len(unknown)
    rest = total_mines - flags
    if rest < 0 or rest > n_unk:
        # 旗与雷数对不上（自动标雷时游戏插的旗不一定是真雷，或识别出错）。
        # 夹到可行区间继续，不要整盘放弃。
        rest = min(max(rest, 0), n_unk)

    p = np.zeros(board.shape, dtype=float)
    if not cons:
        p[board == UNK] = rest / float(n_unk)
        return p

    comps = split_components(cons)
    parts = []
    for cells, cids in comps:
        dist, cellmine = enum_component(cells, cons, cids)
        if dist is None:
            return None
        parts.append((cells, dist, cellmine))

    frontier = set()
    for cells, _, _ in parts:
        frontier.update(cells)
    hard = [x for x in unknown if x not in frontier]
    nh = len(hard)

    logs = [_to_log(dist) for _, dist, _ in parts]
    m = len(parts)
    prefix = [{0: 0.0}]
    for i in range(m):
        prefix.append(_convolve(prefix[-1], logs[i]))
    suffix = [None] * (m + 1)
    suffix[m] = {0: 0.0}
    for i in range(m - 1, -1, -1):
        suffix[i] = _convolve(logs[i], suffix[i + 1])

    all_logs = prefix[m]
    u_lo, u_hi = min(all_logs), max(all_logs)
    rest_eff = min(max(rest, u_lo), u_hi + nh)
    if nh == 0 and rest_eff not in all_logs:
        rest_eff = min(all_logs, key=lambda u: abs(u - rest_eff))

    def log_weight(u):
        return _log_comb(nh, rest_eff - u)

    terms = []
    for u, lv in all_logs.items():
        lw = log_weight(u)
        if lw is not None:
            terms.append(lv + lw)
    if not terms:
        return None
    logZ = terms[0]
    for t in terms[1:]:
        logZ = _logadd(logZ, t)
    if logZ == -math.inf:
        logZ = None

    for i, (cells, dist, cellmine) in enumerate(parts):
        other = _convolve(prefix[i], suffix[i + 1])
        tail = {}
        for uo, lo in other.items():
            for ui in dist:
                lw = log_weight(ui + uo)
                if lw is None:
                    continue
                v = lo + lw
                tail[ui] = _logadd(tail[ui], v) if ui in tail else v
        for j, cell in enumerate(cells):
            num = None
            for k, c in cellmine[j].items():
                if k in tail:
                    v = math.log(c) + tail[k]
                    num = v if num is None else _logadd(num, v)
            if num is None:
                p[cell] = 0.0
            elif logZ is None:
                p[cell] = 1.0
            else:
                p[cell] = min(1.0, max(0.0, math.exp(min(0.0, num - logZ))))

    if nh:
        num = None
        for u, lv in all_logs.items():
            lw = log_weight(u)
            if lw is None:
                continue
            left = rest_eff - u
            if left <= 0:
                continue
            v = lv + lw + math.log(left)
            num = v if num is None else _logadd(num, v)
        hp = 0.0
        if num is not None and logZ is not None:
            # 注意是 exp(min(700, d)) 而不是 exp(min(0, d))：
            # d 是「期望剩余雷数」的对数，可以为正，压成 0 会把概率算小很多。
            hp = math.exp(min(700.0, num - logZ)) / float(nh)
        hp = min(max(hp, 0.0), 1.0)
        for cell in hard:
            p[cell] = hp
    return p


def hint(board, top=3, total_mines=None):
    """推不动时给建议。返回 (entries, info) 或 (None, info)。

    entries: [(行, 列, 是雷概率)]，按概率从低到高（越靠前越安全）
    info:    {'contradictions': n, 'total_mines': n, 'unknown': n} 或 {'error': '...'}
    """
    cons, unknown, contradictions = build_constraints(board)
    if total_mines is None:
        total_mines = TOTAL_MINES if TOTAL_MINES and TOTAL_MINES > 0 else estimate_total_mines(board)
    info = {
        'contradictions': contradictions,
        'total_mines': int(total_mines),
        'unknown': len(unknown),
        'flags': int((board == FLAG).sum()),
    }
    if not unknown:
        info['error'] = '没有未翻开的格子'
        return None, info
    p = solve_prob(board, total_mines)
    if p is None:
        info['error'] = '局面太复杂（分量解数超上限）或约束无解，算不出概率'
        return None, info
    entries = sorted(((float(p[r, c]), int(r), int(c)) for r, c in unknown))
    info['error'] = None
    return [(r, c, prob) for prob, r, c in entries[:top]], info
