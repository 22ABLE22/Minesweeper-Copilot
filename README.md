# mine-sweeping

自动玩网页版扫雷：整屏截图 → 切成单个格子 → CNN 识别棋盘状态 → 确定性推理 → 自动左键点开、右键插旗。

## 目录结构

| 文件 | 作用 |
| --- | --- |
| `saolei.py` | 主程序：截图 → 识别 → 推理 → 点击，并监听全局按键 |
| `config.py` | 棋盘区域、行列数、路径、编码表（**换屏幕或窗口必须改**） |
| `cnn_model.py` | CNN 结构定义，训练与推理共用同一份 |
| `train_img.py` | 用 `imgs/` 与 `samples/auto/` 的样本训练（带数据增强），产出 `recognize_new.keras` |
| `collect.py` | 按数字颜色从真实棋盘采集带标签样本到 `samples/auto/` |
| `calibrate.py` | 交互式标定棋盘区域，点两下直接把结果写回 `config.py` |
| `convert_legacy_model.py` | 可选：把 SavedModel 目录转成 `.keras` |
| `recognize_new.keras` | 已训练好的识别模型（Keras 3 格式） |
| `imgs/` | 训练样本，目录名即类别（`-2`、`-1`、`0`–`9`） |
| `samples/auto/` | 每局结束自动采集的样本（本地生成，不入库） |

## 环境

- Windows 10 / 11（`pyautogui` 的截图与点击在 Windows 上最省事）
- Python 3.10 – 3.12，TensorFlow ≥ 2.16（自带 Keras 3）
- 浏览器缩放 100%，系统显示缩放 100%，游戏窗口位置固定

```powershell
conda create -n mine-sweeping python=3.12 -y
conda activate mine-sweeping
pip install -r requirements.txt
```

## 快速开始

1. 浏览器打开 <https://www.minesweeper.cn/>，用「自定义」把行列数设成和 `config.py` 一致（仓库默认 30 行 × 64 列）；
2. 标定棋盘区域（见下一节）；
3. 开局先随便点一格，把棋盘展开；
4. 运行 `python saolei.py`，把鼠标移到棋盘上，按键操作。

## 一、标定棋盘区域

`BG_REGION = (x, y, w, h)` 是棋盘在**屏幕**上的像素区域：`x, y` 是棋盘左上角坐标，`w, h` 是棋盘宽高（不含标题栏、工具栏）。

交互标定：

```powershell
python calibrate.py --rows 30 --cols 64
```

会弹出一张全屏截图：先左键点棋盘**最左上那一格的左上角**，再点**最右下那一格的右下角**，回车后终端打印 `BG_REGION`、校验单格是否接近正方形，并**直接把结果写回 `config.py`**（加 `--no-write` 则只看不改，用 `git checkout config.py` 撤销）。

没有图形界面时用 `python calibrate.py --snap`，它把截图存到 `screenshots/calib.png`，用画图量出坐标后手填。

`config.py` 自带自检：行高与列宽相差超过 0.5 px 会打印警告，说明 `BG_REGION` / `ROWS` / `COLS` 有一个填错了。

## 二、运行

```powershell
python saolei.py
```

把鼠标移到游戏窗口上（不要停在别的窗口，也不要点终端窗口），然后按键：

| 按键 | 作用 |
| --- | --- |
| `A` | 截一次图、推理一次、点一轮 |
| `B` | 连续执行（逐格右键插旗），直到推理不出可点的格子 |
| `Z` | 同 `B`，但只左键点安全格，旗帜完全交给游戏插（网页勾选「自动标雷」） |
| `C` | 显示截图切图结果（弹窗，用来看切得对不对） |
| `S` | 显示识别结果（弹窗，用来看识别得准不准） |
| `R` | 截图并按预测类别存到 `dir/<类别>/`，用来扩充样本 |
| `Q` | 退出 |

**安全阀**：`pyautogui` 默认开启 FAILSAFE——鼠标快速甩到屏幕**左上角**，程序立刻抛异常停下。识别不准导致乱点时用它保命。

**死局为什么会停**：`B` / `Z` 只在有确定性结论时才动手，推理不出任何可操作的格子就打印提示并停下——进程不退出，按键监听仍然有效。这时自己点开一格（猜一步），再按 `B`（或 `Z`）就接着跑。

### Z 模式（自动标雷）

网页右上角勾选「自动标雷」后，游戏会在你翻开格子时自己插旗：翻开一个格子后，若某个数字周围剩下的未翻开格数正好等于它剩余的雷数，游戏就把这些格子直接标成旗。

所以 Z 模式只做一件事：左键点开推理出的安全格，**完全不发右键**。这不但省掉了每面旗一次的右键点击和等待，还避开了一个坑——网页的右键是「插旗 / 取消」的开关，程序要是对游戏刚替你插好的旗再点一次，旗就被取消了，下一轮又插回来，来回折腾。

不勾这个选项也能按 Z：这时没人插旗，相当于盲扫，速度最快，也不会误标。

## 三、模型文件

主程序与训练脚本都按**脚本所在目录**定位 `recognize_new.keras`，从任何工作目录启动都能找到模型。

模型是 Keras 3 的 `.keras` 格式（V3），这也是 `load_model` 当前支持的原生格式。训练一次即覆盖仓库里的同名文件，不需要改代码。

如果手上只有 SavedModel 目录（`saved_model.pb` + `variables/`），可以转换：

```powershell
pip install tf-keras==2.21.0      # 版本号与 tensorflow 对应
python convert_legacy_model.py recognize_new recognize_new.keras
```

转换脚本不会直接另存为 `.keras`：SavedModel 里的层带有 `batch_input_shape` 参数，Keras 3 读这种文件会报
`Unrecognized keyword arguments passed to Conv2D: {'batch_input_shape': ...}`。
所以它用 `cnn_model.build_model()` 新建同构网络，再逐层 `set_weights` 复制权重。

## 四、训练自己的模型

```powershell
python train_img.py                                   # imgs/ + samples/auto/ 一起训
python train_img.py --epochs 20 --per-class 200       # 每类增强到 200 张
python train_img.py --sources imgs --no-augment       # 只用原始样本
```

默认会把每类用增强补到 150 张（随机平移 ±3px、缩放 0.88–1.12、亮度/对比度扰动、轻微高斯模糊），
验证集只取**未增强**的原始图片（每类 15%），所以打印出来的准确率代表真实分布下的水平。
训练结束会输出混淆矩阵和逐类准确率，低于 95% 的类别会单独列出来——**先看这张表，再拿去实战**。

数据要求：`imgs/<类别>/*.png`，目录名必须是 `config.net_encoder` 的键：

| 目录名 | 含义 |
| --- | --- |
| `-2` | 未翻开的格子 |
| `-1` | 旗子 |
| `0`–`8` | 已翻开、周围八格的地雷数 |
| `9` | 翻开的地雷 |

### 采集真实样本（推荐先做这一步）

数字格的颜色就是数字本身（1 蓝、2 绿、3 红、4 靛蓝、5 暗红、6 青、7 黑），所以
`collect.py` 直接按颜色给格子打标签，**不经过模型预测**——不会出现「6 被认成 4，于是样本写进 4 的目录」这种把错误标签固化下来的情况。

`saolei.py` 在**每局结束**（`B` / `Z` 停下来时）自动跑一次采集，把整盘数字格存进 `samples/auto/<数字>/`，终端会打印本次采到多少张。
也可以在棋盘明确展开时单独跑：

```powershell
python collect.py                 # 抓当前屏幕
python collect.py --sheet         # 顺带输出缩略图拼板，肉眼抽查十秒
python collect.py --dry-run       # 只统计不存图
```

颜色判不准、太淡、被遮挡的格子会被直接丢弃（宁可少收，也不收错标签）。多打几局、每类攒到 100 张以上就值得重训。

`R` 键走的是另一条路：按**模型预测**存进 `dir/<类别>/`，属于伪标签，适合补 `-2`（未翻开）、`-1`（旗子）、`0`（空块）、`9`（地雷）这些形状类；数字类请用 `collect.py`，否则错标的样本会让模型越训越差。

### 重训的完整流程

1. 新机器或换了窗口：`python calibrate.py --rows 30 --cols 64`，结果自动写进 `config.py`；
2. 打几局，让 `saolei.py` 把样本攒进 `samples/auto/`（每类 100 张以上，重点补 5、6、7）；
3. `python train_img.py --epochs 20`，看输出里的混淆矩阵，确认 6 不再被判成 4、数字类都在 96% 以上；
4. 再跑 `python saolei.py` 实战。

## 五、常见问题

**按 A / B 没反应，或者乱点**
棋盘区域或行列数和实际游戏对不上。先按 `C` 看切图：格子错位或一半一半就是 `BG_REGION` 不对；切出来不是 `ROWS × COLS` 个格子就是行列数不对。

**弹出 matplotlib 窗口后像卡住**
`C` / `S` 用独立进程弹窗，关掉窗口就继续，属于正常现象。

**识别出大量错误数字**
多半是显示缩放或浏览器缩放不是 100%，格子被压扁后与训练样本不一致。把缩放调回 100%，或重新标定并补样本重训。

**某几个数字（比如 6）总是认错**
训练样本太少且不平衡：仓库自带的 `imgs/6` 只有 4 张，而 1–4 各 20 张，模型对 6 基本没有泛化能力。打几局让 `samples/auto/` 攒够样本，再 `python train_img.py --epochs 20` 重训，并在混淆矩阵里确认 6 与 4 不再互串。

**`ModuleNotFoundError`**
依赖没装全，重跑 `pip install -r requirements.txt`。

## 六、算法简述

1. **识别**：整屏裁剪出每个格子，缩放到 32×32，一次性成批送进 CNN（比逐格推理快得多），`argmax` 出 11 类，再按 `net_decoder` 还原成 `-2 / -1 / 0..9`。
2. **基础推理**：某数字周围未翻开格数 == 该数字 → 全是旗；某数字周围旗数 == 该数字 → 其余全可点。
3. **子集推理**：对每个数字格收集「周围未翻开格子集合 S 与剩余雷数 n」，若 S1 ⊂ S2：
   - `n2 - n1 == |S2 - S1|` → `S2 - S1` 全是雷；
   - `n1 == n2` → `S2 - S1` 全都不是雷。
   交叉集还能再推一层：`|S1| - |S1 ∩ S2| == n1 - n2` → `S1` 独有的全是雷、`S2` 独有的都可点。
4. **执行**：把 (行, 列) 换算成屏幕坐标 `(x + (c - 0.5) * cols_size, y + (r - 0.5) * row_size)`，左键点开；`B` 模式再逐个右键插旗，`Z` 模式不发右键、交给游戏自动插旗；没有可操作格子时打印提示并停下，等你猜一步后再继续。

推理只用确定性结论，遇到必须猜的局面会停下来。

## License

MIT，见 [LICENSE](LICENSE)。
