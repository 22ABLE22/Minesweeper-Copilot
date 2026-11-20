# mine-sweeping —— 让 AI 替你玩网页版扫雷

对屏幕截图做卷积神经网络识别，再用逻辑推理自动点击 / 插旗。
本仓库在原版（hyper-zlc/mine-sweeping 的那套思路）基础上做了三件事：

1. **适配 Keras 3**：TensorFlow 2.16 起自带 Keras 3，旧的 SavedModel 目录已经加载不了，现在统一用 `.keras` 模型文件；
2. **可复现**：依赖名、路径、标定步骤写成文档和脚本，换台机器照着做就能跑通；
3. **修掉老问题**：模型路径不再依赖当前工作目录、记录样本时会自动建类别目录。

## 目录结构

| 文件 | 作用 |
| --- | --- |
| `saolei.py` | 主程序：截图 → CNN 识别 → 推理 → 自动点击，并监听按键 |
| `config.py` | 棋盘位置、行列数、路径、编码表（**换屏幕/窗口必须改**） |
| `cnn_model.py` | CNN 结构定义，训练与推理共用同一份，避免格式漂移 |
| `train_img.py` | 用 `imgs/` 里的样本训练，产出 `recognize_new.keras` |
| `calibrate.py` | 交互式标定棋盘区域，直接打印可粘贴的配置 |
| `convert_legacy_model.py` | 把老的 SavedModel 目录转成 `.keras`（一般用不到，已提供转换好的模型） |
| `recognize_new.keras` | 已训练好的识别模型（Keras 3 格式） |
| `model_and_data.zip` | 原始训练数据与老模型：解压出 `imgs/` 和 `recognize_new/` |

## 环境

- Windows 10 / 11（`pyautogui` 的点击、截图在 Windows 上最省事）
- Python 3.10 – 3.12，TensorFlow ≥ 2.16（自带 Keras 3）
- 浏览器缩放保持 100%，游戏窗口位置固定

```powershell
conda create -n mine-sweeping python=3.12 -y
conda activate mine-sweeping
pip install -r requirements.txt
```

> 两个常见装错：PyPI 上**没有**名为 `PIL` 和 `tensorflow2` 的包，正确写法是 `pillow` 和 `tensorflow`。
> 代码里 `from PIL import Image` 是 pillow 提供的模块名，装了 pillow 就能用。

把 `model_and_data.zip` 解压到仓库根目录，得到 `imgs/`（训练集）和 `recognize_new/`（老模型）。
只跑推理的话，仓库里已经带了 `recognize_new.keras`，不用解压也行。

## 一、打开游戏

浏览器访问 <https://www.minesweeper.cn/>（旧 README 里的 `www.saolei.org.cn` 已经打不开了）。

- 用「自定义」把行列数设成和 `config.py` 里的 `ROWS` / `COLS` 一致。仓库里提交的那组数是原作者机器上的标定结果，在你机器上**一定要自己标定**。
- 想玩标准难度就改成 `ROWS = 16`、`COLS = 30`，然后重新标定区域。
- 开局先随便点一格把棋盘打开，剩下的交给程序。

## 二、标定棋盘区域（最容易错的一步）

`BG_REGION = (x, y, w, h)` 是棋盘在**屏幕**上的像素区域：`x, y` 是棋盘左上角坐标，`w, h` 是棋盘宽高（不含标题栏、工具栏）。

交互标定：

```powershell
python calibrate.py --rows 25 --cols 53
```

会弹出一张全屏截图，先左键点棋盘**最左上那一格的左上角**，再点**最右下那一格的右下角**，回车后终端会打印可直接粘贴到 `config.py` 的 `BG_REGION`，并校验单格是否接近正方形。

没有图形界面时用 `python calibrate.py --snap`，它会存一张 `screenshots/calib.png`，用画图量坐标后手填。

`config.py` 里还有一个自检：行高和列宽相差超过 0.5px 就会打印警告，说明 `BG_REGION` / `ROWS` / `COLS` 有一个填错了。

## 三、运行

```powershell
python saolei.py
```

把鼠标移到游戏窗口上（不要停在别的窗口，也不要点程序的黑框），然后按键：

| 按键 | 作用 |
| --- | --- |
| `A` | 截一次图、推理一次、点一轮 |
| `B` | 连续执行，直到推理不出可点的格子 |
| `C` | 显示截图结果（会弹窗，用来看切图对不对） |
| `S` | 显示识别结果（弹窗，用来看棋盘识别得准不准） |
| `R` | 截一张图并按预测类别存到 `dir/<类别>/`，用来扩充训练集 |
| `Q` | 退出 |

**安全阀**：`pyautogui` 默认开启 FAILSAFE——只要鼠标快速甩到屏幕**左上角**，程序会立刻抛异常并停下。识别不准导致乱点时用它保命。

## 四、模型与 Keras 3 迁移说明

老版本读的是 SavedModel 目录 `recognize_new/`：

```python
model = load_model('recognize_new')   # Keras 2 时代可以，Keras 3 不行
```

Keras 3 只认 `.keras`（V3）和 `.h5`(legacy)，于是报：

```text
ValueError: File format not supported: filepath=recognize_new.
Keras 3 only supports V3 `.keras` files, legacy H5 format files (`.h5` extension)...
```

现在 `saolei.py` 改成按脚本所在目录加载 `recognize_new.keras`，从任何工作目录启动都不会找不到模型。

如果你手上还有别的老模型要转：

```powershell
pip install tf-keras==2.21.0      # 版本号要和 tensorflow 对应
python convert_legacy_model.py    # recognize_new/ -> recognize_new.keras
```

注意不能简单用 `tf_keras` 直接另存 `.keras`：老模型用 `batch_input_shape` 建的层，Keras 3 读出来会报
`Unrecognized keyword arguments passed to Conv2D: {'batch_input_shape': ...}`。
所以转换脚本是用 `cnn_model.build_model()` 新建同构网络、逐层 `set_weights` 复制权重。
现有 `recognize_new.keras` 就是这么来的，转换后与原模型在样本上输出完全一致（逐元素最大差 0.0）。

## 五、训练自己的模型

```powershell
python train_img.py                       # 默认 1000 样本、4 epoch
python train_img.py --samples 3000 --epochs 8 --no-plot
```

数据目录要求：`imgs/<类别>/*.png`，目录名必须是 `config.net_encoder` 的键：

| 目录名 | 含义 |
| --- | --- |
| `-2` | 未翻开的格子 |
| `-1` | 旗子 |
| `0`–`8` | 已翻开、周围 8 格的地雷数 |
| `9` | 翻开的地雷 |

扩充样本的正确姿势：用 `S` 键先确认程序认得出当前棋盘，再按 `R` 键批量存图到 `dir/`；
人工挑过（删掉明显错标的）后再挪进 `imgs/<类别>/` 重训。

## 六、常见问题

**按 A/B 没反应，鼠标乱点**
棋盘区域或行列数和实际游戏对不上。先按 `C` 看切图：如果格子和棋盘错位/一半一半，就是 `BG_REGION` 不对；如果切出来不是 25×53 个格子，就是 `ROWS`/`COLS` 不对。

**弹出 matplotlib 窗口后程序像卡住**
`C`/`S` 用独立进程弹窗，关掉窗口就会继续，这是正常的。

**识别出大量错误数字**
多半是显示器缩放（125%/150%）或浏览器缩放不是 100%，导致 32×32 的格子被压扁。把系统缩放设成 100% 或重新标定后重训。

**`ModuleNotFoundError: No module named 'PIL'`**
装的是 `PIL` 这个名字（装不上）而不是 `pillow`。`pip install pillow`。

**想换回旧版 TensorFlow**
可以装 `tensorflow==2.15.*`（自带 Keras 2），此时 `saolei.py` 直接读 `recognize_new/` 老模型也能跑，但本仓库的 `.keras` 路径不需要改，两者都兼容。

## 七、算法简述

1. **识别**：整屏裁剪出每个格子，缩放到 32×32，一次性成批送进 CNN（比逐格推理快得多），`argmax` 出 11 类，再按 `net_decoder` 还原成 `-2 / -1 / 0..9`。
2. **基础推理**：某数字周围未翻开格数 == 该数字 → 全是旗；某数字周围旗数 == 该数字 → 其余全可点。
3. **子集推理**：对每个数字格收集"它周围未翻开格子集合 S 与剩余雷数 n"，若 S1 ⊂ S2：
   - `n2 - n1 == |S2 - S1|` → `S2 - S1` 全是雷；
   - `n1 == n2` → `S2 - S1` 全都不是雷。
   交叉集（`S1 ∩ S2`）还能再推出一层：`|S1|-|S1∩S2| == n1-n2` → `S1` 独有的全是雷、`S2` 独有的都可点。
4. **执行**：把 (行, 列) 换算成屏幕坐标 `(x + (c-0.5)*cols_size, y + (r-0.5)*row_size)`，左键点开、右键插旗；没有可操作格子时把鼠标移到左上角提示结束。

推理只用到确定性的结论，遇到"必须猜"的局面会停下来（这也正是需要人接手的时候）。

## License

MIT，见 [LICENSE](LICENSE)。
