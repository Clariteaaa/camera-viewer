# Camera Viewer

迈德威视 MindVision MV-SUA 系列工业相机 Python 驱动。实时预览、二维高斯拟合、激光光斑束腰测量。

## 快速开始

```bash
git clone https://github.com/Clariteaaa/camera-viewer.git
cd camera-viewer
pip install -r requirements.txt
python camera_viewer.py
```

DLL 已内置，无需额外下载 SDK。

## 界面

画面顶部大按钮，鼠标点击操作：

| 按钮 | 功能 |
|------|------|
| **QUIT** | 退出 |
| **SAVE** | 保存当前帧为 PNG |
| **AE** | 自动曝光 开/关 |
| **BG** | 捕获暗场背景 |
| **FIT** | 高斯拟合 开/关 |
| **SET** | 手动输入曝光和增益 |
| **CMAP** | JET 彩色 / 灰度切换 |

## 手动曝光设置

点击 **SET** → 画面底部出现输入栏 → 键盘输入数字 → **Enter** 确认：

1. 输入曝光时间（ms）→ Enter
2. 输入增益（0-300）→ Enter → 生效

ESC 取消，Backspace 删除。视频不中断。

## 高斯拟合

矩方法（Method of Moments），每帧微秒级完成。

1. 挡住激光，点击 **BG** 捕获暗场背景
2. 打开激光，点击 **FIT** 开启拟合
3. 画面叠加：
   - 🟢 十字线 → 光斑中心
   - 🟢 椭圆 → 束腰轮廓（w = 2σ, 1/e² 半径）
   - 🟢 文字 → 中心坐标(px)、束腰(μm)、模场直径 MFD

## 拟合显示

```
Center: (1296.0, 1024.0) px
Waist wx=160.0 um  wy=150.0 um
MFD = 2.480
```

- 束腰从像素换算为 μm（3.2 μm/px）
- MFD = (wx_μm + wy_μm) / 2 / 62.5
- 过曝时显示红色 **OVEREXPOSED** 警告

## 目录

```
camera-viewer/
  camera_viewer.py    # 主程序
  mvsdk.py            # 迈德威视 SDK 绑定
  bin/                # SDK DLL (x64)
  requirements.txt
```
