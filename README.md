# Camera Viewer

迈德威视 MV-SUA 系列工业相机 Python 驱动 — 实时预览 + 二维高斯拟合。

## 快速开始

```bash
git clone https://github.com/Clariteaaa/camera-viewer.git
cd camera-viewer
pip install opencv-python numpy
python camera_viewer.py
```

DLL 已内置在 `bin/`，无需额外下载 SDK。

## 功能

- **实时预览** — 2592×2048 黑白/彩色自动识别
- **图形按钮** — 鼠标点击画面顶部按钮控制，无需键盘焦点
- **自动曝光** — 一键切换手动/自动曝光
- **二维高斯拟合** — 矩方法实时计算激光光斑中心与束腰

## 操作

| 按钮 | 键盘 | 功能 |
|------|------|------|
| **QUIT** | `q` | 退出程序 |
| **SAVE** | `s` | 保存当前帧为 PNG |
| **AE** | `a` | 自动曝光 开/关 |
| **BG** | `b` | 捕获当前帧作为暗场背景 |
| **FIT** | `f` | 高斯拟合 开/关 |

## 高斯拟合使用方法

1. 挡住激光（或关闭），点击 **BG** 捕获暗场背景
2. 打开激光，点击 **FIT** 开启拟合
3. 画面叠加显示：
   - 🟢 十字线 — 光斑中心坐标
   - 🟢 椭圆 — 束腰轮廓（1/e² 强度半径，w = 2σ）
   - 🟢 文字 — 中心 (x₀, y₀)、束腰 (wₓ, wᵧ)、振幅、背景值

## 拟合原理

矩方法（Method of Moments），在光斑周围 256×256 ROI 上计算：

```
总强度    I_total = Σ I(x,y)
中心      x₀ = Σ x·I / I_total,  y₀ = Σ y·I / I_total
方差      σx² = Σ (x-x₀)²·I / I_total
束腰      w = 2σ
```

O(N) 复杂度，每帧微秒级完成，不影响实时帧率。

## 目录结构

```
camera-viewer/
  camera_viewer.py    # 主程序
  mvsdk.py            # 迈德威视 SDK Python 绑定
  bin/                # SDK DLL（64位）
  requirements.txt
  README.md
```

## 依赖

- Python ≥ 3.8
- opencv-python
- numpy
- Windows 10/11（迈德威视 SDK 官方支持）

## 致谢

- 迈德威视 MVCAMSDK
- 官方 Python 示例 `mvsdk.py`
