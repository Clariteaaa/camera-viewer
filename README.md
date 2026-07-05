# 相机数据读取 & 实时可视化

MindVision MV-SUA 系列工业相机 Python 驱动，支持实时显示与二维高斯拟合。

## 功能

- 实时预览（黑白/彩色自动识别）
- 鼠标点击画面按钮控制（QUIT / SAVE / AE / BG / FIT）
- 自动曝光开关
- 二维高斯拟合：激光光斑中心定位 + 束腰测量（矩方法，实时）
- 背景扣除

## 环境要求

- Python 3.8+
- Windows（迈德威视 SDK 仅支持 Windows / Linux）

## 安装

```bash
pip install opencv-python numpy
```

## 相机驱动 & SDK

本程序依赖迈德威视 MVCAMSDK。

1. 从 [迈德威视官网](https://www.mindvision.com.cn/) 下载 **MindVision Camera Platform Setup**（约 180MB）
2. 安装后，SDK 位于安装目录下
3. 将以下 DLL 复制到 `bin/` 目录：
   - `SDK/X64/MVCAMSDK_X64.dll`
   - `SDK/X64/MVImageProcess_X64.DLL`
4. （Linux 用户：安装 `libMVSDK.so` 并置于系统库路径）

## 运行

```bash
python camera_viewer.py
```

## 操作

| 按钮 | 键盘 | 功能 |
|------|------|------|
| QUIT | `q` | 退出 |
| SAVE | `s` | 保存当前帧 |
| AE | `a` | 自动曝光 开/关 |
| BG | `b` | 捕获当前帧作为暗场背景 |
| FIT | `f` | 高斯拟合 开/关 |

## 拟合说明

拟合使用矩方法（Method of Moments），毫秒级完成：

- 先点击 **BG** 捕获背景帧（挡住激光）
- 再打开激光，点击 **FIT** 开启实时拟合
- 画面叠加绿色十字线（中心）、椭圆（束腰 1/e² 半径）、参数文字

束腰定义：`w = 2σ`（强度降至中心 1/e² ≈ 13.5% 处的半径）。
