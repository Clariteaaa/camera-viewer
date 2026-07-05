"""
相机数据读取 & 实时可视化
MindVision MVSUA505GM-T1V-C (5MP 黑白 USB3.0)

操作: 点击画面上方的按钮，或键盘 q/s/a
"""

import os

_here = os.path.dirname(os.path.abspath(__file__))
_dll_dir = os.path.join(_here, "bin")
if os.path.exists(_dll_dir):
    os.add_dll_directory(_dll_dir)
    os.environ["PATH"] = _dll_dir + os.pathsep + os.environ.get("PATH", "")

import cv2
import numpy as np
import mvsdk
import platform
import time
from collections import deque


# ============================================================
# 按钮系统：在画面上画可点击的按钮
# ============================================================

class Button:
    """画面上的可点击按钮"""
    def __init__(self, name, x, y, w, h, color=(60, 60, 60), hover_color=(100, 100, 100)):
        self.name = name
        self.x, self.y = x, y
        self.w, self.h = w, h
        self.color = color
        self.hover_color = hover_color

    def contains(self, px, py):
        return self.x <= px <= self.x + self.w and self.y <= py <= self.y + self.h

    def draw(self, img, hover=False):
        c = self.hover_color if hover else self.color
        cv2.rectangle(img, (self.x, self.y), (self.x + self.w, self.y + self.h), c, -1)
        cv2.rectangle(img, (self.x, self.y), (self.x + self.w, self.y + self.h), (180, 180, 180), 1)
        text_size = cv2.getTextSize(self.name, cv2.FONT_HERSHEY_SIMPLEX, 0.75, 2)[0]
        tx = self.x + (self.w - text_size[0]) // 2
        ty = self.y + (self.h + text_size[1]) // 2
        cv2.putText(img, self.name, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)


class ButtonBar:
    """按钮栏"""
    def __init__(self, width):
        self.buttons = []
        self.hover_idx = -1
        self.bar_h = 56
        self.bar_bg = (40, 40, 40)

    def add(self, name):
        btn_w = 130
        gap = 10
        x = gap + len(self.buttons) * (btn_w + gap)
        y = 8
        self.buttons.append(Button(name, x, y, btn_w, self.bar_h - 16))
        return self

    def check_hover(self, px, py, frame_h):
        """检测鼠标悬停，返回被悬停的按钮索引"""
        # 如果画面被 resize 了，需要换算坐标
        # 这里 py 是相对于窗口的坐标，需要判断是否在按钮栏区域
        self.hover_idx = -1
        for i, btn in enumerate(self.buttons):
            if btn.contains(px, py):
                self.hover_idx = i
                return i
        return -1

    def draw(self, img):
        """在图像顶部画按钮栏"""
        h, w = img.shape[:2]
        # 顶部半透明背景条
        overlay = img.copy()
        cv2.rectangle(overlay, (0, 0), (w, self.bar_h), self.bar_bg, -1)
        cv2.addWeighted(overlay, 0.75, img, 0.25, 0, img)
        for i, btn in enumerate(self.buttons):
            btn.draw(img, hover=(i == self.hover_idx))


# 鼠标回调状态
mouse_state = {"x": 0, "y": 0, "clicked_btn": None}

def mouse_callback(event, x, y, flags, param):
    mouse_state["x"] = x
    mouse_state["y"] = y
    if event == cv2.EVENT_LBUTTONDOWN:
        mouse_state["clicked_btn"] = param.check_hover(x, y, 0)


# ============================================================
# 二维高斯拟合（矩方法）
# ============================================================

def gaussian_2d_moments(img, roi_size=256):
    """
    矩方法二维高斯拟合。返回 (x0, y0, wx, wy, amplitude, background)
    x0, y0: 光斑中心（全局坐标）
    wx, wy: 束腰 (1/e² 半径, 即 2σ)
    返回 None 如果拟合失败
    """
    h, w = img.shape

    # 1. 找峰值位置作为粗定位
    max_val = img.max()
    if max_val <= 0:
        return None
    cy, cx = np.unravel_index(np.argmax(img), img.shape)

    # 2. 裁切 ROI
    half = roi_size // 2
    x1 = max(0, cx - half)
    x2 = min(w, cx + half)
    y1 = max(0, cy - half)
    y2 = min(h, cy + half)
    roi = img[y1:y2, x1:x2].astype(np.float64)

    # 3. 估算背景（ROI 边缘中值）
    border = np.concatenate([
        roi[0, :], roi[-1, :], roi[:, 0], roi[:, -1]
    ])
    bg = np.median(border)

    # 扣除背景
    roi_sub = roi - bg
    roi_sub[roi_sub < 0] = 0

    total = roi_sub.sum()
    if total <= 0:
        return None

    # 4. 矩方法
    yy, xx = np.mgrid[0:roi_sub.shape[0], 0:roi_sub.shape[1]]
    x0_local = (xx * roi_sub).sum() / total
    y0_local = (yy * roi_sub).sum() / total

    dx = xx - x0_local
    dy = yy - y0_local
    var_x = (dx * dx * roi_sub).sum() / total
    var_y = (dy * dy * roi_sub).sum() / total

    if var_x <= 0 or var_y <= 0:
        return None

    sigma_x = np.sqrt(var_x)
    sigma_y = np.sqrt(var_y)

    # 全局坐标
    x0 = x1 + x0_local
    y0 = y1 + y0_local
    wx = 2.0 * sigma_x    # 1/e² 半径
    wy = 2.0 * sigma_y

    amplitude = roi_sub.max()

    return x0, y0, wx, wy, amplitude, bg


def draw_fit_overlay(display, x0, y0, wx, wy, amp, bg):
    """在图像上画拟合结果叠加"""
    xc, yc = int(round(x0)), int(round(y0))
    h, w = display.shape[:2]

    # 十字线（绿）
    cross_len = max(int(wx * 1.2), 20)
    cv2.line(display, (max(0, xc - cross_len), yc),
             (min(w - 1, xc + cross_len), yc), (0, 255, 0), 1)
    cv2.line(display, (xc, max(0, yc - cross_len)),
             (xc, min(h - 1, yc + cross_len)), (0, 255, 0), 1)

    # 束腰椭圆 (1/e²)
    if wx > 0 and wy > 0:
        axes = (int(wx), int(wy))
        if axes[0] > 0 and axes[1] > 0:
            cv2.ellipse(display, (xc, yc), axes, 0, 0, 360, (0, 255, 0), 1)

    # HUD 文字
    lines = [
        f"Center: ({x0:.1f}, {y0:.1f})",
        f"Beam waist wx={wx:.1f} wy={wy:.1f} px",
        f"Amplitude: {amp:.0f} | BG: {bg:.0f}",
    ]
    for i, line in enumerate(lines):
        cv2.putText(display, line, (10, 130 + i * 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)


# ============================================================
# 主程序
# ============================================================

def main():
    # --- 1. 枚举相机 ---
    DevList = mvsdk.CameraEnumerateDevice()
    if len(DevList) < 1:
        print("❌ 未找到相机！请检查 USB 连接。")
        return

    for i, dev in enumerate(DevList):
        print(f"  [{i}] {dev.GetFriendlyName()}  SN:{dev.GetSn()}  {dev.GetPortType()}")

    dev_info = DevList[0]
    print(f"\n✅ {dev_info.GetFriendlyName()}")

    # --- 2. 打开相机 ---
    try:
        hCamera = mvsdk.CameraInit(dev_info, -1, -1)
    except mvsdk.CameraException as e:
        print(f"❌ 打开失败: {e.message}")
        return

    # --- 3. 获取能力 ---
    cap = mvsdk.CameraGetCapability(hCamera)
    mono = (cap.sIspCapacity.bMonoSensor != 0)
    max_w = cap.sResolutionRange.iWidthMax
    max_h = cap.sResolutionRange.iHeightMax
    print(f"  {max_w}x{max_h}, {'黑白' if mono else '彩色'}")

    # --- 4. ISP 设置 ---
    if mono:
        mvsdk.CameraSetIspOutFormat(hCamera, mvsdk.CAMERA_MEDIA_TYPE_MONO8)
    else:
        mvsdk.CameraSetIspOutFormat(hCamera, mvsdk.CAMERA_MEDIA_TYPE_BGR8)

    # --- 5. 采集模式 ---
    mvsdk.CameraSetTriggerMode(hCamera, 0)
    mvsdk.CameraSetAeState(hCamera, 0)
    mvsdk.CameraSetExposureTime(hCamera, 30 * 1000)

    # 预设 AE 参数（开 AE 时生效，防闪烁）
    mvsdk.CameraSetAeTarget(hCamera, 128)                         # 目标亮度 128（中灰）
    mvsdk.CameraSetAeExposureRange(hCamera, 100, 100 * 1000)      # 曝光范围 0.1ms ~ 100ms
    mvsdk.CameraSetAeAnalogGainRange(hCamera, 0, 100)             # 增益范围 0~100
    mvsdk.CameraSetAntiFlick(hCamera, True)                       # 启用抗闪烁

    mvsdk.CameraPlay(hCamera)
    print("  采集开始\n")

    buf_size = max_w * max_h * (1 if mono else 3)
    pFrameBuffer = mvsdk.CameraAlignMalloc(buf_size, 16)

    # --- 6. 状态 ---
    fps_buf = deque(maxlen=30)
    t_last = time.perf_counter()
    t_fps_update = t_last
    fps_display = 0
    ae_state = False
    manual_exposure_us = 30 * 1000  # 默认 30ms
    manual_gain = 100               # 默认增益（0-300，黑白相机典型值）

    # --- 7. 窗口 + 按钮 ---
    win_name = f"Camera - {dev_info.GetFriendlyName()}"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win_name, 1280, 960)

    # 手动曝光/增益滑块（仅 AE 关闭时生效）
    def exp_callback(v):
        pass  # 在主循环里读取
    def gain_callback(v):
        pass

    cv2.createTrackbar("Exposure(ms)", win_name, 30, 200, exp_callback)
    cv2.createTrackbar("Gain", win_name, 100, 300, gain_callback)

    # 创建按钮栏
    btn_bar = ButtonBar(1280)
    btn_bar.add("QUIT").add("SAVE").add("AE").add("BG").add("FIT")
    cv2.setMouseCallback(win_name, mouse_callback, btn_bar)

    print(" QUIT=退出 SAVE=保存 AE=自动曝光 BG=捕获背景 FIT=拟合开关")
    print(" 手动模式下拖动上方滑块调节曝光/增益")
    print("=" * 50)

    prev_exp_tb = 30   # 上次滑块值
    prev_gain_tb = 100

    # --- 8. 主循环 ---
    while True:
        # 取帧 + ISP
        try:
            pRawData, FrameHead = mvsdk.CameraGetImageBuffer(hCamera, 200)
        except mvsdk.CameraException as e:
            if e.error_code != mvsdk.CAMERA_STATUS_TIME_OUT:
                print(f"取帧错误: {e.error_code}")
            # 仍需处理鼠标事件
            cv2.waitKey(1)
            continue

        mvsdk.CameraImageProcess(hCamera, pRawData, pFrameBuffer, FrameHead)
        mvsdk.CameraReleaseImageBuffer(hCamera, pRawData)

        if platform.system() == "Windows":
            mvsdk.CameraFlipFrameBuffer(pFrameBuffer, FrameHead, 1)

        # numpy 转换
        frame_data = (mvsdk.c_ubyte * FrameHead.uBytes).from_address(pFrameBuffer)
        frame = np.frombuffer(frame_data, dtype=np.uint8)
        frame = frame.reshape((
            FrameHead.iHeight, FrameHead.iWidth,
            1 if FrameHead.uiMediaType == mvsdk.CAMERA_MEDIA_TYPE_MONO8 else 3
        ))

        # FPS
        now = time.perf_counter()
        fps_buf.append(now - t_last)
        t_last = now
        if now - t_fps_update > 0.5 and len(fps_buf) > 0:
            fps_display = 1.0 / (sum(fps_buf) / len(fps_buf))
            t_fps_update = now

        # 显示准备
        if mono:
            display = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        else:
            display = frame

        # --- 背景扣除 + 拟合 ---
        if fit_state and bg_frame is not None:
            # 用 float 做减法，避免溢出
            sub = frame.astype(np.float32).squeeze() - bg_frame.astype(np.float32)
            sub[sub < 0] = 0
            sub_u8 = sub.astype(np.uint8)
            fit_result = gaussian_2d_moments(sub_u8)
        elif fit_state and bg_frame is None:
            fit_result = gaussian_2d_moments(frame.squeeze())
        else:
            fit_result = None

        # --- 画按钮栏 ---
        btn_bar.draw(display)

        # --- 手动曝光/增益（AE 关闭时响应滑块） ---
        if not ae_state:
            exp_tb = cv2.getTrackbarPos("Exposure(ms)", win_name)
            gain_tb = cv2.getTrackbarPos("Gain", win_name)
            exp_tb = max(1, exp_tb)
            if exp_tb != prev_exp_tb or gain_tb != prev_gain_tb:
                prev_exp_tb = exp_tb
                prev_gain_tb = gain_tb
                manual_exposure_us = exp_tb * 1000
                manual_gain = gain_tb
                mvsdk.CameraSetExposureTime(hCamera, manual_exposure_us)
                mvsdk.CameraSetAnalogGain(hCamera, manual_gain)

        # --- 过曝检测 ---
        raw = frame.squeeze()
        overexposed = np.any(raw >= 255)

        # --- HUD 信息（按钮下方） ---
        exp_tb = cv2.getTrackbarPos("Exposure(ms)", win_name)
        gain_tb = cv2.getTrackbarPos("Gain", win_name)
        if ae_state:
            ae_str = "AE: ON"
        else:
            ae_str = f"Exp={exp_tb}ms Gain={gain_tb}"

        status_parts = [
            f"FPS: {fps_display:.1f}",
            f"{FrameHead.iWidth}x{FrameHead.iHeight}",
            "MONO8" if mono else "BGR8",
            ae_str,
            f"BG: {'OK' if bg_frame is not None else '--'}",
            f"FIT: {'ON' if fit_state else 'OFF'}",
        ]
        cv2.putText(display, " | ".join(status_parts),
                    (10, 76), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2, cv2.LINE_AA)

        if overexposed:
            cv2.putText(display, "! OVEREXPOSED !",
                        (10, 115), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3, cv2.LINE_AA)

        # --- 画拟合叠加 ---
        if fit_result is not None:
            x0, y0, wx, wy, amp, fit_bg = fit_result
            draw_fit_overlay(display, x0, y0, wx, wy, amp, fit_bg)

        cv2.imshow(win_name, display)

        # --- 检测按钮点击 ---
        clicked = mouse_state.get("clicked_btn")
        if clicked is not None:
            btn_name = btn_bar.buttons[clicked].name
            mouse_state["clicked_btn"] = None  # 消费掉

            if "QUIT" in btn_name:
                print("🛑 用户点击 QUIT")
                break
            elif "SAVE" in btn_name:
                fname = f"capture_{time.strftime('%Y%m%d_%H%M%S')}.png"
                cv2.imwrite(fname, frame.squeeze())
                print(f"💾 已保存: {fname}")
            elif btn_name == "AE":
                ae_state = not ae_state
                mvsdk.CameraSetAeState(hCamera, 1 if ae_state else 0)
                if not ae_state:
                    mvsdk.CameraSetExposureTime(hCamera, manual_exposure_us)
                    mvsdk.CameraSetAnalogGain(hCamera, manual_gain)
                print(f"🔆 自动曝光: {'ON' if ae_state else 'OFF'}")
            elif "BG" in btn_name:
                bg_frame = frame.squeeze().astype(np.float32)
                print(f"📷 背景已捕获 (均值={bg_frame.mean():.1f})")
            elif "FIT" in btn_name:
                fit_state = not fit_state
                print(f"📐 拟合: {'ON' if fit_state else 'OFF'}")

        # --- 键盘事件（备用） ---
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("🛑 键盘退出")
            break
        elif key == ord('s'):
            fname = f"capture_{time.strftime('%Y%m%d_%H%M%S')}.png"
            cv2.imwrite(fname, frame if mono else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
            print(f"💾 已保存: {fname}")
        elif key == ord('a'):
            ae_state = not ae_state
            mvsdk.CameraSetAeState(hCamera, 1 if ae_state else 0)
            if not ae_state:
                mvsdk.CameraSetExposureTime(hCamera, manual_exposure_us)
                mvsdk.CameraSetAnalogGain(hCamera, manual_gain)
            print(f"🔆 自动曝光: {'ON' if ae_state else 'OFF'}")
        elif key == ord('b'):
            bg_frame = frame.squeeze().astype(np.float32)
            print(f"📷 背景已捕获 (均值={bg_frame.mean():.1f})")
        elif key == ord('f'):
            fit_state = not fit_state
            print(f"📐 拟合: {'ON' if fit_state else 'OFF'}")

    # --- 9. 清理 ---
    mvsdk.CameraUnInit(hCamera)
    mvsdk.CameraAlignFree(pFrameBuffer)
    cv2.destroyAllWindows()
    print(f"\n退出。")


if __name__ == "__main__":
    main()
