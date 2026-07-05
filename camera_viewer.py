"""
相机数据读取 & 实时可视化 — MindVision MV-SUA 系列
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
# 按钮系统
# ============================================================

class Button:
    def __init__(self, name, x, y, w, h, color=(60, 60, 60), hover_color=(120, 120, 120)):
        self.name = name
        self.x, self.y, self.w, self.h = x, y, w, h
        self.color = color
        self.hover_color = hover_color

    def contains(self, px, py):
        return self.x <= px <= self.x + self.w and self.y <= py <= self.y + self.h

    def draw(self, img, hover=False):
        c = self.hover_color if hover else self.color
        cv2.rectangle(img, (self.x, self.y), (self.x + self.w, self.y + self.h), c, -1)
        cv2.rectangle(img, (self.x, self.y), (self.x + self.w, self.y + self.h), (180, 180, 180), 2)
        ts = cv2.getTextSize(self.name, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)[0]
        tx = self.x + (self.w - ts[0]) // 2
        ty = self.y + (self.h + ts[1]) // 2
        cv2.putText(img, self.name, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)


class ButtonBar:
    BTN_W, GAP = 180, 16
    BAR_H = 100

    def __init__(self):
        self.buttons = []
        self.hover_idx = -1

    def add(self, name):
        x = self.GAP + len(self.buttons) * (self.BTN_W + self.GAP)
        self.buttons.append(Button(name, x, 16, self.BTN_W, self.BAR_H - 32))

    def check_hover(self, px, py):
        self.hover_idx = -1
        for i, btn in enumerate(self.buttons):
            if btn.contains(px, py):
                self.hover_idx = i
                return i
        return -1

    def draw(self, img):
        h, w = img.shape[:2]
        overlay = img.copy()
        cv2.rectangle(overlay, (0, 0), (w, self.BAR_H), (35, 35, 35), -1)
        cv2.addWeighted(overlay, 0.75, img, 0.25, 0, img)
        for i, btn in enumerate(self.buttons):
            btn.draw(img, hover=(i == self.hover_idx))


mouse_state = {"clicked_btn": None}

def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        idx = param.check_hover(x, y)
        if idx >= 0:
            mouse_state["clicked_btn"] = idx

# ============================================================
# 二维高斯拟合（矩方法）
# ============================================================

def gaussian_2d_moments(img, roi_size=256):
    h, w = img.shape
    max_val = img.max()
    if max_val <= 0:
        return None
    cy, cx = np.unravel_index(np.argmax(img), img.shape)

    half = roi_size // 2
    x1, x2 = max(0, cx - half), min(w, cx + half)
    y1, y2 = max(0, cy - half), min(h, cy + half)
    roi = img[y1:y2, x1:x2].astype(np.float64)

    border = np.concatenate([roi[0, :], roi[-1, :], roi[:, 0], roi[:, -1]])
    bg = np.median(border)
    roi_sub = roi - bg
    roi_sub[roi_sub < 0] = 0

    total = roi_sub.sum()
    if total <= 0:
        return None

    yy, xx = np.mgrid[0:roi_sub.shape[0], 0:roi_sub.shape[1]]
    x0_loc = (xx * roi_sub).sum() / total
    y0_loc = (yy * roi_sub).sum() / total
    dx, dy = xx - x0_loc, yy - y0_loc
    var_x = (dx * dx * roi_sub).sum() / total
    var_y = (dy * dy * roi_sub).sum() / total
    if var_x <= 0 or var_y <= 0:
        return None

    return (x1 + x0_loc, y1 + y0_loc, 2.0 * np.sqrt(var_x), 2.0 * np.sqrt(var_y),
            roi_sub.max(), bg)


def draw_fit_overlay(display, x0, y0, wx, wy, amp, bg):
    xc, yc = int(round(x0)), int(round(y0))
    h, w = display.shape[:2]
    cross = max(int(wx * 1.2), 30)

    cv2.line(display, (max(0, xc - cross), yc), (min(w - 1, xc + cross), yc), (0, 255, 0), 2)
    cv2.line(display, (xc, max(0, yc - cross)), (xc, min(h - 1, yc + cross)), (0, 255, 0), 2)
    if wx > 0 and wy > 0:
        cv2.ellipse(display, (xc, yc), (int(wx), int(wy)), 0, 0, 360, (0, 255, 0), 2)

    wx_um, wy_um = wx * 3.2, wy * 3.2
    mfd = (wx_um + wy_um) / 2 / 62.5
    lines = [
        f"Center: ({x0:.1f}, {y0:.1f}) px",
        f"Waist wx={wx_um:.1f} um  wy={wy_um:.1f} um",
        f"MFD = {mfd:.3f}",
        f"Amp={amp:.0f}  BG={bg:.0f}",
    ]
    for i, line in enumerate(lines):
        cv2.putText(display, line, (15, 220 + i * 48),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3, cv2.LINE_AA)

# ============================================================
# 主程序
# ============================================================

def main():
    DevList = mvsdk.CameraEnumerateDevice()
    if len(DevList) < 1:
        print("No camera found!")
        return
    for i, dev in enumerate(DevList):
        print(f"  [{i}] {dev.GetFriendlyName()}  SN:{dev.GetSn()}  {dev.GetPortType()}")
    dev_info = DevList[0]
    print(f"\n{dev_info.GetFriendlyName()}")

    try:
        hCamera = mvsdk.CameraInit(dev_info, -1, -1)
    except mvsdk.CameraException as e:
        print(f"CameraInit failed: {e.message}")
        return

    cap = mvsdk.CameraGetCapability(hCamera)
    mono = (cap.sIspCapacity.bMonoSensor != 0)
    max_w, max_h = cap.sResolutionRange.iWidthMax, cap.sResolutionRange.iHeightMax
    print(f"  {max_w}x{max_h}, {'Mono' if mono else 'Color'}")

    mvsdk.CameraSetIspOutFormat(hCamera,
        mvsdk.CAMERA_MEDIA_TYPE_MONO8 if mono else mvsdk.CAMERA_MEDIA_TYPE_BGR8)
    mvsdk.CameraSetTriggerMode(hCamera, 0)
    mvsdk.CameraSetAeState(hCamera, 0)
    mvsdk.CameraSetExposureTime(hCamera, 30 * 1000)
    mvsdk.CameraSetAeTarget(hCamera, 128)
    mvsdk.CameraSetAeExposureRange(hCamera, 100, 100 * 1000)
    mvsdk.CameraSetAeAnalogGainRange(hCamera, 0, 100)
    mvsdk.CameraSetAntiFlick(hCamera, True)
    mvsdk.CameraPlay(hCamera)
    print("  Streaming...\n")

    buf_size = max_w * max_h * (1 if mono else 3)
    pFrameBuffer = mvsdk.CameraAlignMalloc(buf_size, 16)

    fps_buf = deque(maxlen=30)
    t_last = t_fps_update = time.perf_counter()
    fps_display = 0
    ae_state = False
    manual_exposure_us = 30_000
    manual_gain = 100
    fit_state = False
    bg_frame = None
    fit_result = None
    input_mode = None
    input_buf = ""

    win_name = f"Camera - {dev_info.GetFriendlyName()}"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win_name, 1280, 960)

    btn_bar = ButtonBar()
    for name in ["QUIT", "SAVE", "AE", "BG", "FIT", "SET"]:
        btn_bar.add(name)
    cv2.setMouseCallback(win_name, mouse_callback, btn_bar)

    def toggle_ae():
        nonlocal ae_state
        ae_state = not ae_state
        mvsdk.CameraSetAeState(hCamera, 1 if ae_state else 0)
        if not ae_state:
            mvsdk.CameraSetExposureTime(hCamera, manual_exposure_us)
            mvsdk.CameraSetAnalogGain(hCamera, manual_gain)
        print(f"AE: {'ON' if ae_state else 'OFF'}")

    print("QUIT  SAVE  AE  BG  FIT  SET")
    print("=" * 50)

    while True:
        try:
            pRawData, FrameHead = mvsdk.CameraGetImageBuffer(hCamera, 200)
        except mvsdk.CameraException as e:
            if e.error_code != mvsdk.CAMERA_STATUS_TIME_OUT:
                print(f"Grab error: {e.error_code}")
            cv2.waitKey(1)
            continue

        mvsdk.CameraImageProcess(hCamera, pRawData, pFrameBuffer, FrameHead)
        mvsdk.CameraReleaseImageBuffer(hCamera, pRawData)
        if platform.system() == "Windows":
            mvsdk.CameraFlipFrameBuffer(pFrameBuffer, FrameHead, 1)

        frame_data = (mvsdk.c_ubyte * FrameHead.uBytes).from_address(pFrameBuffer)
        frame = np.frombuffer(frame_data, dtype=np.uint8).reshape(
            (FrameHead.iHeight, FrameHead.iWidth,
             1 if FrameHead.uiMediaType == mvsdk.CAMERA_MEDIA_TYPE_MONO8 else 3))

        now = time.perf_counter()
        fps_buf.append(now - t_last)
        t_last = now
        if now - t_fps_update > 0.5 and fps_buf:
            fps_display = 1.0 / (sum(fps_buf) / len(fps_buf))
            t_fps_update = now

        display = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR) if mono else frame

        # 拟合
        if fit_state:
            src = frame.squeeze().astype(np.float32)
            if bg_frame is not None:
                src -= bg_frame
                src[src < 0] = 0
            fit_result = gaussian_2d_moments(src.astype(np.uint8))
        else:
            fit_result = None

        btn_bar.draw(display)

        # HUD
        overexposed = np.any(frame >= 255)
        ae_str = "AE: ON" if ae_state else f"Exp={manual_exposure_us // 1000}ms Gain={manual_gain}"
        hud = f"FPS:{fps_display:.1f} | {FrameHead.iWidth}x{FrameHead.iHeight} | {'Mono' if mono else 'Color'} | {ae_str} | BG:{'OK' if bg_frame is not None else '--'} | FIT:{'ON' if fit_state else 'OFF'}"
        cv2.putText(display, hud, (15, 140), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3, cv2.LINE_AA)

        if overexposed:
            cv2.putText(display, "! OVEREXPOSED !", (15, 190),
                        cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 0, 255), 5, cv2.LINE_AA)

        if input_mode is not None:
            label = "Exposure(ms)" if input_mode == "exposure" else "Gain(0-300)"
            prompt = f"{label}: {input_buf}_  [Enter=OK  BS=del  ESC=cancel]"
            h, w = display.shape[:2]
            overlay = display.copy()
            cv2.rectangle(overlay, (0, h - 80), (w, h), (35, 35, 35), -1)
            cv2.addWeighted(overlay, 0.8, display, 0.2, 0, display)
            cv2.putText(display, prompt, (20, h - 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3, cv2.LINE_AA)

        if fit_result is not None:
            draw_fit_overlay(display, *fit_result)

        cv2.imshow(win_name, display)

        # 按钮点击
        clicked = mouse_state.get("clicked_btn")
        if clicked is not None:
            btn = btn_bar.buttons[clicked].name
            mouse_state["clicked_btn"] = None

            if btn == "QUIT":
                break
            elif btn == "SAVE":
                fname = f"capture_{time.strftime('%Y%m%d_%H%M%S')}.png"
                cv2.imwrite(fname, frame.squeeze())
                print(f"Saved: {fname}")
            elif btn == "AE":
                toggle_ae()
            elif btn == "BG":
                bg_frame = frame.squeeze().astype(np.float32)
                print(f"BG captured (mean={bg_frame.mean():.1f})")
            elif btn == "FIT":
                fit_state = not fit_state
                print(f"FIT: {'ON' if fit_state else 'OFF'}")
            elif btn == "SET":
                input_mode = "exposure"
                input_buf = str(manual_exposure_us // 1000)

        # 键盘
        key = cv2.waitKey(1) & 0xFF

        if input_mode is not None:
            if ord('0') <= key <= ord('9'):
                input_buf += chr(key)
            elif key == 8:
                input_buf = input_buf[:-1]
            elif key == 13:
                try:
                    val = int(input_buf) if input_buf else 0
                except ValueError:
                    val = 0
                if input_mode == "exposure":
                    manual_exposure_us = max(1, val) * 1000
                    if not ae_state:
                        mvsdk.CameraSetExposureTime(hCamera, manual_exposure_us)
                    input_mode = "gain"
                    input_buf = str(manual_gain)
                else:
                    manual_gain = max(0, min(300, val))
                    if not ae_state:
                        mvsdk.CameraSetAnalogGain(hCamera, manual_gain)
                    input_mode = None
                    input_buf = ""
                    print(f"Set: Exp={manual_exposure_us // 1000}ms Gain={manual_gain}")
            elif key == 27:
                input_mode = None
                input_buf = ""
        else:
            if key == ord('q'):
                break
            elif key == ord('s'):
                fname = f"capture_{time.strftime('%Y%m%d_%H%M%S')}.png"
                cv2.imwrite(fname, frame.squeeze())
                print(f"Saved: {fname}")
            elif key == ord('a'):
                toggle_ae()
            elif key == ord('b'):
                bg_frame = frame.squeeze().astype(np.float32)
                print(f"BG captured (mean={bg_frame.mean():.1f})")
            elif key == ord('f'):
                fit_state = not fit_state
                print(f"FIT: {'ON' if fit_state else 'OFF'}")

    mvsdk.CameraUnInit(hCamera)
    mvsdk.CameraAlignFree(pFrameBuffer)
    cv2.destroyAllWindows()
    print("Done.")


if __name__ == "__main__":
    main()
