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
from scipy.optimize import curve_fit

MAG = 3         # 放大镜倍数
LOUPE_SRC = 100  # 放大镜原始裁切尺寸

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


mouse_state = {"clicked_btn": None, "x": 0, "y": 0, "wheel": 0}

def mouse_callback(event, x, y, flags, param):
    mouse_state["x"], mouse_state["y"] = x, y
    if event == cv2.EVENT_LBUTTONDOWN:
        idx = param.check_hover(x, y)
        if idx >= 0:
            mouse_state["clicked_btn"] = idx
    elif event == cv2.EVENT_MOUSEWHEEL:
        mouse_state["wheel"] = 1 if flags > 0 else -1


def _gauss_2d(XY, A, x0, sx, y0, sy, B):
    """二维高斯: I(x,y) = A * exp(-((x-x0)^2/(2*sx^2) + (y-y0)^2/(2*sy^2))) + B"""
    x, y = XY
    return A * np.exp(-((x - x0) ** 2 / (2 * sx ** 2) + (y - y0) ** 2 / (2 * sy ** 2))) + B


def gaussian_fit(img):
    """scipy curve_fit 二维高斯拟合 (LM, 无边界约束, 约 6ms)。返回 (x0, y0, wx, wy) 或 None"""
    h, w = img.shape
    max_val = img.max()
    if max_val <= 0:
        return None
    cy, cx = np.unravel_index(np.argmax(img), img.shape)

    # 矩方法粗估
    half = 64
    x1, x2 = max(0, cx - half), min(w, cx + half)
    y1, y2 = max(0, cy - half), min(h, cy + half)
    roi = img[y1:y2, x1:x2].astype(np.float64)
    border = np.concatenate([roi[0, :], roi[-1, :], roi[:, 0], roi[:, -1]])
    bg0 = np.median(border)
    roi_sub = roi - bg0
    roi_sub[roi_sub < 0] = 0
    total = roi_sub.sum()
    if total <= 0:
        return None
    yy, xx = np.mgrid[0:roi_sub.shape[0], 0:roi_sub.shape[1]]
    cx_loc = (xx * roi_sub).sum() / total
    cy_loc = (yy * roi_sub).sum() / total
    dx, dy = xx - cx_loc, yy - cy_loc
    var_x = (dx * dx * roi_sub).sum() / total
    var_y = (dy * dy * roi_sub).sum() / total
    if var_x <= 0 or var_y <= 0:
        return None
    sx0, sy0 = np.sqrt(var_x), np.sqrt(var_y)
    x0_m = x1 + cx_loc
    y0_m = y1 + cy_loc

    # 3.5 sigma ROI 切取
    margin = int(max(sx0, sy0) * 3.5) + 1
    x1 = max(0, int(x0_m) - margin)
    x2 = min(w, int(x0_m) + margin)
    y1 = max(0, int(y0_m) - margin)
    y2 = min(h, int(y0_m) + margin)
    roi = img[y1:y2, x1:x2].astype(np.float64)

    valid = roi < 255
    if valid.sum() < 50:
        return None

    A0 = max(roi.max() - bg0, 1.0)
    yy, xx = np.mgrid[y1:y2, x1:x2]
    xdata = np.vstack([xx[valid].ravel(), yy[valid].ravel()])
    ydata = roi[valid].ravel()

    try:
        popt, _ = curve_fit(_gauss_2d, xdata, ydata,
                            p0=[A0, x0_m, sx0, y0_m, sy0, bg0],
                            method='lm', maxfev=300)
    except Exception:
        return None

    _, x0, sx, y0, sy, _ = popt
    if sx <= 0 or sy <= 0:
        return None

    y_pred = _gauss_2d(xdata, *popt)
    ss_res = np.sum((ydata - y_pred) ** 2)
    ss_tot = np.sum((ydata - ydata.mean()) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return x0, y0, 4.0 * sx, 4.0 * sy, r2


def draw_fit_overlay(display, x0, y0, wx, wy, r2, cmap_on):
    color = (255, 0, 255) if cmap_on else (0, 255, 0)  # magenta / green
    xc, yc = int(round(x0)), int(round(y0))
    h, w = display.shape[:2]
    cross = max(int(wx * 1.2), 30)

    cv2.line(display, (max(0, xc - cross), yc), (min(w - 1, xc + cross), yc), color, 2)
    cv2.line(display, (xc, max(0, yc - cross)), (xc, min(h - 1, yc + cross)), color, 2)
    if wx > 0 and wy > 0:
        cv2.ellipse(display, (xc, yc), (int(wx), int(wy)), 0, 0, 360, color, 2)

    wx_um, wy_um = wx * 3.2, wy * 3.2
    mfd = (wx_um + wy_um) / 2 / 62.5
    lines = [
        f"Center: ({x0:.1f}, {y0:.1f}) px",
        f"Waist wx={wx_um:.1f} um  wy={wy_um:.1f} um",
        f"MFD = {mfd:.3f}",
        f"R^2 = {r2:.4f}",
    ]
    for i, line in enumerate(lines):
        cv2.putText(display, line, (15, 220 + i * 48),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3, cv2.LINE_AA)


def draw_loupe(display, mouse_x, mouse_y, raw_frame, h, w, cmap_on):
    """在右下角画放大镜，跟随主窗口 cmap"""
    half = LOUPE_SRC // 2
    x1 = max(0, mouse_x - half)
    x2 = min(w, mouse_x + half)
    y1 = max(0, mouse_y - half)
    y2 = min(h, mouse_y + half)
    crop = raw_frame[y1:y2, x1:x2]
    if crop.size == 0:
        return
    sz = int(LOUPE_SRC * MAG)
    zoomed = cv2.resize(crop, (sz, sz), interpolation=cv2.INTER_NEAREST)
    zoomed_bgr = cv2.applyColorMap(zoomed, cv2.COLORMAP_JET) if cmap_on else cv2.cvtColor(zoomed, cv2.COLOR_GRAY2BGR)

    # 十字准星
    ch = zoomed_bgr.shape[0] // 2
    cv2.line(zoomed_bgr, (0, ch), (zoomed_bgr.shape[1], ch), (0, 0, 255), 1)
    cv2.line(zoomed_bgr, (ch, 0), (ch, zoomed_bgr.shape[0]), (0, 0, 255), 1)

    # 右下角放置
    lh, lw = zoomed_bgr.shape[:2]
    dx, dy = display.shape[1] - lw - 10, display.shape[0] - lh - 10
    display[dy:dy + lh, dx:dx + lw] = zoomed_bgr
    # 边框
    cv2.rectangle(display, (dx - 1, dy - 1), (dx + lw, dy + lh), (0, 255, 255), 2)

    # 像素值标签
    px_val = raw_frame[min(mouse_y, h - 1), min(mouse_x, w - 1)]
    label = f"[{mouse_x},{mouse_y}] = {px_val}"
    cv2.putText(display, label, (dx, dy - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)


def main():
    global MAG
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
    mvsdk.CameraSetExposureTime(hCamera, 0.5 * 1000)
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
    manual_exposure_us = 500
    manual_gain = 100
    cmap_on = True
    fit_state = False
    bg_frame = None
    fit_result = None
    input_mode = None
    input_buf = ""

    win_name = f"Camera - {dev_info.GetFriendlyName()}"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win_name, 1280, 960)

    btn_bar = ButtonBar()
    for name in ["QUIT", "SAVE", "AE", "BG", "FIT", "SET", "CMAP"]:
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

        if mono and cmap_on:
            display = cv2.applyColorMap(frame.squeeze(), cv2.COLORMAP_JET)
        elif mono:
            display = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        else:
            display = frame

        raw = frame.squeeze()
        if fit_state:
            src = raw.astype(np.float32)
            if bg_frame is not None:
                src -= bg_frame
                src[src < 0] = 0
            fit_result = gaussian_fit(src.astype(np.uint8))
        else:
            fit_result = None

        btn_bar.draw(display)

        overexposed = np.any(frame >= 255)
        ae_str = "AE: ON" if ae_state else f"Exp={manual_exposure_us / 1000:.1f}ms Gain={manual_gain}"
        cmap_str = "JET" if cmap_on else "Gray"
        hud_color = (255, 0, 255) if cmap_on else (0, 255, 0)
        hud = f"FPS:{fps_display:.1f} | {FrameHead.iWidth}x{FrameHead.iHeight} | {cmap_str} | {ae_str} | MAG:{MAG:.1f}x | BG:{'OK' if bg_frame is not None else '--'} | FIT:{'ON' if fit_state else 'OFF'}"
        cv2.putText(display, hud, (15, 140), cv2.FONT_HERSHEY_SIMPLEX, 1.2, hud_color, 3, cv2.LINE_AA)

        if overexposed:
            cv2.putText(display, "! OVEREXPOSED !", (15, 190),
                        cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 0, 255), 5, cv2.LINE_AA)

        if input_mode is not None:
            labels = {"exposure": "Exposure(ms)", "gain": "Gain(0-300)", "magnification": "Magnification"}
            prompt = f"{labels[input_mode]}: {input_buf}_  [Enter=OK  BS=del  ESC=cancel]"
            h, w = display.shape[:2]
            overlay = display.copy()
            cv2.rectangle(overlay, (0, h - 80), (w, h), (35, 35, 35), -1)
            cv2.addWeighted(overlay, 0.8, display, 0.2, 0, display)
            cv2.putText(display, prompt, (20, h - 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3, cv2.LINE_AA)

        if fit_result is not None:
            draw_fit_overlay(display, *fit_result, cmap_on)

        lx = int(fit_result[0]) if fit_result else mouse_state["x"]
        ly = int(fit_result[1]) if fit_result else mouse_state["y"]
        draw_loupe(display, lx, ly, raw, FrameHead.iHeight, FrameHead.iWidth, cmap_on)

        cv2.imshow(win_name, display)

        clicked = mouse_state.get("clicked_btn")
        if clicked is not None:
            btn = btn_bar.buttons[clicked].name
            mouse_state["clicked_btn"] = None

            if btn == "QUIT":
                break
            elif btn == "SAVE":
                fname = f"capture_{time.strftime('%Y%m%d_%H%M%S')}.png"
                cv2.imwrite(fname, raw)
                print(f"Saved: {fname}")
            elif btn == "AE":
                toggle_ae()
            elif btn == "BG":
                bg_frame = raw.astype(np.float32)
                print(f"BG captured (mean={bg_frame.mean():.1f})")
            elif btn == "FIT":
                fit_state = not fit_state
                print(f"FIT: {'ON' if fit_state else 'OFF'}")
            elif btn == "CMAP":
                cmap_on = not cmap_on
                print(f"Colormap: {'JET' if cmap_on else 'Gray'}")
            elif btn == "SET":
                input_mode = "exposure"
                input_buf = f"{manual_exposure_us / 1000:.1f}"

        key = cv2.waitKey(1) & 0xFF

        if input_mode is not None:
            if ord('0') <= key <= ord('9') or key == ord('.'):
                input_buf += chr(key)
            elif key == 8:
                input_buf = input_buf[:-1]
            elif key == 13:
                try:
                    val = float(input_buf) if input_buf else 0.0
                except ValueError:
                    val = 0.0
                if input_mode == "exposure":
                    manual_exposure_us = max(1, int(val * 1000))
                    if not ae_state:
                        mvsdk.CameraSetExposureTime(hCamera, manual_exposure_us)
                    input_mode = "gain"
                    input_buf = str(manual_gain)
                elif input_mode == "gain":
                    manual_gain = max(0, min(300, int(val)))
                    if not ae_state:
                        mvsdk.CameraSetAnalogGain(hCamera, manual_gain)
                    input_mode = "magnification"
                    input_buf = f"{MAG:.1f}"
                else:
                    MAG = max(1.0, min(20.0, float(val)))
                    input_mode = None
                    input_buf = ""
                    print(f"Set: Exp={manual_exposure_us / 1000:.1f}ms Gain={manual_gain} MAG={MAG:.1f}x")
            elif key == 27:
                input_mode = None
                input_buf = ""

    mvsdk.CameraUnInit(hCamera)
    mvsdk.CameraAlignFree(pFrameBuffer)
    cv2.destroyAllWindows()
    print("Done.")


if __name__ == "__main__":
    main()
