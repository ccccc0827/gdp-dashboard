import os

# =========================
# 防止 OpenCV libGL 錯誤
# =========================
os.environ["QT_QPA_PLATFORM"] = "offscreen"

import streamlit as st
import cv2
import math
import av
import time
import threading
import base64
import numpy as np

from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO
from streamlit_webrtc import webrtc_streamer, WebRtcMode
from pathlib import Path


# =========================
# Page Config
# =========================
st.set_page_config(
    page_title="長照睡姿固定過久警報系統",
    page_icon="🛌",
    layout="wide"
)

# =========================
# Custom CSS
# =========================
st.markdown("""
<style>

.main-title {
    font-size: 2.2rem;
    font-weight: 700;
    color: #1f3c88;
    margin-bottom: 0.2rem;
}

.sub-text {
    color: #5f6b7a;
    font-size: 1rem;
    margin-bottom: 1.2rem;
}

.metric-card {
    background-color: #f8fbff;
    border: 1px solid #dfe8f3;
    border-radius: 14px;
    padding: 16px 20px;
    text-align: center;
}

.metric-label {
    font-size: 0.95rem;
    color: #6b7280;
}

.metric-value {
    font-size: 1.8rem;
    font-weight: 700;
    color: #1f3c88;
}

.alert-box {
    background-color: #fff1f2;
    border: 1px solid #fda4af;
    color: #b91c1c;
    border-radius: 12px;
    padding: 16px;
    font-size: 1.05rem;
    font-weight: 600;
}

.normal-box {
    background-color: #f0fdf4;
    border: 1px solid #86efac;
    color: #166534;
    border-radius: 12px;
    padding: 16px;
    font-size: 1.05rem;
    font-weight: 600;
}

</style>
""", unsafe_allow_html=True)

# =========================
# Title
# =========================
st.markdown(
    '<div class="main-title">🛌 長照睡姿固定過久警報系統</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="sub-text">使用影像分析臥床姿勢變化，協助照護員及早發現長時間未翻身狀況。</div>',
    unsafe_allow_html=True
)

# =========================
# Load YOLO Model
# =========================
@st.cache_resource
def load_model():
    return YOLO("yolov8n-pose.pt")

model = load_model()

# =========================
# Shared State
# =========================
class AppState:
    def __init__(self):
        self.lock = threading.Lock()

        self.current_posture = "無人躺著"
        self.last_posture = "無人躺著"

        self.start_time = time.time()
        self.duration = 0.0

        self.alarm = False
        self.alarm_acknowledged = False

        self.monitoring = False


if "shared_state" not in st.session_state:
    st.session_state.shared_state = AppState()

shared_state = st.session_state.shared_state

if "sound_enabled" not in st.session_state:
    st.session_state.sound_enabled = False

# =========================
# Helper Functions
# =========================
def dist(p1, p2):
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def put_chinese_text(
    img,
    text,
    position,
    text_color=(255, 255, 255),
    font_size=30
):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    pil_img = Image.fromarray(img_rgb)

    draw = ImageDraw.Draw(pil_img)

    try:
        font = ImageFont.truetype("msjh.ttc", font_size)

    except IOError:
        try:
            font = ImageFont.truetype("PingFang.ttc", font_size)

        except IOError:
            font = ImageFont.truetype(
                "NotoSansTC-Regular.ttf",
                font_size
            )

    draw.text(
        position,
        text,
        font=font,
        fill=text_color
    )

    return cv2.cvtColor(
        np.array(pil_img),
        cv2.COLOR_RGB2BGR
    )

# =========================
# Posture Classification
# =========================
def classify_posture(results):

    current_posture = "無人躺著"

    if (
        results[0].keypoints is not None
        and len(results[0].keypoints.xy) > 0
    ):

        kps = results[0].keypoints.xy[0]
        conf = results[0].keypoints.conf[0]

        if conf.max() > 0.5 and len(kps) >= 13:

            shoulder_width = dist(kps[5], kps[6])

            torso_length = (
                dist(kps[5], kps[11]) +
                dist(kps[6], kps[12])
            ) / 2

            is_side = (
                (conf[5] < 0.4 or conf[6] < 0.4)
                or (
                    torso_length > 0
                    and (shoulder_width / torso_length) < 0.5
                )
            )

            if is_side:

                if (
                    (conf[4] + conf[6])
                    > (conf[3] + conf[5]) + 0.2
                ):
                    current_posture = "左側躺"

                elif (
                    (conf[3] + conf[5])
                    > (conf[4] + conf[6]) + 0.2
                ):
                    current_posture = "右側躺"

                else:

                    if dist(kps[0], kps[3]) < dist(kps[0], kps[4]):
                        current_posture = "右側躺"
                    else:
                        current_posture = "左側躺"

            else:
                current_posture = "仰躺"

    return current_posture

# =========================
# Alarm Sound
# =========================
def render_loop_alarm():

    if not st.session_state.sound_enabled:
        st.warning("🔇 請先按左側『啟用警報聲』")
        return

    audio_file = Path("alarm.mp3")

    if not audio_file.exists():
        st.warning("⚠️ 找不到 alarm.mp3")
        return

    audio_bytes = audio_file.read_bytes()

    b64 = base64.b64encode(audio_bytes).decode()

    audio_html = f"""
    <audio autoplay loop>
        <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
    </audio>
    """

    st.components.v1.html(audio_html, height=0)

# =========================
# Sidebar
# =========================
st.sidebar.header("⚙️ 分析設定")

alarm_threshold = st.sidebar.slider(
    "同姿勢維持幾秒觸發警報",
    min_value=3,
    max_value=60,
    value=10,
    step=1
)

# =========================
# Enable Sound
# =========================
if st.sidebar.button("🔊 啟用警報聲"):
    st.session_state.sound_enabled = True
    st.sidebar.success("警報聲已啟用")

st.sidebar.markdown("---")

# =========================
# Start / Stop
# =========================
if st.sidebar.button("▶️ Start"):

    with shared_state.lock:

        shared_state.monitoring = True
        shared_state.start_time = time.time()

        shared_state.duration = 0.0
        shared_state.alarm = False

        shared_state.alarm_acknowledged = False

        shared_state.last_posture = (
            shared_state.current_posture
        )

if st.sidebar.button("⏹ Stop"):

    with shared_state.lock:

        shared_state.monitoring = False

        shared_state.duration = 0.0

        shared_state.alarm = False
        shared_state.alarm_acknowledged = False

        shared_state.current_posture = "無人躺著"
        shared_state.last_posture = "無人躺著"

st.sidebar.markdown("---")

st.sidebar.info(
    "按下 Start 後開始監測；Stop 會停止並重新計算。"
)



# =========================
# Video Processor
# =========================

class PoseVideoProcessor:
    # =========================
    # Right Panel Fragment
    # 只刷新右側摘要，不刷新 WebRTC 影像
    # =========================
    @st.fragment(run_every=1)
    def render_summary_panel():
        st.subheader("2. 摘要資訊")
    
        with shared_state.lock:
            posture_now = shared_state.current_posture
            duration_now = int(shared_state.duration)
            alarm_now = shared_state.alarm
            monitoring_now = shared_state.monitoring
    
        c1, c2, c3 = st.columns(3)
    
        with c1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">目前姿勢</div>
                <div class="metric-value">{posture_now}</div>
            </div>
            """, unsafe_allow_html=True)
    
        with c2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">持續時間</div>
                <div class="metric-value">{duration_now} 秒</div>
            </div>
            """, unsafe_allow_html=True)
    
        with c3:
            system_text = "監測中" if monitoring_now else "停止"
    
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">系統狀態</div>
                <div class="metric-value">{system_text}</div>
            </div>
            """, unsafe_allow_html=True)
    
        st.markdown("<br>", unsafe_allow_html=True)
    
        st.subheader("3. 警報摘要")
    
        if alarm_now:
            st.markdown(f"""
            <div class="alert-box">
                🚨 偵測到姿勢持續超過 {alarm_threshold} 秒，
                請協助翻身。
            </div>
            """, unsafe_allow_html=True)
    
            render_loop_alarm()
    
            if st.button("✅ 確認此資訊", type="primary"):
                with shared_state.lock:
                    shared_state.alarm_acknowledged = True
                    shared_state.alarm = False
    
                st.rerun()
    
        else:
            st.markdown("""
            <div class="normal-box">
                ✅ 目前尚未觸發警報
            </div>
            """, unsafe_allow_html=True)

    def recv(self, frame):

        img = frame.to_ndarray(format="bgr24")

        results = model(img, verbose=False)

        current_posture = classify_posture(results)

        now = time.time()

        with shared_state.lock:

            if shared_state.monitoring:

                if current_posture == shared_state.last_posture:

                    shared_state.duration = (
                        now - shared_state.start_time
                    )

                else:

                    shared_state.last_posture = current_posture
                    shared_state.current_posture = current_posture

                    shared_state.start_time = now
                    shared_state.duration = 0.0

                    shared_state.alarm = False
                    shared_state.alarm_acknowledged = False

                if (
                    shared_state.duration >= alarm_threshold
                    and current_posture != "無人躺著"
                    and not shared_state.alarm_acknowledged
                ):
                    shared_state.alarm = True

                else:

                    if (
                        current_posture == "無人躺著"
                        or shared_state.alarm_acknowledged
                    ):
                        shared_state.alarm = False

                shared_state.current_posture = current_posture

            else:

                shared_state.duration = 0.0
                shared_state.alarm = False

        annotated = results[0].plot()

        with shared_state.lock:

            monitor_text = (
                "監測中"
                if shared_state.monitoring
                else "已停止"
            )

            info_text = (
                f"{monitor_text} | "
                f"姿勢: {shared_state.current_posture} | "
                f"持續時間: {int(shared_state.duration)} 秒"
            )

        cv2.rectangle(
            annotated,
            (20, 20),
            (900, 70),
            (0, 0, 0),
            -1
        )

        annotated = put_chinese_text(
            annotated,
            info_text,
            (30, 25),
            text_color=(255, 255, 255),
            font_size=32
        )

        if shared_state.alarm:

            cv2.rectangle(
                annotated,
                (0, 0),
                (annotated.shape[1], annotated.shape[0]),
                (0, 0, 255),
                10
            )

            cv2.putText(
                annotated,
                "ALARM",
                (30, 140),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.5,
                (0, 0, 255),
                4,
                cv2.LINE_AA
            )

        return av.VideoFrame.from_ndarray(
            annotated,
            format="bgr24"
        )

# =========================
# Layout
# =========================
# =========================
# Right Panel Fragment
# 只刷新右側摘要，不刷新 WebRTC 影像
# =========================
@st.fragment(run_every="1s")
def render_summary_panel():
    st.subheader("2. 摘要資訊")

    with shared_state.lock:
        posture_now = shared_state.current_posture
        duration_now = int(shared_state.duration)
        alarm_now = shared_state.alarm
        monitoring_now = shared_state.monitoring

    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">目前姿勢</div>
            <div class="metric-value">{posture_now}</div>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">持續時間</div>
            <div class="metric-value">{duration_now} 秒</div>
        </div>
        """, unsafe_allow_html=True)

    with c3:
        system_text = "監測中" if monitoring_now else "停止"

        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">系統狀態</div>
            <div class="metric-value">{system_text}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    st.subheader("3. 警報摘要")

    if alarm_now:
        st.markdown(f"""
        <div class="alert-box">
            🚨 偵測到姿勢持續超過 {alarm_threshold} 秒，
            請協助翻身。
        </div>
        """, unsafe_allow_html=True)

        render_loop_alarm()

        if st.button("✅ 確認此資訊", type="primary"):
            with shared_state.lock:
                shared_state.alarm_acknowledged = True
                shared_state.alarm = False

            st.rerun()

    else:
        st.markdown("""
        <div class="normal-box">
            ✅ 目前尚未觸發警報
        </div>
        """, unsafe_allow_html=True)
left_col, right_col = st.columns([1.15, 1.4])


# =========================
# Webcam
# =========================
with left_col:

    st.subheader("1. 即時影像監測")

    webrtc_streamer(
        key="pose-monitor",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration={
            "iceServers": [
                {"urls": ["stun:stun.l.google.com:19302"]}
            ]
        },
        media_stream_constraints={
            "video": True,
            "audio": False
        },
        video_processor_factory=PoseVideoProcessor,
        async_processing=True,
    )

# =========================
# Right Panel
# =========================
with right_col:
    render_summary_panel()

    # =========================
    # Alarm Area
    # =========================
    st.subheader("3. 警報摘要")

    if alarm_now:

        st.markdown(f"""
        <div class="alert-box">
            🚨 偵測到姿勢持續超過 {alarm_threshold} 秒，
            請協助翻身。
        </div>
        """, unsafe_allow_html=True)

        render_loop_alarm()

        if st.button("✅ 確認此資訊", type="primary"):

            with shared_state.lock:

                shared_state.alarm_acknowledged = True
                shared_state.alarm = False

            st.rerun()

    else:

        st.markdown("""
        <div class="normal-box">
            ✅ 目前尚未觸發警報
        </div>
        """, unsafe_allow_html=True)
