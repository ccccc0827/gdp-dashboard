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

from datetime import datetime
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
:root {
    --primary: #1f3c88;
    --primary-light: #eef4ff;
    --muted: #667085;
    --border: #dfe8f3;
    --bg-soft: #f8fbff;
    --green-bg: #ecfdf3;
    --green: #15803d;
    --amber-bg: #fffbeb;
    --amber: #b45309;
    --red-bg: #fff1f2;
    --red: #b91c1c;
    --gray-bg: #f3f4f6;
    --gray: #4b5563;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}

.main-title {
    font-size: 2.45rem;
    font-weight: 800;
    color: var(--primary);
    margin-bottom: 0.25rem;
    letter-spacing: -0.02em;
}

.sub-text {
    color: #5f6b7a;
    font-size: 1.05rem;
    margin-bottom: 1.4rem;
}

.section-title {
    font-size: 1.45rem;
    font-weight: 800;
    color: #1f2937;
    margin: 0.6rem 0 1rem 0;
}

.metric-card {
    background: linear-gradient(180deg, #ffffff 0%, #f8fbff 100%);
    border: 1px solid var(--border);
    border-radius: 18px;
    padding: 18px 20px;
    text-align: center;
    min-height: 118px;
    box-shadow: 0 8px 22px rgba(31, 60, 136, 0.06);
}

.metric-label {
    font-size: 0.95rem;
    color: #6b7280;
    margin-bottom: 0.25rem;
}

.metric-value {
    font-size: 1.8rem;
    font-weight: 800;
    color: var(--primary);
}

.bed-card {
    border-radius: 20px;
    padding: 18px 18px 16px 18px;
    border: 1px solid var(--border);
    box-shadow: 0 10px 26px rgba(31, 60, 136, 0.07);
    min-height: 210px;
    background: #ffffff;
}

.bed-normal {
    border-left: 8px solid #22c55e;
    background: linear-gradient(180deg, #ffffff 0%, #f0fdf4 100%);
}

.bed-warning {
    border-left: 8px solid #f59e0b;
    background: linear-gradient(180deg, #ffffff 0%, #fffbeb 100%);
}

.bed-alert {
    border-left: 8px solid #ef4444;
    background: linear-gradient(180deg, #ffffff 0%, #fff1f2 100%);
}

.bed-id {
    font-size: 1.45rem;
    font-weight: 900;
    color: #1f2937;
    margin-bottom: 0.3rem;
}

.bed-meta {
    color: #475467;
    font-size: 0.98rem;
    line-height: 1.8;
}

.status-pill {
    display: inline-block;
    padding: 5px 11px;
    border-radius: 999px;
    font-size: 0.88rem;
    font-weight: 800;
    margin-bottom: 0.8rem;
}

.status-normal {
    background: var(--green-bg);
    color: var(--green);
}

.status-warning {
    background: var(--amber-bg);
    color: var(--amber);
}

.status-alert {
    background: var(--red-bg);
    color: var(--red);
}

.alert-box {
    background-color: var(--red-bg);
    border: 1px solid #fda4af;
    color: var(--red);
    border-radius: 16px;
    padding: 18px;
    font-size: 1.05rem;
    font-weight: 700;
}

.normal-box {
    background-color: var(--green-bg);
    border: 1px solid #86efac;
    color: var(--green);
    border-radius: 16px;
    padding: 18px;
    font-size: 1.05rem;
    font-weight: 700;
}

.warning-box {
    background-color: var(--amber-bg);
    border: 1px solid #fcd34d;
    color: var(--amber);
    border-radius: 16px;
    padding: 18px;
    font-size: 1.05rem;
    font-weight: 700;
}

.sidebar-note {
    background: #eaf2ff;
    color: #075098;
    padding: 16px;
    border-radius: 14px;
    line-height: 1.7;
    font-weight: 600;
}

.small-muted {
    color: #667085;
    font-size: 0.92rem;
}

</style>
""", unsafe_allow_html=True)


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
        self.alarm_logged = False

        self.monitoring = False
        self.alarm_logs = []


if "shared_state" not in st.session_state:
    st.session_state.shared_state = AppState()

shared_state = st.session_state.shared_state

if "sound_enabled" not in st.session_state:
    st.session_state.sound_enabled = False
# =========================
# State Migration
# 避免 Streamlit 沿用舊版 shared_state 時缺少新欄位
# =========================
if not hasattr(shared_state, "alarm_logs"):
    shared_state.alarm_logs = []

if not hasattr(shared_state, "alarm_logged"):
    shared_state.alarm_logged = False

if not hasattr(shared_state, "alarm_acknowledged"):
    shared_state.alarm_acknowledged = False

if not hasattr(shared_state, "lock"):
    shared_state.lock = threading.Lock()


# =========================
# Helper Functions
# =========================
def dist(p1, p2):
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def format_duration(seconds):
    seconds = int(seconds)

    if seconds < 60:
        return f"{seconds} 秒"

    minutes = seconds // 60
    remain_seconds = seconds % 60

    if minutes < 60:
        return f"{minutes} 分 {remain_seconds} 秒"

    hours = minutes // 60
    remain_minutes = minutes % 60

    return f"{hours} 小時 {remain_minutes} 分"


def get_status(duration, threshold, alarm, monitoring, posture):
    if not monitoring:
        return "停止", "gray"

    if posture == "無人躺著":
        return "無人", "gray"

    if alarm:
        return "需翻身", "alert"

    if threshold > 0 and duration >= threshold * 0.8:
        return "接近警報", "warning"

    return "正常", "normal"


def get_status_label(status_key):
    mapping = {
        "normal": ("正常", "status-normal"),
        "warning": ("接近警報", "status-warning"),
        "alert": ("需翻身", "status-alert"),
        "gray": ("未監測", "status-normal"),
    }
    return mapping.get(status_key, ("正常", "status-normal"))


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
        st.warning("🔇 請先按左側「啟用警報聲」")
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
st.sidebar.header("⚙️ 監測設定")

alarm_threshold = st.sidebar.slider(
    "同姿勢維持幾秒觸發警報",
    min_value=3,
    max_value=7200,
    value=10,
    step=1
)

st.sidebar.caption("正式場域可設定為 7200 秒，也就是約 2 小時。")

if st.sidebar.button("🔊 啟用警報聲"):
    st.session_state.sound_enabled = True
    st.sidebar.success("警報聲已啟用")

st.sidebar.markdown("---")

if st.sidebar.button("▶️ Start"):

    with shared_state.lock:

        shared_state.monitoring = True
        shared_state.start_time = time.time()
        shared_state.duration = 0.0
        shared_state.alarm = False
        shared_state.alarm_acknowledged = False
        shared_state.alarm_logged = False
        shared_state.last_posture = shared_state.current_posture

if st.sidebar.button("⏹ Stop"):

    with shared_state.lock:

        shared_state.monitoring = False
        shared_state.duration = 0.0
        shared_state.alarm = False
        shared_state.alarm_acknowledged = False
        shared_state.alarm_logged = False
        shared_state.current_posture = "無人躺著"
        shared_state.last_posture = "無人躺著"

st.sidebar.markdown("---")

st.sidebar.markdown(
    """
    <div class="sidebar-note">
    按下 Start 後開始監測。<br>
    當同一姿勢維持超過設定時間，系統會觸發警報。<br>
    確認後會重新從 0 秒開始計時。
    </div>
    """,
    unsafe_allow_html=True
)


# =========================
# Video Processor
# =========================
class PoseVideoProcessor:

    def recv(self, frame):

        img = frame.to_ndarray(format="bgr24")
        results = model(img, verbose=False)
        current_posture = classify_posture(results)
        now = time.time()

        with shared_state.lock:

            if shared_state.monitoring:

                if current_posture == shared_state.last_posture:
                    shared_state.duration = now - shared_state.start_time

                else:
                    shared_state.last_posture = current_posture
                    shared_state.current_posture = current_posture
                    shared_state.start_time = now
                    shared_state.duration = 0.0
                    shared_state.alarm = False
                    shared_state.alarm_acknowledged = False
                    shared_state.alarm_logged = False

                if (
                    shared_state.duration >= alarm_threshold
                    and current_posture != "無人躺著"
                ):
                    shared_state.alarm = True

                    if not shared_state.alarm_logged:
                        shared_state.alarm_logs.insert(
                            0,
                            {
                                "時間": datetime.now().strftime("%H:%M:%S"),
                                "床位": "A01",
                                "姿勢": current_posture,
                                "持續時間": format_duration(shared_state.duration),
                                "狀態": "已觸發"
                            }
                        )
                        shared_state.alarm_logged = True

                else:
                    if current_posture == "無人躺著":
                        shared_state.alarm = False
                        shared_state.alarm_logged = False

                shared_state.current_posture = current_posture

            else:
                shared_state.duration = 0.0
                shared_state.alarm = False

            monitor_text = "監測中" if shared_state.monitoring else "已停止"

            info_text = (
                f"{monitor_text} | "
                f"姿勢: {shared_state.current_posture} | "
                f"持續時間: {format_duration(shared_state.duration)}"
            )

            alarm_now = shared_state.alarm

        annotated = results[0].plot()

        cv2.rectangle(
            annotated,
            (20, 20),
            (980, 74),
            (0, 0, 0),
            -1
        )

        annotated = put_chinese_text(
            annotated,
            info_text,
            (30, 25),
            text_color=(255, 255, 255),
            font_size=30
        )

        if alarm_now:
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
                (30, 145),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.6,
                (0, 0, 255),
                4,
                cv2.LINE_AA
            )

        return av.VideoFrame.from_ndarray(
            annotated,
            format="bgr24"
        )


# =========================
# UI Render Functions
# =========================
def get_realtime_state():
    with shared_state.lock:
        return {
            "posture": shared_state.current_posture,
            "duration": int(shared_state.duration),
            "alarm": shared_state.alarm,
            "monitoring": shared_state.monitoring,
            "logs": list(shared_state.alarm_logs),
        }


def build_bed_data():
    live = get_realtime_state()

    status_text, status_key = get_status(
        live["duration"],
        alarm_threshold,
        live["alarm"],
        live["monitoring"],
        live["posture"]
    )

    simulated_beds = [
        {
            "床位": "A01",
            "來源": "即時 YOLO 偵測",
            "姿勢": live["posture"],
            "秒數": live["duration"],
            "狀態": status_text,
            "狀態鍵": status_key,
        },
        {
            "床位": "A02",
            "來源": "模擬資料",
            "姿勢": "左側躺",
            "秒數": int(alarm_threshold * 0.42),
            "狀態": "正常",
            "狀態鍵": "normal",
        },
        {
            "床位": "A03",
            "來源": "模擬資料",
            "姿勢": "仰躺",
            "秒數": int(alarm_threshold * 0.86),
            "狀態": "接近警報",
            "狀態鍵": "warning",
        },
        {
            "床位": "A04",
            "來源": "模擬資料",
            "姿勢": "右側躺",
            "秒數": int(alarm_threshold * 1.12),
            "狀態": "需翻身",
            "狀態鍵": "alert",
        },
    ]

    return simulated_beds


def render_bed_card(bed):
    status_class_map = {
        "normal": "bed-normal",
        "warning": "bed-warning",
        "alert": "bed-alert",
        "gray": "bed-normal",
    }

    pill_map = {
        "normal": "status-normal",
        "warning": "status-warning",
        "alert": "status-alert",
        "gray": "status-normal",
    }

    card_class = status_class_map.get(bed["狀態鍵"], "bed-normal")
    pill_class = pill_map.get(bed["狀態鍵"], "status-normal")

    progress = 0
    if alarm_threshold > 0:
        progress = min(bed["秒數"] / alarm_threshold, 1.0)

    st.markdown(
        f"""
        <div class="bed-card {card_class}">
            <div class="bed-id">🛏️ {bed["床位"]}</div>
            <span class="status-pill {pill_class}">{bed["狀態"]}</span>
            <div class="bed-meta">
                目前姿勢：<b>{bed["姿勢"]}</b><br>
                持續時間：<b>{format_duration(bed["秒數"])}</b><br>
                資料來源：{bed["來源"]}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.progress(progress)


def render_dashboard():
    beds = build_bed_data()

    normal_count = sum(1 for bed in beds if bed["狀態鍵"] == "normal")
    warning_count = sum(1 for bed in beds if bed["狀態鍵"] == "warning")
    alert_count = sum(1 for bed in beds if bed["狀態鍵"] == "alert")

    st.markdown('<div class="section-title">照護站總覽</div>', unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("目前監測床數", f"{len(beds)} 床")
    c2.metric("正常", f"{normal_count} 床")
    c3.metric("接近警報", f"{warning_count} 床")
    c4.metric("需翻身", f"{alert_count} 床")

    st.markdown("")

    cols = st.columns(4)

    for col, bed in zip(cols, beds):
        with col:
            render_bed_card(bed)

    st.caption("A01 為目前即時 YOLO 偵測結果；A02 至 A04 為展示多床監測概念的模擬資料。")



def render_summary_panel():

    state = get_realtime_state()

    posture_now = state["posture"]
    duration_now = state["duration"]
    alarm_now = state["alarm"]
    monitoring_now = state["monitoring"]

    status_text, status_key = get_status(
        duration_now,
        alarm_threshold,
        alarm_now,
        monitoring_now,
        posture_now
    )

    st.markdown('<div class="section-title">A01 即時摘要</div>', unsafe_allow_html=True)

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
            <div class="metric-value">{format_duration(duration_now)}</div>
        </div>
        """, unsafe_allow_html=True)

    with c3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">系統狀態</div>
            <div class="metric-value">{status_text}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("##### 距離警報門檻")
    progress = min(duration_now / alarm_threshold, 1.0) if alarm_threshold > 0 else 0
    st.progress(progress)
    st.caption(f"目前門檻：{format_duration(alarm_threshold)}")

    st.markdown("### 3. 警報摘要")

    if alarm_now:
        st.markdown(f"""
        <div class="alert-box">
            🚨 A01 偵測到姿勢持續超過 {format_duration(alarm_threshold)}，
            請協助翻身或確認狀況。
        </div>
        """, unsafe_allow_html=True)

        render_loop_alarm()

        if st.button("✅ 確認此資訊並重新計時", type="primary"):
            with shared_state.lock:
                current_posture = shared_state.current_posture
                shared_state.alarm = False
                shared_state.alarm_acknowledged = False
                shared_state.alarm_logged = False
                shared_state.start_time = time.time()
                shared_state.duration = 0.0
                shared_state.last_posture = current_posture

                shared_state.alarm_logs.insert(
                    0,
                    {
                        "時間": datetime.now().strftime("%H:%M:%S"),
                        "床位": "A01",
                        "姿勢": current_posture,
                        "持續時間": "已重新計時",
                        "狀態": "已確認"
                    }
                )

            st.rerun()

    elif status_key == "warning":
        st.markdown(f"""
        <div class="warning-box">
            ⚠️ A01 已接近警報門檻，請留意是否需要協助翻身。
        </div>
        """, unsafe_allow_html=True)

    else:
        st.markdown("""
        <div class="normal-box">
            ✅ 目前尚未觸發警報
        </div>
        """, unsafe_allow_html=True)


def render_alarm_log():
    state = get_realtime_state()
    logs = state["logs"]

    st.markdown('<div class="section-title">警報紀錄</div>', unsafe_allow_html=True)

    if logs:
        st.dataframe(logs, use_container_width=True, hide_index=True)
    else:
        st.info("目前尚無警報紀錄。")


def render_system_info():
    st.markdown('<div class="section-title">系統流程與未來擴充</div>', unsafe_allow_html=True)

    st.markdown(
        """
        #### 系統流程
        1. 擷取即時影像  
        2. 使用 YOLO Pose 偵測人體關鍵點  
        3. 根據肩膀、軀幹與臉部關鍵點判斷姿勢  
        4. 計算同一姿勢持續時間  
        5. 超過門檻後觸發警報，照服員確認後重新計時  

        #### 多床監測設計
        目前 A01 使用即時 webcam 進行偵測，A02 至 A04 使用模擬資料展示多床照護站介面。
        未來可將每床 camera 串接為獨立影像來源，例如 USB camera 或 IP camera / RTSP stream，
        並為每張床建立獨立的姿勢、計時與警報狀態。
        """
    )


# =========================
# Header
# =========================
st.markdown(
    '<div class="main-title">🛏️ 長照睡姿固定過久警報系統</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="sub-text">照護站多床監測儀表板，協助照服員快速辨識長時間未翻身住民。</div>',
    unsafe_allow_html=True
)


# =========================
# Tabs
# =========================
tab_overview, tab_live, tab_log, tab_info = st.tabs(
    ["🏥 照護總覽", "🎥 A01 即時監測", "📋 警報紀錄", "ℹ️ 系統說明"]
)

with tab_overview:
    render_dashboard()

with tab_live:
    left_col, right_col = st.columns([1.05, 1.25])

    with left_col:
        st.markdown(
            '<div class="section-title">1. 即時影像監測</div>',
            unsafe_allow_html=True
        )

        ctx = webrtc_streamer(
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

    with right_col:
        summary_placeholder = st.empty()

        st.caption(
            "右側摘要會在 camera 運行時即時更新；若畫面短暫停頓，請以左側影像上方黑框資訊為主。"
        )

    # =========================
    # Live Summary Update
    # 不使用 st.fragment，避免 camera 反覆重啟
    # =========================
    if ctx.state.playing:
        while ctx.state.playing:
            with summary_placeholder.container():
                render_summary_panel()

            time.sleep(1)

    else:
        with summary_placeholder.container():
            render_summary_panel()

with tab_log:
    render_alarm_log()

with tab_info:
    render_system_info()
