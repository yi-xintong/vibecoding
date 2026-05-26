import os
import random
import time
from datetime import datetime

import streamlit as st
from openai import APIConnectionError, APIStatusError, AuthenticationError, OpenAI

# ==========================================
# 1. 核心配置与 API 链接
# ==========================================
def get_api_key() -> str:
    if st.session_state.get("deepseek_api_key", "").strip():
        return st.session_state.deepseek_api_key.strip()
    env_key = os.getenv("DEEPSEEK_API_KEY", "").strip()
    if env_key:
        return env_key
    try:
        return st.secrets["DEEPSEEK_API_KEY"].strip()
    except (KeyError, FileNotFoundError, AttributeError):
        return ""


def get_client(api_key: str) -> OpenAI:
    return OpenAI(
        api_key=api_key,
        base_url="https://api.deepseek.com/v1",
    )


def stream_chat(client: OpenAI, system_prompt: str):
    stream = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": "输出报告。"},
        ],
        stream=True,
    )
    for chunk in stream:
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta.content
        if delta:
            yield delta


def format_api_error(err: Exception) -> str:
    if isinstance(err, AuthenticationError):
        return "API Key 无效或已过期，请到 DeepSeek 控制台重新生成并更新。"
    if isinstance(err, APIConnectionError):
        return "网络连接失败，请检查本机网络或关闭全局代理后重试。"
    if isinstance(err, APIStatusError):
        if err.status_code == 402:
            return "账户余额不足，请前往 DeepSeek 控制台充值。"
        if err.status_code == 429:
            return "请求过于频繁，请稍后再试。"
        return f"API 返回错误（HTTP {err.status_code}）：{err.message}"
    return str(err)


def run_reading(client: OpenAI, system_prompt: str, spinner_text: str) -> tuple[bool, str]:
    with st.spinner(spinner_text):
        try:
            with st.container(border=True):
                full_text = st.write_stream(stream_chat(client, system_prompt))
            if isinstance(full_text, str):
                return True, full_text
            return True, "".join(str(part) for part in full_text)
        except Exception as e:
            st.error(f"连接时空磁场时出现了波动：{format_api_error(e)}（注意关闭全局代理）")
            with st.expander("查看原始报错"):
                st.exception(e)
            return False, ""


def extract_share_actions(full_text: str, section_title: str, fallback: str) -> str:
    """截取锦囊段落，冒号前核心词用于分享文案。"""
    if section_title not in full_text:
        return fallback
    lines = [
        line.strip()
        for line in full_text.split(section_title)[-1].split("\n")
        if line.strip() and not line.startswith("#")
    ]
    if lines:
        return " ".join([line.split("：")[0].split(":")[0].strip() for line in lines[:3]])
    return fallback


def extract_history_preview(full_text: str, max_len: int = 80) -> str:
    """从解读文本中提取锦囊段落预览。"""
    for section_title in ("### 🗝️ 解忧锦囊", "### 🗝️ 双人相处锦囊"):
        if section_title in full_text:
            lines = [
                line.strip()
                for line in full_text.split(section_title, 1)[-1].split("\n")
                if line.strip() and not line.startswith("#")
            ]
            if lines:
                preview = " ".join(lines[:3])
                if len(preview) > max_len:
                    return preview[:max_len] + "..."
                return preview
    lines = [line.strip() for line in full_text.split("\n") if line.strip()]
    preview = " ".join(lines[:3])
    if len(preview) > max_len:
        return preview[:max_len] + "..."
    return preview


def add_history_record(mode: str, textarea_key: str, result_name: str, full_text: str) -> None:
    user_input_raw = st.session_state.get(textarea_key, "").strip()
    record = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "mode": mode,
        "user_input": user_input_raw if user_input_raw else "情绪盲盒",
        "result_name": result_name,
        "preview": extract_history_preview(full_text),
    }
    st.session_state.history = [record] + st.session_state.history[:9]


def show_history_detail(preview: str) -> None:
    st.toast(preview)


def fill_textarea(textarea_key: str, text: str) -> None:
    st.session_state[textarea_key] = text


def render_tag_buttons(textarea_key: str, tags: list[str], tag_key_prefix: str):
    st.markdown(
        "<div style='font-size: 12px; color: #888; margin-top: -10px; margin-bottom: 5px;'>"
        "💡 点击可直接填入：</div>",
        unsafe_allow_html=True,
    )
    col_widths = [max(len(tag), 8) for tag in tags] + [2]
    cols = st.columns(col_widths, gap="small")
    for idx, (col, label) in enumerate(zip(cols[:-1], tags)):
        with col:
            st.button(
                label,
                key=f"{tag_key_prefix}{idx}",
                use_container_width=False,
                on_click=fill_textarea,
                args=(textarea_key, label),
            )


def render_share_block(share_text: str, share_key: str, label: str, height: int = 110):
    st.write("")
    st.markdown("### 💌 分享给同样在纠结的 TA")
    st.text_area(label, value=share_text, height=height, key=share_key)


def reset_dual_ready(mode: str) -> None:
    st.session_state[f"my_ready_{mode}"] = False
    st.session_state[f"ta_ready_{mode}"] = False


def render_dual_ready_controls(mode: str) -> bool:
    my_key = f"my_ready_{mode}"
    ta_key = f"ta_ready_{mode}"
    if my_key not in st.session_state:
        st.session_state[my_key] = False
    if ta_key not in st.session_state:
        st.session_state[ta_key] = False

    col1, col2 = st.columns(2)
    with col1:
        if st.session_state[my_key]:
            st.success("✅ 我准备好啦")
        elif st.button("👤 我准备好啦", key=f"btn_my_ready_{mode}", use_container_width=True):
            st.session_state[my_key] = True
    with col2:
        if st.session_state[ta_key]:
            st.success("✅ TA 已准备好")
        elif st.button("👥 帮 TA 点一下", key=f"btn_ta_ready_{mode}", use_container_width=True):
            st.session_state[ta_key] = True

    st.markdown(
        "<div class='dual-link-hint'>"
        "💡 双人联动说明：本版本为单设备演示，双方点击准备后即可抽卦，跨设备互动将会在未来版本实现~"
        "</div>",
        unsafe_allow_html=True,
    )
    st.write("")

    return st.session_state[my_key] and st.session_state[ta_key]


def render_history_sidebar() -> None:
    st.caption("刷新页面后记录将清空。")
    if not st.session_state.history:
        st.markdown(
            "<p style='color:#999;font-size:13px;margin:8px 0 16px 0;'>"
            "暂无占卜记录，试试抽一卦吧"
            "</p>",
            unsafe_allow_html=True,
        )
    else:
        for idx, record in enumerate(st.session_state.history):
            st.markdown(
                f"<div style='margin:10px 0 6px 0;line-height:1.5;'>"
                f"<span style='font-size:12px;color:#888;'>{record['timestamp']}</span><br>"
                f"<span style='font-size:14px;font-weight:600;color:#333;'>"
                f"{record['mode']} · {record['result_name']}"
                f"</span>"
                f"</div>",
                unsafe_allow_html=True,
            )
            st.caption(f"📝 {record['user_input']}")
            preview_short = record["preview"][:60]
            if len(record["preview"]) > 60:
                preview_short += "..."
            st.markdown(
                f"<p style='font-size:12px;color:#666;margin:0 0 8px 0;line-height:1.5;'>"
                f"{preview_short}"
                f"</p>",
                unsafe_allow_html=True,
            )
            st.button(
                "查看详情",
                key=f"history_detail_{idx}",
                on_click=show_history_detail,
                args=(record["preview"],),
            )
            if idx < len(st.session_state.history) - 1:
                st.divider()


SINGLE_TAGS = ["offer等待焦虑", "人生方向抉择Yes or No", "好无聊我做点什么呢"]
DUAL_TAGS = ["最近总冷战吵架", "异地恋方向迷茫", "关系卡在暧昧期"]
SHARE_LINK = os.getenv("APP_SHARE_URL", "https://vibecoding-7ky3op9bomtwgk9szuqeyf.streamlit.app/")

ICHING_HEXAGRAMS = {
    "乾为天": {"num": 1}, "坤为地": {"num": 2}, "水雷屯": {"num": 3}, "山水蒙": {"num": 4},
    "水天需": {"num": 5}, "天水讼": {"num": 6}, "地水师": {"num": 7}, "水地比": {"num": 8},
    "风天小畜": {"num": 9}, "天泽履": {"num": 10}, "地天泰": {"num": 11}, "天地否": {"num": 12},
    "天火同人": {"num": 13}, "火天大有": {"num": 14}, "地山谦": {"num": 15}, "雷地豫": {"num": 16},
    "泽雷随": {"num": 17}, "山风蛊": {"num": 18}, "地泽临": {"num": 19}, "风地观": {"num": 20},
    "火雷噬嗑": {"num": 21}, "山火贲": {"num": 22}, "山地剥": {"num": 23}, "地雷复": {"num": 24},
    "天雷无妄": {"num": 25}, "山天大畜": {"num": 26}, "山雷颐": {"num": 27}, "泽风大过": {"num": 28},
    "坎为水": {"num": 29}, "离为火": {"num": 30}, "泽山咸": {"num": 31}, "雷风恒": {"num": 32},
    "天山遁": {"num": 33}, "雷天大壮": {"num": 34}, "火地晋": {"num": 35}, "地火明夷": {"num": 36},
    "风火家人": {"num": 37}, "火泽睽": {"num": 38}, "水山蹇": {"num": 39}, "雷水解": {"num": 40},
    "山泽损": {"num": 41}, "风雷益": {"num": 42}, "泽天夬": {"num": 43}, "天风姤": {"num": 44},
    "泽地萃": {"num": 45}, "地风升": {"num": 46}, "泽水困": {"num": 47}, "水风井": {"num": 48},
    "泽火革": {"num": 49}, "火风鼎": {"num": 50}, "震为雷": {"num": 51}, "艮为山": {"num": 52},
    "风山渐": {"num": 53}, "雷泽归妹": {"num": 54}, "雷火丰": {"num": 55}, "火山旅": {"num": 56},
    "巽为风": {"num": 57}, "兑为泽": {"num": 58}, "风水涣": {"num": 59}, "水泽节": {"num": 60},
    "风泽中孚": {"num": 61}, "雷山小过": {"num": 62}, "水火既济": {"num": 63}, "火水未济": {"num": 64},
}

TAROT_CARDS = [
    # 大阿卡纳 22 张
    "愚人", "魔术师", "女祭司", "女皇", "皇帝", "教皇", "恋人", "战车", "力量", "隐士",
    "命运之轮", "正义", "倒吊人", "死神", "节制", "恶魔", "高塔", "星星", "月亮", "太阳",
    "审判", "世界",
    # 权杖组
    "权杖一", "权杖二", "权杖三", "权杖四", "权杖五", "权杖六", "权杖七", "权杖八", "权杖九", "权杖十",
    "权杖侍从", "权杖骑士", "权杖王后", "权杖国王",
    # 圣杯组
    "圣杯一", "圣杯二", "圣杯三", "圣杯四", "圣杯五", "圣杯六", "圣杯七", "圣杯八", "圣杯九", "圣杯十",
    "圣杯侍从", "圣杯骑士", "圣杯王后", "圣杯国王",
    # 宝剑组
    "宝剑一", "宝剑二", "宝剑三", "宝剑四", "宝剑五", "宝剑六", "宝剑七", "宝剑八", "宝剑九", "宝剑十",
    "宝剑侍从", "宝剑骑士", "宝剑王后", "宝剑国王",
    # 星币组
    "星币一", "星币二", "星币三", "星币四", "星币五", "星币六", "星币七", "星币八", "星币九", "星币十",
    "星币侍从", "星币骑士", "星币王后", "星币国王",
]

# ==========================================
# 2. 初始化状态机（4 维度抽牌计数器）
# ==========================================
for key, default in [
    ("se_count", 0),
    ("sw_count", 0),
    ("de_count", 0),
    ("dw_count", 0),
    ("se_ta", ""),
    ("sw_ta", ""),
    ("de_ta", ""),
    ("dw_ta", ""),
    ("history", []),
]:
    if key not in st.session_state:
        st.session_state[key] = default

# ==========================================
# 3. 极简呼吸风与全面视觉调优（CSS）
# ==========================================
st.set_page_config(page_title="随时随地算一卦", page_icon="🔮", layout="centered")

st.markdown(
    """
<style>
    .stApp { background-color: #FAFAFA; }
    .stDeployButton { display: none; }

    [data-baseweb="tab-highlight"] {
        display: none !important;
        background-color: transparent !important;
    }

    /* 快捷标签：单行显示，宽度不足时可左右滑动 */
    div[data-testid="stHorizontalBlock"]:has(div[data-testid="column"]:nth-child(4)) {
        overflow-x: auto;
        flex-wrap: nowrap !important;
        gap: 8px;
        padding-bottom: 8px;
        scroll-behavior: smooth;
        -webkit-overflow-scrolling: touch;
        align-items: flex-start !important;
    }
    div[data-testid="stHorizontalBlock"]:has(div[data-testid="column"]:nth-child(4)) div[data-testid="column"] {
        flex: 0 0 auto !important;
        width: auto !important;
    }
    div[data-testid="stHorizontalBlock"]:has(div[data-testid="column"]:nth-child(4)) .stButton {
        width: fit-content !important;
    }
    div[data-testid="stHorizontalBlock"]:has(div[data-testid="column"]:nth-child(4)) .stButton > button {
        background-color: #F0F2F5 !important;
        color: #666 !important;
        border: none !important;
        white-space: nowrap !important;
        word-break: keep-all !important;
        font-size: 12px !important;
        padding: 2px 8px !important;
        font-weight: 400 !important;
        border-radius: 12px !important;
        width: max-content !important;
        min-height: auto !important;
        height: 24px !important;
        line-height: 1 !important;
    }
    div[data-testid="stHorizontalBlock"]:has(div[data-testid="column"]:nth-child(4)) .stButton > button:hover {
        color: #111 !important;
        background-color: #E4E7ED !important;
    }
    div[data-testid="stHorizontalBlock"]:has(div[data-testid="column"]:nth-child(4)) .stButton > button p,
    div[data-testid="stHorizontalBlock"]:has(div[data-testid="column"]:nth-child(4)) .stButton > button div {
        font-size: 12px !important;
        white-space: nowrap !important;
    }

    /* 二级胶囊 Radio */
    div[data-testid="stRadio"] > label { display: none !important; }
    div[data-testid="stRadio"] [data-testid="stHorizontalBlock"] {
        background: #F0F2F5;
        padding: 4px;
        border-radius: 20px;
        display: inline-flex !important;
        margin-bottom: 25px;
        width: fit-content;
    }
    div[data-testid="stRadio"] label[data-baseweb="radio"] {
        background: transparent !important;
        border: none !important;
        padding: 6px 18px !important;
        border-radius: 16px !important;
        font-size: 13px !important;
        color: #666 !important;
        margin: 0 !important;
    }
    div[data-testid="stRadio"] label[data-baseweb="radio"]:has(input:checked) {
        background: #FFFFFF !important;
        color: #111 !important;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
        font-weight: 700 !important;
    }
    div[data-testid="stRadio"] input[type="radio"] { display: none !important; }

    /* 主操作按钮：绿 / 紫 */
    .btn-east .stButton > button {
        background-color: #2E7D32 !important;
        color: white !important;
        font-weight: bold !important;
        border: none !important;
    }
    .btn-east .stButton > button:hover { background-color: #1B5E20 !important; color: white !important; }
    .btn-west .stButton > button {
        background-color: #6A1B9A !important;
        color: white !important;
        font-weight: bold !important;
        border: none !important;
    }
    .btn-west .stButton > button:hover { background-color: #4A148C !important; color: white !important; }

    div[data-testid="stVerticalBlockBorderWrapper"] {
        background-color: #FFFFFF;
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.02);
    }

    /* 美化切换 Tab */
    .stTabs [data-baseweb="tab-list"] {
        gap: 12px;
        background-color: #f0f2f5;
        padding: 6px;
        border-radius: 40px;
        margin-bottom: 24px;
        width: fit-content;
        margin-left: auto;
        margin-right: auto;
        box-shadow: 0 2px 6px rgba(0,0,0,0.05);
        border-bottom: none !important;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 32px !important;
        padding: 10px 28px !important;
        font-weight: 600;
        font-size: 16px;
        transition: all 0.2s ease;
    }
    .stTabs [aria-selected="true"] {
        background-color: #2E7D32 !important;
        color: white !important;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
    .stTabs [aria-selected="false"]:hover {
        background-color: #e0e2e5;
        color: #111;
    }

    /* 双人模式准备按钮 */
    .dual-ready-btn .stButton > button {
        background-color: #6c757d !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 12px !important;
        font-weight: 600 !important;
        box-shadow: none !important;
    }
    .dual-ready-btn .stButton > button:hover {
        background-color: #5a6268 !important;
        color: #ffffff !important;
    }
    .dual-link-hint {
        background-color: #f5f5f5;
        color: #888888;
        border-radius: 12px;
        padding: 10px 14px;
        font-size: 13px;
        line-height: 1.6;
        margin-top: 8px;
    }

    @media (max-width: 600px) {
        h1 {
            font-size: calc(1.2rem + 3vw);
            white-space: nowrap;
            text-align: center;
            padding: 0 10px;
        }
        .stTabs [data-baseweb="tab"] {
            padding: 6px 12px !important;
            font-size: 13px !important;
        }
        .stButton > button {
            font-size: 14px !important;
            padding: 0.4rem 0.8rem !important;
        }
        div[data-testid="stHorizontalBlock"]:has(div[data-testid="column"]:nth-child(4)) {
            gap: 6px !important;
            flex-wrap: nowrap !important;
            overflow-x: auto;
        }
        div[data-testid="stHorizontalBlock"]:has(div[data-testid="column"]:nth-child(4)) .stButton > button {
            font-size: 11px !important;
            padding: 2px 6px !important;
            height: 22px !important;
        }
        .stTextArea textarea {
            font-size: 14px !important;
        }
        .stMarkdown, .stCaption {
            font-size: 13px !important;
        }
    }
</style>
""",
    unsafe_allow_html=True,
)

with st.sidebar:
    st.subheader("📜 历史记录")
    render_history_sidebar()

api_key = get_api_key()

st.title("🔮 随时随地算一卦")
st.markdown(
    "<div style='color: #444; font-size: 15px; line-height: 1.6; margin-bottom: 25px;'>"
    "你好呀，这里是你的疗愈树洞。<br>"
    "随时随地算一卦，用科学的潜意识投射帮你理清思绪，我们会认真倾听你。"
    "</div>",
    unsafe_allow_html=True,
)

if not api_key:
    st.warning(
        "未检测到 API Key。请在启动前设置环境变量："
        "`export DEEPSEEK_API_KEY='你的key'`"
    )

# ==========================================
# 4. 双层嵌套场景架构
# ==========================================
tab_single, tab_dual = st.tabs(["🧘 单人自愈模式", "💞 双人磁场合盘"])


def single_iching(api_key: str):
    user_input = st.text_area(
        "✍️ 把那些让你焦虑、纠结或期盼的事，都写在这里吧...",
        height=120,
        placeholder="",
        key="se_ta",
    )
    render_tag_buttons("se_ta", SINGLE_TAGS, "tag_se")
    st.write("")

    with st.container():
        st.markdown('<div class="btn-east">', unsafe_allow_html=True)
        draw = st.button("✨ 沉心求一卦", use_container_width=True, key="east_btn_s")
        st.markdown("</div>", unsafe_allow_html=True)

    if not draw:
        return
    if not api_key:
        st.error("请先配置 DeepSeek API Key（环境变量 DEEPSEEK_API_KEY）。")
        return

    st.session_state.se_count += 1
    current_second = int(time.time())
    gua_names = list(ICHING_HEXAGRAMS.keys())
    gua_name = gua_names[current_second % len(gua_names)]
    gua_num = ICHING_HEXAGRAMS[gua_name]["num"]
    st.markdown(
        f"<div style='padding: 15px; border-radius: 8px; background-color: #F0F4F2; "
        f"color: #1E4620; border-left: 5px solid #2E7D32;'>系统为你抽取了卦象：<b>{gua_name}</b></div>",
        unsafe_allow_html=True,
    )

    is_empty = not user_input.strip()
    if is_empty:
        st.caption("🎁 情绪盲盒模式：未填写心事，将为你解读今日卦象运势。")
    prompt_context = (
        "触发【情绪盲盒】。针对抽到的卦象生成【今日运势】。"
        if is_empty
        else f"烦恼是：“{user_input}”。"
    )
    guideline = "从整体能量切入定调。" if is_empty else "温柔拆解烦恼，理清思绪。"

    system_prompt = f"""
    你是一位知心导师。用户抽到《易经》第 {gua_num} 卦【{gua_name}】。{prompt_context}

    【绝不准客套】严禁输出如「准备好、作为导师」等任何开场白！必须直接从第一个标题开始输出！

    ### 📜 卦象原文
    《易经》-第 {gua_num} 卦。
    (核心意象解读)

    ### 🫂 卦象解读
    ({guideline})

    ### 🗝️ 解忧锦囊
    (给出3个今天现实中就能动手的具体破局行动建议)
    """

    ok, full_text = run_reading(get_client(api_key), system_prompt, "☯️ 正在解卦...")
    if ok:
        add_history_record("单人易经", "se_ta", gua_name, full_text)
        actions = extract_share_actions(
            full_text, "### 🗝️ 解忧锦囊", "1. 现实破局 2. 拒绝内耗 3. 理清思绪"
        )
        share_text = (
            f"✨【随时随地算一卦】今日我抽到了《易经》第{gua_num}卦·{gua_name}。"
            f"它给我的锦囊是：{actions}\n\n"
            f"🔮 遇到了生活卡点？点击链接白嫖锦囊 👉 [{SHARE_LINK}]"
        )
        render_share_block(share_text, f"se_s_{st.session_state.se_count}", "长按复制分享：")


def single_tarot(api_key: str):
    user_input = st.text_area(
        "✍️ 把那些让你焦虑、纠结或期盼的事，都写在这里吧...",
        height=120,
        placeholder="",
        key="sw_ta",
    )
    render_tag_buttons("sw_ta", SINGLE_TAGS, "tag_sw")
    st.write("")

    with st.container():
        st.markdown('<div class="btn-west">', unsafe_allow_html=True)
        draw = st.button("🔮 凭直觉抽一牌", use_container_width=True, key="west_btn_s")
        st.markdown("</div>", unsafe_allow_html=True)

    if not draw:
        return
    if not api_key:
        st.error("请先配置 DeepSeek API Key（环境变量 DEEPSEEK_API_KEY）。")
        return

    st.session_state.sw_count += 1
    current_second = int(time.time())
    card = TAROT_CARDS[current_second % len(TAROT_CARDS)]
    st.markdown(
        f"<div style='padding: 15px; border-radius: 8px; background-color: #F3E5F5; "
        f"color: #4A148C; border-left: 5px solid #6A1B9A;'>抽取了塔罗牌：<b>{card}</b></div>",
        unsafe_allow_html=True,
    )

    is_empty = not user_input.strip()
    if is_empty:
        st.caption("🎁 情绪盲盒模式：未填写心事，将为你解读今日牌面运势。")
    prompt_context = (
        "触发【情绪盲盒】。针对塔罗牌生成【今日运势】。"
        if is_empty
        else f"烦恼是：“{user_input}”。"
    )
    guideline = "给出一整天能量定调。" if is_empty else "从潜意识解析烦恼，理清思绪。"

    system_prompt = f"""
    你是一位塔罗疗愈师。用户抽到牌：【{card}】。{prompt_context}

    【绝不准客套】第一行必须直接开始！

    ### 🔮 牌面牌义
    (能量核心词与精神原型)

    ### 🌌 塔罗解读
    ({guideline})

    ### 🗝️ 解忧锦囊
    (3个现实调整行动)
    """

    ok, full_text = run_reading(get_client(api_key), system_prompt, "🌌 感知能量...")
    if ok:
        add_history_record("单人塔罗", "sw_ta", card, full_text)
        actions = extract_share_actions(
            full_text, "### 🗝️ 解忧锦囊", "1. 现实破局 2. 拒绝内耗 3. 理清思绪"
        )
        share_text = (
            f"✨【随时随地算一卦】今日我直觉抽到了塔罗牌·{card}。锦囊是：{actions}\n\n"
            f"🔮 遇到生活卡点？点击链接白嫖锦囊 👉 [{SHARE_LINK}]"
        )
        render_share_block(share_text, f"sw_s_{st.session_state.sw_count}", "长按复制分享：")


def dual_iching(api_key: str):
    user_input = st.text_area(
        "✍️ 把你们当下的相处卡点或共同纠结，都写在这里吧...",
        height=120,
        placeholder="",
        key="de_ta",
    )
    render_tag_buttons("de_ta", DUAL_TAGS, "tag_de")
    st.write("")

    both_ready = render_dual_ready_controls("de")
    draw = False
    if both_ready:
        with st.container():
            st.markdown('<div class="btn-east">', unsafe_allow_html=True)
            draw = st.button("✨ 沉心合一卦", use_container_width=True, key="east_btn_d")
            st.markdown("</div>", unsafe_allow_html=True)

    if not draw:
        return
    if not api_key:
        st.error("请先配置 DeepSeek API Key（环境变量 DEEPSEEK_API_KEY）。")
        return

    st.session_state.de_count += 1
    current_second = int(time.time())
    gua_names = list(ICHING_HEXAGRAMS.keys())
    gua_name = gua_names[current_second % len(gua_names)]
    gua_num = ICHING_HEXAGRAMS[gua_name]["num"]
    st.markdown(
        f"<div style='padding: 15px; border-radius: 8px; background-color: #F0F4F2; "
        f"color: #1E4620; border-left: 5px solid #2E7D32;'>为你们抽取了关系卦象：<b>{gua_name}</b></div>",
        unsafe_allow_html=True,
    )

    is_empty = not user_input.strip()
    if is_empty:
        st.caption("🎁 双人盲盒：未填写卡点，将解读今日相处磁场。")
    prompt_context = (
        "【双人盲盒】解析今天相处运势磁场。"
        if is_empty
        else f"相处卡点是：“{user_input}”。"
    )
    guideline = (
        "解析两股能量今天如何对齐。"
        if is_empty
        else "温柔剖析关系卡顿原因，帮其理清纠缠。"
    )

    system_prompt = f"""
    你是一位双人关系导师。用户测算双人互动抽到卦象：【{gua_name}】。{prompt_context}

    【绝对严禁开场白】必须直接从标题开始！

    ### ☯️ 东方双人磁场
    《易经》-第 {gua_num} 卦。
    (意象定调)

    ### 🫂 双人关系剖析
    ({guideline})

    ### 🗝️ 双人相处锦囊
    (3个现实相处建议)
    """

    ok, full_text = run_reading(get_client(api_key), system_prompt, "☯️ 磁场对齐中...")
    if ok:
        add_history_record("双人易经", "de_ta", gua_name, full_text)
        actions = extract_share_actions(
            full_text, "### 🗝️ 双人相处锦囊", "1. 双人破局 2. 拒绝内耗 3. 理清纠结"
        )
        share_text = (
            f"✨【双人磁场合盘】我和TA今天测了关系，抽到《易经》第{gua_num}卦·{gua_name}。"
            f"锦囊是：{actions}\n\n"
            f"🔮 你们的感情也遇到了卡点？点击链接，一起白嫖双人锦囊 👉 [{SHARE_LINK}]"
        )
        render_share_block(
            share_text, f"de_s_{st.session_state.de_count}", "长按复制发给TA："
        )
        reset_dual_ready("de")


def dual_tarot(api_key: str):
    user_input = st.text_area(
        "✍️ 把你们当下的相处卡点或共同纠结，都写在这里吧...",
        height=120,
        placeholder="",
        key="dw_ta",
    )
    render_tag_buttons("dw_ta", DUAL_TAGS, "tag_dw")
    st.write("")

    both_ready = render_dual_ready_controls("dw")
    draw = False
    if both_ready:
        with st.container():
            st.markdown('<div class="btn-west">', unsafe_allow_html=True)
            draw = st.button("🔮 凭直觉合一牌", use_container_width=True, key="west_btn_d")
            st.markdown("</div>", unsafe_allow_html=True)

    if not draw:
        return
    if not api_key:
        st.error("请先配置 DeepSeek API Key（环境变量 DEEPSEEK_API_KEY）。")
        return

    st.session_state.dw_count += 1
    current_second = int(time.time())
    card = TAROT_CARDS[current_second % len(TAROT_CARDS)]
    st.markdown(
        f"<div style='padding: 15px; border-radius: 8px; background-color: #F3E5F5; "
        f"color: #4A148C; border-left: 5px solid #6A1B9A;'>抽取了塔罗牌：<b>{card}</b></div>",
        unsafe_allow_html=True,
    )

    is_empty = not user_input.strip()
    if is_empty:
        st.caption("🎁 双人盲盒：未填写卡点，将解读今日合盘能量。")
    prompt_context = (
        "【双人合盘盲盒】解析今日能量投射。"
        if is_empty
        else f"相处卡点是：“{user_input}”。"
    )
    guideline = (
        "解析潜意识能量如何交织。"
        if is_empty
        else "温柔客观解构双人互动心理，理清纠结。"
    )

    system_prompt = f"""
    你是一位双人心理疗愈师。测算双人互动抽到牌：【{card}】。{prompt_context}

    【绝对严禁开场白】第一行必须直接开始！

    ### 🃏 西方双人符号
    (象征与精神内核定调)

    ### 🌌 双人关系剖析
    ({guideline})

    ### 🗝️ 双人相处锦囊
    (3个现实相处建议)
    """

    ok, full_text = run_reading(get_client(api_key), system_prompt, "🌌 能量感应中...")
    if ok:
        add_history_record("双人塔罗", "dw_ta", card, full_text)
        actions = extract_share_actions(
            full_text, "### 🗝️ 双人相处锦囊", "1. 双人破局 2. 拒绝内耗 3. 理清纠结"
        )
        share_text = (
            f"✨【双人磁场合盘】我和TA今天测了关系，抽到塔罗牌·{card}。"
            f"锦囊是：{actions}\n\n"
            f"🔮 你们的感情也遇到了卡点？点击链接白嫖双人锦囊 👉 [{SHARE_LINK}]"
        )
        render_share_block(
            share_text, f"dw_s_{st.session_state.dw_count}", "长按复制发给TA："
        )
        reset_dual_ready("dw")


with tab_single:
    tool_s = st.radio(
        "ts",
        ["☯️ 易经哲思", "🃏 塔罗指引"],
        horizontal=True,
        key="tool_s",
        label_visibility="collapsed",
    )
    if tool_s == "☯️ 易经哲思":
        single_iching(api_key)
    else:
        single_tarot(api_key)

with tab_dual:
    tool_d = st.radio(
        "td",
        ["☯️ 易经哲思", "🃏 塔罗指引"],
        horizontal=True,
        key="tool_d",
        label_visibility="collapsed",
    )
    if tool_d == "☯️ 易经哲思":
        dual_iching(api_key)
    else:
        dual_tarot(api_key)
