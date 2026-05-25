import os
import random

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


SINGLE_TAGS = ["offer等待焦虑", "人生方向抉择Yes or No", "好无聊我做点什么呢"]
DUAL_TAGS = ["最近总冷战吵架", "异地恋方向迷茫", "关系卡在暧昧期"]
SHARE_LINK = os.getenv("APP_SHARE_URL", "http://localhost:8501")

ICHING_HEXAGRAMS = {
    "乾为天": {"num": 1},
    "坤为地": {"num": 2},
    "水雷屯": {"num": 3},
    "山水蒙": {"num": 4},
    "地天泰": {"num": 11},
    "天地否": {"num": 12},
    "火天大有": {"num": 14},
    "地山谦": {"num": 15},
    "泽水困": {"num": 47},
    "火山旅": {"num": 56},
    "水火既济": {"num": 63},
    "火水未济": {"num": 64},
}

TAROT_CARDS = [
    "愚人",
    "魔术师",
    "女祭司",
    "女皇",
    "皇帝",
    "教皇",
    "恋人",
    "战车",
    "力量",
    "隐士",
    "命运之轮",
    "正义",
    "倒吊人",
    "死神 (代表转化与新生)",
    "节制",
    "恶魔 (代表直面欲望)",
    "高塔 (代表打破重组)",
    "星星",
    "月亮",
    "太阳",
    "审判",
    "世界",
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
    .stTabs [data-baseweb="tab-list"] {
        gap: 40px;
        margin-bottom: 20px;
        border-bottom: 1px solid #EAEAEA !important;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 12px 24px;
        font-size: 18px;
        font-weight: 600;
        color: #999;
        border-bottom: none !important;
    }
    .stTabs [aria-selected="true"] { color: #111 !important; }

    /* 快捷标签 10px、不换行、容器随文字收缩 */
    div[data-testid="stHorizontalBlock"]:has(div[data-testid="column"]:nth-child(4)) {
        gap: 8px !important;
        align-items: flex-start !important;
    }
    div[data-testid="stHorizontalBlock"]:has(div[data-testid="column"]:nth-child(4)) div[data-testid="column"] {
        width: auto !important;
        flex: 0 0 auto !important;
    }
    div[data-testid="stHorizontalBlock"]:has(div[data-testid="column"]:nth-child(4)) .stButton {
        width: fit-content !important;
    }
    div[data-testid="stHorizontalBlock"]:has(div[data-testid="column"]:nth-child(4)) .stButton > button {
        background-color: #F0F2F5 !important;
        color: #666 !important;
        border: none !important;
        padding: 2px 6px !important;
        font-size: 10px !important;
        font-weight: 400 !important;
        border-radius: 12px !important;
        white-space: nowrap !important;
        word-break: keep-all !important;
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
        font-size: 10px !important;
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
</style>
""",
    unsafe_allow_html=True,
)

with st.sidebar:
    st.subheader("🔑 API 配置")
    st.text_input(
        "DeepSeek API Key",
        type="password",
        key="deepseek_api_key",
        placeholder="sk-...（未填则使用环境变量）",
        help="启动前也可执行：export DEEPSEEK_API_KEY='你的key'",
    )

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
        "未检测到 API Key。请在左侧侧边栏填写，或在启动前执行："
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
        st.error("请先配置 DeepSeek API Key（侧边栏或环境变量）。")
        return

    st.session_state.se_count += 1
    gua_name = random.choice(list(ICHING_HEXAGRAMS.keys()))
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
        st.error("请先配置 DeepSeek API Key（侧边栏或环境变量）。")
        return

    st.session_state.sw_count += 1
    card = random.choice(TAROT_CARDS)
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

    with st.container():
        st.markdown('<div class="btn-east">', unsafe_allow_html=True)
        draw = st.button("✨ 沉心合一卦", use_container_width=True, key="east_btn_d")
        st.markdown("</div>", unsafe_allow_html=True)

    if not draw:
        return
    if not api_key:
        st.error("请先配置 DeepSeek API Key（侧边栏或环境变量）。")
        return

    st.session_state.de_count += 1
    gua_name = random.choice(list(ICHING_HEXAGRAMS.keys()))
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


def dual_tarot(api_key: str):
    user_input = st.text_area(
        "✍️ 把你们当下的相处卡点或共同纠结，都写在这里吧...",
        height=120,
        placeholder="",
        key="dw_ta",
    )
    render_tag_buttons("dw_ta", DUAL_TAGS, "tag_dw")
    st.write("")

    with st.container():
        st.markdown('<div class="btn-west">', unsafe_allow_html=True)
        draw = st.button("🔮 凭直觉合一牌", use_container_width=True, key="west_btn_d")
        st.markdown("</div>", unsafe_allow_html=True)

    if not draw:
        return
    if not api_key:
        st.error("请先配置 DeepSeek API Key（侧边栏或环境变量）。")
        return

    st.session_state.dw_count += 1
    card = random.choice(TAROT_CARDS)
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
