import base64
import json
import os
import random
import re
import secrets
import time
from datetime import datetime

import html
from urllib.parse import urlencode

import streamlit as st
from openai import APIConnectionError, APIStatusError, AuthenticationError, OpenAI

# ==========================================
# 1. 核心配置
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
    return OpenAI(api_key=api_key, base_url="https://api.deepseek.com/v1")


def stream_chat(client: OpenAI, system_prompt: str):
    stream = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": "输出报告。"},
        ],
        stream=True,
        temperature=0.7,
    )
    for chunk in stream:
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta.content
        if delta:
            yield delta


def format_api_error(err: Exception) -> str:
    if isinstance(err, AuthenticationError):
        return "🔐 API Key 无效或已过期。"
    if isinstance(err, APIConnectionError):
        return "🌐 网络连接失败，请检查网络或关闭代理。"
    if isinstance(err, APIStatusError):
        if err.status_code == 402:
            return "💳 账户余额不足。"
        if err.status_code == 429:
            return "⏳ 请求过于频繁。"
        return f"API 错误（HTTP {err.status_code}）"
    return str(err)


def run_reading(
    client: OpenAI,
    system_prompt: str,
    spinner_text: str = "",
    *,
    tool: str = "iching",
    show_spinner: bool = True,
) -> tuple[bool, str]:
    """流式解读：先出现绿/紫解读框，再在框内逐段填入文字。"""

    def _stream_into_panel() -> str:
        slot = st.empty()
        slot.markdown(_reading_panel_skeleton_html(tool), unsafe_allow_html=True)
        raw_buf = ""
        for delta in stream_chat(client, system_prompt):
            raw_buf += delta
            display = format_reading_display_text(raw_buf, streaming=True)
            inner = (
                _format_reading_sections_html(display)
                if display.strip()
                else _reading_panel_skeleton_inner(tool)
            )
            slot.markdown(build_reading_panel_html(inner, tool), unsafe_allow_html=True)
        return raw_buf

    try:
        if show_spinner and spinner_text:
            with st.spinner(spinner_text):
                return True, _stream_into_panel()
        return True, _stream_into_panel()
    except Exception as e:
        st.error(f"连接 AI 服务时出现问题：{format_api_error(e)}")
        with st.expander("查看原始报错"):
            st.exception(e)
        return False, ""


# ==========================================
# 2. 安全过滤
# ==========================================
FORBIDDEN_PATTERNS = [
    "ignore previous instructions", "ignore all previous instructions",
    "system prompt", "you are now", "DAN", "jailbreak",
]


def sanitize_input(text: str, max_len: int = 500) -> tuple[bool, str, str]:
    if not text:
        return True, "", ""
    text = text.strip()
    if len(text) > max_len:
        return False, "", f"输入过长，请控制在 {max_len} 字以内"
    lower = text.lower()
    for p in FORBIDDEN_PATTERNS:
        if p in lower:
            return False, "", "检测到可疑输入，请重新描述"
    return True, text, ""


# ==========================================
# 3. 数据：64卦（含 binary / 卦辞 / 爻辞 / 象辞）
# ==========================================
ICHING_HEXAGRAMS = {
    "乾为天": {
        "num": 1, "binary": "111111", "keywords": ["刚健", "创始", "领导力"], "element": "天",
        "gua_ci": "元亨利贞。", "xiang_ci": "天行健，君子以自强不息。",
        "yao_ci": ["初九：潜龙勿用。", "九二：见龙在田，利见大人。", "九三：君子终日乾乾，夕惕若厉，无咎。", "九四：或跃在渊，无咎。", "九五：飞龙在天，利见大人。", "上九：亢龙有悔。"],
    },
    "坤为地": {
        "num": 2, "binary": "000000", "keywords": ["柔顺", "承载", "包容"], "element": "地",
        "gua_ci": "元亨，利牝马之贞。", "xiang_ci": "地势坤，君子以厚德载物。",
        "yao_ci": ["初六：履霜，坚冰至。", "六二：直方大，不习无不利。", "六三：含章可贞。或从王事，无成有终。", "六四：括囊，无咎无誉。", "六五：黄裳，元吉。", "上六：龙战于野，其血玄黄。"],
    },
    "水雷屯": {
        "num": 3, "binary": "010001", "keywords": ["初生", "艰难", "孕育"], "element": "水/雷",
        "gua_ci": "元亨利贞，勿用有攸往，利建侯。", "xiang_ci": "云雷屯，君子以经纶。",
        "yao_ci": ["初九：磐桓，利居贞，利建侯。", "六二：屯如邅如，乘马班如。匪寇婚媾，女子贞不字，十年乃字。", "六三：即鹿无虞，惟入于林中。君子几不如舍，往吝。", "六四：乘马班如，求婚媾，往吉，无不利。", "九五：屯其膏，小贞吉，大贞凶。", "上六：乘马班如，泣血涟如。"],
    },
    "山水蒙": {
        "num": 4, "binary": "100010", "keywords": ["启蒙", "教育", "困惑"], "element": "山/水",
        "gua_ci": "亨。匪我求童蒙，童蒙求我。", "xiang_ci": "山下出泉，蒙。君子以果行育德。",
        "yao_ci": ["初六：发蒙，利用刑人，用说桎梏，以往吝。", "九二：包蒙吉，纳妇吉，子克家。", "六三：勿用取女，见金夫，不有躬，无攸利。", "六四：困蒙，吝。", "六五：童蒙，吉。", "上九：击蒙，不利为寇，利御寇。"],
    },
    "水天需": {
        "num": 5, "binary": "010111", "keywords": ["等待", "耐心", "蓄力"], "element": "水/天",
        "gua_ci": "有孚，光亨，贞吉。利涉大川。", "xiang_ci": "云上于天，需。君子以饮食宴乐。",
        "yao_ci": ["初九：需于郊，利用恒，无咎。", "九二：需于沙，小有言，终吉。", "九三：需于泥，致寇至。", "六四：需于血，出自穴。", "九五：需于酒食，贞吉。", "上六：入于穴，有不速之客三人来，敬之终吉。"],
    },
    "天水讼": {
        "num": 6, "binary": "111010", "keywords": ["争执", "诉讼", "明辨"], "element": "天/水",
        "gua_ci": "有孚，窒惕，中吉，终凶。", "xiang_ci": "天与水违行，讼。君子以作事谋始。",
        "yao_ci": ["初六：不永所事，小有言，终吉。", "九二：不克讼，归而逋，其邑人三百户，无眚。", "六三：食旧德，贞厉，终吉。或从王事，无成。", "九四：不克讼，复即命渝，安贞吉。", "九五：讼，元吉。", "上九：或锡之鞶带，终朝三褫之。"],
    },
    "地水师": {
        "num": 7, "binary": "000010", "keywords": ["军队", "组织", "纪律"], "element": "地/水",
        "gua_ci": "贞，丈人吉，无咎。", "xiang_ci": "地中有水，师。君子以容民畜众。",
        "yao_ci": ["初六：师出以律，否臧凶。", "九二：在师中吉，无咎，王三锡命。", "六三：师或舆尸，凶。", "六四：师左次，无咎。", "六五：田有禽，利执言，无咎。长子帅师，弟子舆尸，贞凶。", "上六：大君有命，开国承家，小人勿用。"],
    },
    "水地比": {
        "num": 8, "binary": "010000", "keywords": ["亲比", "合作", "依附"], "element": "水/地",
        "gua_ci": "吉。原筮元永贞，无咎。不宁方来，后夫凶。", "xiang_ci": "地上有水，比。先王以建万国，亲诸侯。",
        "yao_ci": ["初六：有孚比之，无咎。有孚盈缶，终来有它，吉。", "六二：比之自内，贞吉。", "六三：比之匪人。", "六四：外比之，贞吉。", "九五：显比，王用三驱，失前禽。邑人不诫，吉。", "上六：比之无首，凶。"],
    },
    "风天小畜": {
        "num": 9, "binary": "110111", "keywords": ["蓄积", "小有", "蓄养"], "element": "风/天",
        "gua_ci": "亨。密云不雨，自我西郊。", "xiang_ci": "风行天上，小畜。君子以懿文德。",
        "yao_ci": ["初九：复自道，何其咎，吉。", "九二：牵复，吉。", "九三：舆说辐，夫妻反目。", "六四：有孚，血去惕出，无咎。", "九五：有孚挛如，富以其邻。", "上九：既雨既处，尚德载，妇贞厉。月几望，君子征凶。"],
    },
    "天泽履": {
        "num": 10, "binary": "111011", "keywords": ["践行", "礼仪", "谨慎"], "element": "天/泽",
        "gua_ci": "履虎尾，不咥人，亨。", "xiang_ci": "上天下泽，履。君子以辨上下，定民志。",
        "yao_ci": ["初九：素履，往无咎。", "九二：履道坦坦，幽人贞吉。", "六三：眇能视，跛能履，履虎尾，咥人，凶。武人为于大君。", "九四：履虎尾，愬愬，终吉。", "九五：夬履，贞厉。", "上九：视履考祥，其旋元吉。"],
    },
    "地天泰": {
        "num": 11, "binary": "000111", "keywords": ["通泰", "安泰", "交流"], "element": "地/天",
        "gua_ci": "小往大来，吉亨。", "xiang_ci": "天地交，泰。后以财成天地之道，辅相天地之宜，以左右民。",
        "yao_ci": ["初九：拔茅茹，以其汇，征吉。", "九二：包荒，用冯河，不遐遗，朋亡，得尚于中行。", "九三：无平不陂，无往不复，艰贞无咎。勿恤其孚，于食有福。", "六四：翩翩，不富以其邻，不戒以孚。", "六五：帝乙归妹，以祉元吉。", "上六：城复于隍，勿用师。自邑告命，贞吝。"],
    },
    "天地否": {
        "num": 12, "binary": "111000", "keywords": ["闭塞", "隔阂", "隐忍"], "element": "天/地",
        "gua_ci": "否之匪人，不利君子贞，大往小来。", "xiang_ci": "天地不交，否。君子以俭德辟难，不可荣以禄。",
        "yao_ci": ["初六：拔茅茹，以其汇，贞吉亨。", "六二：包承，小人吉，大人否，亨。", "六三：包羞。", "九四：有命无咎，畴离祉。", "九五：休否，大人吉。其亡其亡，系于苞桑。", "上九：倾否，先否后喜。"],
    },
    "天火同人": {
        "num": 13, "binary": "111101", "keywords": ["团结", "同人", "共识"], "element": "天/火",
        "gua_ci": "同人于野，亨。利涉大川，利君子贞。", "xiang_ci": "天与火，同人。君子以类族辨物。",
        "yao_ci": ["初九：同人于门，无咎。", "六二：同人于宗，吝。", "九三：伏戎于莽，升其高陵，三岁不兴。", "九四：乘其墉，弗克攻，吉。", "九五：同人，先号咷而后笑，大师克相遇。", "上九：同人于郊，无悔。"],
    },
    "火天大有": {
        "num": 14, "binary": "101111", "keywords": ["丰盛", "收获", "光明"], "element": "火/天",
        "gua_ci": "元亨。", "xiang_ci": "火在天上，大有。君子以遏恶扬善，顺天休命。",
        "yao_ci": ["初九：无交害，匪咎，艰则无咎。", "九二：大车以载，有攸往，无咎。", "九三：公用亨于天子，小人弗克。", "九四：匪其彭，无咎。", "六五：厥孚交如，威如，吉。", "上九：自天佑之，吉无不利。"],
    },
    "地山谦": {
        "num": 15, "binary": "000100", "keywords": ["谦虚", "退让", "内修"], "element": "地/山",
        "gua_ci": "亨，君子有终。", "xiang_ci": "地中有山，谦。君子以裒多益寡，称物平施。",
        "yao_ci": ["初六：谦谦君子，用涉大川，吉。", "六二：鸣谦，贞吉。", "九三：劳谦，君子有终，吉。", "六四：无不利，撝谦。", "六五：不富以其邻，利用侵伐，无不利。", "上六：鸣谦，利用行师，征邑国。"],
    },
    "雷地豫": {
        "num": 16, "binary": "001000", "keywords": ["愉悦", "预备", "和乐"], "element": "雷/地",
        "gua_ci": "利建侯行师。", "xiang_ci": "雷出地奋，豫。先王以作乐崇德，殷荐之上帝，以配祖考。",
        "yao_ci": ["初六：鸣豫，凶。", "六二：介于石，不终日，贞吉。", "六三：盱豫，悔。迟有悔。", "九四：由豫，大有得。勿疑，朋盍簪。", "六五：贞疾，恒不死。", "上六：冥豫，成有渝，无咎。"],
    },
    "泽雷随": {
        "num": 17, "binary": "011001", "keywords": ["随从", "顺势", "变通"], "element": "泽/雷",
        "gua_ci": "元亨利贞，无咎。", "xiang_ci": "泽中有雷，随。君子以向晦入宴息。",
        "yao_ci": ["初九：官有渝，贞吉。出门交有功。", "六二：系小子，失丈夫。", "六三：系丈夫，失小子。随有求得，利居贞。", "九四：随有获，贞凶。有孚在道，以明，何咎。", "九五：孚于嘉，吉。", "上六：拘系之，乃从维之。王用亨于西山。"],
    },
    "山风蛊": {
        "num": 18, "binary": "100110", "keywords": ["腐败", "革新", "整治"], "element": "山/风",
        "gua_ci": "元亨，利涉大川。先甲三日，后甲三日。", "xiang_ci": "山下有风，蛊。君子以振民育德。",
        "yao_ci": ["初六：干父之蛊，有子，考无咎，厉终吉。", "九二：干母之蛊，不可贞。", "九三：干父之蛊，小有悔，无大咎。", "六四：裕父之蛊，往见吝。", "六五：干父之蛊，用誉。", "上九：不事王侯，高尚其事。"],
    },
    "地泽临": {
        "num": 19, "binary": "000011", "keywords": ["临近", "督导", "审视"], "element": "地/泽",
        "gua_ci": "元，亨，利，贞。至于八月有凶。", "xiang_ci": "泽上有地，临。君子以教思无穷，容保民无疆。",
        "yao_ci": ["初九：咸临，贞吉。", "九二：咸临，吉无不利。", "六三：甘临，无攸利。既忧之，无咎。", "六四：至临，无咎。", "六五：知临，大君之宜，吉。", "上六：敦临，吉无咎。"],
    },
    "风地观": {
        "num": 20, "binary": "110000", "keywords": ["观察", "瞻仰", "自省"], "element": "风/地",
        "gua_ci": "盥而不荐，有孚颙若。", "xiang_ci": "风行地上，观。先王以省方，观民设教。",
        "yao_ci": ["初六：童观，小人无咎，君子吝。", "六二：窥观，利女贞。", "六三：观我生，进退。", "六四：观国之光，利用宾于王。", "九五：观我生，君子无咎。", "上九：观其生，君子无咎。"],
    },
    "火雷噬嗑": {
        "num": 21, "binary": "101001", "keywords": ["咬合", "决断", "刑罚"], "element": "火/雷",
        "gua_ci": "亨。利用狱。", "xiang_ci": "雷电噬嗑。先王以明罚敕法。",
        "yao_ci": ["初九：屦校灭趾，无咎。", "六二：噬肤灭鼻，无咎。", "六三：噬腊肉，遇毒，小吝，无咎。", "九四：噬干胏，得金矢，利艰贞，吉。", "六五：噬干肉，得黄金，贞厉，无咎。", "上九：何校灭耳，凶。"],
    },
    "山火贲": {
        "num": 22, "binary": "100101", "keywords": ["装饰", "文明", "节制"], "element": "山/火",
        "gua_ci": "亨。小利有所往。", "xiang_ci": "山下有火，贲。君子以明庶政，无敢折狱。",
        "yao_ci": ["初九：贲其趾，舍车而徒。", "六二：贲其须。", "九三：贲如濡如，永贞吉。", "六四：贲如皤如，白马翰如，匪寇婚媾。", "六五：贲于丘园，束帛戋戋，吝，终吉。", "上九：白贲，无咎。"],
    },
    "山地剥": {
        "num": 23, "binary": "100000", "keywords": ["剥落", "衰退", "根基"], "element": "山/地",
        "gua_ci": "不利有攸往。", "xiang_ci": "山附于地，剥。上以厚下，安宅。",
        "yao_ci": ["初六：剥床以足，蔑贞凶。", "六二：剥床以辨，蔑贞凶。", "六三：剥之，无咎。", "六四：剥床以肤，凶。", "六五：贯鱼，以宫人宠，无不利。", "上九：硕果不食，君子得舆，小人剥庐。"],
    },
    "地雷复": {
        "num": 24, "binary": "000001", "keywords": ["复兴", "回归", "循环"], "element": "地/雷",
        "gua_ci": "亨。出入无疾，朋来无咎。", "xiang_ci": "雷在地中，复。先王以至日闭关，商旅不行，后不省方。",
        "yao_ci": ["初九：不远复，无祗悔，元吉。", "六二：休复，吉。", "六三：频复，厉无咎。", "六四：中行独复。", "六五：敦复，无悔。", "上六：迷复，凶，有灾眚。用行师，终有大败，以其国君，凶。至于十年，不克征。"],
    },
    "天雷无妄": {
        "num": 25, "binary": "111001", "keywords": ["无妄", "真实", "自然"], "element": "天/雷",
        "gua_ci": "元亨利贞。其匪正有眚，不利有攸往。", "xiang_ci": "天下雷行，物与无妄。先王以茂对时，育万物。",
        "yao_ci": ["初九：无妄，往吉。", "六二：不耕获，不菑畲，则利有攸往。", "六三：无妄之灾，或系之牛，行人之得，邑人之灾。", "九四：可贞，无咎。", "九五：无妄之疾，勿药有喜。", "上九：无妄，行有眚，无攸利。"],
    },
    "山天大畜": {
        "num": 26, "binary": "100111", "keywords": ["蓄积", "止健", "蓄德"], "element": "山/天",
        "gua_ci": "利贞，不家食吉，利涉大川。", "xiang_ci": "天在山中，大畜。君子以多识前言往行，以畜其德。",
        "yao_ci": ["初九：有厉利已。", "九二：舆说輹。", "九三：良马逐，利艰贞。曰闲舆卫，利有攸往。", "六四：童牛之牿，元吉。", "六五：豮豕之牙，吉。", "上九：何天之衢，亨。"],
    },
    "山雷颐": {
        "num": 27, "binary": "100001", "keywords": ["颐养", "自养", "养生"], "element": "山/雷",
        "gua_ci": "贞吉。观颐，自求口实。", "xiang_ci": "山下有雷，颐。君子以慎言语，节饮食。",
        "yao_ci": ["初九：舍尔灵龟，观我朵颐，凶。", "六二：颠颐，拂经，于丘颐，征凶。", "六三：拂颐，贞凶，十年勿用，无攸利。", "六四：颠颐吉，虎视眈眈，其欲逐逐，无咎。", "六五：拂经，居贞吉，不可涉大川。", "上九：由颐，厉吉，利涉大川。"],
    },
    "泽风大过": {
        "num": 28, "binary": "011110", "keywords": ["大过", "非常", "冒险"], "element": "泽/风",
        "gua_ci": "栋桡，利有攸往，亨。", "xiang_ci": "泽灭木，大过。君子以独立不惧，遁世无闷。",
        "yao_ci": ["初六：藉用白茅，无咎。", "九二：枯杨生稊，老夫得其女妻，无不利。", "九三：栋桡，凶。", "九四：栋隆，吉，有它吝。", "九五：枯杨生华，老妇得其士夫，无咎无誉。", "上六：过涉灭顶，凶，无咎。"],
    },
    "坎为水": {
        "num": 29, "binary": "010010", "keywords": ["险陷", "重重", "智慧"], "element": "水",
        "gua_ci": "习坎，有孚，维心亨，行有尚。", "xiang_ci": "水洊至，习坎。君子以常德行，习教事。",
        "yao_ci": ["初六：习坎，入于坎窞，凶。", "九二：坎有险，求小得。", "六三：来之坎坎，险且枕，入于坎窞，勿用。", "六四：樽酒簋贰，用缶，纳约自牖，终无咎。", "九五：坎不盈，祗既平，无咎。", "上六：系用徽纆，寘于丛棘，三岁不得，凶。"],
    },
    "离为火": {
        "num": 30, "binary": "101101", "keywords": ["附着", "光明", "文明"], "element": "火",
        "gua_ci": "利贞，亨。畜牝牛，吉。", "xiang_ci": "明两作，离。大人以继明照于四方。",
        "yao_ci": ["初九：履错然，敬之，无咎。", "六二：黄离，元吉。", "九三：日昃之离，不鼓缶而歌，则大耋之嗟，凶。", "九四：突如其来如，焚如，死如，弃如。", "六五：出涕沱若，戚嗟若，吉。", "上九：王用出征，有嘉折首，获匪其丑，无咎。"],
    },
    "泽山咸": {
        "num": 31, "binary": "011100", "keywords": ["感应", "交感", "情感"], "element": "泽/山",
        "gua_ci": "亨，利贞，取女吉。", "xiang_ci": "山上有泽，咸。君子以虚受人。",
        "yao_ci": ["初六：咸其拇。", "六二：咸其腓，凶，居吉。", "九三：咸其股，执其随，往吝。", "九四：贞吉悔亡，憧憧往来，朋从尔思。", "九五：咸其脢，无悔。", "上六：咸其辅，颊，舌。"],
    },
    "雷风恒": {
        "num": 32, "binary": "001110", "keywords": ["恒久", "稳定", "坚持"], "element": "雷/风",
        "gua_ci": "亨，无咎，利贞，利有攸往。", "xiang_ci": "雷风，恒。君子以立不易方。",
        "yao_ci": ["初六：浚恒，贞凶，无攸利。", "九二：悔亡。", "九三：不恒其德，或承之羞，贞吝。", "九四：田无禽。", "六五：恒其德，贞，妇人吉，夫子凶。", "上六：振恒，凶。"],
    },
    "天山遁": {
        "num": 33, "binary": "111100", "keywords": ["退避", "隐退", "明哲"], "element": "天/山",
        "gua_ci": "亨，小利贞。", "xiang_ci": "天下有山，遁。君子以远小人，不恶而严。",
        "yao_ci": ["初六：遁尾，厉，勿用有攸往。", "六二：执之用黄牛之革，莫之胜说。", "九三：系遁，有疾厉，畜臣妾吉。", "九四：好遁，君子吉，小人否。", "九五：嘉遁，贞吉。", "上九：肥遁，无不利。"],
    },
    "雷天大壮": {
        "num": 34, "binary": "001111", "keywords": ["壮盛", "强盛", "进退"], "element": "雷/天",
        "gua_ci": "利贞。", "xiang_ci": "雷在天上，大壮。君子以非礼弗履。",
        "yao_ci": ["初九：壮于趾，征凶，有孚。", "九二：贞吉。", "九三：小人用壮，君子用罔，贞厉。羝羊触藩，羸其角。", "九四：贞吉悔亡，藩决不羸，壮于大舆之輹。", "六五：丧羊于易，无悔。", "上六：羝羊触藩，不能退，不能遂，无攸利，艰则吉。"],
    },
    "火地晋": {
        "num": 35, "binary": "101000", "keywords": ["晋升", "进步", "光明"], "element": "火/地",
        "gua_ci": "康侯用锡马蕃庶，昼日三接。", "xiang_ci": "明出地上，晋。君子以自昭明德。",
        "yao_ci": ["初六：晋如，摧如，贞吉。罔孚，裕无咎。", "六二：晋如，愁如，贞吉。受兹介福，于其王母。", "六三：众允，悔亡。", "九四：晋如鼫鼠，贞厉。", "六五：悔亡，失得勿恤，往吉无不利。", "上九：晋其角，维用伐邑，厉吉无咎，贞吝。"],
    },
    "地火明夷": {
        "num": 36, "binary": "000101", "keywords": ["受伤", "晦暗", "韬光"], "element": "地/火",
        "gua_ci": "利艰贞。", "xiang_ci": "明入地中，明夷。君子以莅众，用晦而明。",
        "yao_ci": ["初九：明夷于飞，垂其翼。君子于行，三日不食，有攸往，主人有言。", "六二：明夷，夷于左股，用拯马壮，吉。", "九三：明夷于南狩，得其大首，不可疾贞。", "六四：入于左腹，获明夷之心，出于门庭。", "六五：箕子之明夷，利贞。", "上六：不明晦，初登于天，后入于地。"],
    },
    "风火家人": {
        "num": 37, "binary": "110101", "keywords": ["家庭", "伦理", "内政"], "element": "风/火",
        "gua_ci": "利女贞。", "xiang_ci": "风自火出，家人。君子以言有物，而行有恒。",
        "yao_ci": ["初九：闲有家，悔亡。", "六二：无攸遂，在中馈。", "九三：家人嗃嗃，悔厉吉；妇子嘻嘻，终吝。", "六四：富家，大吉。", "九五：王假有家，勿恤吉。", "上九：有孚威如，终吉。"],
    },
    "火泽睽": {
        "num": 38, "binary": "101011", "keywords": ["乖违", "背离", "求同"], "element": "火/泽",
        "gua_ci": "小事吉。", "xiang_ci": "上火下泽，睽。君子以同而异。",
        "yao_ci": ["初九：悔亡，丧马勿逐，自复；见恶人无咎。", "九二：遇主于巷，无咎。", "六三：见舆曳，其牛掣，其人天且劓，无初有终。", "九四：睽孤，遇元夫，交孚，厉无咎。", "六五：悔亡，厥宗噬肤，往何咎。", "上九：睽孤，见豕负涂，载鬼一车，先张之弧，后说之弧，匪寇婚媾，往遇雨则吉。"],
    },
    "水山蹇": {
        "num": 39, "binary": "010100", "keywords": ["蹇难", "险阻", "反思"], "element": "水/山",
        "gua_ci": "利西南，不利东北；利见大人，贞吉。", "xiang_ci": "山上有水，蹇。君子以反身修德。",
        "yao_ci": ["初六：往蹇，来誉。", "六二：王臣蹇蹇，匪躬之故。", "九三：往蹇来反。", "六四：往蹇来连。", "九五：大蹇朋来。", "上六：往蹇来硕，吉；利见大人。"],
    },
    "雷水解": {
        "num": 40, "binary": "001010", "keywords": ["解脱", "释放", "雨过天晴"], "element": "雷/水",
        "gua_ci": "利西南，无所往，其来复吉。有攸往，夙吉。", "xiang_ci": "雷雨作，解。君子以赦过宥罪。",
        "yao_ci": ["初六：无咎。", "九二：田获三狐，得黄矢，贞吉。", "六三：负且乘，致寇至，贞吝。", "九四：解而拇，朋至斯孚。", "六五：君子维有解，吉；有孚于小人。", "上六：公用射隼，于高墉之上，获之，无不利。"],
    },
    "山泽损": {
        "num": 41, "binary": "100011", "keywords": ["减损", "牺牲", "增益"], "element": "山/泽",
        "gua_ci": "有孚，元吉，无咎，可贞，利有攸往。", "xiang_ci": "山下有泽，损。君子以惩忿窒欲。",
        "yao_ci": ["初九：已事遄往，无咎，酌损之。", "九二：利贞，征凶，弗损益之。", "六三：三人行，则损一人；一人行，则得其友。", "六四：损其疾，使遄有喜，无咎。", "六五：或益之，十朋之龟弗克违，元吉。", "上九：弗损益之，无咎，贞吉，利有攸往，得臣无家。"],
    },
    "风雷益": {
        "num": 42, "binary": "110001", "keywords": ["增益", "受益", "施惠"], "element": "风/雷",
        "gua_ci": "利有攸往，利涉大川。", "xiang_ci": "风雷，益。君子以见善则迁，有过则改。",
        "yao_ci": ["初九：利用为大作，元吉，无咎。", "六二：或益之，十朋之龟弗克违，永贞吉。王用享于帝，吉。", "六三：益之用凶事，无咎。有孚中行，告公用圭。", "六四：中行，告公从。利用为依迁国。", "九五：有孚惠心，勿问元吉。有孚惠我德。", "上九：莫益之，或击之，立心勿恒，凶。"],
    },
    "泽天夬": {
        "num": 43, "binary": "011111", "keywords": ["决断", "果决", "清除"], "element": "泽/天",
        "gua_ci": "扬于王庭，孚号，有厉，告自邑，不利即戎，利有攸往。", "xiang_ci": "泽上于天，夬。君子以施禄及下，居德则忌。",
        "yao_ci": ["初九：壮于前趾，往不胜为吝。", "九二：惕号，莫夜有戎，勿恤。", "九三：壮于頄，有凶。君子夬夬，独行遇雨，若濡有愠，无咎。", "九四：臀无肤，其行次且。牵羊悔亡，闻言不信。", "九五：苋陆夬夬，中行无咎。", "上六：无号，终有凶。"],
    },
    "天风姤": {
        "num": 44, "binary": "111110", "keywords": ["相遇", "邂逅", "防范"], "element": "天/风",
        "gua_ci": "女壮，勿用取女。", "xiang_ci": "天下有风，姤。后以施命诰四方。",
        "yao_ci": ["初六：系于金柅，贞吉，有攸往，见凶，羸豕孚蹢躅。", "九二：包有鱼，无咎，不利宾。", "九三：臀无肤，其行次且，厉，无大咎。", "九四：包无鱼，起凶。", "九五：以杞包瓜，含章，有陨自天。", "上九：姤其角，吝，无咎。"],
    },
    "泽地萃": {
        "num": 45, "binary": "011000", "keywords": ["聚集", "荟萃", "团结"], "element": "泽/地",
        "gua_ci": "亨。王假有庙，利见大人，亨，利贞。", "xiang_ci": "泽上于地，萃。君子以除戎器，戒不虞。",
        "yao_ci": ["初六：有孚不终，乃乱乃萃，若号一握为笑，勿恤，往无咎。", "六二：引吉，无咎，孚乃利用禴。", "六三：萃如，嗟如，无攸利，往无咎，小吝。", "九四：大吉，无咎。", "九五：萃有位，无咎，匪孚，元永贞，悔亡。", "上六：齎咨涕洟，无咎。"],
    },
    "地风升": {
        "num": 46, "binary": "000110", "keywords": ["上升", "成长", "积小"], "element": "地/风",
        "gua_ci": "元亨，用见大人，勿恤，南征吉。", "xiang_ci": "地中生木，升。君子以顺德，积小以高大。",
        "yao_ci": ["初六：允升，大吉。", "九二：孚乃利用禴，无咎。", "九三：升虚邑。", "六四：王用亨于岐山，吉无咎。", "六五：贞吉，升阶。", "上六：冥升，利于不息之贞。"],
    },
    "泽水困": {
        "num": 47, "binary": "011010", "keywords": ["困穷", "困境", "守正"], "element": "泽/水",
        "gua_ci": "亨，贞，大人吉，无咎，有言不信。", "xiang_ci": "泽无水，困。君子以致命遂志。",
        "yao_ci": ["初六：臀困于株木，入于幽谷，三岁不觌。", "九二：困于酒食，朱绂方来，利用享祀，征凶，无咎。", "六三：困于石，据于蒺藜，入于其宫，不见其妻，凶。", "九四：来徐徐，困于金车，吝，有终。", "九五：劓刖，困于赤绂，乃徐有说，利用祭祀。", "上六：困于葛藟，于臲卼，曰动悔。有悔，征吉。"],
    },
    "水风井": {
        "num": 48, "binary": "010110", "keywords": ["井养", "源泉", "无穷"], "element": "水/风",
        "gua_ci": "改邑不改井，无丧无得，往来井井。", "xiang_ci": "木上有水，井。君子以劳民劝相。",
        "yao_ci": ["初六：井泥不食，旧井无禽。", "九二：井谷射鲋，瓮敝漏。", "九三：井渫不食，为我心恻，可用汲，王明，并受其福。", "六四：井甃，无咎。", "九五：井冽，寒泉食。", "上六：井收勿幕，有孚元吉。"],
    },
    "泽火革": {
        "num": 49, "binary": "011101", "keywords": ["变革", "革命", "去旧"], "element": "泽/火",
        "gua_ci": "已日乃孚，元亨利贞，悔亡。", "xiang_ci": "泽中有火，革。君子以治历明时。",
        "yao_ci": ["初九：巩用黄牛之革。", "六二：已日乃革之，征吉，无咎。", "九三：征凶，贞厉，革言三就，有孚。", "九四：悔亡，有孚改命，吉。", "九五：大人虎变，未占有孚。", "上六：君子豹变，小人革面，征凶，居贞吉。"],
    },
    "火风鼎": {
        "num": 50, "binary": "101110", "keywords": ["鼎新", "稳定", "中正"], "element": "火/风",
        "gua_ci": "元吉，亨。", "xiang_ci": "木上有火，鼎。君子以正位凝命。",
        "yao_ci": ["初六：鼎颠趾，利出否，得妾以其子，无咎。", "九二：鼎有实，我仇有疾，不我能即，吉。", "九三：鼎耳革，其行塞，雉膏不食，方雨亏悔，终吉。", "九四：鼎折足，覆公餗，其形渥，凶。", "六五：鼎黄耳金铉，利贞。", "上九：鼎玉铉，大吉，无不利。"],
    },
    "震为雷": {
        "num": 51, "binary": "001001", "keywords": ["震动", "惊恐", "奋起"], "element": "雷",
        "gua_ci": "亨。震来虩虩，笑言哑哑。", "xiang_ci": "洊雷，震。君子以恐惧修省。",
        "yao_ci": ["初九：震来虩虩，后笑言哑哑，吉。", "六二：震来厉，亿丧贝，跻于九陵，勿逐，七日得。", "六三：震苏苏，震行无眚。", "九四：震遂泥。", "六五：震往来厉，亿无丧，有事。", "上六：震索索，视矍矍，征凶。震不于其躬，于其邻，无咎。婚媾有言。"],
    },
    "艮为山": {
        "num": 52, "binary": "100100", "keywords": ["静止", "止步", "稳重"], "element": "山",
        "gua_ci": "艮其背，不获其身，行其庭，不见其人，无咎。", "xiang_ci": "兼山，艮。君子以思不出其位。",
        "yao_ci": ["初六：艮其趾，无咎，利永贞。", "六二：艮其腓，不拯其随，其心不快。", "九三：艮其限，列其夤，厉薰心。", "六四：艮其身，无咎。", "六五：艮其辅，言有序，悔亡。", "上九：敦艮，吉。"],
    },
    "风山渐": {
        "num": 53, "binary": "110100", "keywords": ["渐进", "循序", "台阶"], "element": "风/山",
        "gua_ci": "女归吉，利贞。", "xiang_ci": "山上有木，渐。君子以居贤德，善俗。",
        "yao_ci": ["初六：鸿渐于干，小子厉，有言，无咎。", "六二：鸿渐于磐，饮食衎衎，吉。", "九三：鸿渐于陆，夫征不复，妇孕不育，凶；利御寇。", "六四：鸿渐于木，或得其桷，无咎。", "九五：鸿渐于陵，妇三岁不孕，终莫之胜，吉。", "上九：鸿渐于逵，其羽可用为仪，吉。"],
    },
    "雷泽归妹": {
        "num": 54, "binary": "001011", "keywords": ["婚嫁", "归宿", "少女"], "element": "雷/泽",
        "gua_ci": "征凶，无攸利。", "xiang_ci": "泽上有雷，归妹。君子以永终知敝。",
        "yao_ci": ["初九：归妹以娣，跛能履，征吉。", "九二：眇能视，利幽人之贞。", "六三：归妹以须，反归以娣。", "九四：归妹愆期，迟归有时。", "六五：帝乙归妹，其君之袂，不如其娣之袂良，月几望，吉。", "上六：女承筐无实，士刲羊无血，无攸利。"],
    },
    "雷火丰": {
        "num": 55, "binary": "001101", "keywords": ["丰盛", "盛大", "日中"], "element": "雷/火",
        "gua_ci": "亨，王假之，勿忧，宜日中。", "xiang_ci": "雷电皆至，丰。君子以折狱致刑。",
        "yao_ci": ["初九：遇其配主，虽旬无咎，往有尚。", "六二：丰其蔀，日中见斗，往得疑疾，有孚发若，吉。", "九三：丰其沛，日中见沬，折其右肱，无咎。", "九四：丰其蔀，日中见斗，遇其夷主，吉。", "六五：来章，有庆誉，吉。", "上六：丰其屋，蔀其家，窥其户，阒其无人，三岁不觌，凶。"],
    },
    "火山旅": {
        "num": 56, "binary": "101100", "keywords": ["旅行", "漂泊", "不安"], "element": "火/山",
        "gua_ci": "小亨，旅贞吉。", "xiang_ci": "山上有火，旅。君子以明慎用刑，而不留狱。",
        "yao_ci": ["初六：旅琐琐，斯其所取灾。", "六二：旅即次，怀其资，得童仆贞。", "九三：旅焚其次，丧其童仆，贞厉。", "九四：旅于处，得其资斧，我心不快。", "六五：射雉一矢亡，终以誉命。", "上九：鸟焚其巢，旅人先笑后号咷。丧牛于易，凶。"],
    },
    "巽为风": {
        "num": 57, "binary": "110110", "keywords": ["顺从", "进入", "谦逊"], "element": "风",
        "gua_ci": "小亨，利有攸往，利见大人。", "xiang_ci": "随风，巽。君子以申命行事。",
        "yao_ci": ["初六：进退，利武人之贞。", "九二：巽在床下，用史巫纷若，吉无咎。", "九三：频巽，吝。", "六四：悔亡，田获三品。", "九五：贞吉悔亡，无不利。无初有终，先庚三日，后庚三日，吉。", "上九：巽在床下，丧其资斧，贞凶。"],
    },
    "兑为泽": {
        "num": 58, "binary": "011011", "keywords": ["喜悦", "口舌", "和悦"], "element": "泽",
        "gua_ci": "亨，利贞。", "xiang_ci": "丽泽，兑。君子以朋友讲习。",
        "yao_ci": ["初九：和兑，吉。", "九二：孚兑，吉，悔亡。", "六三：来兑，凶。", "九四：商兑，未宁，介疾有喜。", "九五：孚于剥，有厉。", "上六：引兑。"],
    },
    "风水涣": {
        "num": 59, "binary": "110010", "keywords": ["涣散", "离散", "凝聚"], "element": "风/水",
        "gua_ci": "亨。王假有庙，利涉大川，利贞。", "xiang_ci": "风行水上，涣。先王以享于帝立庙。",
        "yao_ci": ["初六：用拯马壮，吉。", "九二：涣奔其机，悔亡。", "六三：涣其躬，无悔。", "六四：涣其群，元吉；涣有丘，匪夷所思。", "九五：涣汗其大号，涣王居，无咎。", "上九：涣其血，去逖出，无咎。"],
    },
    "水泽节": {
        "num": 60, "binary": "010011", "keywords": ["节制", "节约", "制度"], "element": "水/泽",
        "gua_ci": "亨。苦节不可贞。", "xiang_ci": "泽上有水，节。君子以制数度，议德行。",
        "yao_ci": ["初九：不出户庭，无咎。", "九二：不出门庭，凶。", "六三：不节若，则嗟若，无咎。", "六四：安节，亨。", "九五：甘节，吉；往有尚。", "上六：苦节，贞凶，悔亡。"],
    },
    "风泽中孚": {
        "num": 61, "binary": "110011", "keywords": ["诚信", "孚信", "内心"], "element": "风/泽",
        "gua_ci": "豚鱼吉，利涉大川，利贞。", "xiang_ci": "泽上有风，中孚。君子以议狱缓死。",
        "yao_ci": ["初九：虞吉，有他不燕。", "九二：鸣鹤在阴，其子和之，我有好爵，吾与尔靡之。", "六三：得敌，或鼓或罢，或泣或歌。", "六四：月几望，马匹亡，无咎。", "九五：有孚挛如，无咎。", "上九：翰音登于天，贞凶。"],
    },
    "雷山小过": {
        "num": 62, "binary": "001100", "keywords": ["小过", "矫枉", "小事"], "element": "雷/山",
        "gua_ci": "亨，利贞，可小事，不可大事。", "xiang_ci": "山上有雷，小过。君子以行过乎恭，丧过乎哀，用过乎俭。",
        "yao_ci": ["初六：飞鸟以凶。", "六二：过其祖，遇其妣；不及其君，遇其臣；无咎。", "九三：弗过防之，从或戕之，凶。", "九四：无咎，弗过遇之。往厉必戒，勿用永贞。", "六五：密云不雨，自我西郊，公弋取彼在穴。", "上六：弗遇过之，飞鸟离之，凶，是谓灾眚。"],
    },
    "水火既济": {
        "num": 63, "binary": "010101", "keywords": ["成功", "完成", "初吉"], "element": "水/火",
        "gua_ci": "亨，小利贞，初吉终乱。", "xiang_ci": "水在火上，既济。君子以思患而豫防之。",
        "yao_ci": ["初九：曳其轮，濡其尾，无咎。", "六二：妇丧其茀，勿逐，七日得。", "九三：高宗伐鬼方，三年克之，小人勿用。", "六四：繻有衣袽，终日戒。", "九五：东邻杀牛，不如西邻之禴祭，实受其福。", "上六：濡其首，厉。"],
    },
    "火水未济": {
        "num": 64, "binary": "101010", "keywords": ["未竟", "初始", "无限"], "element": "火/水",
        "gua_ci": "亨，小狐汔济，濡其尾，无攸利。", "xiang_ci": "火在水上，未济。君子以慎辨物居方。",
        "yao_ci": ["初六：濡其尾，吝。", "九二：曳其轮，贞吉。", "六三：未济，征凶，利涉大川。", "九四：贞吉，悔亡，震用伐鬼方，三年有赏于大国。", "六五：贞吉，无悔，君子之光，有孚，吉。", "上九：有孚于饮酒，无咎，濡其首，有孚失是。"],
    },
}


# ==========================================
# 4. 数据：22张大阿卡纳
# ==========================================
TAROT_MAJOR = [
    {"name": "愚人", "element": "风", "keywords": ["开始", "冒险", "天真"]},
    {"name": "魔术师", "element": "风", "keywords": ["创造", "意志", "显化"]},
    {"name": "女祭司", "element": "水", "keywords": ["直觉", "神秘", "内在"]},
    {"name": "女皇", "element": "土", "keywords": ["丰饶", "母性", "自然"]},
    {"name": "皇帝", "element": "火", "keywords": ["权威", "结构", "父性"]},
    {"name": "教皇", "element": "土", "keywords": ["传统", "信仰", "指导"]},
    {"name": "恋人", "element": "风", "keywords": ["选择", "爱情", "结合"]},
    {"name": "战车", "element": "水", "keywords": ["胜利", "意志", "控制"]},
    {"name": "力量", "element": "火", "keywords": ["勇气", "耐心", "内在力量"]},
    {"name": "隐士", "element": "土", "keywords": ["内省", "孤独", "智慧"]},
    {"name": "命运之轮", "element": "火", "keywords": ["转变", "周期", "命运"]},
    {"name": "正义", "element": "风", "keywords": ["公正", "因果", "平衡"]},
    {"name": "倒吊人", "element": "水", "keywords": ["牺牲", "暂停", "视角转换"]},
    {"name": "死神", "element": "水", "keywords": ["结束", "转化", "新生"]},
    {"name": "节制", "element": "火", "keywords": ["平衡", "调和", "中庸"]},
    {"name": "恶魔", "element": "土", "keywords": ["束缚", "欲望", "物质"]},
    {"name": "高塔", "element": "火", "keywords": ["突变", "觉醒", "崩塌"]},
    {"name": "星星", "element": "风", "keywords": ["希望", "灵感", "宁静"]},
    {"name": "月亮", "element": "水", "keywords": ["幻觉", "恐惧", "潜意识"]},
    {"name": "太阳", "element": "火", "keywords": ["喜悦", "成功", "活力"]},
    {"name": "审判", "element": "火", "keywords": ["重生", "觉醒", "召唤"]},
    {"name": "世界", "element": "土", "keywords": ["完成", "圆满", "整合"]},
]
for _i, _card in enumerate(TAROT_MAJOR):
    _card["image"] = f"https://sacred-texts.com/tarot/pkt/img/ar{_i:02d}.jpg"

SINGLE_TAGS = ["offer等待焦虑", "人生方向抉择Yes or No", "好无聊我做点什么呢"]
DUAL_TAGS = ["最近总冷战吵架", "异地恋方向迷茫", "关系卡在暧昧期"]
SHARE_LINK = os.getenv("APP_SHARE_URL", "https://vibecoding-7ky3op9bomtwgk9szuqeyf.streamlit.app/")

# ==========================================
# 5. 状态初始化
# ==========================================
for key, default in [
    ("flow_step", 1),
    ("flow_choice", None),
    ("iching_ritual", None),
    ("tarot_ritual", None),
]:
    if key not in st.session_state:
        st.session_state[key] = default

# ==========================================
# 6. 金钱卦引擎
# ==========================================
def cast_yao():
    """摇一爻：3枚铜钱"""
    coins = ["字" if secrets.randbelow(2) == 0 else "背" for _ in range(3)]
    zi = coins.count("字")
    if zi == 3:
        return {"symbol": "⚋", "yin_yang": "阴", "change": True, "coins": coins, "name": "老阴"}
    elif zi == 0:
        return {"symbol": "⚊", "yin_yang": "阳", "change": True, "coins": coins, "name": "老阳"}
    elif zi == 2:
        return {"symbol": "⚋", "yin_yang": "阴", "change": False, "coins": coins, "name": "少阴"}
    else:
        return {"symbol": "⚊", "yin_yang": "阳", "change": False, "coins": coins, "name": "少阳"}


def build_hex_result_from_yaos(yaos: list) -> dict:
    """由已摇出的六爻计算本卦、之卦及变爻信息"""
    ben_binary = "".join(["1" if y["yin_yang"] == "阳" else "0" for y in reversed(yaos)])
    zhi_binary = "".join([
        "1" if ((y["yin_yang"] == "阴" and y["change"]) or (y["yin_yang"] == "阳" and not y["change"])) else "0"
        for y in reversed(yaos)
    ])

    ben_gua = None
    zhi_gua = None
    for name, data in ICHING_HEXAGRAMS.items():
        if data.get("binary") == ben_binary:
            ben_gua = name
        if data.get("binary") == zhi_binary:
            zhi_gua = name

    has_change = any(y["change"] for y in yaos)
    change_positions = [y["position"] for y in yaos if y["change"]]

    change_yao_texts = []
    if has_change and ben_gua:
        hex_data = ICHING_HEXAGRAMS[ben_gua]
        for pos in change_positions:
            change_yao_texts.append({
                "position": pos,
                "text": hex_data["yao_ci"][pos - 1],
                "from": "阴" if yaos[pos - 1]["yin_yang"] == "阴" else "阳",
                "to": "阳" if yaos[pos - 1]["yin_yang"] == "阴" else "阴",
            })

    return {
        "ben_gua": ben_gua,
        "zhi_gua": zhi_gua if has_change else None,
        "ben_binary": ben_binary,
        "zhi_binary": zhi_binary,
        "yaos": yaos,
        "has_change": has_change,
        "change_count": len(change_positions),
        "change_yao_texts": change_yao_texts,
    }


def format_coins_line(coins: list[str]) -> str:
    return " ".join(f"🪙{c}" for c in coins)


def render_yao_stack(yaos: list, title: str = "已摇出的爻（自下而上）") -> None:
    if not yaos:
        return
    st.markdown(f"<div style='text-align:center;margin:12px 0;'><b>{title}</b></div>", unsafe_allow_html=True)
    yao_lines = []
    for y in reversed(yaos):
        change_class = "yao-change" if y["change"] else ""
        yao_lines.append(
            f'<div class="yao-line"><span class="{change_class}">{y["symbol"]}</span> '
            f'<span style="font-size:12px;color:#666;">第{y["position"]}爻 · {y["name"]}</span></div>'
        )
    st.markdown(
        f'<div class="guayupai-yao-display yao-display">{"".join(yao_lines)}</div>',
        unsafe_allow_html=True,
    )


def _normalize_flow_step(step) -> int | None:
    if step is None:
        return None
    if step == "crystal":
        return 1
    if step == "doors":
        return 2
    if str(step) == "3":
        return 3
    try:
        n = int(step)
        if n in (1, 2, 3):
            return n
    except (TypeError, ValueError):
        pass
    return None


def _coerce_flow_step(step) -> int:
    """将 session 中的 flow_step 规范为 1 / 2 / 3。"""
    normalized = _normalize_flow_step(step)
    if normalized is not None:
        return normalized
    return 1


def _is_ritual_flow_step(step) -> bool:
    return step == 3 or str(step) == "3"


def reset_flow_to_step1() -> None:
    st.session_state.flow_step = 1
    st.session_state.flow_choice = None
    st.session_state.flow_tool = None
    prefixes = (
        "iching_ritual_",
        "tarot_ritual_",
        "outcome_",
        "dilemma_text_",
        "dilemma_widget_",
        "ritual_gen_",
        "tarot_busy_",
    )
    for key in list(st.session_state.keys()):
        if key.startswith(prefixes):
            del st.session_state[key]
    _clear_tarot_query_params()
    for key in ("flow_step", "flow_tool", "flow_choice", "dilemma", "ritual_reset", "view", "share"):
        if key in st.query_params:
            del st.query_params[key]


def init_iching_ritual(ritual: dict) -> None:
    ritual.setdefault("iching_step", 0)
    ritual.setdefault("iching_yaos", [])
    ritual.setdefault("hex", None)


def render_hex_summary_card(hex_result: dict) -> None:
    """组卦结果卡：本卦卦辞 + 变爻得卦 + 变爻爻辞（小字紧跟标题）"""
    ben_gua = hex_result["ben_gua"]
    zhi_gua = hex_result.get("zhi_gua")
    ben_data = ICHING_HEXAGRAMS.get(ben_gua, {})

    small = "font-size:12px;line-height:1.7;opacity:0.82;color:#1E4620;margin:0;"
    parts = [
        f'<div class="card-title" style="color:#1E4620;">'
        f"组成本卦：{ben_gua}（第{ben_data.get('num', '?')}卦）</div>",
        f'<p style="{small}margin-top:6px;">{ben_data.get("gua_ci", "")}</p>',
    ]

    if zhi_gua:
        parts.append(
            f'<div class="card-title" style="margin-top:12px;color:#1E4620;">'
            f"变爻得卦：{zhi_gua}</div>"
        )
        for ct in hex_result.get("change_yao_texts", []):
            parts.append(
                f'<p style="{small}margin-top:6px;">'
                f"第{ct['position']}爻（{ct['from']}→{ct['to']}）：{ct['text']}</p>"
            )
    else:
        parts.append(f'<p style="{small}margin-top:10px;">无变爻，事态稳定</p>')

    st.markdown(
        f'<div class="result-card result-east">{"".join(parts)}</div>',
        unsafe_allow_html=True,
    )


# ==========================================
# 7. 塔罗仪式引擎
# ==========================================
def shuffle_tarot(seed: int | None = None):
    """洗牌：22张大阿卡纳；传入 seed 可复现同一副牌序（用于 URL 恢复）。"""
    deck = TAROT_MAJOR.copy()
    if seed is not None:
        rng = random.Random(seed)
        for i in range(len(deck) - 1, 0, -1):
            j = rng.randint(0, i)
            deck[i], deck[j] = deck[j], deck[i]
    else:
        for i in range(len(deck) - 1, 0, -1):
            j = secrets.randbelow(i + 1)
            deck[i], deck[j] = deck[j], deck[i]
    return deck


# ==========================================
# 8. Prompt 工程
# ==========================================
def _parse_iching_trigrams(element: str) -> tuple[str, str]:
    if "/" in element:
        upper, lower = element.split("/", 1)
        return upper.strip(), lower.strip()
    return element, element


def build_prompt(
    name,
    info,
    user_input,
    is_iching=False,
    is_dual=False,
    is_random=False,
):
    """
    统一 Prompt 生成
    name: 卦名/牌名
    info: 字典，包含元素/关键词/上下卦等
    is_iching: True=易经, False=塔罗
    is_dual: True=双人模式
    """
    if is_iching:
        info_block = (
            f"上下卦：{info['upper']} + {info['lower']}\n"
            f"五行：{info['element']}\n"
            f"核心意象：{', '.join(info['keywords'])}"
        )
    else:
        info_block = (
            f"元素：{info['element']}\n"
            f"核心关键词：{', '.join(info['keywords'])}"
        )

    dilemma = (user_input or "").strip()
    if dilemma:
        context = (
            f"【用户心里的纠结】\n「{dilemma}」\n\n"
            "务必紧扣以上问题解读：卦象/牌面要与用户描述对上号，"
            "先回应用户正在纠结什么，再给牌义和建议；不要只泛泛讲牌面含义。"
        )
    else:
        context = (
            "用户没填具体问题，请结合抽到的结果给贴近生活的直觉指引，语气像朋友聊天。"
        )
    if is_dual:
        context += "\n（双人关系视角，多从「你们」的互动感受说起，别站队评判谁对谁错。）"
    if is_random:
        context += "\n（用户选了随便看看；若上方有具体问题，仍要优先回应。）"

    action = "摇出了" if is_iching else "翻到了"
    yi_section = "☯️ 卦象" if is_iching else "🔮 牌面"
    read_section = "📜 卦象" if is_iching else "🌌 塔罗"

    return f"""你是一位温柔的朋友，懂点塔罗和易经，说话不绕弯子，也不用学术词吓人。

用户{action}【{name}】
{info_block}

{context}

【输出要求】
1. 严禁开场白、问候语、自我介绍
2. 严禁"潜意识""阴影""能量""疗愈""接纳""觉察"等词
3. 像朋友聊天，不是论文
4. 温柔询问，不下结论，用"是不是""感觉""可能"
5. 短句，像微信聊天
6. 严禁 Markdown 代码块
7. 每个章节必须严格按指定标题格式

### {yi_section}牌义
（1-2个核心词，50字内，人话解释）

### {read_section}解读
（温柔拆解，像朋友说话："你{action}{name}，感觉你最近..."）

### 💬 试试看
（3条建议，像朋友随口说的，正常人会真的做）

好的例子：
- 先给他发条消息，随便说点啥
- 明天出去走走，不用想正事
- 这周找个时间，把想骂的话写备忘录里

不好的例子：
- 今晚睡前，写下你最近对关系最恐惧的三个场景（像作业）
- 用玩笑口吻向对方说：我们来玩个游戏（像剧本）
- 取一枚硬币，在掌心默念...（像仪式）

检查：如果建议需要解释"预期效果"才能懂，或者你自己都不会真的做，就重写。
"""


def build_iching_prompt(
    hex_result,
    user_input,
    is_empty,
    is_dual=False,
    is_random=False,
):
    ben = hex_result["ben_gua"]
    zhi = hex_result.get("zhi_gua")
    ben_data = ICHING_HEXAGRAMS.get(ben, {})
    upper, lower = _parse_iching_trigrams(ben_data.get("element", ""))

    info = {
        "upper": upper,
        "lower": lower,
        "element": ben_data.get("element", ""),
        "keywords": ben_data.get("keywords", []),
    }
    prompt = build_prompt(
        ben,
        info,
        user_input or "",
        is_iching=True,
        is_dual=is_dual,
        is_random=is_random,
    )

    supplement = (
        f"\n\n【起卦详情】\n"
        f"第{ben_data.get('num')}卦 · 卦辞：{ben_data.get('gua_ci', '')}\n"
        f"象辞：{ben_data.get('xiang_ci', '')}"
    )
    if zhi:
        supplement += f"\n变爻得卦：{zhi}"
    if hex_result.get("change_yao_texts"):
        supplement += "\n变爻爻辞："
        for ct in hex_result["change_yao_texts"]:
            supplement += f"\n第{ct['position']}爻（{ct['from']}→{ct['to']}）：{ct['text']}"
    return prompt + supplement


def build_tarot_prompt(
    card,
    user_input,
    is_empty,
    is_dual=False,
    is_random=False,
):
    info = {
        "upper": "",
        "lower": "",
        "element": card.get("element", "未知"),
        "keywords": card.get("keywords", []),
    }
    return build_prompt(
        card["name"],
        info,
        user_input or "",
        is_iching=False,
        is_dual=is_dual,
        is_random=is_random,
    )


# ==========================================
# 9. 工具函数
# ==========================================
TAROT_ROMAN = [
    "0", "I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X",
    "XI", "XII", "XIII", "XIV", "XV", "XVI", "XVII", "XVIII", "XIX", "XX", "XXI",
]


def extract_section_text(full_text: str, heading: str) -> str:
    pattern = rf"{re.escape(heading)}\s*\n(.*?)(?=\n### |\Z)"
    m = re.search(pattern, full_text, re.DOTALL)
    return m.group(1).strip() if m else ""


_SHARE_TAIL_RE = re.compile(
    r"\n###\s*(?:"
    r"💎\s*今日金句|📝\s*日记体分享|✨\s*金句体分享|📕\s*小红书分享|"
    r"💎|📝|✨|📕"
    r")[^\n]*\Z"
)


_SHARE_SECTION_MARKERS = (
    r"###\s*💎\s*今日金句",
    r"###\s*📝\s*日记体分享",
    r"###\s*✨\s*金句体分享",
    r"###\s*📕\s*小红书分享",
    r"💎\s*今日金句",
    r"📝\s*日记体分享",
    r"✨\s*金句体分享",
    r"📕\s*小红书分享",
)


def format_reading_display_text(full_text: str, *, streaming: bool = False) -> str:
    """解读正文：彻底去掉金句/分享体段落（仅分享截图卡展示金句）。"""
    text = full_text
    cut_at = len(text)
    for marker in _SHARE_SECTION_MARKERS:
        m = re.search(marker, text)
        if m and m.start() < cut_at:
            cut_at = m.start()
    text = text[:cut_at].rstrip()
    for heading in (
        "### 💎 今日金句",
        "### 📝 日记体分享",
        "### ✨ 金句体分享",
        "### 📕 小红书分享",
        "💎 今日金句",
        "📝 日记体分享",
        "✨ 金句体分享",
        "📕 小红书分享",
    ):
        pattern = rf"(?:\n|^)\s*{re.escape(heading)}\s*\n.*?(?=\n### |\n💎 |\n📝 |\n✨ |\n📕 |\Z)"
        text = re.sub(pattern, "", text, flags=re.DOTALL)
    if streaming:
        text = _SHARE_TAIL_RE.sub("", text).rstrip()
    return text.rstrip()


def get_outcome_display_text(outcome: dict) -> str:
    stored = outcome.get("display_text")
    if stored:
        return stored
    return format_reading_display_text(outcome.get("full_text", ""))


def _reading_section_titles(tool: str) -> list[str]:
    if tool == "iching":
        return ["☯️ 卦象牌义", "📜 卦象解读", "💬 试试看"]
    return ["🔮 牌面牌义", "🌌 塔罗解读", "💬 试试看"]


def _reading_panel_skeleton_inner(tool: str) -> str:
    shimmer_bg = (
        "rgba(46,125,50,0.08)"
        if tool == "iching"
        else "rgba(106,27,154,0.1)"
    )
    parts = []
    for title in _reading_section_titles(tool):
        parts.append(
            '<div style="margin-bottom:16px;min-height:52px;">'
            f'<div style="font-weight:700;font-size:16px;margin-bottom:8px;">'
            f"{html.escape(title)}</div>"
            f'<div style="height:36px;border-radius:6px;background:linear-gradient(90deg,'
            f"transparent 25%,{shimmer_bg} 50%,transparent 75%);"
            'background-size:200% 100%;animation:readingShimmer 1.8s ease infinite;"></div>'
            "</div>"
        )
    return "".join(parts)


def build_reading_panel_html(inner_html: str, tool: str) -> str:
    return (
        f'<div class="guayupai-reading-panel guayupai-reading-panel--{tool}">'
        f"{inner_html}</div>"
    )


def _reading_panel_skeleton_html(tool: str) -> str:
    return build_reading_panel_html(_reading_panel_skeleton_inner(tool), tool)


def _format_reading_sections_html(text: str) -> str:
    """将 ### 牌面牌义 / 解读 / 试试看 转为带标题的正文 HTML。"""
    blocks = re.split(r"\n(?=### )", (text or "").strip())
    sections: list[str] = []
    for block in blocks:
        block = block.strip()
        if not block:
            continue
        m = re.match(r"^###\s*(.+?)\s*\n(.*)", block, re.DOTALL)
        if m:
            title, body = m.group(1).strip(), m.group(2).strip()
        else:
            title, body = "", block
        body_html = html.escape(body).replace("\n", "<br>")
        sections.append(
            '<div style="margin-bottom:16px;">'
            f'<div class="card-title" style="margin-bottom:8px;">'
            f"{html.escape(title)}</div>"
            f'<div style="line-height:1.8;">{body_html}</div>'
            "</div>"
        )
    return "".join(sections)


def render_reading_panel(text: str, tool: str) -> None:
    """解读正文外框（易经浅绿 / 塔罗淡紫，对应牌义·解读·试试看三块）。"""
    if not (text or "").strip():
        st.markdown(_reading_panel_skeleton_html(tool), unsafe_allow_html=True)
        return
    st.markdown(
        build_reading_panel_html(_format_reading_sections_html(text), tool),
        unsafe_allow_html=True,
    )


def extract_golden_quote(full_text: str, is_iching: bool) -> str:
    quote = extract_section_text(full_text, "### 💎 今日金句")
    quote = re.sub(r"^[-*>\s#]+", "", quote).strip()
    quote = quote.replace("「", "").replace("」", "").strip('"').strip()
    if quote:
        return quote[:20]
    read_heading = "### 📜 卦象解读" if is_iching else "### 🌌 塔罗解读"
    body = extract_section_text(full_text, read_heading)
    body = re.sub(r"^[-*>\s]+", "", body)
    return (body[:25] + "…") if len(body) > 25 else body or "答案在心里，只是需要被看见。"


def extract_three_tips(full_text: str) -> list[str]:
    section = extract_section_text(full_text, "### 💬 试试看")
    lines = []
    for line in section.split("\n"):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        line = re.sub(r"^[\d\-\.\)、\s]+", "", line).strip()
        if line:
            lines.append(line)
    defaults = [
        "今天先迈出一小步，比想一百步有用。",
        "别急着给自己下结论，感受比答案重要。",
        "对自己温柔一点，你已经很努力了。",
    ]
    for i in range(3):
        if i >= len(lines):
            lines.append(defaults[i])
    return lines[:3]


def _question_summary(user_input: str, max_len: int = 15) -> str:
    text = (user_input or "").strip() or "心里那点事"
    return text if len(text) <= max_len else text[:max_len] + "..."


def _tarot_card_index(name: str) -> int:
    for i, c in enumerate(TAROT_MAJOR):
        if c["name"] == name:
            return i
    return 0


def _build_yao_lines_html_from_yaos(yaos: list) -> str:
    rows = []
    for y in reversed(yaos):
        is_yang = y["yin_yang"] == "阳"
        changed = y.get("change", False)
        color = "#fff" if changed else "#d4af37"
        glow = "box-shadow:0 0 8px rgba(255,255,255,0.4);" if changed else ""
        if is_yang:
            rows.append(f'<div class="sc-yang" style="background:{color};{glow}"></div>')
        else:
            rows.append(
                f'<div class="sc-yin"><span style="background:{color};{glow}"></span>'
                f'<span style="background:{color};{glow}"></span></div>'
            )
    return "".join(rows)


def _mode_key_from_uid(uid: str) -> str:
    for prefix in ("iching_", "tarot_"):
        if uid.startswith(prefix):
            return uid[len(prefix):] or "single"
    return "single"


def _pack_share_payload(outcome: dict) -> str:
    """将分享卡数据编码进 URL（新标签页独立 session 也能展示）。"""
    tool = outcome["tool"]
    payload: dict = {
        "t": tool,
        "m": _mode_key_from_uid(outcome.get("uid", "")),
        "n": outcome["result_name"],
        "q": (outcome.get("golden_quote") or "")[:280],
        "k": (outcome.get("keywords") or [])[:3],
    }
    if tool == "iching" and outcome.get("hex_result"):
        hr = outcome["hex_result"]
        payload["yg"] = [
            {
                "s": y["symbol"],
                "y": y["yin_yang"],
                "c": bool(y.get("change")),
                "p": y["position"],
            }
            for y in hr.get("yaos", [])
        ]
        ben_data = ICHING_HEXAGRAMS.get(hr.get("ben_gua", ""), {})
        payload["bn"] = ben_data.get("num")
    elif tool == "tarot":
        card = outcome.get("card") or {}
        payload["ci"] = card.get("image", "")
    raw = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    return base64.urlsafe_b64encode(raw.encode("utf-8")).decode().rstrip("=")


def _unpack_share_payload(token: str) -> dict | None:
    try:
        pad = "=" * (-len(token) % 4)
        data = base64.urlsafe_b64decode(token + pad)
        return json.loads(data.decode("utf-8"))
    except (json.JSONDecodeError, ValueError, UnicodeDecodeError):
        return None


def build_share_page_href(outcome: dict) -> str:
    return f"?view=share&share={_pack_share_payload(outcome)}"


def build_ritual_reset_href(tool: str, mode_key: str) -> str:
    """分享页「再来一卦/再抽一张」：回到第三步对应工具的起始页并重置仪式。"""
    return "?" + urlencode(
        {
            "flow_step": "3",
            "flow_tool": tool,
            "flow_choice": mode_key,
            "ritual_reset": "1",
        }
    )


def _kwargs_from_share_payload(p: dict) -> dict | None:
    tool = p.get("t")
    if tool not in ("iching", "tarot"):
        return None
    name = p.get("n") or ""
    kwargs: dict = {
        "tool": tool,
        "result_name": name,
        "golden_quote": p.get("q") or "",
        "keywords": p.get("k") or [],
        "hex_result": None,
        "card": None,
    }
    if tool == "iching":
        yaos = [
            {
                "symbol": y["s"],
                "yin_yang": y["y"],
                "change": y["c"],
                "position": y["p"],
            }
            for y in (p.get("yg") or [])
        ]
        kwargs["hex_result"] = {"ben_gua": name, "yaos": yaos}
    else:
        kwargs["card"] = {
            "name": name,
            "image": p.get("ci") or "",
            "keywords": p.get("k") or [],
        }
    return kwargs


def render_share_standalone_page() -> None:
    """分享专用页：新标签打开，只展示分享卡片。"""
    token = _qp_get("share")
    if not token:
        st.error("分享链接无效或已过期")
        return
    packed = _unpack_share_payload(token)
    if not packed:
        st.error("无法解析分享内容，请从结果页重新点击分享")
        return
    kwargs = _kwargs_from_share_payload(packed)
    if not kwargs:
        st.error("分享数据不完整")
        return

    st.markdown(
        """
<style>
.stApp { background-color: #F5F2EC !important; }
section[data-testid="stSidebar"] { display: none !important; }
div[data-testid="stToolbar"] { display: none !important; }
.block-container { max-width: 420px !important; padding-top: 2rem !important; }
</style>
""",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<div style='text-align:center;margin-bottom:12px;'>"
        "<div class='guayupai-h1-wrap' style='display:inline-block;text-align:left;'>"
        "<h1 class='guayupai-h1'>🔮 <span class='brand-name'>卦与牌</span></h1>"
        "<div class='guayupai-h1-line'></div>"
        "</div>"
        "<div class='guayupai-muted' style='margin-top:10px;'>长按下方卡片截图分享</div>"
        "</div>",
        unsafe_allow_html=True,
    )
    render_share_snapshot_card(**kwargs)
    st.markdown(
        '<p style="text-align:center;font-size:12px;color:#888;margin:16px 0 8px;">'
        "💡 可分享到朋友圈、小红书，或发给同样在纠结的朋友</p>",
        unsafe_allow_html=True,
    )

    tool = kwargs["tool"]
    mode_key = packed.get("m") or "single"
    reset_label = "🪙 再来一卦" if tool == "iching" else "🃏 再抽一张"
    st.link_button(
        reset_label,
        build_ritual_reset_href(tool, mode_key),
        use_container_width=True,
        key=f"share_reset_{tool}_{mode_key}",
    )


def _render_share_new_tab_link(outcome: dict, uid: str) -> None:
    """新标签打开分享页；易经=深绿，塔罗=深紫。"""
    href = html.escape(build_share_page_href(outcome), quote=True)
    cls = f"guayupai-share-newtab-{uid}"
    is_iching = outcome.get("tool") == "iching"
    theme_cls = "guayupai-btn-iching" if is_iching else "guayupai-btn-purple"
    if is_iching:
        inline_bg = (
            "background:linear-gradient(135deg,#2E7D32,#1B5E20);"
            "box-shadow:0 4px 12px rgba(46,125,50,0.2);"
        )
    else:
        inline_bg = (
            "background:linear-gradient(135deg,#6A1B9A,#4A148C);"
            "box-shadow:0 4px 12px rgba(106,27,154,0.2);"
        )
    st.markdown(
        f"""
<a class="{cls} {theme_cls}" href="{href}" target="_blank" rel="noopener noreferrer"
  style="display:flex;align-items:center;justify-content:center;width:100%;min-height:52px;
  box-sizing:border-box;text-decoration:none;padding:0.5rem 1rem;color:#fff;font-weight:500;
  border-radius:14px;border:none;{inline_bg}">
  💌 分享给同样纠结的 TA
</a>
""",
        unsafe_allow_html=True,
    )


def _share_card_footer_html() -> str:
    return """
  <div style="padding:16px;text-align:center;border-top:1px solid rgba(255,255,255,0.08);">
    <div style="font-size:20px;letter-spacing:16px;">🚶 💭 🌱</div>
    <div style="font-size:9px;color:rgba(255,255,255,0.25);margin-top:10px;">from <span class="brand-name">卦与牌</span> 🔮</div>
  </div>"""


def _keyword_pills_html(keywords: list) -> str:
    return "".join(
        f'<span style="display:inline-block;padding:3px 10px;border-radius:20px;'
        f"background:rgba(255,255,255,0.08);color:rgba(255,255,255,0.85);font-size:11px;"
        f'border:1px solid rgba(255,255,255,0.15);margin:0 3px;">{html.escape(k)}</span>'
        for k in keywords[:3]
    )


def render_share_snapshot_card(
    *,
    tool: str,
    result_name: str,
    golden_quote: str,
    keywords: list,
    hex_result: dict | None = None,
    card: dict | None = None,
) -> None:
    date_str = datetime.now().strftime("%Y.%m.%d")
    quote = html.escape(golden_quote)
    is_iching = tool == "iching"

    if is_iching and hex_result:
        ben_data = ICHING_HEXAGRAMS.get(hex_result["ben_gua"], {})
        num = ben_data.get("num", "?")
        yao_html = _build_yao_lines_html_from_yaos(hex_result["yaos"])
        body_mid = f"""
<div style="padding:20px;text-align:center;">
  <div style="display:flex;flex-direction:column;align-items:center;gap:8px;">{yao_html}</div>
  <div style="font-size:22px;color:#fff;margin-top:14px;font-weight:600;">{html.escape(result_name)}</div>
  <div style="font-size:11px;color:rgba(255,255,255,0.6);margin-top:4px;">第{num}卦 · {html.escape(result_name)}</div>
  <div style="margin-top:12px;">{_keyword_pills_html(keywords)}</div>
</div>"""
        shell_style = (
            "background:linear-gradient(165deg,#1a2f23 0%,#2d4a3e 60%,#1a2f23 100%);"
            "border:1px solid rgba(212,175,55,0.25);"
        )
        st.markdown(
            f"""
<div class="guayupai-iching-share-card" style="width:100%;max-width:360px;margin:0 auto 8px;border-radius:20px;overflow:hidden;
  box-shadow:0 12px 40px rgba(0,0,0,0.15);{shell_style}">
  <div style="display:flex;justify-content:space-between;align-items:center;padding:18px 20px;
    backdrop-filter:blur(10px);background:rgba(0,0,0,0.15);">
    <span class="brand-name" style="color:#d4af37;font-size:17px;letter-spacing:3px;">卦与牌</span>
    <span style="font-size:11px;color:rgba(255,255,255,0.5);letter-spacing:1px;">{date_str}</span>
  </div>
  {body_mid}
  <div style="padding:0 20px 16px;text-align:center;">
    <div style="font-size:10px;color:#d4af37;letter-spacing:2px;">今日直觉</div>
    <div style="margin:6px 0;line-height:1.2;">
      <span style="font-size:20px;color:#d4af37;vertical-align:top;">"</span>
      <span class="quote-text" style="color:rgba(255,255,255,0.95);">{quote}</span>
      <span style="font-size:20px;color:#d4af37;vertical-align:top;">"</span>
    </div>
  </div>
  {_share_card_footer_html()}
</div>
<style>
.sc-yang {{ width:52px;height:5px;border-radius:2px;margin:6px auto; }}
.sc-yin {{ width:52px;display:flex;justify-content:center;gap:8px;margin:6px auto; }}
.sc-yin span {{ width:22px;height:5px;border-radius:2px;display:block; }}
</style>
""",
            unsafe_allow_html=True,
        )
        return

    card_data = card or {}
    idx = _tarot_card_index(card_data.get("name", result_name))
    img = html.escape(card_data.get("image", ""))
    body_mid = f"""
<div style="padding:20px;text-align:center;">
  <img src="{img}" alt="{html.escape(result_name)}" loading="lazy"
    style="width:110px;border-radius:10px;border:2px solid #a8bdd8;box-shadow:0 8px 20px rgba(0,0,0,0.4);object-fit:cover;display:block;margin:0 auto;"
    onerror="this.style.display='none';this.nextElementSibling.style.display='flex';"/>
  <div style="display:none;width:110px;height:165px;margin:0 auto;border-radius:10px;border:2px solid #a8bdd8;
    background:#162544;align-items:center;justify-content:center;font-size:13px;color:#fff;">{html.escape(result_name)}</div>
  <div style="font-size:20px;color:#fff;margin-top:12px;font-weight:600;">{html.escape(result_name)}</div>
  <div style="font-size:11px;color:rgba(255,255,255,0.6);margin-top:4px;">{TAROT_ROMAN[idx]} · {html.escape(result_name)}</div>
  <div style="margin-top:12px;">{_keyword_pills_html(keywords)}</div>
</div>"""
    shell_style = (
        "border:1px solid rgba(160,185,225,0.25) !important;"
    )
    st.markdown(
        f"""
<div class="guayupai-tarot-share-card" style="width:100%;max-width:360px;min-height:420px;margin:0 auto 8px;border-radius:20px;overflow:hidden;
  box-shadow:0 12px 40px rgba(0,0,0,0.15);{shell_style}">
  <div style="display:flex;justify-content:space-between;align-items:center;padding:18px 20px;
    backdrop-filter:blur(10px);background:rgba(0,0,0,0.15);">
    <span class="brand-name" style="color:#d4af37;font-size:17px;letter-spacing:3px;">卦与牌</span>
    <span style="font-size:11px;color:rgba(255,255,255,0.5);letter-spacing:1px;">{date_str}</span>
  </div>
  {body_mid}
  <div style="padding:0 20px 16px;text-align:center;">
    <div style="font-size:10px;color:#d4af37;letter-spacing:2px;">今日直觉</div>
    <div style="margin:6px 0;line-height:1.2;">
      <span style="font-size:20px;color:#d4af37;vertical-align:top;">"</span>
      <span class="quote-text" style="color:rgba(255,255,255,0.95);">{quote}</span>
      <span style="font-size:20px;color:#d4af37;vertical-align:top;">"</span>
    </div>
  </div>
  {_share_card_footer_html()}
</div>
""",
        unsafe_allow_html=True,
    )


def save_reading_outcome(
    *,
    uid: str,
    tool: str,
    full_text: str,
    user_input: str,
    result_name: str,
    mode_name: str,
    hex_result: dict | None = None,
    card: dict | None = None,
) -> dict:
    is_iching = tool == "iching"
    golden_quote = extract_golden_quote(full_text, is_iching)
    tips = extract_three_tips(full_text)
    outcome = {
        "uid": uid,
        "tool": tool,
        "full_text": full_text,
        "display_text": format_reading_display_text(full_text),
        "user_input": user_input,
        "result_name": result_name,
        "mode_name": mode_name,
        "golden_quote": golden_quote,
        "tips": tips,
        "hex_result": hex_result,
        "card": card,
        "keywords": (
            ICHING_HEXAGRAMS.get(hex_result["ben_gua"], {}).get("keywords", [])
            if hex_result
            else (card or {}).get("keywords", [])
        ),
    }
    st.session_state[f"outcome_{uid}"] = outcome
    return outcome


def render_reading_outcome_footer(
    uid: str,
    tool: str,
    _ritual_key: str,
    _mode_key: str,
) -> None:
    outcome = st.session_state.get(f"outcome_{uid}")
    if not outcome:
        return
    _render_share_new_tab_link(outcome, uid)


# 10. UI 渲染
# ==========================================
def _theme_tokens(theme_tool: str | None) -> dict[str, str]:
    """全站暖米色背景；accent 仅用于按钮/边框等点缀，不再区分页面底色。"""
    page_bg = "#F5F2EC"
    if theme_tool == "tarot":
        return {
            "bg": page_bg,
            "accent": "#6A1B9A",
            "accent_dark": "#4A148C",
            "accent_rgb": "106,27,154",
            "flow": "tarot",
        }
    if theme_tool == "iching":
        return {
            "bg": page_bg,
            "accent": "#2E7D32",
            "accent_dark": "#1B5E20",
            "accent_rgb": "46,125,50",
            "flow": "iching",
        }
    return {
        "bg": page_bg,
        "accent": "#5C5C5C",
        "accent_dark": "#444444",
        "accent_rgb": "90,90,90",
        "flow": "neutral",
    }


def _inject_streamlit_shell_fix() -> None:
    """折叠按钮在 Streamlit 外壳（父文档），需在 parent/top 注入样式并移除节点。"""
    st.markdown(
        """
<script>
(function() {
  var BAD = /keyboard_double_arrow_(right|left)/i;
  var HIDE_SEL = [
    "section[data-testid=stSidebar]",
    "[data-testid=collapsedControl]",
    "[data-testid=stSidebarCollapsedControl]",
    "[data-testid=stSidebarCollapseButton]",
    "[data-testid=stExpandSidebarButton]",
    "[data-testid=stCollapseSidebarButton]",
    "section[data-testid=stSidebar] button[kind=header]",
    "section[data-testid=stSidebar] button[kind=headerNoPadding]"
  ].join(",");

  var STYLE_ID = "gp-hide-sidebar-toggle-style";
  var STYLE_TEXT = HIDE_SEL + "{display:none!important;visibility:hidden!important;width:0!important;height:0!important;max-width:0!important;max-height:0!important;overflow:hidden!important;opacity:0!important;pointer-events:none!important;position:fixed!important;left:-99999px!important;font-size:0!important;line-height:0!important;}";

  function allDocs() {
    var docs = [], seen = new Set();
    function add(d) {
      if (!d || !d.documentElement || seen.has(d)) return;
      seen.add(d);
      docs.push(d);
    }
    try { add(window.top && window.top.document); } catch (e) {}
    try { if (window.parent && window.parent.document) add(window.parent.document); } catch (e) {}
    add(document);
    docs.forEach(function(d) {
      try {
        d.querySelectorAll("iframe").forEach(function(f) {
          try { add(f.contentDocument); } catch (e) {}
        });
      } catch (e) {}
    });
    return docs;
  }

  function injectStyle(doc) {
    if (!doc || doc.getElementById(STYLE_ID)) return;
    var el = doc.createElement("style");
    el.id = STYLE_ID;
    el.textContent = STYLE_TEXT;
    (doc.head || doc.documentElement).appendChild(el);
  }

  function removeBadNodes(doc) {
    if (!doc || !doc.body) return;
    try {
      doc.querySelectorAll(HIDE_SEL).forEach(function(el) { el.remove(); });
    } catch (e) {}
    try {
      var sidebar = doc.querySelector("section[data-testid=stSidebar]");
      if (sidebar) {
        sidebar.querySelectorAll("button").forEach(function(btn) {
          var t = (btn.innerText || btn.textContent || "").trim();
          if (BAD.test(t)) btn.remove();
        });
      }
    } catch (e) {}
    try {
      var walker = doc.createTreeWalker(doc.body, NodeFilter.SHOW_TEXT);
      var node;
      while ((node = walker.nextNode())) {
        var t = (node.textContent || "").trim();
        if (!BAD.test(t)) continue;
        var host = node.parentElement;
        if (!host) continue;
        var btn = host.closest ? host.closest("button") : null;
        if (btn) btn.remove();
        else if (t.length < 48) host.remove();
      }
    } catch (e) {}
  }

  function run() {
    allDocs().forEach(function(doc) {
      injectStyle(doc);
      removeBadNodes(doc);
    });
  }

  run();
  var until = Date.now() + 8000;
  function loop() {
    run();
    if (Date.now() < until) requestAnimationFrame(loop);
  }
  requestAnimationFrame(loop);
  [50, 150, 400, 800, 1500, 3000, 5000].forEach(function(ms) { setTimeout(run, ms); });

  try {
    var targets = [];
    allDocs().forEach(function(doc) {
      if (doc.body) targets.push(doc.body);
    });
    if (window.MutationObserver && targets.length) {
      var obs = new MutationObserver(run);
      targets.forEach(function(t) { obs.observe(t, { childList: true, subtree: true }); });
    }
  } catch (e) {}
})();
</script>
""",
        unsafe_allow_html=True,
    )


def render_css(theme_tool: str | None = None, *, share_page: bool = False) -> None:
    """全局「仪式空间」视觉：全站暖米色背景 + 衬线标题 / 无衬线正文；分享卡样式优先级最高。

    按钮配色约定（写死在 CSS + JS，覆盖 Streamlit 默认红色 primary）：
    - 抽卦 / 易经仪式：开始起卦、摇爻、看看结果 → 深绿渐变 #2E7D32 → #1B5E20
    - 抽牌 / 塔罗仪式：开始洗牌等 → 深蓝紫渐变 #6A1B9A → #4A148C
    - 其余（场景选择、分享、再来一卦）：深紫；返回 ← 为次要描边按钮
    """
    t = _theme_tokens(theme_tool)
    glass_main = "" if share_page else """
    .main .block-container {
        background: rgba(255,255,255,0.88) !important;
        backdrop-filter: blur(16px) !important;
        -webkit-backdrop-filter: blur(16px) !important;
        border-radius: 20px !important;
        border: 1px solid rgba(0,0,0,0.05) !important;
        box-shadow: 0 8px 32px rgba(0,0,0,0.06), 0 2px 8px rgba(0,0,0,0.03) !important;
        padding: 28px !important;
        max-width: 720px !important;
        animation: gpFadeIn 0.3s ease-out !important;
    }
    """
    st.markdown(
        f"""
<style>
    :root {{
        --gp-bg: #F5F2EC;
        --gp-font-serif: Georgia, 'STSong', 'SimSun', 'Songti SC', serif;
        --gp-font-sans: -apple-system, 'PingFang SC', 'Microsoft YaHei', sans-serif;
        --gp-accent: {t["accent"]};
        --gp-accent-dark: {t["accent_dark"]};
        --gp-accent-rgb: {t["accent_rgb"]};
    }}

    /* ========== 全站字体：衬线标题 / 无衬线正文 ========== */
    h1, h2, h3,
    .brand-name, .step-title, .step-title .big,
    .card-title, .quote-text,
    .guayupai-h1, .guayupai-h1 .brand-name,
    [data-testid="stMain"] h1,
    [data-testid="stMain"] h2,
    [data-testid="stMain"] h3,
    [data-testid="stMain"] .guayupai-h1,
    [data-testid="stMain"] .brand-name,
    [data-testid="stMain"] .step-title,
    [data-testid="stMain"] .step-title .big,
    [data-testid="stMain"] .card-title,
    div[data-testid="stMarkdownContainer"] h1,
    div[data-testid="stMarkdownContainer"] .guayupai-h1,
    div[data-testid="stMarkdownContainer"] .brand-name {{
        font-family: var(--gp-font-serif) !important;
    }}
    .stApp,
    [data-testid="stAppViewContainer"],
    [data-testid="stMain"],
    [data-testid="stSidebar"],
    .stApp p, .stApp label,
    .stApp button, .stApp input, .stApp textarea,
    .stApp span:not(.brand-name):not(.quote-text),
    .guayupai-lead, .guayupai-muted,
    .guayupai-ritual-panel__desc,
    .guayupai-reading-panel, .result-card,
    div[data-testid="stTextArea"] textarea,
    .stTextArea textarea,
    [data-testid="stMain"] .stMarkdown {{
        font-family: var(--gp-font-sans) !important;
    }}
    [data-testid="stMain"] .stMarkdown h1,
    [data-testid="stMain"] .stMarkdown h2,
    [data-testid="stMain"] .stMarkdown h3,
    [data-testid="stMain"] .stMarkdown .guayupai-h1,
    [data-testid="stMain"] .stMarkdown .brand-name,
    [data-testid="stMain"] .stMarkdown .step-title,
    [data-testid="stMain"] .stMarkdown .card-title,
    [data-testid="stMain"] .stMarkdown .quote-text {{
        font-family: var(--gp-font-serif) !important;
    }}
    h1, .guayupai-h1 {{
        font-size: 32px !important;
        font-weight: 600 !important;
        letter-spacing: 4px !important;
        line-height: 1.3 !important;
    }}
    h2, .step-title .big {{
        font-size: 20px !important;
        font-weight: 600 !important;
        line-height: 1.5 !important;
    }}
    h3, .card-title, .guayupai-ritual-panel__title {{
        font-size: 18px !important;
        font-weight: 600 !important;
        line-height: 1.45 !important;
    }}
    .step-title .small,
    .guayupai-muted, .guayupai-ritual-panel__desc {{
        font-size: 13px !important;
    }}
    .quote-text {{
        font-size: 15px !important;
        line-height: 1.7 !important;
    }}
    [data-testid="stMain"] p,
    .guayupai-lead,
    .guayupai-reading-panel,
    .result-card p {{
        font-size: 15px !important;
        line-height: 1.8 !important;
    }}
    [data-testid="stMain"] button,
    [data-testid="stMain"] [data-testid="stBaseButton-primary"],
    [data-testid="stMain"] [data-testid="stBaseButton-secondary"],
    div[data-testid="stButton"] > button {{
        font-size: 16px !important;
        font-weight: 500 !important;
    }}
    .stTextArea textarea,
    div[data-testid="stTextArea"] textarea {{
        font-size: 15px !important;
    }}
    .stTextArea textarea::placeholder {{
        font-size: 15px !important;
    }}
    .stTextArea label {{
        font-size: 15px !important;
    }}

    @keyframes gpFadeIn {{
        from {{ opacity: 0; transform: translateY(6px); }}
        to {{ opacity: 1; transform: translateY(0); }}
    }}
    @keyframes readingShimmer {{
        0% {{ background-position: 100% 0; }}
        100% {{ background-position: -100% 0; }}
    }}

    .stApp {{
        background-color: #F5F2EC !important;
        background-image: none !important;
    }}
    /* 覆盖 Streamlit 默认红色 primary 主题变量 */
    .stApp.guayupai-flow-iching {{
        --primary-color: #2E7D32 !important;
        --primary-color-background: #2E7D32 !important;
        --primary-color-text: #ffffff !important;
    }}
    .stApp.guayupai-flow-tarot {{
        --primary-color: #6A1B9A !important;
        --primary-color-background: #6A1B9A !important;
        --primary-color-text: #ffffff !important;
    }}
    .stApp.guayupai-flow-neutral {{
        --primary-color: #6A1B9A !important;
        --primary-color-background: #6A1B9A !important;
    }}
    .stDeployButton {{ display: none !important; }}

    /* 无侧边栏：隐藏侧栏区域及折叠按钮（避免 Material 图标名露字） */
    section[data-testid="stSidebar"],
    section[data-testid="stSidebar"] > div,
    [data-testid="collapsedControl"],
    [data-testid="stSidebarCollapsedControl"],
    [data-testid="stSidebarCollapseButton"],
    [data-testid="stExpandSidebarButton"],
    [data-testid="stCollapseSidebarButton"],
    button[data-testid="stExpandSidebarButton"],
    button[data-testid="stCollapseSidebarButton"] {{
        display: none !important;
        visibility: hidden !important;
        width: 0 !important;
        max-width: 0 !important;
        height: 0 !important;
        overflow: hidden !important;
        opacity: 0 !important;
        pointer-events: none !important;
    }}

    {glass_main}

    /* 页面标题 */
    .guayupai-h1-wrap {{ margin: 0 0 8px 0; }}
    .guayupai-h1 {{
        color: #1a1a1a !important;
        margin: 0 !important;
        padding: 0 !important;
    }}
    .guayupai-h1-line {{
        width: 32px;
        height: 2px;
        background: var(--gp-accent);
        margin-top: 10px;
    }}
    .guayupai-lead {{
        color: #444;
        margin-bottom: 20px;
    }}
    .guayupai-muted {{
        color: #888;
    }}

    /* 步骤标题 */
    .step-title {{ text-align: center; margin: 24px 0 20px 0; }}
    .step-title .big {{
        color: #333;
    }}
    .step-title .small {{
        color: #888;
        margin-top: 8px;
        line-height: 1.6;
    }}

    /* 玻璃功能卡 */
    .guayupai-glass,
    .guayupai-ritual-panel,
    .guayupai-yao-display,
    .guayupai-coin-result {{
        background: rgba(255,255,255,0.88) !important;
        backdrop-filter: blur(16px) !important;
        -webkit-backdrop-filter: blur(16px) !important;
        border-radius: 20px !important;
        border: 1px solid rgba(0,0,0,0.05) !important;
        box-shadow: 0 8px 32px rgba(0,0,0,0.06), 0 2px 8px rgba(0,0,0,0.03) !important;
        transition: transform 0.3s ease, box-shadow 0.3s ease !important;
    }}
    @media (hover: hover) {{
        .guayupai-glass:hover,
        .guayupai-ritual-panel:hover {{
            transform: translateY(-2px);
            box-shadow: 0 12px 40px rgba(0,0,0,0.08), 0 4px 12px rgba(0,0,0,0.04) !important;
        }}
    }}

    .guayupai-ritual-panel {{
        text-align: center;
        padding: 32px 24px;
        margin: 16px 0;
    }}
    .guayupai-ritual-panel__icon {{ font-size: 48px; margin-bottom: 12px; }}
    .guayupai-ritual-panel__desc {{ color: #888; margin-top: 8px; line-height: 1.6; }}
    .guayupai-ritual-panel--iching .guayupai-ritual-panel__title {{ color: #1E4620; }}
    .guayupai-ritual-panel--tarot .guayupai-ritual-panel__title {{ color: #4A148C; }}

    /* 输入框 */
    .stTextArea textarea,
    div[data-testid="stTextArea"] textarea {{
        background: #FAFAFA !important;
        border: 1.5px solid rgba(0,0,0,0.08) !important;
        border-radius: 14px !important;
        padding: 16px !important;
        color: #333 !important;
        line-height: 1.6 !important;
    }}
    .stTextArea textarea::placeholder {{
        color: #bbb !important;
    }}
    .stTextArea textarea:focus {{
        border-color: var(--gp-accent) !important;
        box-shadow: 0 0 0 4px rgba(var(--gp-accent-rgb), 0.08) !important;
        outline: none !important;
    }}
    .stTextArea label {{
        color: #444 !important;
    }}

    /* ========== 按钮主题：抽卦=深绿 / 抽牌=深紫 / 其余默认深紫 ========== */
    .gp-btn-purple {{
        background: linear-gradient(135deg, #6A1B9A, #4A148C) !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 14px !important;
        font-weight: 500 !important;
        box-shadow: 0 4px 12px rgba(106,27,154,0.2) !important;
        transition: all 0.2s ease !important;
    }}
    .gp-btn-purple:hover {{
        transform: translateY(-1px) !important;
        box-shadow: 0 6px 16px rgba(106,27,154,0.3) !important;
        color: #ffffff !important;
    }}
    a.guayupai-btn-purple,
    [data-testid="stMain"] a[data-testid="stBaseLinkButton"] {{
        background: linear-gradient(135deg, #6A1B9A, #4A148C) !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 14px !important;
        font-weight: 500 !important;
        box-shadow: 0 4px 12px rgba(106,27,154,0.2) !important;
    }}
    a.guayupai-btn-iching {{
        background: linear-gradient(135deg, #2E7D32, #1B5E20) !important;
        background-color: #2E7D32 !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 14px !important;
        font-weight: 500 !important;
        box-shadow: 0 4px 12px rgba(46,125,50,0.2) !important;
    }}
    a.guayupai-btn-iching:hover {{
        box-shadow: 0 6px 16px rgba(46,125,50,0.3) !important;
        color: #ffffff !important;
    }}

    /* 通用：默认深紫（Step1、分享、再来一卦等） */
    .stApp.guayupai-flow-neutral [data-testid="stMain"] button:not([kind="secondary"]):not(.gp-btn-secondary),
    .stApp.guayupai-flow-neutral [data-testid="stMain"] [data-testid="stBaseButton-primary"],
    .stApp.guayupai-flow-neutral [data-testid="stMain"] .stButton button:not([kind="secondary"]),
    button.gp-btn-purple,
    button.tarot-theme,
    div[data-testid="stButton"] > button.tarot-theme {{
        background: linear-gradient(135deg, #6A1B9A, #4A148C) !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 14px !important;
        font-weight: 500 !important;
        box-shadow: 0 4px 12px rgba(106,27,154,0.2) !important;
        transition: all 0.2s ease !important;
    }}

    /* 易经流程：抽卦仪式按钮 — 全部深绿（开始起卦 / 摇爻 / 看看结果） */
    .stApp.guayupai-flow-iching [data-testid="stMain"] button:not([kind="secondary"]):not(.gp-btn-secondary),
    .stApp.guayupai-flow-iching [data-testid="stMain"] [data-testid="stBaseButton-primary"],
    .stApp.guayupai-flow-iching [data-testid="stMain"] .stButton button:not([kind="secondary"]),
    .stApp.guayupai-flow-iching [data-testid="stMain"] button.iching-theme,
    .stApp.guayupai-flow-iching [data-testid="stMain"] button.gp-btn-iching,
    button.iching-theme,
    button.gp-btn-iching,
    div[data-testid="stButton"] > button.iching-theme {{
        background: linear-gradient(135deg, #2E7D32, #1B5E20) !important;
        background-color: #2E7D32 !important;
        background-image: linear-gradient(135deg, #2E7D32, #1B5E20) !important;
        color: #ffffff !important;
        border: none !important;
        border-color: transparent !important;
        border-radius: 14px !important;
        font-weight: 500 !important;
        box-shadow: 0 4px 12px rgba(46,125,50,0.2) !important;
        transition: all 0.2s ease !important;
    }}
    .stApp.guayupai-flow-iching [data-testid="stMain"] button:not([kind="secondary"]):hover,
    .stApp.guayupai-flow-iching [data-testid="stMain"] button.iching-theme:hover,
    button.gp-btn-iching:hover {{
        transform: translateY(-1px) !important;
        box-shadow: 0 6px 16px rgba(46,125,50,0.3) !important;
        color: #ffffff !important;
    }}

    /* 塔罗流程：抽牌仪式按钮 — 全部深蓝紫（开始洗牌等） */
    .stApp.guayupai-flow-tarot [data-testid="stMain"] button:not([kind="secondary"]):not(.gp-btn-secondary),
    .stApp.guayupai-flow-tarot [data-testid="stMain"] [data-testid="stBaseButton-primary"],
    .stApp.guayupai-flow-tarot [data-testid="stMain"] .stButton button:not([kind="secondary"]),
    .stApp.guayupai-flow-tarot [data-testid="stMain"] button.tarot-theme,
    .stApp.guayupai-flow-tarot [data-testid="stMain"] button.gp-btn-tarot {{
        background: linear-gradient(135deg, #6A1B9A, #4A148C) !important;
        background-color: #6A1B9A !important;
        background-image: linear-gradient(135deg, #6A1B9A, #4A148C) !important;
        color: #ffffff !important;
        border: none !important;
        border-color: transparent !important;
        border-radius: 14px !important;
        font-weight: 500 !important;
        box-shadow: 0 4px 12px rgba(106,27,154,0.2) !important;
        transition: all 0.2s ease !important;
    }}
    .stApp.guayupai-flow-tarot [data-testid="stMain"] button:not([kind="secondary"]):hover,
    .stApp.guayupai-flow-tarot [data-testid="stMain"] button.tarot-theme:hover {{
        transform: translateY(-1px) !important;
        box-shadow: 0 6px 16px rgba(106,27,154,0.3) !important;
    }}

    [data-testid="stMain"] button:active,
    [data-testid="stMain"] [data-testid="stBaseButton-primary"]:active {{
        transform: scale(0.97) !important;
    }}

    /* 次要按钮：返回 */
    [data-testid="stMain"] button[kind="secondary"],
    [data-testid="stMain"] [data-testid="stBaseButton-secondary"],
    button.gp-btn-secondary {{
        background: transparent !important;
        border: 1.5px solid rgba(0,0,0,0.15) !important;
        color: #666 !important;
        box-shadow: none !important;
        min-height: 44px !important;
        border-radius: 14px !important;
        font-weight: 500 !important;
    }}
    [data-testid="stMain"] button[kind="secondary"]:hover {{
        background: rgba(0,0,0,0.03) !important;
        transform: none !important;
        box-shadow: none !important;
    }}

    /* 仪式主按钮尺寸 */
    .stApp.guayupai-flow-iching [data-testid="stMain"] button:not([kind="secondary"]),
    .stApp.guayupai-flow-tarot [data-testid="stMain"] button:not([kind="secondary"]),
    .stApp.guayupai-flow-iching [data-testid="stMain"] [data-testid="stBaseButton-primary"],
    .stApp.guayupai-flow-tarot [data-testid="stMain"] [data-testid="stBaseButton-primary"] {{
        height: 52px !important;
        min-height: 52px !important;
        letter-spacing: 0.5px !important;
    }}

    /* Step1：三个场景 — 深紫大卡片 */
    .guayupai-step-choices--step1 [data-testid="column"] button,
    .guayupai-step-choices--step1 [data-testid="column"] [data-testid="stBaseButton-primary"] {{
        padding: 32px 16px !important;
        font-size: 16px !important;
        text-align: center !important;
        line-height: 1.7 !important;
        white-space: pre-line !important;
        border-radius: 20px !important;
        min-height: 140px !important;
        background: linear-gradient(135deg, #6A1B9A, #4A148C) !important;
        color: #ffffff !important;
        border: none !important;
        box-shadow: 0 4px 12px rgba(106,27,154,0.2) !important;
    }}

    /* Step2 左门：摇铜钱 — 深绿 */
    .guayupai-step-choices--step2 [data-testid="column"]:nth-child(1) button,
    .guayupai-step-choices--step2 [data-testid="column"]:nth-child(1) [data-testid="stBaseButton-primary"] {{
        background: linear-gradient(135deg, #2E7D32, #1B5E20) !important;
        box-shadow: 0 4px 12px rgba(46,125,50,0.2) !important;
        color: #ffffff !important;
        padding: 32px 16px !important;
        font-size: 16px !important;
        text-align: center !important;
        line-height: 1.7 !important;
        white-space: pre-line !important;
        border-radius: 20px !important;
        min-height: 140px !important;
        border: none !important;
    }}
    .guayupai-step-choices--step2 [data-testid="column"]:nth-child(1) button:hover {{
        box-shadow: 0 6px 16px rgba(46,125,50,0.3) !important;
    }}
    /* Step2 右门：翻张牌 — 深紫 */
    .guayupai-step-choices--step2 [data-testid="column"]:nth-child(2) button,
    .guayupai-step-choices--step2 [data-testid="column"]:nth-child(2) [data-testid="stBaseButton-primary"] {{
        padding: 32px 16px !important;
        font-size: 16px !important;
        text-align: center !important;
        line-height: 1.7 !important;
        white-space: pre-line !important;
        border-radius: 20px !important;
        min-height: 140px !important;
        background: linear-gradient(135deg, #6A1B9A, #4A148C) !important;
        color: #ffffff !important;
        border: none !important;
        box-shadow: 0 4px 12px rgba(106,27,154,0.2) !important;
    }}

    /* 爻象 / 铜钱 */
    .guayupai-yao-display {{
        text-align: center;
        padding: 20px;
        margin: 12px 0;
        border-left: 4px solid var(--gp-accent) !important;
    }}
    .yao-line {{
        font-size: 24px;
        line-height: 1.6;
        font-family: monospace;
    }}
    .yao-change {{ color: #E65100; font-weight: 700; }}
    .guayupai-coin-result {{
        text-align: center;
        padding: 16px;
        margin: 8px 0;
    }}
    .coin-coins {{ font-size: 20px; letter-spacing: 4px; }}
    .coin-name {{ font-size: 14px; font-weight: 600; color: #333; margin-top: 4px; }}

    /* 解读面板 */
    .guayupai-reading-panel {{
        padding: 20px !important;
        margin: 12px 0 !important;
        border-radius: 20px !important;
        background: rgba(255,255,255,0.88) !important;
        backdrop-filter: blur(16px) !important;
        border: 1px solid rgba(0,0,0,0.05) !important;
        box-shadow: 0 8px 32px rgba(0,0,0,0.06), 0 2px 8px rgba(0,0,0,0.03) !important;
    }}
    .guayupai-reading-panel--iching {{
        color: #1E4620 !important;
        border-left: 5px solid #2E7D32 !important;
    }}
    .guayupai-reading-panel--tarot {{
        color: #4A148C !important;
        border-left: 5px solid #6A1B9A !important;
    }}

    /* 结果卡片（卦/牌摘要） */
    .result-card {{
        padding: 16px;
        border-radius: 20px;
        margin: 12px 0;
        border-left: 5px solid;
        background: rgba(255,255,255,0.88) !important;
        backdrop-filter: blur(16px) !important;
        border: 1px solid rgba(0,0,0,0.05) !important;
        box-shadow: 0 8px 32px rgba(0,0,0,0.06), 0 2px 8px rgba(0,0,0,0.03) !important;
    }}
    .result-east {{ color: #1E4620; border-left-color: #2E7D32 !important; }}
    .result-west {{ color: #4A148C; border-left-color: #6A1B9A !important; }}

    /* st.info 融入背景 */
    .main [data-testid="stAlert"] {{
        background: rgba(255,255,255,0.75) !important;
        backdrop-filter: blur(10px) !important;
        border-radius: 14px !important;
        border: 1px solid rgba(0,0,0,0.05) !important;
    }}

    @media (max-width: 640px) {{
        .main .block-container {{ padding: 20px !important; }}
        .guayupai-h1 {{ font-size: 26px !important; letter-spacing: 3px !important; }}
        .step-title .big {{ font-size: 18px !important; }}
        .guayupai-ritual-panel {{ padding: 24px 16px; }}
        .main .stButton > button[kind="primary"] {{
            height: 48px !important;
            min-height: 48px !important;
        }}
        .guayupai-tarot-result-card img.tarot-result-img {{
            width: 80px !important;
        }}
        .guayupai-tarot-result-card .tarot-result-title {{
            font-size: 16px !important;
        }}
    }}

    /* ========== 分享卡：最高优先级，不被全局覆盖 ========== */
    .guayupai-tarot-share-card {{
        background-color: #0a1528 !important;
        background-image:
            linear-gradient(165deg, #0a1528 0%, #162544 60%, #0a1528 100%),
            radial-gradient(0.5px 0.5px at 18% 22%, rgba(180,205,255,0.5), transparent),
            radial-gradient(0.5px 0.5px at 72% 38%, rgba(180,205,255,0.45), transparent),
            radial-gradient(0.5px 0.5px at 45% 78%, rgba(180,205,255,0.4), transparent) !important;
        border-radius: 20px !important;
        box-shadow: 0 12px 40px rgba(0,0,0,0.15) !important;
    }}
    div[data-testid="stMarkdownContainer"]:has(.guayupai-tarot-share-card),
    div[data-testid="stMarkdownContainer"]:has(.guayupai-iching-share-card) {{
        background: transparent !important;
        border: none !important;
        padding: 0 !important;
        box-shadow: none !important;
        backdrop-filter: none !important;
    }}
    .guayupai-iching-share-card {{
        border-radius: 20px !important;
        box-shadow: 0 12px 40px rgba(0,0,0,0.15) !important;
    }}
</style>
<div class="guayupai-flow-{t["flow"]}" style="display:none" aria-hidden="true"></div>
""",
        unsafe_allow_html=True,
    )
    st.markdown(
        f"""
<script>
(function() {{
  var root = window.parent?.document || document;
  var app = root.querySelector(".stApp");
  if (!app) return;
  app.classList.remove("guayupai-flow-iching", "guayupai-flow-tarot", "guayupai-flow-neutral");
  app.classList.add("guayupai-flow-{t["flow"]}");

  function isSecondary(btn) {{
    var k = btn.getAttribute("kind");
    var tid = btn.getAttribute("data-testid") || "";
    return k === "secondary" || tid === "stBaseButton-secondary" || (btn.innerText || "").trim() === "←";
  }}

  function tagButtonThemes() {{
    var main = root.querySelector('[data-testid="stMain"]') || root.querySelector("section.main");
    if (!main) return;
    var ichingRe = /iching|step2_iching/i;
    var tarotRe = /tarot|step2_tarot/i;
    var ichingText = /开始起卦|摇第\\d*爻|看看结果|摇铜钱/;
    var tarotText = /开始洗牌|翻张牌/;
    var purpleText = /再来一卦|再抽一张/;
    var shareText = /分享给/;

    main.querySelectorAll("button").forEach(function(btn) {{
      if (isSecondary(btn)) {{
        btn.classList.add("gp-btn-secondary");
        return;
      }}
      btn.classList.remove("iching-theme", "tarot-theme", "gp-btn-iching", "gp-btn-tarot", "gp-btn-purple");

      var label = (btn.innerText || "").replace(/\\s+/g, "");
      var theme = "purple";
      if (shareText.test(label) && app.classList.contains("guayupai-flow-iching")) {{
        theme = "iching";
      }} else if (purpleText.test(label) || shareText.test(label)) {{
        theme = "purple";
      }} else {{
        var node = btn;
        for (var i = 0; i < 14 && node; i++, node = node.parentElement) {{
          var blob = (node.id || "") + (node.getAttribute("data-testid") || "") + (node.className || "");
          if (ichingRe.test(blob)) {{ theme = "iching"; break; }}
          if (tarotRe.test(blob)) {{ theme = "tarot"; break; }}
        }}
        if (theme === "purple") {{
          if (app.classList.contains("guayupai-flow-iching") || ichingText.test(label)) {{
            theme = "iching";
          }} else if (app.classList.contains("guayupai-flow-tarot") || tarotText.test(label)) {{
            theme = "tarot";
          }}
        }}
      }}

      var greenGrad = "linear-gradient(135deg, #2E7D32, #1B5E20)";
      var purpleGrad = "linear-gradient(135deg, #6A1B9A, #4A148C)";
      if (theme === "iching") {{
        btn.classList.add("iching-theme", "gp-btn-iching");
        btn.style.setProperty("background", greenGrad, "important");
        btn.style.setProperty("background-color", "#2E7D32", "important");
        btn.style.setProperty("background-image", greenGrad, "important");
        btn.style.setProperty("color", "#ffffff", "important");
        btn.style.setProperty("border", "none", "important");
        btn.style.setProperty("box-shadow", "0 4px 12px rgba(46,125,50,0.2)", "important");
      }} else {{
        btn.classList.add(theme === "tarot" ? "tarot-theme gp-btn-tarot" : "gp-btn-purple");
        btn.style.setProperty("background", purpleGrad, "important");
        btn.style.setProperty("background-color", "#6A1B9A", "important");
        btn.style.setProperty("background-image", purpleGrad, "important");
        btn.style.setProperty("color", "#ffffff", "important");
        btn.style.setProperty("border", "none", "important");
        btn.style.setProperty("box-shadow", "0 4px 12px rgba(106,27,154,0.2)", "important");
      }}
      btn.style.setProperty("border-radius", "14px", "important");
      btn.style.setProperty("font-weight", "500", "important");
      if (app.classList.contains("guayupai-flow-iching") || app.classList.contains("guayupai-flow-tarot")) {{
        btn.style.setProperty("min-height", "52px", "important");
        btn.style.setProperty("height", "52px", "important");
      }}
    }});
  }}

  tagButtonThemes();
  [80, 200, 500, 1000].forEach(function(ms) {{ setTimeout(tagButtonThemes, ms); }});
  try {{
    if (app && window.MutationObserver) {{
      new MutationObserver(function() {{ tagButtonThemes(); }}).observe(app, {{ childList: true, subtree: true }});
    }}
  }} catch (e) {{}}
}})();
</script>
""",
        unsafe_allow_html=True,
    )


def _inject_iching_ritual_button_css() -> None:
    st.markdown(
        """
<style>
.stApp.guayupai-flow-iching [data-testid="stMain"] div[data-testid="stButton"] > button:not([kind="secondary"]),
.stApp.guayupai-flow-iching [data-testid="stMain"] [data-testid="stBaseButton-primary"] {
  background: linear-gradient(135deg, #2E7D32, #1B5E20) !important;
  background-color: #2E7D32 !important;
  background-image: linear-gradient(135deg, #2E7D32, #1B5E20) !important;
  color: #ffffff !important;
  border: none !important;
  box-shadow: 0 4px 12px rgba(46,125,50,0.2) !important;
}
</style>
""",
        unsafe_allow_html=True,
    )


def _inject_tarot_ritual_button_css() -> None:
    st.markdown(
        """
<style>
.stApp.guayupai-flow-tarot [data-testid="stMain"] div[data-testid="stButton"] > button:not([kind="secondary"]),
.stApp.guayupai-flow-tarot [data-testid="stMain"] [data-testid="stBaseButton-primary"] {
  background: linear-gradient(135deg, #6A1B9A, #4A148C) !important;
  background-color: #6A1B9A !important;
  background-image: linear-gradient(135deg, #6A1B9A, #4A148C) !important;
  color: #ffffff !important;
  border: none !important;
  box-shadow: 0 4px 12px rgba(106,27,154,0.2) !important;
}
</style>
""",
        unsafe_allow_html=True,
    )


def _card_button_label(emoji: str, title: str, desc: str) -> str:
    if desc:
        return f"{emoji}\n\n{title}\n{desc}"
    return f"{emoji}\n\n{title}"


def render_step1() -> None:
    """第1步：选择场景。"""
    st.markdown(
        '<div class="guayupai-lead">'
        "当你搜遍小红书、问遍朋友，还是选不出来——<br>"
        "摇一卦，让直觉帮你拍板。"
        "</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        """
    <div class="step-title">
        <div class="big">此刻，你想探索什么？</div>
        <div class="small">选择一个与你当下状态共鸣的入口</div>
    </div>
    """,
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="guayupai-step-choices guayupai-step-choices--step1">',
        unsafe_allow_html=True,
    )

    c1, c2, c3 = st.columns(3)
    choices = [
        (c1, "🌙", "一个人", "静静理清思绪", "single"),
        (c2, "🫂", "和TA", "一起找个第三方视角", "dual"),
        (c3, "🌀", "随便看看", "让直觉指引你", "random"),
    ]

    for col, emoji, title, desc, value in choices:
        with col:
            if st.button(
                _card_button_label(emoji, title, desc),
                key=f"step1_{value}",
                use_container_width=True,
            ):
                st.session_state.flow_step = 2
                st.session_state.flow_choice = value
                st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)


def render_step2() -> None:
    """第2步：选择工具。"""
    choice = st.session_state.get("flow_choice")

    titles = {
        "single": "一个人时，你更想...",
        "dual": "和TA一起时，你们更想...",
        "random": "让直觉为你选择...",
    }

    st.markdown(
        f"""
    <div class="step-title">
        <div class="big">{titles.get(choice, "你想...")}</div>
    </div>
    """,
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="guayupai-step-choices guayupai-step-choices--step2">',
        unsafe_allow_html=True,
    )

    c1, c2 = st.columns(2)

    with c1:
        if st.button(
            _card_button_label("☯️", "摇铜钱（东方易经智慧）", ""),
            key="step2_iching",
            use_container_width=True,
        ):
            st.session_state.flow_step = 3
            st.session_state.flow_tool = "iching"
            st.rerun()

    with c2:
        if st.button(
            _card_button_label("🃏", "翻张牌（西方塔罗指引）", ""),
            key="step2_tarot",
            use_container_width=True,
        ):
            st.session_state.flow_step = 3
            st.session_state.flow_tool = "tarot"
            st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)


def format_mode_display_name(tool: str, is_dual: bool, is_random: bool) -> str:
    if is_random:
        scene = "随便看看"
    elif is_dual:
        scene = "双人"
    else:
        scene = "单人"
    kind = "易经" if tool == "iching" else "塔罗"
    return f"{scene}{kind}"


def _dilemma_input_label(is_dual: bool, is_random: bool = False) -> str:
    if is_random:
        return "✍️ 把让你纠结的事写在这里..."
    if is_dual:
        return "✍️ 把让你纠结的事写在这里（关系卡在暧昧期、异地恋方向迷茫...）"
    return "✍️ 把让你纠结的事写在这里（offer等待焦虑、人生方向选择...）"


def dilemma_storage_key(mode_key: str) -> str:
    return f"dilemma_text_{mode_key}"


def dilemma_widget_key(mode_key: str) -> str:
    return f"dilemma_widget_{mode_key}"


def get_user_dilemma(mode_key: str) -> str:
    return (st.session_state.get(dilemma_storage_key(mode_key)) or "").strip()


def _save_user_dilemma(mode_key: str, text: str) -> None:
    st.session_state[dilemma_storage_key(mode_key)] = str(text or "").strip()


def _persist_dilemma_from_widget(mode_key: str) -> None:
    """起卦/洗牌/解卦前同步输入框内容，避免点击按钮后问题丢失。"""
    wkey = dilemma_widget_key(mode_key)
    text = st.session_state.get(wkey, get_user_dilemma(mode_key))
    _save_user_dilemma(mode_key, text)
    _sync_dilemma_query_param(mode_key)


def _hide_dilemma_widget(mode_key: str) -> None:
    """起卦/洗牌开始后隐藏输入框（内容已存入 session）。"""
    st.session_state.pop(dilemma_widget_key(mode_key), None)


def _sync_dilemma_query_param(mode_key: str) -> None:
    """铺牌/解读流程中把问题写入 URL，整页刷新或选牌后仍能恢复。"""
    dilemma = get_user_dilemma(mode_key)
    if dilemma:
        st.query_params["dilemma"] = dilemma
    elif "dilemma" in st.query_params:
        del st.query_params["dilemma"]


def render_dilemma_textarea(is_dual: bool, mode_key: str, is_random: bool = False) -> str:
    """渲染纠结输入框，并持久化到 session（选牌链接跳转后仍能读到）。"""
    wkey = dilemma_widget_key(mode_key)
    sk = dilemma_storage_key(mode_key)

    if wkey not in st.session_state and st.session_state.get(sk):
        st.session_state[wkey] = st.session_state[sk]

    def _on_dilemma_change() -> None:
        _save_user_dilemma(mode_key, st.session_state.get(wkey, ""))
        _sync_dilemma_query_param(mode_key)

    st.text_area(
        _dilemma_input_label(is_dual, is_random),
        height=100,
        key=wkey,
        on_change=_on_dilemma_change,
    )
    _save_user_dilemma(mode_key, st.session_state.get(wkey, ""))
    _sync_dilemma_query_param(mode_key)
    return get_user_dilemma(mode_key)


def resolve_flow_mode(choice: str | None) -> tuple[str, bool, bool]:
    """返回 (mode_key, is_dual, is_random)。"""
    c = choice or "single"
    if c == "dual":
        return "dual", True, False
    if c == "random":
        return "random", False, True
    return "single", False, False


# ==========================================
# 11. 易经仪式页面
# ==========================================
def render_iching_ritual(
    api_key,
    mode_key: str = "single",
    is_dual: bool = False,
    is_random: bool = False,
):
    _inject_iching_ritual_button_css()
    ritual_key = f"iching_ritual_{mode_key}"
    gen = st.session_state.get(f"ritual_gen_{mode_key}", 0)

    if ritual_key not in st.session_state:
        st.session_state[ritual_key] = {
            "step": "idle",
            "iching_step": 0,
            "iching_yaos": [],
            "hex": None,
        }
    ritual = st.session_state[ritual_key]
    init_iching_ritual(ritual)

    uid = f"iching_{mode_key}"
    outcome = st.session_state.get(f"outcome_{uid}")

    if outcome:
        hex_result = outcome.get("hex_result") or ritual.get("hex")
        if hex_result:
            render_yao_stack(hex_result["yaos"], title="六爻齐备（自下而上）")
            render_hex_summary_card(hex_result)
        render_reading_panel(get_outcome_display_text(outcome), "iching")
        render_reading_outcome_footer(uid, "iching", ritual_key, mode_key)
        return

    if ritual["step"] == "idle":
        render_dilemma_textarea(is_dual, mode_key, is_random)
        st.markdown("""
        <div class="guayupai-ritual-panel guayupai-glass guayupai-ritual-panel--iching">
            <div class="guayupai-ritual-panel__icon">☯️</div>
            <div class="guayupai-ritual-panel__title card-title">金钱卦 · 六爻起卦</div>
            <div class="guayupai-ritual-panel__desc">3枚铜钱，分6次摇出，自下而上定六爻</div>
        </div>
        """, unsafe_allow_html=True)

        if st.button(
            "🔮 开始起卦",
            use_container_width=True,
            key=f"iching_start_{mode_key}_{gen}",
        ):
            _persist_dilemma_from_widget(mode_key)
            _hide_dilemma_widget(mode_key)
            ritual["step"] = "casting"
            ritual["iching_step"] = 0
            ritual["iching_yaos"] = []
            ritual["hex"] = None
            st.rerun()

    elif ritual["step"] == "casting":
        yaos = ritual["iching_yaos"]
        render_yao_stack(yaos)

        if yaos:
            last = yaos[-1]
            st.markdown(
                f"<div class='guayupai-coin-result coin-result'><div class='coin-coins'>"
                f"{format_coins_line(last['coins'])}</div>"
                f"<div class='coin-name'>第{last['position']}爻 · {last['name']}</div></div>",
                unsafe_allow_html=True,
            )
            if last["change"]:
                st.markdown(
                    "<p style='color:#E65100;font-weight:700;text-align:center;margin:8px 0;'>变爻</p>",
                    unsafe_allow_html=True,
                )

        next_yao = len(yaos) + 1
        if len(yaos) < 6:
            if st.button(
                f"🪙 摇第{next_yao}爻",
                use_container_width=True,
                key=f"iching_cast_{mode_key}_{len(yaos)}",
            ):
                yao = cast_yao()
                yao["position"] = next_yao
                ritual["iching_yaos"].append(yao)
                ritual["iching_step"] = len(ritual["iching_yaos"])
                if len(ritual["iching_yaos"]) == 6:
                    ritual["hex"] = build_hex_result_from_yaos(ritual["iching_yaos"])
                    ritual["step"] = "complete"
                st.rerun()

    elif ritual["step"] == "complete":
        hex_result = ritual["hex"]
        ben_gua = hex_result["ben_gua"]

        render_yao_stack(hex_result["yaos"], title="六爻齐备（自下而上）")
        render_hex_summary_card(hex_result)

        if st.button(
            "✨ 看看结果",
            use_container_width=True,
            key=f"iching_read_{mode_key}_{gen}",
        ):
            _persist_dilemma_from_widget(mode_key)
            user_input = get_user_dilemma(mode_key)
            hex_result = ritual["hex"]
            ben_gua = hex_result["ben_gua"]
            ben_data = ICHING_HEXAGRAMS.get(ben_gua, {})
            if not api_key:
                st.error("请先配置 DeepSeek API Key")
                return

            passed, sanitized, err = sanitize_input(user_input)
            if not passed:
                st.error(err)
                return

            is_empty = not sanitized
            system_prompt = build_iching_prompt(
                hex_result, sanitized, is_empty, is_dual, is_random
            )

            client = get_client(api_key)
            ok, full_text = run_reading(
                client, system_prompt, "☯️ 正在解卦...", tool="iching"
            )

            if ok:
                mode_name = format_mode_display_name("iching", is_dual, is_random)
                uid = f"iching_{mode_key}"
                save_reading_outcome(
                    uid=uid,
                    tool="iching",
                    full_text=full_text,
                    user_input=sanitized,
                    result_name=ben_gua,
                    mode_name=mode_name,
                    hex_result=hex_result,
                )
                st.rerun()


# ==========================================
# 12. 塔罗仪式页面
# ==========================================
def _qp_get(key: str) -> str | None:
    val = st.query_params.get(key)
    if val is None:
        return None
    if isinstance(val, list):
        return val[0] if val else None
    return str(val)


def _clear_tarot_query_params() -> None:
    for key in ("tarot_pick", "tarot_mode", "tarot_seed", "dilemma"):
        if key in st.query_params:
            del st.query_params[key]


def _apply_ritual_reset_from_query_params() -> None:
    """分享页点「再来一卦/再抽一张」后，回到对应工具起始并重置仪式。"""
    if _qp_get("ritual_reset") != "1":
        return
    tool = _qp_get("flow_tool")
    mode_key = _qp_get("flow_choice") or "single"
    if tool == "iching":
        reset_iching_ritual(f"iching_ritual_{mode_key}", mode_key)
    elif tool == "tarot":
        reset_tarot_ritual(f"tarot_ritual_{mode_key}", mode_key)
    if "ritual_reset" in st.query_params:
        del st.query_params["ritual_reset"]


def _sync_flow_from_query_params() -> None:
    """点击牌后整页刷新时，从 URL 恢复流程步骤（避免跳回首页）。"""
    step = _qp_get("flow_step")
    if step is not None:
        normalized = _normalize_flow_step(step)
        if normalized is not None:
            st.session_state.flow_step = normalized
    tool = _qp_get("flow_tool")
    if tool:
        st.session_state.flow_tool = tool
    choice = _qp_get("flow_choice")
    if choice:
        st.session_state.flow_choice = choice
    dilemma = _qp_get("dilemma")
    mode = _qp_get("tarot_mode") or _qp_get("flow_choice")
    if dilemma and mode:
        _save_user_dilemma(mode, dilemma)
        wkey = dilemma_widget_key(mode)
        if wkey not in st.session_state:
            st.session_state[wkey] = dilemma


def _set_tarot_spread_query_params(mode_key: str, seed: int) -> None:
    st.query_params["flow_step"] = "3"
    st.query_params["flow_tool"] = "tarot"
    st.query_params["flow_choice"] = mode_key
    st.query_params["tarot_seed"] = str(seed)
    st.query_params["tarot_mode"] = mode_key
    _sync_dilemma_query_param(mode_key)
    if "tarot_pick" in st.query_params:
        del st.query_params["tarot_pick"]


def _tarot_pick_href(idx: int, mode_key: str, ritual: dict) -> str:
    """选牌链接：带上流程步骤 + 洗牌种子 + 用户问题，刷新后仍能解读。"""
    params: dict[str, str] = {
        "flow_step": "3",
        "flow_tool": "tarot",
        "flow_choice": mode_key,
        "tarot_pick": str(idx),
        "tarot_mode": mode_key,
    }
    seed = ritual.get("seed")
    if seed is not None:
        params["tarot_seed"] = str(seed)
    dilemma = get_user_dilemma(mode_key)
    if dilemma:
        params["dilemma"] = dilemma
    return f"?{urlencode(params)}"


def _restore_tarot_ritual_from_query(ritual: dict, mode_key: str) -> None:
    """仅在有选牌/铺牌流程时恢复牌堆，避免「再来一张」后 URL 残留 seed 又跳回铺牌页。"""
    if _qp_get("tarot_mode") not in (None, mode_key):
        return
    if ritual.get("step") == "idle" and _qp_get("tarot_pick") is None:
        return
    seed_s = _qp_get("tarot_seed")
    if not seed_s:
        return
    try:
        seed = int(seed_s)
    except ValueError:
        return
    if ritual.get("deck") is None:
        ritual["deck"] = shuffle_tarot(seed)
        ritual["seed"] = seed
        if ritual.get("step") in ("idle", "spreading"):
            ritual["step"] = "spread"


def reset_iching_ritual(ritual_key: str, mode_key: str) -> None:
    st.session_state[ritual_key] = {
        "step": "idle",
        "iching_step": 0,
        "iching_yaos": [],
        "hex": None,
    }
    uid = f"iching_{mode_key}"
    st.session_state.pop(f"outcome_{uid}", None)
    st.session_state.pop(dilemma_storage_key(mode_key), None)
    st.session_state.pop(dilemma_widget_key(mode_key), None)
    st.session_state[f"ritual_gen_{mode_key}"] = (
        st.session_state.get(f"ritual_gen_{mode_key}", 0) + 1
    )


def reset_tarot_ritual(ritual_key: str, mode_key: str) -> None:
    _clear_tarot_query_params()
    st.session_state[ritual_key] = {
        "step": "idle",
        "deck": None,
        "selected": None,
        "flipped": False,
        "show_deal": False,
        "seed": None,
        "auto_read": False,
    }
    uid = f"tarot_{mode_key}"
    st.session_state.pop(f"outcome_{uid}", None)
    st.session_state.pop(f"tarot_busy_{mode_key}", None)
    st.session_state.pop(dilemma_storage_key(mode_key), None)
    st.session_state.pop(dilemma_widget_key(mode_key), None)
    st.session_state[f"ritual_gen_{mode_key}"] = (
        st.session_state.get(f"ritual_gen_{mode_key}", 0) + 1
    )


def _apply_tarot_query_pick(ritual: dict, mode_key: str) -> None:
    if _qp_get("tarot_mode") != mode_key:
        return
    pick = _qp_get("tarot_pick")
    if pick is None or ritual.get("deck") is None:
        return
    try:
        idx = int(pick)
    except (TypeError, ValueError):
        return
    if 0 <= idx < len(ritual["deck"]):
        ritual["selected"] = idx
        ritual["flipped"] = True
        ritual["show_deal"] = False
        ritual["step"] = "revealed"
        ritual["auto_read"] = True


def _init_tarot_ritual(ritual: dict) -> None:
    ritual.setdefault("flipped", False)
    ritual.setdefault("show_deal", False)
    ritual.setdefault("seed", None)
    ritual.setdefault("auto_read", False)


def _inject_tarot_image_preload(deck: list) -> None:
    seen: set[str] = set()
    tags = []
    for card in deck:
        url = card.get("image", "")
        if url and url not in seen:
            seen.add(url)
            tags.append(f'<link rel="preload" as="image" href="{html.escape(url)}">')
    if tags:
        st.markdown("".join(tags), unsafe_allow_html=True)


def render_tarot_result_card(card: dict) -> None:
    """塔罗翻牌结果卡（对应易经 render_hex_summary_card / result-east）。"""
    idx = _tarot_card_index(card.get("name", ""))
    roman = html.escape(TAROT_ROMAN[idx])
    name = html.escape(card.get("name", ""))
    img = html.escape(card.get("image", ""))
    element = html.escape(card.get("element", "未知"))
    kw = "，".join(html.escape(k) for k in (card.get("keywords") or [])[:3])
    meta = f"{element}元素 · {kw}" if kw else f"{element}元素"

    st.markdown(
        f"""
<div class="result-card result-west guayupai-tarot-result-card" style="
  width:100%;max-width:100%;
  border-left:5px solid #6A1B9A !important;
  color:#4A148C !important;
">
  <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:14px;">
    <span class="card-title tarot-result-title" style="color:#4A148C;">你抽到了：</span>
    <span style="font-size:14px;color:rgba(74,20,140,0.55);">{roman}</span>
  </div>
  <div style="display:flex;flex-direction:column;align-items:center;text-align:center;">
    <img class="tarot-result-img" src="{img}" alt="{name}" loading="eager"
      style="width:100px;border-radius:10px;border:2px solid #b39ddb;
      box-shadow:0 4px 12px rgba(106,27,154,0.2);object-fit:cover;display:block;"
      onerror="this.style.display='none';this.nextElementSibling.style.display='block';"/>
    <div style="display:none;font-size:18px;font-weight:600;color:#4A148C;margin:12px 0;">{name}</div>
    <div style="font-size:20px;font-weight:600;color:#4A148C;margin-top:10px;">{name}</div>
    <div style="font-size:13px;color:rgba(74,20,140,0.72);margin-top:6px;">{meta}</div>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )


def _inject_tarot_pick_dilemma_script() -> None:
    """选牌链接点击时从页面输入框读取最新问题（避免未失焦时 href 里仍是旧值）。"""
    st.markdown(
        """
<script>
(function () {
  function readDilemma() {
    var areas = document.querySelectorAll('[data-testid="stTextArea"] textarea');
    if (!areas.length) return "";
    return (areas[areas.length - 1].value || "").trim();
  }
  document.querySelectorAll("a.tarot-card-3d").forEach(function (a) {
    a.addEventListener("click", function () {
      var text = readDilemma();
      if (!text) return;
      try {
        var u = new URL(a.getAttribute("href"), window.location.origin);
        u.searchParams.set("dilemma", text);
        a.setAttribute("href", u.pathname + u.search);
      } catch (e) {}
    }, true);
  });
})();
</script>
""",
        unsafe_allow_html=True,
    )


def _render_tarot_fan_stage(deck: list, mode_key: str, ritual: dict) -> None:
    _save_user_dilemma(mode_key, st.session_state.get(dilemma_widget_key(mode_key), ""))
    _sync_dilemma_query_param(mode_key)
    _inject_tarot_image_preload(deck)
    selected = ritual.get("selected")
    if selected is None:
        hint = "🌙 滑动浏览，点击一张与你能量共振的牌"
    else:
        hint = "✨ 已为你锁定此牌，解读在下方生成"
    st.markdown(
        f"<p style='text-align:center;color:#4A148C;font-size:14px;margin:8px 0 4px;'>{hint}</p>",
        unsafe_allow_html=True,
    )
    render_tarot_fan(deck, mode_key, ritual)
    if selected is None:
        _inject_tarot_pick_dilemma_script()
    if ritual.get("show_deal"):
        ritual["show_deal"] = False


def _execute_tarot_auto_read(
    api_key: str,
    user_input: str,
    card: dict,
    is_dual: bool,
    is_random: bool,
    mode_key: str,
    ritual: dict,
) -> None:
    if not api_key:
        st.error("请先配置 DeepSeek API Key")
        ritual["auto_read"] = False
        return

    passed, sanitized, err = sanitize_input(user_input)
    if not passed:
        st.error(err)
        ritual["auto_read"] = False
        return

    busy_key = f"tarot_busy_{mode_key}"
    if st.session_state.get(busy_key):
        return

    ritual["step"] = "revealed"
    st.session_state[busy_key] = True

    is_empty = not sanitized
    system_prompt = build_tarot_prompt(
        card, sanitized, is_empty, is_dual, is_random
    )
    client = get_client(api_key)
    ok, full_text = run_reading(
        client,
        system_prompt,
        show_spinner=False,
        tool="tarot",
    )

    st.session_state.pop(busy_key, None)
    ritual["auto_read"] = False
    if "tarot_pick" in st.query_params:
        del st.query_params["tarot_pick"]
    _sync_dilemma_query_param(mode_key)

    if ok:
        ritual["step"] = "reading"
        mode_name = format_mode_display_name("tarot", is_dual, is_random)
        uid = f"tarot_{mode_key}"
        save_reading_outcome(
            uid=uid,
            tool="tarot",
            full_text=full_text,
            user_input=sanitized,
            result_name=card["name"],
            mode_name=mode_name,
            card=card,
        )
        st.rerun()
    else:
        ritual["step"] = "spread"


def _build_tarot_fan_markup(
    deck: list,
    mode_key: str,
    ritual: dict,
    selected: int | None,
    flipped: bool,
    show_deal: bool,
) -> str:
    """主页面 HTML（非 iframe），用 <a href> 触发 Streamlit query_params 选牌。"""
    n = len(deck)
    mid = (n - 1) / 2.0
    pickable = selected is None
    cards = []
    for i, card in enumerate(deck):
        offset = i - mid
        rotate = max(-12, min(12, int(round(offset * 2))))
        margin = "-15px" if i > 0 else "0"
        name = html.escape(card["name"])
        img = html.escape(card.get("image", ""))
        is_sel = selected == i
        flipped_cls = " flipped" if is_sel and flipped else ""
        selected_cls = " selected" if is_sel and flipped else ""
        dim_cls = " dimmed" if selected is not None and not is_sel else ""
        inner = (
            f'<div class="tarot-card-inner">'
            f'<div class="card-face-back"><span class="back-symbol">🌙</span></div>'
            f'<div class="card-face-front">'
            f'<img src="{img}" alt="{name}" loading="lazy" '
            f'onerror="this.style.display=\'none\';this.nextElementSibling.style.display=\'flex\';" />'
            f'<span class="front-fallback">{name}</span>'
            f"</div></div>"
        )
        style = f"--rot:{rotate}deg;margin-left:{margin};z-index:{i + 1};"
        cls = f"tarot-card-3d{flipped_cls}{selected_cls}{dim_cls}"
        if pickable:
            href = html.escape(_tarot_pick_href(i, mode_key, ritual), quote=True)
            cards.append(
                f'<a class="{cls}" href="{href}" style="{style}">{inner}</a>'
            )
        else:
            cards.append(f'<div class="{cls}" style="{style}">{inner}</div>')

    deal_cls = " tarot-fan--deal-in" if show_deal else ""
    cards_html = "".join(cards)
    return f"""
<style>
.tarot-fan-wrap {{
  width: 100%;
  max-height: 200px;
  padding: 8px 4px 12px;
  overflow: hidden;
}}
.tarot-fan-scroll {{
  display: flex;
  align-items: flex-end;
  justify-content: center;
  min-width: min-content;
  padding: 0 24px 6px;
  overflow-x: auto;
  overflow-y: hidden;
  -webkit-overflow-scrolling: touch;
  scrollbar-width: none;
}}
.tarot-fan-scroll::-webkit-scrollbar {{ display: none; }}
.tarot-fan {{
  display: flex;
  align-items: flex-end;
  justify-content: center;
  padding-bottom: 4px;
}}
.tarot-fan--deal-in .tarot-card-3d {{
  animation: tarotDealIn 0.55s ease-out backwards;
}}
.tarot-fan--deal-in .tarot-card-3d:nth-child(odd) {{ animation-delay: 0.02s; }}
.tarot-fan--deal-in .tarot-card-3d:nth-child(even) {{ animation-delay: 0.05s; }}
@keyframes tarotDealIn {{
  from {{ transform: rotate(var(--rot)) scale(0.5); opacity: 0; }}
  to {{ transform: rotate(var(--rot)) scale(1); opacity: 1; }}
}}
a.tarot-card-3d {{
  text-decoration: none;
  color: inherit;
  display: block;
  flex: 0 0 auto;
}}
.tarot-card-3d {{
  flex: 0 0 auto;
  width: 80px;
  height: 120px;
  perspective: 800px;
  cursor: pointer;
  transform: rotate(var(--rot));
  transition: transform 0.35s ease, opacity 0.35s ease, margin 0.35s ease;
}}
.tarot-card-3d.dimmed {{ opacity: 0.3; pointer-events: none; }}
.tarot-card-3d.selected {{
  z-index: 50 !important;
  transform: rotate(0deg) scale(1.5);
  margin-left: 8px !important;
  margin-right: 8px !important;
}}
.tarot-card-inner {{
  width: 100%;
  height: 100%;
  position: relative;
  transform-style: preserve-3d;
  transition: transform 0.6s ease;
}}
.tarot-card-3d.flipped .tarot-card-inner {{
  transform: rotateY(180deg);
  animation: tarotFlipIn 0.6s ease;
}}
@keyframes tarotFlipIn {{
  from {{ transform: rotateY(0deg); }}
  to {{ transform: rotateY(180deg); }}
}}
.card-face-back, .card-face-front {{
  position: absolute;
  inset: 0;
  border-radius: 8px;
  backface-visibility: hidden;
  overflow: hidden;
}}
.card-face-back {{
  background: radial-gradient(circle at 35% 30%, #5c4bb5 0%, #2a1f6e 55%, #1a1248 100%);
  border: 1px solid #d4af37;
  display: flex;
  align-items: center;
  justify-content: center;
  box-shadow: 0 4px 12px rgba(74,20,140,0.35);
}}
.back-symbol {{
  font-size: 28px;
  opacity: 0.85;
  text-shadow: 0 0 12px rgba(255,215,0,0.4);
}}
.card-face-front {{
  transform: rotateY(180deg);
  background: #1a1248;
  border: 1px solid #d4af37;
  display: flex;
  align-items: center;
  justify-content: center;
}}
.card-face-front img {{
  width: 100%;
  height: 100%;
  object-fit: cover;
  border-radius: 7px;
}}
.front-fallback {{
  display: none;
  width: 100%;
  height: 100%;
  align-items: center;
  justify-content: center;
  padding: 6px;
  text-align: center;
  font-size: 11px;
  font-weight: 700;
  color: #f3e5f5;
  line-height: 1.3;
}}
@media (max-width: 640px) {{
  .tarot-card-3d {{ width: 60px; height: 90px; }}
  .back-symbol {{ font-size: 22px; }}
}}
</style>
<div class="tarot-fan-wrap">
  <div class="tarot-fan-scroll">
    <div class="tarot-fan{deal_cls}">{cards_html}</div>
  </div>
</div>
"""


def render_tarot_fan(deck: list, mode_key: str, ritual: dict) -> None:
    markup = _build_tarot_fan_markup(
        deck,
        mode_key,
        ritual,
        ritual.get("selected"),
        ritual.get("flipped", False),
        ritual.get("show_deal", False),
    )
    st.markdown(markup, unsafe_allow_html=True)


def render_tarot_ritual(
    api_key,
    mode_key: str = "single",
    is_dual: bool = False,
    is_random: bool = False,
):
    _inject_tarot_ritual_button_css()
    ritual_key = f"tarot_ritual_{mode_key}"
    gen = st.session_state.get(f"ritual_gen_{mode_key}", 0)

    if ritual_key not in st.session_state:
        st.session_state[ritual_key] = {
            "step": "idle",
            "deck": None,
            "selected": None,
            "flipped": False,
            "show_deal": False,
            "seed": None,
            "auto_read": False,
        }
    ritual = st.session_state[ritual_key]
    _init_tarot_ritual(ritual)
    _restore_tarot_ritual_from_query(ritual, mode_key)
    _apply_tarot_query_pick(ritual, mode_key)

    uid = f"tarot_{mode_key}"
    outcome = st.session_state.get(f"outcome_{uid}")
    selected = ritual.get("selected")

    if outcome:
        deck = ritual.get("deck")
        if deck:
            _render_tarot_fan_stage(deck, mode_key, ritual)
        result_card = outcome.get("card")
        if not result_card and deck and ritual.get("selected") is not None:
            result_card = deck[ritual["selected"]]
        if result_card:
            render_tarot_result_card(result_card)
        render_reading_panel(get_outcome_display_text(outcome), "tarot")
        render_reading_outcome_footer(uid, "tarot", ritual_key, mode_key)
        return

    if ritual["step"] == "idle":
        render_dilemma_textarea(is_dual, mode_key, is_random)
        st.markdown("""
        <div class="guayupai-ritual-panel guayupai-glass guayupai-ritual-panel--tarot">
            <div class="guayupai-ritual-panel__icon">🃏</div>
            <div class="guayupai-ritual-panel__title card-title">22张韦特塔罗 · 大阿卡纳</div>
            <div class="guayupai-ritual-panel__desc">扇形铺牌，滑动浏览，凭直觉选一张</div>
        </div>
        """, unsafe_allow_html=True)

        if st.button(
            "🔮 开始洗牌",
            use_container_width=True,
            key=f"tarot_start_{mode_key}_{gen}",
        ):
            _persist_dilemma_from_widget(mode_key)
            _hide_dilemma_widget(mode_key)
            seed = secrets.randbelow(2**31)
            ritual["step"] = "spreading"
            ritual["seed"] = seed
            ritual["deck"] = shuffle_tarot(seed)
            ritual["selected"] = None
            ritual["flipped"] = False
            ritual["auto_read"] = False
            ritual["show_deal"] = True
            st.session_state.pop(f"tarot_busy_{mode_key}", None)
            _set_tarot_spread_query_params(mode_key, seed)
            st.rerun()

    elif ritual["step"] in ("spreading", "spread", "revealed", "reading"):
        deck = ritual["deck"]
        if not deck:
            ritual["step"] = "idle"
            st.rerun()
            return

        if ritual["step"] == "spreading":
            ritual["step"] = "spread"

        selected = ritual.get("selected")

        # ① 未选牌：扇形抽牌
        if selected is None:
            _render_tarot_fan_stage(deck, mode_key, ritual)
            return

        # ② 已选牌：保留牌阵 → 紫色结果卡 → 下方牌面牌义/解读
        if ritual["step"] == "spread" and ritual.get("auto_read"):
            ritual["step"] = "revealed"
        card = deck[selected]
        _persist_dilemma_from_widget(mode_key)
        user_input = get_user_dilemma(mode_key)

        _render_tarot_fan_stage(deck, mode_key, ritual)
        render_tarot_result_card(card)

        if ritual.get("auto_read"):
            _execute_tarot_auto_read(
                api_key, user_input, card, is_dual, is_random, mode_key, ritual
            )


# ==========================================
# 13. 主入口
# ==========================================
def main():
    st.set_page_config(
        page_title="卦与牌",
        page_icon="🔮",
        layout="centered",
        initial_sidebar_state="collapsed",
    )
    if _qp_get("view") == "share":
        render_css(share_page=True)
        render_share_standalone_page()
        return

    _sync_flow_from_query_params()
    _apply_ritual_reset_from_query_params()

    step = _coerce_flow_step(st.session_state.get("flow_step", 1))
    st.session_state.flow_step = step
    theme_tool = st.session_state.get("flow_tool") if _is_ritual_flow_step(step) else None
    render_css(theme_tool)
    _inject_streamlit_shell_fix()

    api_key = get_api_key()

    c1, c2 = st.columns([1, 10])
    with c1:
        if step > 1 and st.button("←", type="secondary", key="back_top"):
            reset_flow_to_step1()
            st.rerun()
    with c2:
        st.markdown(
            '<div class="guayupai-h1-wrap">'
            '<h1 class="guayupai-h1">🔮 <span class="brand-name">卦与牌</span></h1>'
            '<div class="guayupai-h1-line"></div>'
            "</div>",
            unsafe_allow_html=True,
        )

    if not api_key:
        st.info("请设置环境变量 DEEPSEEK_API_KEY，或在 Streamlit secrets 中配置。")

    # 流程控制
    choice = st.session_state.get("flow_choice")
    tool = st.session_state.get("flow_tool")

    if step == 1:
        render_step1()
    elif step == 2:
        render_step2()
    elif _is_ritual_flow_step(step):
        mode_key, is_dual, is_random = resolve_flow_mode(choice)
        # 「随便看看」只影响解读语气，不覆盖第二步用户选的易经/塔罗

        if tool == "iching":
            render_iching_ritual(api_key, mode_key, is_dual, is_random)
        elif tool == "tarot":
            render_tarot_ritual(api_key, mode_key, is_dual, is_random)


main()
