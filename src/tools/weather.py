"""
=============================================================================
天气查询工具

基于 Open-Meteo 免费天气 API（无需 API Key）：
    - 天气数据:  https://api.open-meteo.com/v1/forecast
    - 地理编码:  https://geocoding-api.open-meteo.com/v1/search （支持中文城市名）

流程：中文城市名 → 地理编码拿经纬度 → 查询实时天气 + 未来 N 天预报。

设计要点：
    - httpx 与应用同栈，能继承系统代理（与 BailianLLM 一致）
    - 15 秒超时防止卡住工具循环；失败返回 {"error": ...} 让 LLM 自愈
    - WMO 天气代码映射为人类可读的中文描述
=============================================================================
"""

import httpx

from src.utils.logger import setup_logger

logger = setup_logger(__name__)

# Open-Meteo API 端点
GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"
WEATHER_URL = "https://api.open-meteo.com/v1/forecast"

# 请求超时（秒）
TIMEOUT = 15

# ---------------------------------------------------------------------------
# 本地中国城市经纬度映射（兜底）
#
# Open-Meteo 地理编码对部分中文地名支持不佳（如"晋江""泉州"返回 0 结果）。
# 查无结果时回退到此表，保证常见中国城市可正常查询。
# 数据源：约略城市中心坐标（纬度, 经度）。
# ---------------------------------------------------------------------------
CN_CITIES: dict[str, tuple[float, float]] = {
    # 直辖市
    "北京": (39.9042, 116.4074),
    "上海": (31.2304, 121.4737),
    "天津": (39.0842, 117.2010),
    "重庆": (29.5630, 106.5516),
    # 省会 / 自治区首府
    "广州": (23.1291, 113.2644),
    "杭州": (30.2741, 120.1551),
    "南京": (32.0603, 118.7969),
    "武汉": (30.5928, 114.3055),
    "成都": (30.5728, 104.0668),
    "西安": (34.3416, 108.9398),
    "郑州": (34.7466, 113.6254),
    "济南": (36.6512, 117.1201),
    "沈阳": (41.8057, 123.4315),
    "长春": (43.8171, 125.3235),
    "哈尔滨": (45.8038, 126.5349),
    "石家庄": (38.0428, 114.5149),
    "太原": (37.8706, 112.5489),
    "呼和浩特": (40.8415, 111.7519),
    "兰州": (36.0611, 103.8343),
    "西宁": (36.6171, 101.7782),
    "银川": (38.4872, 106.2309),
    "乌鲁木齐": (43.8256, 87.6168),
    "拉萨": (29.6520, 91.1721),
    "昆明": (25.0389, 102.7183),
    "贵阳": (26.6470, 106.6302),
    "南宁": (22.8170, 108.3665),
    "海口": (20.0440, 110.1999),
    "福州": (26.0745, 119.2965),
    "长沙": (28.2282, 112.9388),
    "南昌": (28.6829, 115.8582),
    "合肥": (31.8206, 117.2272),
    # 计划单列市 / 经济发达城市
    "深圳": (22.5431, 114.0579),
    "青岛": (36.0671, 120.3826),
    "大连": (38.9140, 121.6147),
    "厦门": (24.4798, 118.0894),
    "宁波": (29.8683, 121.5440),
    "苏州": (31.2989, 120.5853),
    "无锡": (31.4912, 120.3119),
    "温州": (28.0000, 120.6721),
    "佛山": (23.0218, 113.1219),
    "东莞": (23.0207, 113.7518),
    "珠海": (22.2707, 113.5767),
    "中山": (22.5176, 113.3928),
    "惠州": (23.1115, 114.4152),
    # 常见查询（Open-Meteo 中文名识别不到的重点补充）
    "泉州": (24.9139, 118.5858),
    "晋江": (24.8198, 118.5742),
    "泉州晋江": (24.8198, 118.5742),
    "莆田": (25.4540, 119.0078),
    "漳州": (24.5130, 117.6470),
    "宁德": (26.6657, 119.5277),
    "龙岩": (25.0916, 117.0172),
    "三明": (26.2636, 117.6387),
    "南平": (26.6418, 118.1781),
    "金华": (29.0791, 119.6474),
    "台州": (28.6560, 121.4206),
    "嘉兴": (30.7450, 120.7562),
    "绍兴": (30.0303, 120.5802),
    "湖州": (30.8924, 120.0868),
    "芜湖": (31.3528, 118.4331),
    "扬州": (32.3932, 119.4127),
    "徐州": (34.2044, 117.2857),
    "常州": (31.8107, 119.9741),
    "南通": (31.9802, 120.8943),
    "盐城": (33.3495, 120.1627),
    "洛阳": (34.6181, 112.4540),
    "烟台": (37.4638, 121.4479),
    "潍坊": (36.7069, 119.1618),
    "临沂": (35.1047, 118.3564),
    "唐山": (39.6305, 118.1802),
    "保定": (38.8740, 115.4646),
    "廊坊": (39.5378, 116.6837),
    "桂林": (25.2736, 110.2900),
    "柳州": (24.3264, 109.4158),
    "三亚": (18.2528, 109.5119),
    "咸阳": (34.3294, 108.7093),
    "绵阳": (31.4680, 104.6796),
    "宜宾": (28.7521, 104.6430),
    "泸州": (28.8717, 105.4423),
    "遵义": (27.7256, 106.9272),
    "襄阳": (32.0091, 112.1226),
    "宜昌": (30.6919, 111.2865),
    "岳阳": (29.3572, 113.1289),
    "株洲": (27.8274, 113.1340),
    "汕头": (23.3535, 116.6820),
    "湛江": (21.2707, 110.3594),
    "茂名": (21.6629, 110.9256),
    "江门": (22.5787, 113.0815),
    "北海": (21.4810, 109.1206),
    "大理": (25.6065, 100.2676),
    "丽江": (26.8721, 100.2300),
    "黄山": (29.7147, 118.3376),
    "张家界": (29.1170, 110.4792),
    "威海": (37.5131, 122.1204),
    "日照": (35.4167, 119.5269),
    "徐州": (34.2044, 117.2857),
}

# 行政后缀：查询时去掉再匹配本地表（如"晋江市"→"晋江"）
# 注意：不含"州"——广州/泉州等大量城市名本身以"州"结尾，去掉会导致匹配失败
_CITY_SUFFIXES = ("市", "县", "区", "盟")


def _normalize_city_name(name: str) -> str:
    """去掉行政后缀，返回规范城市名。"""
    name = name.strip()
    for suf in _CITY_SUFFIXES:
        if name.endswith(suf) and len(name) > len(suf):
            return name[: -len(suf)]
    return name

# WMO 天气代码 → 中文描述（Open-Meteo 返回的 weather_code）
WMO_CODES = {
    0: "晴",
    1: "大部晴朗",
    2: "局部多云",
    3: "阴天",
    45: "雾",
    48: "雾凇",
    51: "小毛毛雨",
    53: "毛毛雨",
    55: "大毛毛雨",
    56: "冻毛毛雨",
    57: "强冻毛毛雨",
    61: "小雨",
    63: "中雨",
    65: "大雨",
    66: "冻雨",
    67: "强冻雨",
    71: "小雪",
    73: "中雪",
    75: "大雪",
    77: "雪粒",
    80: "小阵雨",
    81: "中阵雨",
    82: "强阵雨",
    85: "小阵雪",
    86: "强阵雪",
    95: "雷阵雨",
    96: "雷阵雨伴小冰雹",
    99: "雷阵雨伴强冰雹",
}


def _geocode(city: str) -> dict | None:
    """中文城市名 → 经纬度。

    优先级：本地 CN_CITIES 表 → Open-Meteo 地理编码（支持中文/英文）。
    Open-Meteo 对部分中文地名（如"晋江""泉州"）识别不到，故用本地表兜底。
    """
    norm = _normalize_city_name(city)

    # 1. 本地表兜底
    if norm in CN_CITIES:
        lat, lon = CN_CITIES[norm]
        return {
            "name": norm,
            "country": "中国",
            "latitude": lat,
            "longitude": lon,
            "timezone": "auto",
        }

    # 2. Open-Meteo 地理编码
    try:
        resp = httpx.get(
            GEOCODING_URL,
            params={"name": city, "count": 1, "language": "zh", "format": "json"},
            timeout=TIMEOUT,
        )
        resp.raise_for_status()
        results = resp.json().get("results") or []
        if not results:
            return None
        r = results[0]
        return {
            "name": r.get("name", city),
            "country": r.get("country", ""),
            "latitude": r["latitude"],
            "longitude": r["longitude"],
            "timezone": r.get("timezone", "auto"),
        }
    except Exception as e:
        logger.warning(f"地理编码失败 [{city}]: {e}")
        return None


def _weather_code_text(code) -> str:
    """WMO 天气代码 → 中文描述。"""
    try:
        return WMO_CODES.get(int(code), f"代码 {code}")
    except (ValueError, TypeError):
        return f"代码 {code}"


async def get_weather(city: str, days: int = 1) -> dict:
    """
    查询指定城市的实时天气与未来预报。

    Args:
        city: 城市名（中文/英文均可，如"北京"、"shanghai"）
        days: 预报天数 1~7，默认 1（仅当天）

    Returns:
        dict:
            - city / country / latitude / longitude
            - current: {temperature, weather, weather_code, time}
            - daily:   [{date, temp_max, temp_min, weather}]
            - error（失败时）
    """
    days = max(1, min(int(days or 1), 7))

    loc = _geocode(city)
    if loc is None:
        return {"error": f"无法定位城市 '{city}'，请检查城市名是否正确"}

    try:
        params = {
            "latitude": loc["latitude"],
            "longitude": loc["longitude"],
            "timezone": loc.get("timezone", "auto"),
            "current": "temperature_2m,weather_code,wind_speed_10m,relative_humidity_2m",
            "daily": "temperature_2m_max,temperature_2m_min,weather_code",
            "forecast_days": days,
        }
        resp = httpx.get(WEATHER_URL, params=params, timeout=TIMEOUT)
        resp.raise_for_status()
        data = resp.json()

        # 实时天气
        cur = data.get("current", {})
        current = {
            "temperature": cur.get("temperature_2m"),
            "weather": _weather_code_text(cur.get("weather_code")),
            "weather_code": cur.get("weather_code"),
            "wind_speed": cur.get("wind_speed_10m"),
            "humidity": cur.get("relative_humidity_2m"),
            "time": cur.get("time", ""),
        }

        # 未来 N 天预报
        daily = data.get("daily", {})
        daily_list = []
        for i in range(min(days, len(daily.get("time", [])))):
            daily_list.append({
                "date": daily.get("time", [])[i],
                "temp_max": daily.get("temperature_2m_max", [])[i],
                "temp_min": daily.get("temperature_2m_min", [])[i],
                "weather": _weather_code_text(daily.get("weather_code", [])[i]),
            })

        return {
            "city": loc["name"],
            "country": loc.get("country", ""),
            "latitude": loc["latitude"],
            "longitude": loc["longitude"],
            "current": current,
            "daily": daily_list,
        }
    except httpx.TimeoutException:
        return {"error": "天气服务请求超时，请稍后重试"}
    except Exception as e:
        logger.warning(f"天气查询失败 [{city}]: {e}")
        return {"error": f"天气查询失败: {e}"}
