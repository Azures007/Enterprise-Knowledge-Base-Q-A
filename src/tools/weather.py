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
    """中文城市名 → 经纬度。"""
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
