"""Sayaç sorularını uygun MCP aracına yönlendiren servis."""

import re
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Tuple
from zoneinfo import ZoneInfo

from services.mcp_service import MCPClient, result_data, tool_error_message


METRIC_LABELS = {
    "energy_total_kwh": ("Toplam enerji", "kWh"),
    "daily_energy_kwh": ("Günlük enerji", "kWh"),
    "weekly_energy_kwh": ("Haftalık enerji", "kWh"),
    "monthly_energy_kwh": ("Aylık enerji", "kWh"),
    "voltage_l1_v": ("L1 gerilim", "V"),
    "voltage_l2_v": ("L2 gerilim", "V"),
    "voltage_l3_v": ("L3 gerilim", "V"),
    "current_l1_a": ("L1 akım", "A"),
    "current_l2_a": ("L2 akım", "A"),
    "current_l3_a": ("L3 akım", "A"),
    "power_factor_l1": ("L1 güç faktörü", ""),
    "power_factor_l2": ("L2 güç faktörü", ""),
    "power_factor_l3": ("L3 güç faktörü", ""),
    "frequency_hz": ("Frekans", "Hz"),
}

LOCAL_TIMEZONE = ZoneInfo("Europe/Istanbul")


def today_time_range(query: str) -> Tuple[datetime, datetime] | None:
    """'Bugün 9-10' benzeri bir ifadeyi yerel zaman aralığına dönüştürür."""
    normalized = query.casefold()
    if "bugün" not in normalized and "bugun" not in normalized:
        return None
    match = re.search(
        r"(\d{1,2})(?:[.:](\d{2}))?\s*"
        r"(?:-|–|—|ile)\s*"
        r"(\d{1,2})(?:[.:](\d{2}))?",
        normalized,
    )
    if not match:
        return None
    start_hour = int(match.group(1))
    start_minute = int(match.group(2) or 0)
    end_hour = int(match.group(3))
    end_minute = int(match.group(4) or 0)
    if not (0 <= start_hour <= 23 and 0 <= end_hour <= 23):
        return None
    if not (0 <= start_minute <= 59 and 0 <= end_minute <= 59):
        return None
    today = datetime.now(LOCAL_TIMEZONE)
    start = today.replace(
        hour=start_hour, minute=start_minute, second=0, microsecond=0
    )
    end = today.replace(
        hour=end_hour, minute=end_minute, second=0, microsecond=0
    )
    if end <= start:
        return None
    return start, end


def query_time_range(query: str) -> Tuple[datetime, datetime, str]:
    """Son 31 gündeki tarihi ve isteğe bağlı saat aralığını çözümler."""
    now = datetime.now(LOCAL_TIMEZONE)
    normalized = query.casefold()
    date_match = re.search(
        r"\b(?:(\d{1,2})[./](\d{1,2})[./](\d{4})|"
        r"(\d{4})-(\d{1,2})-(\d{1,2}))\b",
        normalized,
    )
    if date_match:
        if date_match.group(1):
            day, month, year = map(int, date_match.group(1, 2, 3))
        else:
            year, month, day = map(int, date_match.group(4, 5, 6))
        try:
            selected = datetime(year, month, day, tzinfo=LOCAL_TIMEZONE)
        except ValueError as exc:
            raise ValueError("Geçersiz tarih yazıldı.") from exc
    else:
        selected = now.replace(hour=0, minute=0, second=0, microsecond=0)
    today = now.replace(hour=0, minute=0, second=0, microsecond=0)
    age_days = (today.date() - selected.date()).days
    if age_days < 0:
        raise ValueError("Gelecekteki bir tarih sorgulanamaz.")
    if age_days > 31:
        raise ValueError("Yalnızca son 31 gün içindeki tarihler sorgulanabilir.")
    time_source = re.sub(
        r"\b(?:\d{1,2}[./]\d{1,2}[./]\d{4}|\d{4}-\d{1,2}-\d{1,2})\b",
        " ",
        normalized,
    )
    time_match = re.search(
        r"(\d{1,2})(?:[.:](\d{2}))?\s*(?:-|–|—|ile)\s*"
        r"(\d{1,2})(?:[.:](\d{2}))?",
        time_source,
    )
    if time_match:
        sh, sm = int(time_match.group(1)), int(time_match.group(2) or 0)
        eh, em = int(time_match.group(3)), int(time_match.group(4) or 0)
        if not (0 <= sh <= 23 and 0 <= eh <= 23 and 0 <= sm <= 59 and 0 <= em <= 59):
            raise ValueError("Geçersiz saat aralığı yazıldı.")
        start = selected.replace(hour=sh, minute=sm)
        end = selected.replace(hour=eh, minute=em)
        if end <= start:
            raise ValueError("Saat aralığının bitişi başlangıçtan sonra olmalıdır.")
        label = f"{selected:%d.%m.%Y} {start:%H:%M}-{end:%H:%M}"
    else:
        start = selected
        end = min(selected + timedelta(days=1), now)
        label = f"{selected:%d.%m.%Y}"
    return start, end, label


def explicit_query_dates(query: str) -> List[datetime]:
    """Sorgudaki açık tarihleri tekrarları korumadan yerel tarihe dönüştürür."""
    matches = re.findall(
        r"\b(?:(\d{1,2})[./](\d{1,2})[./](\d{4})|"
        r"(\d{4})-(\d{1,2})-(\d{1,2}))\b",
        query,
    )
    dates: List[datetime] = []
    for first_day, first_month, first_year, iso_year, iso_month, iso_day in matches:
        if first_day:
            day, month, year = int(first_day), int(first_month), int(first_year)
        else:
            year, month, day = int(iso_year), int(iso_month), int(iso_day)
        try:
            value = datetime(year, month, day, tzinfo=LOCAL_TIMEZONE)
        except ValueError:
            continue
        if value not in dates:
            dates.append(value)
    return dates


def requested_floor(query: str) -> str:
    """3. kat, 3 katta veya 3. kattaki ifadelerinden katı çıkarır."""
    match = re.search(r"\b(\d{1,3})\.?\s*kat", query.casefold())
    return match.group(1) if match else ""


def requested_coordinates(query: str) -> Tuple[float | None, float | None]:
    """40.99758, 29.101014 biçimindeki koordinat çiftini çıkarır."""
    match = re.search(
        r"(?<!\d)(-?\d{1,2}\.\d+)\s*[,;]\s*(-?\d{1,3}\.\d+)(?!\d)",
        query,
    )
    if not match:
        return None, None
    return float(match.group(1)), float(match.group(2))


def requested_threshold(query: str) -> Tuple[float | None, str | None]:
    """230 V üstü veya 5 A altı gibi eşik ifadelerini çözümler."""
    match = re.search(
        r"(\d+(?:[.,]\d+)?)\s*(?:v|a|hz)?(?:['’](?:u|ü|ı|i|nun|nün|nın|nin))?"
        r"[^\d]{0,35}?"
        r"(üst(?:ü|ünde|üne)?|üzer(?:i|inde|ine)?|aş(?:tı|an|mış)?|"
        r"alt(?:ı|ında|ına)?|düş(?:tü|en|müş)|düşük)",
        query.casefold(),
    )
    if not match:
        return None, None
    threshold = float(match.group(1).replace(",", "."))
    word = match.group(2)
    comparison = "lt" if word.startswith(("alt", "düş")) else "gt"
    return threshold, comparison


def requested_analysis_metric(query: str) -> str:
    """Zaman sözcüklerini enerji metriği sanmadan açıkça istenen metriği seçer."""
    normalized = query.casefold()
    phase_match = re.search(r"\bl([123])\b", normalized)
    phase = phase_match.group(1) if phase_match else "1"
    if "gerilim" in normalized or "voltaj" in normalized:
        return f"voltage_l{phase}_v"
    if "akım" in normalized or "amper" in normalized:
        return f"current_l{phase}_a"
    if "güç faktörü" in normalized or "power factor" in normalized or "cos" in normalized:
        return f"power_factor_l{phase}"
    if "frekans" in normalized or re.search(r"\bhz\b", normalized):
        return "frequency_hz"
    if "aylık enerji" in normalized or "aylık tüketim" in normalized:
        return "monthly_energy_kwh"
    if "haftalık enerji" in normalized or "haftalık tüketim" in normalized:
        return "weekly_energy_kwh"
    if "günlük enerji" in normalized or "günlük tüketim" in normalized:
        return "daily_energy_kwh"
    return "energy_total_kwh"


def requested_bulk_metrics(query: str) -> List[str]:
    """Toplu sorgudaki doğal dil ifadelerini MCP metriklerine dönüştürür."""

    normalized = query.casefold()
    metrics: List[str] = []

    def add(*keys: str) -> None:
        for key in keys:
            if key not in metrics:
                metrics.append(key)

    if "günlük" in normalized or "bugün" in normalized:
        add("daily_energy_kwh")
    if "haftalık" in normalized or "hafta" in normalized:
        add("weekly_energy_kwh")
    if "aylık" in normalized or "ay" in normalized:
        add("monthly_energy_kwh")
    if "toplam enerji" in normalized or "toplam tüketim" in normalized:
        add("energy_total_kwh")
    if "gerilim" in normalized or "voltaj" in normalized:
        add("voltage_l1_v", "voltage_l2_v", "voltage_l3_v")
    if "akım" in normalized or "amper" in normalized:
        add("current_l1_a", "current_l2_a", "current_l3_a")
    if "güç faktörü" in normalized or "cos" in normalized:
        add("power_factor_l1", "power_factor_l2", "power_factor_l3")
    if "frekans" in normalized or "hz" in normalized:
        add("frequency_hz")
    if not metrics:
        add("daily_energy_kwh")
    return metrics


def query_meter_names(query: str) -> List[str]:
    """Sorgudan desteklenen sayaç adı biçimlerini çıkarır."""

    matches = re.findall(
        r"\b(?:ESM-\d+|4001001_\d+)\b",
        query,
        flags=re.IGNORECASE,
    )
    return list(dict.fromkeys(matches))


def display_metric_value(metric: str, value: Any) -> str:
    """Metrik değerini etiketi ve birimiyle kullanıcıya hazırlar."""

    label, unit = METRIC_LABELS.get(metric, (metric, ""))
    rendered = "-" if value is None else str(value)
    suffix = f" {unit}" if unit else ""
    return f"{label}: {rendered}{suffix}"


def answer_meter_question( #sayaç sorusu için gereken aracı çalıştırır
    query: str, #kullanıcıya gösterilecek yanıt
    mcp: MCPClient, #arayüzde gösterilebilecek mcp events
) -> Tuple[str, List[Dict[str, Any]]]:
    
    events: List[Dict[str, Any]] = []
    normalized_query = query.casefold()

    def call_meter_tool(
        name: str,
        arguments: Dict[str, Any],
    ) -> Dict[str, Any]:
        events.append({
            "type": "tool_call",
            "tool": name,
            "arguments": arguments,
        })
        try:
            result = mcp.call_tool(name, arguments)
        except Exception as exc:
            result = {"isError": True, "error": str(exc)}
        events.append({
            "type": "tool_result",
            "tool": name,
            "ok": not bool(result.get("isError", False)),
            "result": result,
        })
        return result

    wants_comparison = any(phrase in normalized_query for phrase in (
        "karşılaştır", "kıyasla", "arasındaki fark", "hangisi daha",
    ))
    wants_week_comparison = (
        "geçen hafta" in normalized_query
        and "bu hafta" in normalized_query
    )
    wants_month_comparison = (
        "geçen ay" in normalized_query
        and "bu ay" in normalized_query
    )
    wants_period_comparison = (
        wants_week_comparison or wants_month_comparison
    )
    requested_phases = set(re.findall(r"\bl([123])\b", normalized_query))
    wants_phase_comparison = (
        wants_comparison
        and len(requested_phases) >= 2
        and any(term in normalized_query for term in (
            "gerilim", "voltaj", "akım", "amper", "güç faktörü", "cos",
        ))
    )
    wants_ranking = any(phrase in normalized_query for phrase in (
        "en yüksek", "en düşük", "büyükten küçüğe",
        "küçükten büyüğe", "sırala", "sıralama",
    ))
    wants_anomalies = any(phrase in normalized_query for phrase in (
        "veri göndermeyen", "okuma hatası", "hatalı sayaç",
        "anormal", "anomali", "faz dengesiz", "gerilim sınırı",
        "normal aralığın dışında", "güç faktörü düşük",
        "frekans sınırı", "telemetrisi eksik",
    ))
    wants_grouping = any(phrase in normalized_query for phrase in (
        "katlara göre", "kata göre", "hangi katta", "katta olduğu",
        "katta olduğunu", "kat listesi", "kat bazında",
        "güç kaynağına göre",
        "duruma göre", "hata koduna göre", "hata kodlarına göre",
        "faz sayısına göre", "grupla", "gruplandır",
    ))
    wants_fleet_summary = any(phrase in normalized_query for phrase in (
        "tüm sayaçların durumu", "sayaçların genel durumu",
        "genel özet", "filo özeti", "kaç sayaç güncel",
        "kaç sayaç veri", "toplam günlük tüketim",
        "toplam haftalık tüketim", "toplam aylık tüketim",
    ))
    wants_transmission_count = any(phrase in normalized_query for phrase in (
        "kaç kere veri", "kaç kez veri", "kaç defa veri",
        "kaç kayıt", "veri sayısı", "telemetri sayısı",
    ))
    wants_data_quality = any(phrase in normalized_query for phrase in (
        "null veri", "null değer", "eksik veri", "boş veri",
        "veri kalitesi", "alan eksik", "telemetri eksik",
    ))
    wants_metric_analysis = any(phrase in normalized_query for phrase in (
        "minimum", "min ", "en düşük değer", "maksimum", "max ",
        "en yüksek değer", "ortalama", "kaç kez aşt", "kaç kere aşt",
        "kaç defa aşt", "kaç kez üst", "kaç kere üst", "kaç kez alt",
        "eşik", "hangi saatte en yüksek", "hangi saatte en düşük",
    )) or bool(re.search(
        r"kaç\s+(?:kez|kere|defa).{0,40}(?:aşt|üst|alt|düş)",
        normalized_query,
    ))
    date_mentions = explicit_query_dates(query)
    floor_filter = requested_floor(query)
    latitude_filter, longitude_filter = requested_coordinates(query)
    bulk_threshold, bulk_comparison = requested_threshold(query)
    wants_bulk_threshold_analysis = (
        bool(date_mentions)
        and bulk_threshold is not None
        and bulk_comparison is not None
        and not query_meter_names(query)
        and any(term in normalized_query for term in (
            "frekans", "hz", "gerilim", "voltaj", "akım", "amper",
            "güç faktörü", "power factor",
        ))
        and any(term in normalized_query for term in (
            "sırala", "listele", "göster", "bul", "hangi sayaç",
        ))
    )
    wants_location_list = (
        bool(floor_filter) or latitude_filter is not None
    ) and any(phrase in normalized_query for phrase in (
        "listele", "listesi", "bul", "göster", "hangi sayaç",
    ))
    wants_location_energy_ranking = (
        bool(date_mentions)
        and any(term in normalized_query for term in ("katları", "katlara", "kat bazında"))
        and any(term in normalized_query for term in ("enerji", "tüketim", "kwh"))
        and any(term in normalized_query for term in ("sırala", "en yüksek", "en fazla", "karşılaştır"))
    )
    wants_date_energy_comparison = (
        len(date_mentions) >= 2
        and wants_comparison
        and any(term in normalized_query for term in ("enerji", "tüketim", "kwh"))
    )
    wants_seven_day_peak = (
        any(phrase in normalized_query for phrase in ("son 7 gün", "son yedi gün"))
        and any(phrase in normalized_query for phrase in ("en fazla", "en yüksek", "maksimum"))
    )
    wants_thirty_day_summary = (
        any(phrase in normalized_query for phrase in ("son 30 gün", "son otuz gün"))
        and any(phrase in normalized_query for phrase in ("ortalama", "en yüksek", "en fazla"))
    )
    wants_interval_energy_ranking = (
        bool(date_mentions)
        and any(stem in normalized_query for stem in ("sayaç", "sayac"))
        and any(phrase in normalized_query for phrase in (
            "en çok enerji tüketen", "en fazla enerji tüketen",
            "en yüksek tüketimli", "enerji tüketen 5", "tüketen ilk",
        ))
    )

    if wants_bulk_threshold_analysis:
        try:
            start_at, end_at, range_label = query_time_range(query)
        except ValueError as exc:
            return str(exc), events
        metric = requested_analysis_metric(query)
        result = call_meter_tool(
            "find_meters_by_metric_threshold",
            {
                "metric": metric,
                "start_ts": int(start_at.timestamp() * 1000),
                "end_ts": int(end_at.timestamp() * 1000),
                "threshold": bulk_threshold,
                "comparison": bulk_comparison,
                "floor": floor_filter,
                "limit": 100,
            },
        )
        error = tool_error_message(result)
        if error:
            return f"Toplu eşik analizi yapılamadı: {error}", events
        data = result_data(result)
        if data.get("error"):
            return str(data["error"]), events
        meters = data.get("meters", [])
        label, unit = METRIC_LABELS.get(metric, (metric, ""))
        direction = "altına düşen" if bulk_comparison == "lt" else "üstüne çıkan"
        location_text = f" {floor_filter}. katta" if floor_filter else ""
        if not meters:
            return (
                f"{range_label}{location_text} {label} değeri "
                f"{bulk_threshold:g} {unit} {direction} sayaç bulunmadı.\n"
                f"• İncelenen sayaç: {data.get('evaluated_meter_count', 0)}\n"
                f"• Bu aralıkta ölçümü bulunmayan sayaç: {data.get('no_data_meter_count', 0)}",
                events,
            )
        lines = [
            f"{range_label}{location_text} {label} değeri "
            f"{bulk_threshold:g} {unit} {direction} sayaçlar:"
        ]
        for index, meter in enumerate(meters, 1):
            lines.append(
                f"{index}. {meter.get('name', 'Adsız sayaç')}: "
                f"{meter.get('threshold_match_count', 0)} ölçüm"
            )
        lines.append(f"Toplam eşleşen sayaç: {data.get('matched_meter_count', len(meters))}")
        return "\n".join(lines), events

    if wants_location_energy_ranking:
        try:
            start_at, end_at, range_label = query_time_range(query)
        except ValueError as exc:
            return str(exc), events
        result = call_meter_tool(
            "rank_meter_locations_by_energy",
            {
                "start_ts": int(start_at.timestamp() * 1000),
                "end_ts": int(end_at.timestamp() * 1000),
                "attribute": "floor",
            },
        )
        error = tool_error_message(result)
        if error:
            return f"Kat bazlı enerji sıralaması yapılamadı: {error}", events
        data = result_data(result)
        if data.get("error"):
            return str(data["error"]), events
        groups = data.get("groups", [])
        if not groups:
            evaluated = data.get("evaluated_meter_count", 0)
            insufficient = data.get("no_energy_data_meter_count", evaluated)
            return (
                f"{range_label} için kat bazında hesaplanabilir enerji tüketimi bulunamadı.\n"
                f"• İncelenen sayaç: {evaluated}\n"
                f"• Bu zaman aralığında yeterli e_tkwh verisi olmayan sayaç: {insufficient}",
                events,
            )
        lines = [f"{range_label} kat bazlı enerji tüketimi:"]
        for index, group in enumerate(groups, 1):
            floor_value = group.get("value")
            floor_label = "Tanımsız" if floor_value is None else f"{floor_value}. kat"
            lines.append(
                f"{index}. {floor_label}: {group.get('consumption_kwh', '-')} kWh "
                f"({group.get('meter_count', 0)} sayaç)"
            )
        return "\n".join(lines), events

    if wants_location_list:
        result = call_meter_tool(
            "find_meters_by_location",
            {
                "floor": floor_filter,
                "latitude": latitude_filter,
                "longitude": longitude_filter,
            },
        )
        error = tool_error_message(result)
        if error:
            return f"Konuma göre sayaçlar alınamadı: {error}", events
        data = result_data(result)
        if data.get("error"):
            return str(data["error"]), events
        lines = [f"Konum filtresine uyan {data.get('count', 0)} sayaç bulundu:"]
        for meter in data.get("meters", []):
            lines.append(
                f"• {meter.get('name', 'Adsız sayaç')} — "
                f"Kat: {meter.get('floor', '-')}, "
                f"Konum: {meter.get('latitude', '-')}, {meter.get('longitude', '-')}"
            )
        return "\n".join(lines), events

    if wants_interval_energy_ranking:
        try:
            start_at, end_at, range_label = query_time_range(query)
        except ValueError as exc:
            return str(exc), events
        limit_match = re.search(r"(?:ilk|en çok|en fazla)?\s*(\d{1,3})\s+saya", normalized_query)
        limit = min(max(int(limit_match.group(1)), 1), 100) if limit_match else 5
        ranking_result = call_meter_tool(
            "rank_meters_by_interval_energy",
            {
                "start_ts": int(start_at.timestamp() * 1000),
                "end_ts": int(end_at.timestamp() * 1000),
                "limit": limit,
                "floor": floor_filter,
            },
        )
        error = tool_error_message(ranking_result)
        if error:
            return f"Zaman aralığı tüketim sıralaması yapılamadı: {error}", events
        ranking = result_data(ranking_result)
        if ranking.get("error"):
            return str(ranking["error"]), events
        meters = ranking.get("meters", [])
        evaluated = ranking.get("evaluated_meter_count", 0)
        insufficient = ranking.get("no_energy_data_meter_count", evaluated)
        if not meters:
            if floor_filter and evaluated == 0:
                return f"{floor_filter}. katta eşleşen sayaç bulunamadı.", events
            location_text = f" {floor_filter}. katta" if floor_filter else ""
            return (
                f"{range_label}{location_text} için tüketimi hesaplanabilen sayaç bulunamadı.\n"
                f"• Eşleşen sayaç: {evaluated}\n"
                f"• Bu zaman aralığında yeterli e_tkwh verisi olmayan sayaç: {insufficient}",
                events,
            )
        location_suffix = f" {floor_filter}. katta" if floor_filter else ""
        lines = [f"{range_label}{location_suffix} tüketimi en yüksek {limit} sayaç:"]
        for index, meter in enumerate(meters, 1):
            lines.append(
                f"{index}. {meter.get('name', 'Adsız sayaç')}: "
                f"{meter.get('consumption_kwh', '-')} kWh"
            )
        return "\n".join(lines), events

    if (
        wants_comparison
        and not wants_period_comparison
        and not wants_phase_comparison
        and not wants_date_energy_comparison
    ):
        meter_names = query_meter_names(query)
        if len(meter_names) < 2:
            return (
                "Karşılaştırma için en az iki tam sayaç adı yazın.",
                events,
            )
        metrics = requested_bulk_metrics(query)
        comparison_result = call_meter_tool(
            "compare_meter_devices",
            {"meter_names": meter_names, "metrics": metrics},
        )
        error = tool_error_message(comparison_result)
        if error:
            return f"Sayaç karşılaştırması yapılamadı: {error}", events
        comparison = result_data(comparison_result)
        if comparison.get("error"):
            return str(comparison["error"]), events
        lines = ["Sayaç karşılaştırması:"]
        for meter in comparison.get("meters", []):
            lines.append(f"\n{meter.get('name', 'Adsız sayaç')}:")
            values = meter.get("values", {})
            for metric in metrics:
                lines.append(f"• {display_metric_value(metric, values.get(metric))}")
        missing = comparison.get("missing_meter_names", [])
        if missing:
            lines.append(f"\nBulunamayan sayaçlar: {', '.join(map(str, missing))}")
        return "\n".join(lines), events

    if (
        wants_ranking
        and not wants_phase_comparison
        and not wants_metric_analysis
        and not wants_seven_day_peak
        and not wants_thirty_day_summary
        and not wants_interval_energy_ranking
    ):
        requested_metrics = requested_bulk_metrics(query)
        if "aylık" in normalized_query or "aylik" in normalized_query:
            metric = "monthly_energy_kwh"
        elif "haftalık" in normalized_query or "haftalik" in normalized_query:
            metric = "weekly_energy_kwh"
        elif "günlük" in normalized_query or "gunluk" in normalized_query:
            metric = "daily_energy_kwh"
        else:
            metric = requested_metrics[0]
        order = "asc" if any(phrase in normalized_query for phrase in (
            "en düşük", "küçükten büyüğe",
        )) else "desc"
        limit_match = re.search(
            r"(?:en yüksek|en düşük|ilk)\s+(\d{1,3})",
            normalized_query,
        )
        limit = min(max(int(limit_match.group(1)), 1), 100) if limit_match else 10
        ranking_result = call_meter_tool(
            "rank_meters_by_metric",
            {"metric": metric, "order": order, "limit": limit},
        )
        error = tool_error_message(ranking_result)
        if error:
            return f"Sayaç sıralaması yapılamadı: {error}", events
        ranking = result_data(ranking_result)
        if ranking.get("error"):
            return str(ranking["error"]), events
        label, unit = METRIC_LABELS.get(metric, (metric, ""))
        lines = [f"{label} ölçümüne göre sayaç sıralaması:"]
        ranked_meters = ranking.get("meters", [])
        for index, meter in enumerate(ranked_meters, 1):
            suffix = f" {unit}" if unit else ""
            value = meter.get("value")
            if isinstance(value, float):
                rendered_value = f"{value:.3f}".rstrip("0").rstrip(".")
            else:
                rendered_value = str(value if value is not None else "-")
            lines.append(
                f"{index}. {meter.get('name', 'Adsız sayaç')}: "
                f"{rendered_value}{suffix}"
            )

        supplemental_metrics = [
            item for item in requested_metrics if item != metric
        ]
        top_count = 0
        top_match = re.search(r"ilk\s+(\d{1,2})", normalized_query)
        if top_match:
            top_count = int(top_match.group(1))
        else:
            number_words = {
                "bir": 1, "iki": 2, "üç": 3, "uc": 3,
                "dört": 4, "dort": 4, "beş": 5, "bes": 5,
            }
            for word, number in number_words.items():
                if f"ilk {word}" in normalized_query:
                    top_count = number
                    break

        if supplemental_metrics and top_count and ranked_meters:
            selected = ranked_meters[:min(top_count, len(ranked_meters))]
            selected_names = [
                str(item.get("name", ""))
                for item in selected
                if item.get("name")
            ]
            detail_result = call_meter_tool(
                "compare_meter_devices",
                {
                    "meter_names": selected_names,
                    "metrics": supplemental_metrics,
                },
            )
            detail_error = tool_error_message(detail_result)
            if detail_error:
                lines.append(f"\nEk enerji değerleri alınamadı: {detail_error}")
            else:
                details = result_data(detail_result)
                lines.append(f"\nİlk {len(selected_names)} sayacın ek değerleri:")
                for item in details.get("meters", []):
                    lines.append(f"\n{item.get('name', 'Adsız sayaç')}:")
                    values = item.get("values", {})
                    for extra_metric in supplemental_metrics:
                        lines.append(
                            f"• {display_metric_value(extra_metric, values.get(extra_metric))}"
                        )
        return "\n".join(lines), events

    if wants_anomalies:
        anomaly_types = []
        anomaly_phrases = (
            ("stale", ("veri göndermeyen", "güncel olmayan")),
            ("read_error", ("okuma hatası", "hatalı sayaç")),
            ("voltage_out_of_range", ("gerilim sınırı", "normal aralığın dışında")),
            ("voltage_imbalance", ("gerilim dengesiz", "faz dengesiz")),
            ("current_imbalance", ("akım dengesiz",)),
            ("low_power_factor", ("güç faktörü düşük",)),
            ("frequency_out_of_range", ("frekans sınırı",)),
            ("missing_telemetry", ("telemetrisi eksik",)),
        )
        for anomaly_type, phrases in anomaly_phrases:
            if any(phrase in normalized_query for phrase in phrases):
                anomaly_types.append(anomaly_type)
        anomaly_result = call_meter_tool(
            "find_meter_anomalies",
            {"anomaly_types": anomaly_types or None},
        )
        error = tool_error_message(anomaly_result)
        if error:
            return f"Sayaç anomali taraması yapılamadı: {error}", events
        anomalies = result_data(anomaly_result)
        if anomalies.get("error"):
            return str(anomalies["error"]), events
        findings = anomalies.get("findings", [])
        lines = [f"{len(findings)} sayaçta anomali bulundu:"]
        for finding in findings:
            types = ", ".join(
                str(item.get("type", "bilinmeyen"))
                for item in finding.get("anomalies", [])
            )
            lines.append(f"• {finding.get('name', 'Adsız sayaç')}: {types}")
        return "\n".join(lines), events

    if wants_grouping:
        include_meter_names = any(phrase in normalized_query for phrase in (
            "liste", "listele", "göster", "hangi",
        ))
        if "kat" in normalized_query:
            attribute = "floor"
        elif "güç kaynağı" in normalized_query:
            attribute = "power_source"
        elif "hata kod" in normalized_query:
            attribute = "error_code"
        elif "faz say" in normalized_query:
            attribute = "phase_count"
        elif "durum" in normalized_query:
            attribute = "status_text"
        else:
            attribute = "active"
        grouping_result = call_meter_tool(
            "group_meters_by_attribute",
            {
                "attribute": attribute,
                "include_meter_names": include_meter_names,
            },
        )
        error = tool_error_message(grouping_result)
        if error:
            return f"Sayaçlar gruplanamadı: {error}", events
        grouping = result_data(grouping_result)
        if grouping.get("error"):
            return str(grouping["error"]), events
        attribute_labels = {
            "floor": "kat",
            "power_source": "güç kaynağı",
            "error_code": "hata kodu",
            "phase_count": "faz sayısı",
            "status_text": "durum",
            "active": "aktiflik",
        }
        label = attribute_labels.get(attribute, attribute)
        lines = [f"Sayaçların {label} alanına göre dağılımı:"]
        for group in grouping.get("groups", []):
            value = group.get("value")
            rendered = "Tanımsız" if value is None else str(value)
            lines.append(f"• {rendered}: {group.get('count', 0)} sayaç")
            if include_meter_names:
                for meter_name in group.get("meters", []):
                    lines.append(f"  - {meter_name}")
        return "\n".join(lines), events

    if wants_fleet_summary:
        fleet_result = call_meter_tool(
            "get_meter_fleet_summary",
            {"stale_after_minutes": 15},
        )
        error = tool_error_message(fleet_result)
        if error:
            return f"Sayaç genel özeti alınamadı: {error}", events
        fleet = result_data(fleet_result)
        totals = fleet.get("energy_totals_kwh", {})
        lines = [
            "Sayaç filosu özeti:",
            f"• Toplam sayaç: {fleet.get('total_meter_count', 0)}",
            f"• Güncel veri gönderen: {fleet.get('fresh_meter_count', 0)}",
            f"• Güncel olmayan: {fleet.get('stale_meter_count', 0)}",
            f"• Okuma hatası bulunan: {fleet.get('read_error_count', 0)}",
            f"• Toplam günlük enerji: {totals.get('daily_energy_kwh', '-')} kWh",
            f"• Toplam haftalık enerji: {totals.get('weekly_energy_kwh', '-')} kWh",
            f"• Toplam aylık enerji: {totals.get('monthly_energy_kwh', '-')} kWh",
        ]
        return "\n".join(lines), events

    list_result = call_meter_tool( #tüm sayaçlar listelenir
        "list_meter_devices",
        {"page_size": 20, "search": ""},
    )
    error = tool_error_message(list_result)
    if error:
        return f"Sayaçlar ThingsBoard'dan alınamadı: {error}", events

    meters = result_data(list_result).get("meters", [])
    if not isinstance(meters, list) or not meters:
        return "ThingsBoard hesabında meter profiline ait sayaç bulunamadı.", events

    wants_list = any(word in normalized_query for word in (
        "listele", "listesi", "hangi sayaç", "kaç sayaç",
        "sayaclari", "sayaçları",
    ))
    wants_reading = any(word in normalized_query for word in (
        "güncel", "anlık", "değer", "ölçüm", "güç", "enerji",
        "gerilim", "voltaj", "akım", "frekans", "bağlantı",
        "telemetri",
    ))

    if wants_list and not wants_reading:
        lines = [f"ThingsBoard hesabında {len(meters)} sayaç bulundu:"]
        for index, meter in enumerate(meters, 1):
            lines.append(
                f"{index}. {meter.get('name', 'Adsız sayaç')} "
                f"(ID: {meter.get('device_id', '-')})"
            )
        return "\n".join(lines), events

    named_meters = []
    for candidate in meters:
        if not isinstance(candidate, dict):
            continue

        identifiers = (
            str(candidate.get("name") or "").strip(),
            str(candidate.get("label") or "").strip(),
        )
        if any(
            identifier and identifier.casefold() in normalized_query
            for identifier in identifiers
        ):
            named_meters.append(candidate)

    if len(named_meters) > 1:
        matching_names = ", ".join(
            str(candidate.get("name") or "Adsız sayaç")
            for candidate in named_meters
        )
        return (
            "Sorgudaki sayaç ifadesi birden fazla cihazla eşleşti. "
            f"Lütfen tam sayaç adını yazın: {matching_names}.",
            events,
        )

    if named_meters:
        meter = named_meters[0]
    elif len(meters) == 1:
        meter = meters[0]
    else:
        example_names = ", ".join(
            str(candidate.get("name") or "Adsız sayaç")
            for candidate in meters[:5]
            if isinstance(candidate, dict)
        )
        remaining_count = max(0, len(meters) - 5)
        if remaining_count:
            example_names += f" ve {remaining_count} sayaç daha"
        return (
            "Hangi sayacı sorgulamak istediğinizi belirtin. "
            f"Örneğin: {example_names}.",
            events,
        )

    meter_name = str(meter.get("name", "Sayaç"))
    device_id = str(meter.get("device_id", ""))
    if not device_id:
        return "Sayaç kimliği ThingsBoard yanıtında bulunamadı.", events

    if wants_date_energy_comparison:
        selected_dates = date_mentions[:2]
        summaries = []
        for selected_date in selected_dates:
            day_end = selected_date + timedelta(days=1)
            result = call_meter_tool(
                "get_meter_energy_summary",
                {
                    "device_id": device_id,
                    "start_ts": int(selected_date.timestamp() * 1000),
                    "end_ts": int(day_end.timestamp() * 1000),
                },
            )
            error = tool_error_message(result)
            if error:
                return f"{selected_date:%d.%m.%Y} tüketimi alınamadı: {error}", events
            summary = result_data(result)
            if summary.get("error"):
                return f"{selected_date:%d.%m.%Y}: {summary['error']}", events
            summaries.append((selected_date, float(summary["consumption_kwh"])))
        first_date, first_value = summaries[0]
        second_date, second_value = summaries[1]
        difference = second_value - first_value
        percentage = (difference / first_value * 100) if first_value else None
        direction = "fazla" if difference > 0 else "az" if difference < 0 else "aynı"
        return (
            f"{meter_name} günlük enerji karşılaştırması:\n"
            f"• {first_date:%d.%m.%Y}: {first_value:.3f} kWh\n"
            f"• {second_date:%d.%m.%Y}: {second_value:.3f} kWh\n"
            f"• Fark: {abs(difference):.3f} kWh ({direction})\n"
            f"• Değişim: {abs(percentage):.2f}% {direction}" if percentage is not None else
            f"{meter_name} günlük enerji karşılaştırması:\n"
            f"• {first_date:%d.%m.%Y}: {first_value:.3f} kWh\n"
            f"• {second_date:%d.%m.%Y}: {second_value:.3f} kWh\n"
            f"• Fark: {abs(difference):.3f} kWh ({direction})\n"
            "• Değişim yüzdesi: İlk gün tüketimi sıfır olduğu için hesaplanamadı.",
            events,
        )

    if wants_seven_day_peak or wants_thirty_day_summary:
        now_at = datetime.now(LOCAL_TIMEZONE)
        today_start = now_at.replace(hour=0, minute=0, second=0, microsecond=0)
        day_count = 30 if wants_thirty_day_summary else 7
        start_at = today_start - timedelta(days=day_count - 1)
        series_result = call_meter_tool(
            "get_meter_daily_energy_series",
            {
                "device_id": device_id,
                "start_ts": int(start_at.timestamp() * 1000),
                "end_ts": int(now_at.timestamp() * 1000),
            },
        )
        error = tool_error_message(series_result)
        if error:
            return f"Günlük enerji serisi alınamadı: {error}", events
        series = result_data(series_result)
        if series.get("error"):
            return str(series["error"]), events
        maximum = series.get("maximum_consumption_day") or {}
        lines = [f"{meter_name} son {day_count} günlük enerji analizi:"]
        if wants_thirty_day_summary:
            lines.append(
                f"• Günlük ortalama: "
                f"{series.get('average_daily_consumption_kwh', '-')} kWh"
            )
        lines.extend((
            f"• En yüksek tüketimli gün: {maximum.get('date', '-')}",
            f"• O günkü tüketim: {maximum.get('consumption_kwh', '-')} kWh",
            f"• Geçerli gün sayısı: {series.get('valid_day_count', 0)}",
        ))
        return "\n".join(lines), events

    wants_context = any(word in normalized_query for word in (
        "konum", "bina", "kat", "model", "seri numarası",
        "seri no", "firmware", "nerede", "cihaz bilgi",
        "cihaz özellik", "attribute", "özellik",
    ))
    wants_connection = any(word in normalized_query for word in (
        "çevrim içi", "çevrimiçi", "çevrim dışı", "offline",
        "online", "bağlantı durumu", "bağlantı bilgi",
        "son veri zamanı", "ne zaman veri",
        "veri ne zaman", "veri gelmiş",
    ))
    wants_history = any(word in normalized_query for word in (
        "geçmiş", "değişmiş", "değişim", "son bir saat",
        "son 24 saat", "grafik", "trend",
    ))
    wants_summary = any(word in normalized_query for word in (
        "bugün tüketti", "bugünkü tüketim", "tüketim kaç",
        "tükettiği enerji", "enerji özeti",
    ))

    context_labels = (
        ("building", "Bina", ("bina",)),
        ("floor", "Kat", ("kat",)),
        ("location", "Konum", ("konum", "nerede")),
        ("model", "Model", ("model",)),
        ("firmwareVersion", "Firmware", ("firmware",)),
        ("meterType", "Faz sayısı", ("sayaç tipi", "faz")),
        ("serialNumber", "Seri numarası", ("seri numarası", "seri no")),
        ("powerSource", "Güç kaynağı", ("güç kaynağı",)),
        ("statusText", "Cihaz durumu", ("durum",)),
        ("readOk", "Okuma başarılı", ("okuma",)),
    )
    requested_context_keys = [
        key
        for key, _label, phrases in context_labels
        if any(phrase in normalized_query for phrase in phrases)
    ]

    #kullanıcı yalnızca "cihaz bilgileri" dediyse tüm bağlamı göster.
    if wants_context and not requested_context_keys:
        requested_context_keys = [
            key for key, _label, _phrases in context_labels
        ]

    if wants_context:
        context_result = call_meter_tool(
            "get_meter_context",
            {"device_id": device_id},
        )
        error = tool_error_message(context_result)
        if error:
            return f"Sayaç bağlam bilgileri alınamadı: {error}", events

        attributes = result_data(context_result).get("attributes", {})
        if not isinstance(attributes, dict):
            attributes = {}
        lines = [f"{meter_name} cihaz bilgileri:"]
        for key, label, _phrases in context_labels:
            if key not in requested_context_keys:
                continue
            value = attributes.get(key)
            if value is not None:
                lines.append(f"• {label}: {value}")
        if len(lines) == 1:
            lines.append("• Cihaz için ek bağlam bilgisi tanımlanmamış.")
        return "\n".join(lines), events

    if wants_connection:
        connection_result = call_meter_tool(
            "get_meter_connection_status",
            {"device_id": device_id},
        )
        error = tool_error_message(connection_result)
        if error:
            return f"Bağlantı durumu alınamadı: {error}", events

        status = result_data(connection_result)
        reported_status = status.get(
            "reported_status",
            "bilinmiyor",
        )
        last_telemetry_at = status.get(
            "last_telemetry_at",
            "bilinmiyor",
        )
        freshness = status.get(
            "freshness",
            "Veri güncelliği hesaplanamadı.",
        )

        return (
            f"{meter_name} bağlantı bilgisi:\n"
            f"• Cihazın son bildirdiği durum: {reported_status}\n"
            f"• Son telemetri zamanı: {last_telemetry_at}\n"
            f"• Veri güncelliği: {freshness}",
            events,
        )

    now_ms = int(time.time() * 1000)

    if wants_transmission_count:
        try:
            start_at, end_at, range_label = query_time_range(query)
        except ValueError as exc:
            return str(exc), events
        history_result = call_meter_tool(
            "get_meter_history",
            {
                "device_id": device_id,
                "metric": "device_timestamp",
                "start_ts": int(start_at.timestamp() * 1000),
                "end_ts": int(end_at.timestamp() * 1000),
                "limit": 1000,
            },
        )
        error = tool_error_message(history_result)
        if error:
            return f"Telemetri kayıtları alınamadı: {error}", events
        history = result_data(history_result)
        if history.get("error"):
            return str(history["error"]), events
        points = history.get("points", [])
        lines = [
            f"{meter_name}, {range_label} aralığında "
            f"{len(points)} kez veri göndermiştir."
        ]
        if points:
            first_ts = int(points[0].get("timestamp", 0)) / 1000
            last_ts = int(points[-1].get("timestamp", 0)) / 1000
            first_at = datetime.fromtimestamp(first_ts, LOCAL_TIMEZONE)
            last_at = datetime.fromtimestamp(last_ts, LOCAL_TIMEZONE)
            lines.extend((
                f"• İlk kayıt: {first_at:%H:%M:%S}",
                f"• Son kayıt: {last_at:%H:%M:%S}",
            ))
        return "\n".join(lines), events

    if wants_data_quality:
        try:
            start_at, end_at, range_label = query_time_range(query)
        except ValueError as exc:
            return str(exc), events
        quality_result = call_meter_tool(
            "analyze_meter_data_quality",
            {
                "device_id": device_id,
                "start_ts": int(start_at.timestamp() * 1000),
                "end_ts": int(end_at.timestamp() * 1000),
            },
        )
        error = tool_error_message(quality_result)
        if error:
            return f"Veri kalitesi incelenemedi: {error}", events
        quality = result_data(quality_result)
        if quality.get("error"):
            return str(quality["error"]), events
        metrics = quality.get("metrics", {})
        problems = [
            (metric, details)
            for metric, details in metrics.items()
            if int(details.get("problem_count", 0) or 0) > 0
        ]
        lines = [
            f"{meter_name} veri kalitesi ({range_label}):",
            f"• Toplam paket: {quality.get('baseline_record_count', 0)}",
            f"• Tam paket: {quality.get('complete_record_count', 0)}",
            f"• Eksik/null içeren paket: {quality.get('incomplete_record_count', 0)}",
        ]
        if problems:
            lines.append("• Sorunlu alanlar:")
            for metric, details in problems:
                label = METRIC_LABELS.get(metric, (metric, ""))[0]
                lines.append(
                    f"  - {label}: {details.get('null_count', 0)} null, "
                    f"{details.get('missing_count', 0)} eksik"
                )
        else:
            lines.append("• İncelenen temel alanlarda null veya eksik veri bulunmadı.")
        return "\n".join(lines), events

    if wants_metric_analysis:
        try:
            start_at, end_at, range_label = query_time_range(query)
        except ValueError as exc:
            return str(exc), events
        metric = requested_analysis_metric(query)
        threshold, comparison = requested_threshold(query)
        analysis_result = call_meter_tool(
            "analyze_meter_metric",
            {
                "device_id": device_id,
                "metric": metric,
                "start_ts": int(start_at.timestamp() * 1000),
                "end_ts": int(end_at.timestamp() * 1000),
                "threshold": threshold,
                "comparison": comparison,
            },
        )
        error = tool_error_message(analysis_result)
        if error:
            return f"Ölçüm analizi yapılamadı: {error}", events
        analysis = result_data(analysis_result)
        if analysis.get("error"):
            return str(analysis["error"]), events
        label, unit = METRIC_LABELS.get(metric, (metric, ""))
        suffix = f" {unit}" if unit else ""

        def render_point(point: Any) -> str:
            if not isinstance(point, dict):
                return "-"
            timestamp = point.get("timestamp")
            when = "bilinmiyor"
            if timestamp is not None:
                when = datetime.fromtimestamp(
                    int(timestamp) / 1000, LOCAL_TIMEZONE
                ).strftime("%H:%M:%S")
            return f"{point.get('value', '-')}{suffix} ({when})"

        lines = [
            f"{meter_name} {label} analizi ({range_label}):",
            f"• Ölçüm sayısı: {analysis.get('numeric_point_count', 0)}",
            f"• Minimum: {render_point(analysis.get('minimum'))}",
            f"• Maksimum: {render_point(analysis.get('maximum'))}",
            f"• Ortalama: {analysis.get('average', '-')}{suffix}",
        ]
        if threshold is not None and comparison:
            direction = "üstündeki" if comparison == "gt" else "altındaki"
            lines.append(
                f"• {threshold:g}{suffix} {direction} ölçüm: "
                f"{analysis.get('threshold_match_count', 0)}"
            )
        return "\n".join(lines), events

    if wants_period_comparison:
        now = datetime.now()
        if wants_week_comparison:
            current_period_start = now.replace(
                hour=0,
                minute=0,
                second=0,
                microsecond=0,
            )
            current_period_start -= timedelta(
                days=current_period_start.weekday()
            )
            previous_period_start = (
                current_period_start - timedelta(days=7)
            )
            period_title = "haftalık"
            previous_label = "Geçen hafta"
            current_label = "Bu hafta (şu ana kadar)"
        else:
            current_period_start = now.replace(
                day=1,
                hour=0,
                minute=0,
                second=0,
                microsecond=0,
            )
            previous_month_last_day = (
                current_period_start - timedelta(days=1)
            )
            previous_period_start = previous_month_last_day.replace(
                day=1,
                hour=0,
                minute=0,
                second=0,
                microsecond=0,
            )
            period_title = "aylık"
            previous_label = "Geçen ay"
            current_label = "Bu ay (şu ana kadar)"

        current_period_start_ms = int(
            current_period_start.timestamp() * 1000
        )
        previous_period_start_ms = int(
            previous_period_start.timestamp() * 1000
        )

        previous_result = call_meter_tool(
            "get_meter_energy_summary",
            {
                "device_id": device_id,
                "start_ts": previous_period_start_ms,
                "end_ts": current_period_start_ms - 1,
            },
        )
        error = tool_error_message(previous_result)
        if error:
            return f"{previous_label} enerji özeti alınamadı: {error}", events

        current_result = call_meter_tool(
            "get_meter_energy_summary",
            {
                "device_id": device_id,
                "start_ts": current_period_start_ms,
                "end_ts": now_ms,
            },
        )
        error = tool_error_message(current_result)
        if error:
            return f"{current_label} enerji özeti alınamadı: {error}", events

        previous = result_data(previous_result)
        current = result_data(current_result)
        if previous.get("error"):
            return str(previous["error"]), events
        if current.get("error"):
            return str(current["error"]), events

        previous_value = previous.get("consumption_kwh")
        current_value = current.get("consumption_kwh")
        difference_text = "hesaplanamadı"
        percentage_text = "hesaplanamadı"
        try:
            difference = float(current_value) - float(previous_value)
            difference_text = f"{difference:.3f} kWh"
            if float(previous_value) != 0:
                percentage_text = f"{(difference / float(previous_value)) * 100:.2f}%"
        except (TypeError, ValueError):
            pass

        return (
            f"{meter_name} {period_title} enerji karşılaştırması:\n"
            f"• {previous_label}: {previous_value if previous_value is not None else '-'} kWh\n"
            f"• {current_label}: {current_value if current_value is not None else '-'} kWh\n"
            f"• Fark: {difference_text}\n"
            f"• Değişim: {percentage_text}",
            events,
        )

    if wants_summary:
        today_start = datetime.now().replace(
            hour=0,
            minute=0,
            second=0,
            microsecond=0,
        )
        summary_result = call_meter_tool(
            "get_meter_energy_summary",
            {
                "device_id": device_id,
                "start_ts": int(today_start.timestamp() * 1000),
                "end_ts": now_ms,
            },
        )
        error = tool_error_message(summary_result)
        if error:
            return f"Enerji özeti alınamadı: {error}", events

        summary = result_data(summary_result)
        if summary.get("error"):
            return str(summary["error"]), events
        return (
            f"{meter_name} bugünkü enerji özeti:\n"
            f"• Başlangıç: {summary.get('start_energy_kwh', '-')} kWh\n"
            f"• Güncel: {summary.get('end_energy_kwh', '-')} kWh\n"
            f"• Tüketim: {summary.get('consumption_kwh', '-')} kWh",
            events,
        )

    if wants_history:
        hours = 24 if "24" in normalized_query else 1
        if any(word in normalized_query for word in ("gerilim", "voltaj")):
            metric = "voltage_v"
        elif any(word in normalized_query for word in ("akım", "amper")):
            metric = "current_a"
        else:
            metric = "energy_total_kwh"
        history_result = call_meter_tool(
            "get_meter_history",
            {
                "device_id": device_id,
                "metric": metric,
                "start_ts": now_ms - (hours * 60 * 60 * 1000),
                "end_ts": now_ms,
            },
        )
        error = tool_error_message(history_result)
        if error:
            return f"Sayaç geçmişi alınamadı: {error}", events

        history = result_data(history_result)
        points = history.get("points", [])
        if not points:
            return (
                f"Son {hours} saat için {metric} geçmiş verisi bulunamadı.",
                events,
            )
        return (
            f"{meter_name} için son {hours} saatlik {metric} geçmişi:\n"
            f"• Veri noktası sayısı: {len(points)}\n"
            f"• İlk değer: {points[0].get('value', '-')}\n"
            f"• Son değer: {points[-1].get('value', '-')}",
            events,
        )

    reading_result = call_meter_tool(
        "get_latest_meter_reading",
        {"device_id": device_id},
    )
    error = tool_error_message(reading_result)
    if error:
        return f"Sayaç verileri alınamadı: {error}", events

    measurements = result_data(reading_result).get("measurements", {})
    if not isinstance(measurements, dict):
        measurements = {}

    def measurement_value(key: str) -> str:
        item = measurements.get(key)
        if isinstance(item, dict):
            value = item.get("value")
        else:
            value = item
        return "-" if value is None else str(value)

    if wants_phase_comparison:
        if any(term in normalized_query for term in ("gerilim", "voltaj")):
            metric_prefix = "voltage_l"
            metric_suffix = "_v"
            metric_label = "gerilim"
            unit = "V"
        elif any(term in normalized_query for term in ("akım", "amper")):
            metric_prefix = "current_l"
            metric_suffix = "_a"
            metric_label = "akım"
            unit = "A"
        else:
            metric_prefix = "power_factor_l"
            metric_suffix = ""
            metric_label = "güç faktörü"
            unit = ""

        phase_values = []
        lines = [f"{meter_name} faz {metric_label} karşılaştırması:"]
        for phase in sorted(requested_phases):
            key = f"{metric_prefix}{phase}{metric_suffix}"
            value = measurement_value(key)
            suffix = f" {unit}" if unit else ""
            lines.append(f"• L{phase}: {value}{suffix}")
            try:
                phase_values.append((phase, float(value)))
            except ValueError:
                pass

        if len(phase_values) >= 2:
            lowest = min(phase_values, key=lambda item: item[1])
            highest = max(phase_values, key=lambda item: item[1])
            difference = highest[1] - lowest[1]
            suffix = f" {unit}" if unit else ""
            lines.append(
                f"• En yüksek: L{highest[0]} ({highest[1]:.3f}{suffix})"
            )
            lines.append(
                f"• En düşük: L{lowest[0]} ({lowest[1]:.3f}{suffix})"
            )
            lines.append(f"• Fazlar arası fark: {difference:.3f}{suffix}")
        return "\n".join(lines), events

    measurement_labels = (
        ("energy_total_kwh", "Toplam enerji", "kWh", ("enerji", "kwh")),
        ("daily_energy_kwh", "Günlük enerji", "kWh", ("günlük", "bugün")),
        ("weekly_energy_kwh", "Haftalık enerji", "kWh", ("haftalık", "hafta")),
        ("monthly_energy_kwh", "Aylık enerji", "kWh", ("aylık", "ay")),
        ("tariff_1_energy_kwh", "Tarife 1 enerji", "kWh", ("tarife 1",)),
        ("tariff_2_energy_kwh", "Tarife 2 enerji", "kWh", ("tarife 2",)),
        ("tariff_3_energy_kwh", "Tarife 3 enerji", "kWh", ("tarife 3",)),
        ("tariff_4_energy_kwh", "Tarife 4 enerji", "kWh", ("tarife 4",)),
        ("voltage_l1_v", "L1 gerilim", "V", ("gerilim", "voltaj", "l1")),
        ("voltage_l2_v", "L2 gerilim", "V", ("gerilim", "voltaj", "l2")),
        ("voltage_l3_v", "L3 gerilim", "V", ("gerilim", "voltaj", "l3")),
        ("current_l1_a", "L1 akım", "A", ("akım", "amper", "l1")),
        ("current_l2_a", "L2 akım", "A", ("akım", "amper", "l2")),
        ("current_l3_a", "L3 akım", "A", ("akım", "amper", "l3")),
        ("frequency_hz", "Frekans", "Hz", ("frekans", "hz")),
        ("power_factor_l1", "L1 güç faktörü", "", ("güç faktörü", "cos", "l1")),
        ("power_factor_l2", "L2 güç faktörü", "", ("güç faktörü", "cos", "l2")),
        ("power_factor_l3", "L3 güç faktörü", "", ("güç faktörü", "cos", "l3")),
        ("connection_status", "Bağlantı durumu", "", ("bağlantı", "online", "offline")),
    )
    requested_measurement_keys = [
        key
        for key, _label, _unit, phrases in measurement_labels
        if any(phrase in normalized_query for phrase in phrases)
    ]
    if not requested_measurement_keys:
        requested_measurement_keys = [
            key for key, _label, _unit, _phrases in measurement_labels
        ]

    lines = [f"{meter_name} güncel değerleri:"]
    for key, label, unit, _phrases in measurement_labels:
        if key not in requested_measurement_keys:
            continue
        suffix = f" {unit}" if unit else ""
        lines.append(
            f"• {label}: {measurement_value(key)}{suffix}"
        )
    return "\n".join(lines), events
