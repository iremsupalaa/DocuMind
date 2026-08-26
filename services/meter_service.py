"""Sayaç sorularını uygun MCP aracına yönlendiren servis."""

import re
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Tuple

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
        "katlara göre", "kata göre", "güç kaynağına göre",
        "duruma göre", "hata koduna göre", "hata kodlarına göre",
        "faz sayısına göre", "grupla", "gruplandır",
    ))
    wants_fleet_summary = any(phrase in normalized_query for phrase in (
        "tüm sayaçların durumu", "sayaçların genel durumu",
        "genel özet", "filo özeti", "kaç sayaç güncel",
        "kaç sayaç veri", "toplam günlük tüketim",
        "toplam haftalık tüketim", "toplam aylık tüketim",
    ))

    if (
        wants_comparison
        and not wants_period_comparison
        and not wants_phase_comparison
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

    if wants_ranking:
        metric = requested_bulk_metrics(query)[0]
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
        for index, meter in enumerate(ranking.get("meters", []), 1):
            suffix = f" {unit}" if unit else ""
            lines.append(
                f"{index}. {meter.get('name', 'Adsız sayaç')}: "
                f"{meter.get('value', '-')}{suffix}"
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
            {"attribute": attribute, "include_meter_names": False},
        )
        error = tool_error_message(grouping_result)
        if error:
            return f"Sayaçlar gruplanamadı: {error}", events
        grouping = result_data(grouping_result)
        if grouping.get("error"):
            return str(grouping["error"]), events
        lines = [f"Sayaçların {attribute} alanına göre dağılımı:"]
        for group in grouping.get("groups", []):
            value = group.get("value")
            rendered = "Tanımsız" if value is None else str(value)
            lines.append(f"• {rendered}: {group.get('count', 0)} sayaç")
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
            return "-" if value is None else str(value)
        return "-"

    if wants_phase_comparison:
        if any(term in normalized_query for term in ("gerilim", "voltaj")):
            metric_prefix = "voltage_l"
            metric_label = "gerilim"
            unit = "V"
        elif any(term in normalized_query for term in ("akım", "amper")):
            metric_prefix = "current_l"
            metric_label = "akım"
            unit = "A"
        else:
            metric_prefix = "power_factor_l"
            metric_label = "güç faktörü"
            unit = ""

        phase_values = []
        lines = [f"{meter_name} faz {metric_label} karşılaştırması:"]
        for phase in sorted(requested_phases):
            key = f"{metric_prefix}{phase}"
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
