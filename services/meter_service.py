"""Sayaç sorularını uygun MCP aracına yönlendiren servis."""

import time
from datetime import datetime
from typing import Any, Dict, List, Tuple

from services.mcp_service import MCPClient, result_data, tool_error_message


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
    ))

    if wants_list and not wants_reading:
        lines = [f"ThingsBoard hesabında {len(meters)} sayaç bulundu:"]
        for index, meter in enumerate(meters, 1):
            lines.append(
                f"{index}. {meter.get('name', 'Adsız sayaç')} "
                f"(ID: {meter.get('device_id', '-')})"
            )
        return "\n".join(lines), events

    meter = meters[0]
    meter_name = str(meter.get("name", "Sayaç"))
    device_id = str(meter.get("device_id", ""))
    if not device_id:
        return "Sayaç kimliği ThingsBoard yanıtında bulunamadı.", events

    wants_context = any(word in normalized_query for word in (
        "konum", "bina", "kat", "model", "seri numarası",
        "seri no", "firmware", "nerede",
    ))
    wants_connection = any(word in normalized_query for word in (
        "çevrim içi", "çevrimiçi", "çevrim dışı", "offline",
        "online", "son telemetri", "bağlantı durumu",
        "bağlantı bilgi", "son veri", "ne zaman veri",
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
        ("meterType", "Sayaç tipi", ("sayaç tipi", "faz")),
        ("serialNumber", "Seri numarası", ("seri numarası", "seri no")),
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
        metric = "voltage_v" if any(word in normalized_query for word in (
            "gerilim", "voltaj",
        )) else "active_power_kw"
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
            return str(item.get("value", "-"))
        return "-"

    measurement_labels = (
        ("energy_total_kwh", "Toplam enerji", "kWh", ("enerji", "kwh")),
        ("active_power_kw", "Anlık aktif güç", "kW", ("güç", "kw")),
        ("voltage_v", "Gerilim", "V", ("gerilim", "voltaj")),
        ("current_a", "Akım", "A", ("akım", "amper")),
        ("frequency_hz", "Frekans", "Hz", ("frekans", "hz")),
        ("power_factor", "Güç faktörü", "", ("güç faktörü", "cos")),
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
