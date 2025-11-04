from __future__ import annotations
# =====================================================
# FastAPI: Chat único con Function Calling + Redis + Recurrentes + Pomodoro + Presets (Memoria)
# - /task?task=...
# - Salida siempre a MacroDroid (sin emojis/markdown)
# - "Bonitos" via LLM (intermedios/finales/confirmación). Recurrencia y Pomodoro incluidos.
# - Recurrentes:
#     * Modo intervalo: cada N minutos/horas/días; optional "hasta <YYYY-MM-DD HH:MM>"
#     * Modo diario fijo: todos los días a HH:MM
# - Pomodoro: fases trabajo/descanso corto/largo, pausables/cancelables
# - Presets: guardar/activar/listar/borrar (pomodoro, recurrentes y one-shot)
# - Confirmación al vencer: preguntar si ya lo hiciste, pedir duración; insistir con frecuencia creciente
# - "pendientes": lista numerada simple, sin IDs
# - Zona horaria: America/Bogota
# - MacroDroid: se envía chat_entry, require_response=true/false, require_sound=true/false
# Requisitos: pip install fastapi uvicorn requests redis
# Env: OPENAI_API_KEY, REDIS_URL, opcional MACRODROID_BASE
# Ejecuta: uvicorn app:app --host 0.0.0.0 --port 8000
# =====================================================
from fastapi import FastAPI, Query, HTTPException
from urllib.parse import quote_plus
from datetime import datetime, timedelta, timezone, time as dtime
from typing import Dict, List, Optional, Tuple
import os, json, time, threading, re, requests, redis

# -------- Config --------
TZ = timezone(timedelta(hours=-5))  # America/Bogota
OPENAI_URL = "https://api.openai.com/v1/chat/completions"
MODEL = "gpt-4o-mini"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MACRODROID_BASE = os.getenv(
    "MACRODROID_BASE",
    "https://trigger.macrodroid.com/52971f1c-b406-48e5-9f4d-dfb857c2e33e/chatbot_entry",
)
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

if not OPENAI_API_KEY:
    raise RuntimeError("Falta OPENAI_API_KEY")

# -------- Redis --------
r = redis.from_url(REDIS_URL, decode_responses=True)

# Claves Redis (one-shot)
KEY_CHAT_HISTORY   = "chat:history"                 # LIST JSON: {role, content, ts}
KEY_REM_Z          = "reminders:zset"               # ZSET: score=epoch next_ping_at / next_nag_at
KEY_REM_SEQ        = "reminders:seq"                # contador autoincremental compartido
KEY_REM_HASH       = lambda rid: f"reminders:{rid}" # HASH por id
KEY_REM_PENDING    = "reminders:pending"            # SET ids pending/await_confirm
KEY_REM_ALL        = "reminders:all"                # SET con TODOS los ids (cualquier estado)

# Claves Redis (recurrentes / pomodoro)
KEY_REC_Z          = "recurring:zset"               # ZSET próximo disparo
KEY_REC_SET        = "recurring:active"             # SET ids activos (recurrente/pomodoro)
KEY_REC_HASH       = lambda rid: f"recurring:{rid}" # HASH por id
KEY_REC_ALL        = "recurring:all"                # SET con TODOS los ids recurrentes/pomodoro

# -------- Memoria de presets --------
KEY_PRESET_SET           = "presets:names"               # SET de nombres (normalizados)
KEY_PRESET_HASH          = lambda name: f"preset:{name}" # HASH con metadatos
PRESET_KIND_POMODORO     = "pomodoro"
PRESET_KIND_RECURRING    = "recurring"
PRESET_KIND_ONESHOT      = "oneshot"

# Mínimo 5 minutos entre avisos
MIN_INTERVAL_SEC = 5 * 60

# Escalada de insistencia tras el vencimiento (minutos)
NAG_SCHEDULE_MIN = [15, 10, 5, 3, 2, 1]  # luego queda 1 min constante

# -------- Utilidades --------
def now_ts() -> float:
    return datetime.now(TZ).timestamp()

def ts_str_col(ts: float) -> str:
    return datetime.fromtimestamp(ts, TZ).strftime("%Y-%m-%d %H:%M:%S")

def parse_iso_local(iso_: str) -> Optional[float]:
    try:
        dt = datetime.strptime(iso_, "%Y-%m-%d %H:%M").replace(tzinfo=TZ)
        return dt.timestamp()
    except Exception:
        return None

def today_range() -> Tuple[float,float]:
    now = datetime.now(TZ)
    start = datetime(now.year, now.month, now.day, 0, 0, tzinfo=TZ)
    end   = start + timedelta(days=1) - timedelta(seconds=1)
    return (start.timestamp(), end.timestamp())

def push_history(role: str, content: str) -> None:
    item = {"role": role, "content": content, "ts": ts_str_col(now_ts())}
    r.lpush(KEY_CHAT_HISTORY, json.dumps(item))
    r.ltrim(KEY_CHAT_HISTORY, 0, 49)  # máx 50 mensajes

def history_for_openai(n: int = 20) -> List[Dict]:
    raw = r.lrange(KEY_CHAT_HISTORY, 0, n - 1)
    items = list(reversed([json.loads(x) for x in raw]))
    out = []
    for it in items:
        role = it.get("role", "user")
        if role not in ("system", "user", "assistant"):
            role = "user"
        out.append({"role": role, "content": it.get("content", "")})
    return out

def sanitize_for_tts(text: str) -> str:
    text = re.sub(r"[\u2600-\u27BF\uE000-\uF8FF\U0001F000-\U0001FAFF]+", "", text)  # emojis/símbolos
    text = re.sub(r"[\\*`_~•●■▪▫►✔✅➜➤➔➖➕]+", "", text)                           # decoradores
    text = re.sub(r"^[\-\+\>]\s*", "", text, flags=re.MULTILINE)                    # bullets ascii
    text = re.sub(r"\s+", " ", text).strip()
    return text

# ---- Inferencia de si requiere respuesta del usuario ----
CLASSIFIER_SYSTEM = (
    "Eres un clasificador en español. Devuelve SOLO true o false (minúsculas), "
    "sin explicación. Respondes true si el MENSAJE DEL ASISTENTE espera una "
    "respuesta del usuario (p.ej. hace una pregunta, pide confirmación, ofrece "
    "opciones, solicita datos, dice '¿ya lo hiciste?', 'quieres', 'puedo', etc.). "
    "Respondes false si NO se espera respuesta (avisos informativos, listados, "
    "confirmaciones de sistema). Considera también el MENSAJE DEL USUARIO como contexto."
)

QUESTION_RX = re.compile(
    r"[¿?]|(confirma|confirmas|quieres|deseas|prefieres|indica|elige|elige una opción|te parece|puedo|debo|quedo atento|avísame|avisame|dime|cuál|cual|cuándo|cuando|dónde|donde|ya lo hiciste|aplazar|cancelar)\b",
    re.I
)

def _infer_require_response_fallback(user_text: str, assistant_text: str) -> bool:
    a = (assistant_text or "").strip()
    if not a:
        return False
    if QUESTION_RX.search(a):
        return True
    if re.search(r"\b(pendiente|pendientes|mis recordatorios|lista presets|listar presets)\b", user_text or "", re.I):
        return False
    return False

def infer_require_response(user_text: str, assistant_text: str) -> bool:
    a = (assistant_text or "").strip()
    if not a:
        return False
    try:
        headers = {
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "Content-Type": "application/json",
        }
        user_payload = json.dumps({
            "mensaje_usuario": (user_text or "").strip(),
            "mensaje_asistente": a
        }, ensure_ascii=False)

        payload = {
            "model": MODEL,
            "messages": [
                {"role": "system", "content": CLASSIFIER_SYSTEM},
                {"role": "user", "content": user_payload}
            ],
            "temperature": 0.0,
            "max_tokens": 3,
        }
        resp = requests.post(OPENAI_URL, json=payload, headers=headers, timeout=10)
        if resp.ok:
            out = (resp.json()["choices"][0]["message"]["content"] or "").strip().lower()
            if out in ("true", "false"):
                return out == "true"
            if out in ("sí", "si", "yes", "1"):
                return True
            if out in ("no", "0"):
                return False
    except Exception:
        pass
    return _infer_require_response_fallback(user_text, a)

def send_to_macrodroid(text: str, require_response: bool, require_sound: bool) -> None:
    plain = sanitize_for_tts(text)
    rr = "true" if require_response else "false"
    rs = "true" if require_sound else "false"
    url = (
        f"{MACRODROID_BASE}"
        f"?chat_entry={quote_plus(plain)}"
        f"&require_response={quote_plus(rr)}"
        f"&require_sound={quote_plus(rs)}"
    )
    resp = requests.post(url, timeout=10)
    if not resp.ok:
        raise HTTPException(status_code=502, detail=f"MacroDroid respondió {resp.status_code}")

# -------- Mensajes “bonitos” con LLM --------
def llm_rewrite(kind: str, base_text: str, mins_left: Optional[int] = None, phase: Optional[str] = None, nag_level: int = 0) -> str:
    system = (
        "Eres un asistente que redacta recordatorios en español, tono cordial y motivador. "
        "No uses emojis ni markdown. Una sola oración clara y breve. "
        "Dirígete a 'senor Andres'."
    )
    if kind == "intermediate":
        user = f"Redacta un aviso intermedio indicando que faltan {mins_left} minutos y recordando: '{base_text}'."
    elif kind == "final":
        user = f"Redacta el aviso final recordando: '{base_text}', con un toque motivador breve."
    elif kind == "recurring":
        user = f"Escribe un aviso breve para recordar: '{base_text}', tono cordial y directo."
    elif kind in ("confirm","confirm_nag"):
        tono = "amable" if nag_level <= 1 else ("más directo" if nag_level <= 3 else "enfático pero respetuoso")
        user = (
            f"Redacta en tono {tono} una pregunta breve para confirmar si ya realizó: '{base_text}'. "
            "Si ya lo hizo, pídale que indique cuánto se tardó. "
            "Si aún no puede, ofrézcale aplazar unos minutos o cancelar."
        )
    else:  # pomodoro
        if phase == "work":
            user = f"Anuncia el inicio de un bloque de trabajo recordando: '{base_text}', con tono motivador breve."
        elif phase == "short_break":
            user = "Anuncia un descanso corto para respirar y estirar, con tono amable y breve."
        else:
            user = "Anuncia un descanso largo para recuperar energía, con tono amable y breve."

    try:
        headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}
        payload = {
            "model": MODEL,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "temperature": 0.5,
            "max_tokens": 80,
        }
        resp = requests.post(OPENAI_URL, json=payload, headers=headers, timeout=15)
        if resp.ok:
            txt = resp.json()["choices"][0]["message"]["content"].strip()
            return sanitize_for_tts(txt)[:400]
    except Exception:
        pass

    # Fallbacks
    if kind == "intermediate":
        return f"Hola senor Andres, en {mins_left} minutos por favor recuerde: {base_text}."
    if kind == "final":
        return f"Hola senor Andres, no olvide por favor que debe {base_text}. Es importante para mantener una buena disciplina."
    if kind == "recurring":
        return f"Hola senor Andres, por favor recuerde: {base_text}."
    if kind in ("confirm","confirm_nag"):
        return f"Hola senor Andres, ¿ya realizó {base_text}? Si ya lo hizo cuénteme cuánto se tardó; si no puede ahora, puedo aplazarlo o cancelarlo."
    # pomodoro
    if phase == "work":
        return f"Hola senor Andres, iniciemos el bloque de trabajo: {base_text}."
    if phase == "short_break":
        return "Hola senor Andres, tome un descanso corto para respirar y estirar."
    return "Hola senor Andres, tome un descanso largo para recuperar energía."

# -------- One-shot (mitades y confirmación final) --------
def reminder_doc(rid: str, text: str, due_at: float, next_ping_at: float, status: str = "pending", created_at: Optional[float] = None, nag_level: int = 0) -> Dict:
    return {
        "id": rid,
        "text": text,
        "status": status,  # pending | await_confirm | done | cancelled
        "created_at": str(created_at if created_at else now_ts()),
        "done_at": "",            # ts cuando se marcó done
        "cancelled_at": "",       # ts cuando se canceló
        "due_at": str(due_at),
        "next_ping_at": str(next_ping_at),  # próximo aviso intermedio o próxima insistencia
        "nag_level": str(nag_level),        # sólo en await_confirm
    }

def schedule_from_natural(text: str) -> Optional[int]:
    m = re.search(r"\ben\s+(\d+)\s*(minutos|min|hora|horas)\b", text, re.IGNORECASE)
    if not m:
        return None
    qty = int(m.group(1))
    unit = m.group(2).lower()
    return qty * 60 if unit.startswith("min") else qty * 3600

# ---- Clasificador “pendientes” ----
PENDING_CLASSIFIER_SYSTEM = (
    "Eres un clasificador en español. Devuelve SOLO true o false en minúsculas. "
    "Responde true si el MENSAJE DEL USUARIO está pidiendo ver sus recordatorios, "
    "tareas pendientes, 'mis recordatorios', 'qué me falta', 'qué tengo por hacer', "
    "'lista de pendientes', 'qué hay pendiente', 'qué tengo hoy', etc. "
    "Responde false en cualquier otro caso."
)

_PENDING_RX_FALLBACK = re.compile(
    r"\b("
    r"pendiente[s]?|mis\s+recordatorios|recordatorios\s+pendientes|tareas\s+pendientes|"
    r"lista\s+de\s+pendientes|qué\s+me\s+falta|que\s+me\s+falta|qué\s+tengo\s+por\s+hacer|"
    r"que\s+tengo\s+por\s+hacer|qué\s+hay\s+pendiente|que\s+hay\s+pendiente|"
    r"qué\s+tengo\s+hoy|que\s+tengo\s+hoy|mis\s+tareas|ver\s+pendientes|"
    r"mostrar\s+pendientes|muéstrame\s+pendientes|muestrame\s+pendientes|"
    r"qué\s+falta\s+por\s+hacer|que\s+falta\s+por\s+hacer"
    r")\b",
    re.I
)

def infer_is_pending_request(user_text: str) -> bool:
    txt = (user_text or "").strip()
    if not txt:
        return False
    try:
        headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}
        payload = {
            "model": MODEL,
            "messages": [
                {"role": "system", "content": PENDING_CLASSIFIER_SYSTEM},
                {"role": "user", "content": txt}
            ],
            "temperature": 0.0,
            "max_tokens": 3,
        }
        resp = requests.post(OPENAI_URL, json=payload, headers=headers, timeout=10)
        if resp.ok:
            out = (resp.json()["choices"][0]["message"]["content"] or "").strip().lower()
            if out in ("true", "false"):
                return out == "true"
            if out in ("sí", "si", "yes", "1"):
                return True
            if out in ("no", "0"):
                return False
    except Exception:
        pass
    return bool(_PENDING_RX_FALLBACK.search(txt))

# ---- Clasificador HISTÓRICO (atajos de servidor) ----
_HISTORY_RX_ALL = re.compile(r"\b(historial|mostrar\s+todo|todo\s+el\s+historial|mostrar\s+t[oó]do)\b", re.I)
_HISTORY_RX_DONE_TODAY = re.compile(r"\b(hechos\s+hoy|completados\s+hoy|lo\s+que\s+hice\s+hoy|qu[eé]\s+hice\s+hoy|qu[eé]\s+complet[eé]\s+hoy)\b", re.I)
_HISTORY_RX_CANCELLED = re.compile(r"\b(cancelados|canceladas|recordatorios\s+cancelados)\b", re.I)

def infer_history_intent(user_text: str) -> Optional[str]:
    txt = (user_text or "").strip()
    if not txt:
        return None
    if _HISTORY_RX_DONE_TODAY.search(txt):
        return "done_today"
    if _HISTORY_RX_CANCELLED.search(txt):
        return "cancelled"
    if _HISTORY_RX_ALL.search(txt):
        return "all"
    # variantes genéricas
    if re.search(r"\b(pasados|anteriores|hist[oó]ricos|historia)\b", txt, re.I):
        return "all"
    return None

SYSTEM_PROMPT = (
    "Eres un asistente en español, claro y sin adornos. "
    "No uses emojis, asteriscos, guiones ni decoraciones. Zona horaria: America/Bogota. "
    "Cuando el usuario diga 'ya hice X', marca ese recordatorio como hecho (usa mark_done_by_text). "
    "Si pide 'pendientes', llama a list_pending_reminders y list_recurring, y responde con lista numerada simple: "
    "'1. <texto> - para  MM-DD HH:MM  - falta <tiempo>' para one-shot, y "
    "'recurrente: <texto> - siguiente  MM-DD HH:MM  - cada <X> - hasta <fecha>' para recurrentes. "
    "Si pide reactivar, usa reactivate_reminder o reactivate_by_text. "
    "Para frases del estilo 'cada N minutos/horas/días' usa create_recurring; "
    "si incluye 'hasta <fecha>' mapea a until_iso. "
    "Si menciona 'pomodoro', crea uno con create_pomodoro; permite pausa/reanudar/cancelar. "
    "=== Confirmación al vencer === "
    "Cuando yo te avise que algo venció, no lo marques como hecho; pregúntale al usuario si ya lo hizo y, si ya lo hizo, "
    "pídele que indique cuánto se tardó. Si no puede hacerlo ahora, ofrécele aplazar ('aplazar 10 minutos') o cancelar ('cancelar <texto>'). "
    "Para aplazar por texto usa snooze_by_text; para cancelar por texto usa cancel_reminder_by_text. "
    "=== Presets === "
    "Si el usuario dice 'guarda/crear preset' con nombre y tipo (pomodoro/recordatorio/one-shot) usa las funciones save_*_preset. "
    "Si dice 'activa/usar/arranca' + nombre, decide si es pomodoro/recurring/oneshot según lo guardado y usa start_*_preset. "
    "Si pide 'lista presets', llama a list_presets y responde un resumen simple. "
    "=== HISTÓRICO y EDICIÓN === "
    "Si pide 'mostrar todo', 'hechos hoy', 'cancelados', o 'historial', usa las funciones list_all_reminders, list_done_today, list_cancelled. "
    "Si pide 'editar' o 'mueve X a HH:MM' o 'cambia el texto a ...', usa edit_reminder_by_text o edit_recurring_by_text según corresponda. "
    "Para diarios a hora fija usa edit_recurring_* con daily_time='HH:MM'. "
)

PENDING_RX = re.compile(r"\b(pendiente|pendientes|tareas pendientes|mis recordatorios)\b", re.I)

# -------- Planificador de pings --------
def schedule_first_ping(due_at: float) -> float:
    now = now_ts()
    remaining = max(0, due_at - now)
    half = remaining / 2.0
    if half >= MIN_INTERVAL_SEC:
        return now + half
    pre = due_at - MIN_INTERVAL_SEC
    if pre > now:
        return pre
    return now + MIN_INTERVAL_SEC

def schedule_next_ping(due_at: float) -> Optional[float]:
    now = now_ts()
    remaining = due_at - now
    if remaining <= MIN_INTERVAL_SEC:
        return None
    half = remaining / 2.0
    next_at = now + max(half, MIN_INTERVAL_SEC)
    if next_at >= due_at:
        return None
    return next_at

def schedule_next_nag(nag_level: int) -> float:
    minutes = NAG_SCHEDULE_MIN[min(nag_level, len(NAG_SCHEDULE_MIN)-1)]
    return max(60, minutes * 60)

# -------- DB helpers: One-shot --------
def create_reminder_db(text: str, due_ts: float) -> str:
    rid = str(r.incr(KEY_REM_SEQ))
    first_ping = schedule_first_ping(due_ts)
    doc = reminder_doc(rid, text, due_ts, first_ping, status="pending")
    r.hset(KEY_REM_HASH(rid), mapping=doc)
    r.zadd(KEY_REM_Z, {rid: float(doc["next_ping_at"])})
    r.sadd(KEY_REM_PENDING, rid)
    r.sadd(KEY_REM_ALL, rid)
    return rid

def list_pending_db() -> List[Dict]:
    ids = sorted(list(r.smembers(KEY_REM_PENDING)), key=lambda x: int(x))
    out = []
    for rid in ids:
        data = r.hgetall(KEY_REM_HASH(rid))
        if not data or data.get("status") not in ("pending","await_confirm"):
            continue
        out.append({
            "id": rid,
            "text": data.get("text"),
            "due_at_col": ts_str_col(float(data.get("due_at", str(now_ts())))),
            "next_ping_at_col": ts_str_col(float(data.get("next_ping_at", str(now_ts())))),
            "status": data.get("status"),
            "type": "one_shot",
        })
    return out

def cancel_reminder_db(rid: str) -> bool:
    h = KEY_REM_HASH(rid)
    if not r.exists(h): return False
    r.hset(h, mapping={"status": "cancelled", "cancelled_at": str(now_ts())})
    r.srem(KEY_REM_PENDING, rid)
    r.zrem(KEY_REM_Z, rid)
    r.sadd(KEY_REM_ALL, rid)
    return True

def mark_done_db(rid: str) -> bool:
    h = KEY_REM_HASH(rid)
    if not r.exists(h): return False
    r.hset(h, mapping={"status": "done", "done_at": str(now_ts())})
    r.srem(KEY_REM_PENDING, rid)
    r.zrem(KEY_REM_Z, rid)
    r.sadd(KEY_REM_ALL, rid)
    return True

def find_by_text_candidate(q: str) -> Optional[str]:
    qn = q.lower().strip()
    ids = sorted(list(r.smembers(KEY_REM_ALL)), key=lambda x: int(x))
    for rid in reversed(ids):
        data = r.hgetall(KEY_REM_HASH(rid))
        if not data: continue
        if qn in (data.get("text","").lower()):
            return rid
    return None

def mark_done_by_text(q: str) -> Optional[str]:
    rid = find_by_text_candidate(q)
    if not rid: return None
    if mark_done_db(rid): return rid
    return None

def reactivate_by_id(rid: str, in_minutes: Optional[int] = None) -> bool:
    h = KEY_REM_HASH(rid)
    if not r.exists(h): return False
    data = r.hgetall(h)
    text = data.get("text", "")
    due = now_ts() + (max(1, in_minutes) * 60 if in_minutes is not None else 10 * 60)
    first = schedule_first_ping(due)
    r.hset(h, mapping={"status": "pending", "due_at": str(due), "next_ping_at": str(first), "text": text, "nag_level": "0"})
    r.sadd(KEY_REM_PENDING, rid)
    r.zadd(KEY_REM_Z, {rid: first})
    r.sadd(KEY_REM_ALL, rid)
    return True

def reactivate_by_text(q: str, in_minutes: Optional[int] = None) -> Optional[str]:
    rid = find_by_text_candidate(q)
    if not rid: return None
    ok = reactivate_by_id(rid, in_minutes=in_minutes)
    return rid if ok else None

def snooze_db(rid: str, minutes: int) -> bool:
    h = KEY_REM_HASH(rid)
    if not r.exists(h): return False
    due = now_ts() + max(1, minutes) * 60
    next_ping = schedule_first_ping(due)
    r.hset(h, mapping={"status": "pending", "due_at": str(due), "next_ping_at": str(next_ping), "nag_level": "0"})
    r.sadd(KEY_REM_PENDING, rid)
    r.zadd(KEY_REM_Z, {rid: next_ping})
    r.sadd(KEY_REM_ALL, rid)
    return True

def edit_reminder_db(rid: str, new_text: Optional[str]=None, when_iso: Optional[str]=None, in_minutes: Optional[int]=None) -> bool:
    h = KEY_REM_HASH(rid)
    if not r.exists(h): return False
    d = r.hgetall(h)
    text = new_text.strip() if (isinstance(new_text,str) and new_text.strip()) else d.get("text","")
    if in_minutes is not None and int(in_minutes) >= 1:
        due = now_ts() + int(in_minutes) * 60
    elif when_iso:
        parsed = parse_iso_local(when_iso)
        due = parsed if parsed else now_ts() + 10*60
    else:
        # solo cambia el texto
        r.hset(h, mapping={"text": text})
        r.sadd(KEY_REM_ALL, rid)
        return True
    first = schedule_first_ping(due)
    r.hset(h, mapping={
        "text": text, "status":"pending", "due_at": str(due), "next_ping_at": str(first), "nag_level":"0"
    })
    r.sadd(KEY_REM_PENDING, rid)
    r.zadd(KEY_REM_Z, {rid: first})
    r.sadd(KEY_REM_ALL, rid)
    return True

def edit_reminder_by_text(q: str, new_text: Optional[str]=None, when_iso: Optional[str]=None, in_minutes: Optional[int]=None) -> Optional[str]:
    rid = find_by_text_candidate(q)
    if not rid: return None
    ok = edit_reminder_db(rid, new_text=new_text, when_iso=when_iso, in_minutes=in_minutes)
    return rid if ok else None

# -------- Recurrencia --------
def interval_human(sec: int) -> str:
    if sec % 86400 == 0: return f"{sec//86400} días"
    if sec % 3600 == 0:  return f"{sec//3600} horas"
    return f"{max(1,sec//60)} minutos"

def parse_recurring(text: str) -> Optional[Dict]:
    # Modo intervalo
    m = re.search(r"\bcada\s+(\d+)\s*(minutos|min|horas|hora|d[ií]as|d[ií]a)\b", text, re.I)
    if m:
        qty = int(m.group(1))
        unit = m.group(2).lower()
        if unit.startswith("min"): interval = qty * 60
        elif unit.startswith("hora"): interval = qty * 3600
        else: interval = qty * 86400
        interval = max(interval, MIN_INTERVAL_SEC)
        um = re.search(r"\bhasta\s+(\d{4}-\d{2}-\d{2})(?:\s+(\d{2}:\d{2}))?\b", text, re.I)
        until_ts = None
        if um:
            date_part = um.group(1)
            time_part = um.group(2) or "23:59"
            try:
                dt = datetime.strptime(f"{date_part} {time_part}", "%Y-%m-%d %H:%M").replace(tzinfo=TZ)
                until_ts = dt.timestamp()
            except Exception:
                until_ts = None
        return {"mode":"interval", "interval_sec": interval, "until_ts": until_ts}

    # Modo diario fijo: “cada día a las 3 am”, “todos los días 03:00”
    m2 = re.search(r"(cada\s+d[ií]a|todos?\s+los\s+d[ií]as).*(\d{1,2})[:\.]?(\d{2})?\s*(am|pm)?", text, re.I)
    if m2:
        hh = int(m2.group(2))
        mm = int(m2.group(3) or "0")
        ap = (m2.group(4) or "").lower()
        if ap == "pm" and hh < 12: hh += 12
        if ap == "am" and hh == 12: hh = 0
        hh = max(0, min(23, hh))
        mm = max(0, min(59, mm))
        return {"mode":"daily_time", "hh": hh, "mm": mm}

    return None

def next_daily_time_epoch(hh: int, mm: int, from_ts: Optional[float]=None) -> float:
    base = datetime.fromtimestamp(from_ts or now_ts(), TZ)
    target_today = datetime(base.year, base.month, base.day, hh, mm, tzinfo=TZ)
    if target_today.timestamp() > (from_ts or now_ts()):
        return target_today.timestamp()
    tomorrow = target_today + timedelta(days=1)
    return tomorrow.timestamp()

def create_recurring_db(text: str, interval_sec: int, until_ts: Optional[float]) -> str:
    rid = str(r.incr(KEY_REM_SEQ))
    due = now_ts() + interval_sec
    doc = {
        "id": rid, "kind": "recurring", "text": text,
        "mode": "interval",
        "interval_sec": str(interval_sec),
        "until_ts": str(until_ts) if until_ts else "",
        "hhmm": "",    # reservado para modo daily_time (HH:MM)
        "next_at": str(due),
        "status": "active"
    }
    r.hset(KEY_REC_HASH(rid), mapping=doc)
    r.sadd(KEY_REC_SET, rid)
    r.zadd(KEY_REC_Z, {rid: due})
    r.sadd(KEY_REC_ALL, rid)
    return rid

def create_recurring_daily_time_db(text: str, hh: int, mm: int) -> str:
    rid = str(r.incr(KEY_REM_SEQ))
    due = next_daily_time_epoch(hh, mm)
    doc = {
        "id": rid, "kind": "recurring", "text": text,
        "mode": "daily_time",
        "interval_sec": "",
        "until_ts": "",
        "hhmm": f"{hh:02d}:{mm:02d}",
        "next_at": str(due),
        "status": "active"
    }
    r.hset(KEY_REC_HASH(rid), mapping=doc)
    r.sadd(KEY_REC_SET, rid)
    r.zadd(KEY_REC_Z, {rid: due})
    r.sadd(KEY_REC_ALL, rid)
    return rid

def list_recurring_db() -> List[Dict]:
    ids = sorted(list(r.smembers(KEY_REC_SET)), key=lambda x: int(x))
    out = []
    for rid in ids:
        h = KEY_REC_HASH(rid)
        if not r.exists(h): continue
        d = r.hgetall(h)
        if d.get("status") != "active": continue
        mode = d.get("mode","interval")
        until = d.get("until_ts")
        if mode == "interval":
            interval = int(float(d.get("interval_sec","0")))
            cada = interval_human(interval)
        else:
            cada = f"diariamente a las {d.get('hhmm','--:--')}"
        out.append({
            "id": rid, "text": d.get("text",""), "type": "recurring",
            "next_at_col": ts_str_col(float(d.get("next_at", str(now_ts())))),
            "interval_human": cada,
            "until_col": ts_str_col(float(until)) if until else "indefinido",
            "status": "active",
            "mode": mode
        })
    return out

def cancel_recurring_db(rid: str) -> bool:
    h = KEY_REC_HASH(rid)
    if not r.exists(h): return False
    r.hset(h, mapping={"status":"cancelled"})
    r.srem(KEY_REC_SET, rid)
    r.zrem(KEY_REC_Z, rid)
    r.sadd(KEY_REC_ALL, rid)
    return True

def pause_recurring_db(rid: str) -> bool:
    h = KEY_REC_HASH(rid)
    if not r.exists(h): return False
    r.hset(h, mapping={"status":"paused"})
    r.zrem(KEY_REC_Z, rid)
    r.sadd(KEY_REC_ALL, rid)
    return True

def resume_recurring_db(rid: str) -> bool:
    h = KEY_REC_HASH(rid)
    if not r.exists(h): return False
    d = r.hgetall(h)
    if d.get("status") not in ("paused","active"): return False
    mode = d.get("mode","interval")
    if mode == "interval":
        interval = int(float(d.get("interval_sec","600")))
        nxt = now_ts() + interval
    else:
        hh, mm = map(int, (d.get("hhmm","00:00").split(":")))
        nxt = next_daily_time_epoch(hh, mm)
    r.hset(h, mapping={"status":"active","next_at":str(nxt)})
    r.sadd(KEY_REC_SET, d["id"])
    r.zadd(KEY_REC_Z, {d["id"]: nxt})
    r.sadd(KEY_REC_ALL, d["id"])
    return True

def find_recurring_by_text(q: str) -> Optional[str]:
    qn = q.lower().strip()
    ids = sorted(list(r.smembers(KEY_REC_ALL)), key=lambda x: int(x))
    for rid in reversed(ids):
        d = r.hgetall(KEY_REC_HASH(rid))
        if not d: continue
        if qn in (d.get("text","").lower()): return rid
    return None

def edit_recurring_db(
    rid: str,
    new_text: Optional[str]=None,
    interval_minutes: Optional[int]=None,
    until_iso: Optional[str]=None,
    daily_time: Optional[str]=None
) -> bool:
    h = KEY_REC_HASH(rid)
    if not r.exists(h): return False
    d = r.hgetall(h)
    mode = d.get("mode","interval")
    text = new_text.strip() if (isinstance(new_text,str) and new_text.strip()) else d.get("text","")

    update = {"text": text}

    if daily_time:
        # Cambiamos a modo diario fijo HH:MM
        try:
            hh, mm = map(int, daily_time.split(":"))
            hh = max(0,min(23,hh)); mm = max(0,min(59,mm))
        except Exception:
            return False
        nxt = next_daily_time_epoch(hh, mm)
        update.update({
            "mode":"daily_time", "hhmm": f"{hh:02d}:{mm:02d}",
            "interval_sec":"", "until_ts":"", "next_at": str(nxt), "status":"active"
        })
        r.hset(h, mapping=update)
        r.sadd(KEY_REC_SET, rid)
        r.zadd(KEY_REC_Z, {rid: nxt})
        r.sadd(KEY_REC_ALL, rid)
        return True

    # Si no hay daily_time, podemos seguir en modo intervalo (o convertir a intervalo)
    if interval_minutes is not None:
        interval_sec = max(MIN_INTERVAL_SEC, int(interval_minutes)*60)
        nxt = now_ts() + interval_sec
        update.update({"mode":"interval", "interval_sec": str(interval_sec), "next_at": str(nxt), "status":"active"})
    # until opcional
    if until_iso:
        uts = parse_iso_local(until_iso)
        update.update({"until_ts": str(uts) if uts else ""})

    r.hset(h, mapping=update)
    if update.get("next_at"):
        r.zadd(KEY_REC_Z, {rid: float(update["next_at"])})
        r.sadd(KEY_REC_SET, rid)
    r.sadd(KEY_REC_ALL, rid)
    return True

def edit_recurring_by_text(
    q: str,
    new_text: Optional[str]=None,
    interval_minutes: Optional[int]=None,
    until_iso: Optional[str]=None,
    daily_time: Optional[str]=None
) -> Optional[str]:
    rid = find_recurring_by_text(q)
    if not rid: return None
    ok = edit_recurring_db(rid, new_text=new_text, interval_minutes=interval_minutes, until_iso=until_iso, daily_time=daily_time)
    return rid if ok else None

# -------- Pomodoro --------
def create_pomodoro_db(
    base_text: str,
    work_min: int = 25,
    short_break_min: int = 5,
    long_break_min: int = 15,
    cycles_total: int = 4,
    long_every: int = 4
) -> str:
    rid = str(r.incr(KEY_REM_SEQ))
    now = now_ts()
    doc = {
        "id": rid, "kind": "pomodoro", "status": "active", "text": base_text,
        "work_min": str(max(1, work_min)),
        "short_break_min": str(max(1, short_break_min)),
        "long_break_min": str(max(1, long_break_min)),
        "cycles_total": str(max(1, cycles_total)),
        "long_every": str(max(1, long_every)),
        "cycle": "1",
        "phase": "work",
        "next_at": str(now),
    }
    r.hset(KEY_REC_HASH(rid), mapping=doc)
    r.sadd(KEY_REC_SET, rid)
    r.zadd(KEY_REC_Z, {rid: now})
    r.sadd(KEY_REC_ALL, rid)
    return rid

def cancel_pomodoro_db(rid: str) -> bool:
    return cancel_recurring_db(rid)

def pause_pomodoro_db(rid: str) -> bool:
    return pause_recurring_db(rid)

def resume_pomodoro_db(rid: str) -> bool:
    return resume_recurring_db(rid)

# -------- Listados / Histórico --------
def _fmt_time_left(target_ts: float) -> str:
    now = now_ts()
    diff = max(0, int(target_ts - now))
    mins = diff // 60
    hrs  = mins // 60
    if hrs >= 1:
        rem_m = mins % 60
        return f"{hrs} horas {rem_m} minutos"
    return f"{mins} minutos"

def format_pending_plain(items_oneshot: List[Dict], items_rec: List[Dict]) -> str:
    lines: List[str] = []

    for i, it in enumerate(items_oneshot, start=1):
        text = (it.get("text") or "").strip()
        due_at = it.get("due_at_col")
        try:
            dt = datetime.strptime(due_at, "%Y-%m-%d %H:%M:%S").replace(tzinfo=TZ)
            left = _fmt_time_left(dt.timestamp())
        except Exception:
            left = "desconocido"
        status = it.get("status","pending")
        lines.append(f"{i}. {text} - para {due_at} - falta {left}" + ("" if status=="pending" else " - esperando confirmación"))

    base = len(lines)
    for j, it in enumerate(items_rec, start=1):
        text = (it.get("text") or "").strip()
        next_at = it.get("next_at_col")
        cada = it.get("interval_human")
        hasta = it.get("until_col")
        idx = base + j
        lines.append(f"{idx}. recurrente: {text} - siguiente {next_at} - cada {cada} - hasta {hasta}")

    if not lines:
        return "no tienes recordatorios pendientes"
    return "\n".join(lines)

def list_all_oneshot_db() -> List[Dict]:
    ids = sorted(list(r.smembers(KEY_REM_ALL)), key=lambda x: int(x))
    out = []
    for rid in ids:
        d = r.hgetall(KEY_REM_HASH(rid))
        if not d: continue
        out.append({
            "id": rid,
            "text": d.get("text",""),
            "status": d.get("status",""),
            "created_at": ts_str_col(float(d.get("created_at", now_ts()))),
            "due_at": ts_str_col(float(d.get("due_at", now_ts()))),
            "done_at": ts_str_col(float(d.get("done_at"))) if d.get("done_at") else "",
            "cancelled_at": ts_str_col(float(d.get("cancelled_at"))) if d.get("cancelled_at") else "",
        })
    return out

def list_done_today_db() -> List[Dict]:
    start, end = today_range()
    ids = sorted(list(r.smembers(KEY_REM_ALL)), key=lambda x: int(x))
    out = []
    for rid in ids:
        d = r.hgetall(KEY_REM_HASH(rid))
        if not d or d.get("status") != "done" or not d.get("done_at"): continue
        ts = float(d["done_at"])
        if start <= ts <= end:
            out.append({"id": rid, "text": d.get("text",""), "done_at": ts_str_col(ts)})
    return out

def list_cancelled_db() -> List[Dict]:
    ids = sorted(list(r.smembers(KEY_REM_ALL)), key=lambda x: int(x))
    out = []
    for rid in ids:
        d = r.hgetall(KEY_REM_HASH(rid))
        if not d or d.get("status") != "cancelled": continue
        out.append({"id": rid, "text": d.get("text",""), "cancelled_at": ts_str_col(float(d.get("cancelled_at", now_ts())))})
    return out

def format_all_plain(oneshot: List[Dict]) -> str:
    if not oneshot:
        return "no hay historial de recordatorios"
    lines = []
    for i, it in enumerate(oneshot, start=1):
        base = f"{i}. {it['text']} - creado {it['created_at']} - vence {it['due_at']} - estado {it['status']}"
        if it.get("done_at"): base += f" - hecho {it['done_at']}"
        if it.get("cancelled_at"): base += f" - cancelado {it['cancelled_at']}"
        lines.append(base)
    return "\n".join(lines)

def format_done_today_plain(items: List[Dict]) -> str:
    if not items: return "hoy no has completado recordatorios"
    lines = [f"{i}. {it['text']} - hecho {it['done_at']}" for i,it in enumerate(items, start=1)]
    return "\n".join(lines)

def format_cancelled_plain(items: List[Dict]) -> str:
    if not items: return "no tienes recordatorios cancelados"
    lines = [f"{i}. {it['text']} - cancelado {it['cancelled_at']}" for i,it in enumerate(items, start=1)]
    return "\n".join(lines)

# -------- PRESETS (implementación completa) --------
def _norm_preset_name(name: str) -> str:
    return re.sub(r"\s+", "_", (name or "").strip().lower())

def list_presets_db() -> List[Dict]:
    names = sorted(list(r.smembers(KEY_PRESET_SET)))
    out: List[Dict] = []
    for nm in names:
        hkey = KEY_PRESET_HASH(nm)
        if not r.exists(hkey): 
            r.srem(KEY_PRESET_SET, nm)
            continue
        d = r.hgetall(hkey)
        kind = d.get("kind")
        # devolvemos un resumen simple y datos clave para arrancar
        if kind == PRESET_KIND_POMODORO:
            out.append({
                "name": nm, "kind": kind,
                "text": d.get("text",""),
                "work_min": int(d.get("work_min","25")),
                "short_break_min": int(d.get("short_break_min","5")),
                "long_break_min": int(d.get("long_break_min","15")),
                "cycles_total": int(d.get("cycles_total","4")),
                "long_every": int(d.get("long_every","4")),
            })
        elif kind == PRESET_KIND_RECURRING:
            out.append({
                "name": nm, "kind": kind,
                "text": d.get("text",""),
                "interval_minutes": int(d.get("interval_minutes","10")),
                "until_iso": d.get("until_iso",""),
                "mode": d.get("mode","interval"),   # interval | daily_time
                "hhmm": d.get("hhmm",""),
            })
        else:  # one-shot
            out.append({
                "name": nm, "kind": kind,
                "text": d.get("text",""),
                "in_minutes": int(d.get("in_minutes","10")),
            })
    return out

def delete_preset_db(name: str) -> bool:
    nm = _norm_preset_name(name)
    hkey = KEY_PRESET_HASH(nm)
    existed = r.delete(hkey) > 0
    r.srem(KEY_PRESET_SET, nm)
    return existed

# -- guardar presets --
def save_pomodoro_preset(name: str, text: str, work_min: int, short_break_min: int, long_break_min: int, cycles_total: int, long_every: int) -> bool:
    nm = _norm_preset_name(name)
    payload = {
        "kind": PRESET_KIND_POMODORO,
        "text": (text or "").strip(),
        "work_min": str(max(1, work_min)),
        "short_break_min": str(max(1, short_break_min)),
        "long_break_min": str(max(1, long_break_min)),
        "cycles_total": str(max(1, cycles_total)),
        "long_every": str(max(1, long_every)),
        "updated_at": str(now_ts()),
    }
    r.hset(KEY_PRESET_HASH(nm), mapping=payload)
    r.sadd(KEY_PRESET_SET, nm)
    return True

def save_recurring_preset(name: str, text: str, interval_minutes: int, until_iso: Optional[str]) -> bool:
    nm = _norm_preset_name(name)
    im = max(5, int(interval_minutes))
    payload = {
        "kind": PRESET_KIND_RECURRING,
        "text": (text or "").strip(),
        "mode": "interval",
        "interval_minutes": str(im),
        "until_iso": (until_iso or ""),
        "hhmm": "",
        "updated_at": str(now_ts()),
    }
    r.hset(KEY_PRESET_HASH(nm), mapping=payload)
    r.sadd(KEY_PRESET_SET, nm)
    return True

def save_oneshot_preset(name: str, text: str, in_minutes: int) -> bool:
    nm = _norm_preset_name(name)
    im = max(1, int(in_minutes))
    payload = {
        "kind": PRESET_KIND_ONESHOT,
        "text": (text or "").strip(),
        "in_minutes": str(im),
        "updated_at": str(now_ts()),
    }
    r.hset(KEY_PRESET_HASH(nm), mapping=payload)
    r.sadd(KEY_PRESET_SET, nm)
    return True

# -- activar presets --
def start_pomodoro_preset(name: str) -> Optional[str]:
    nm = _norm_preset_name(name)
    d = r.hgetall(KEY_PRESET_HASH(nm))
    if not d or d.get("kind") != PRESET_KIND_POMODORO: return None
    return create_pomodoro_db(
        base_text=d.get("text",""),
        work_min=int(d.get("work_min","25")),
        short_break_min=int(d.get("short_break_min","5")),
        long_break_min=int(d.get("long_break_min","15")),
        cycles_total=int(d.get("cycles_total","4")),
        long_every=int(d.get("long_every","4")),
    )

def start_recurring_preset(name: str) -> Optional[str]:
    nm = _norm_preset_name(name)
    d = r.hgetall(KEY_PRESET_HASH(nm))
    if not d or d.get("kind") != PRESET_KIND_RECURRING: return None
    text = d.get("text","")
    mode = d.get("mode","interval")
    if mode == "daily_time" and d.get("hhmm"):
        hh, mm = map(int, d["hhmm"].split(":"))
        return create_recurring_daily_time_db(text, hh, mm)
    # interval
    interval_minutes = int(d.get("interval_minutes","10"))
    until_ts = parse_iso_local(d.get("until_iso","")) if d.get("until_iso") else None
    return create_recurring_db(text, max(5, interval_minutes)*60, until_ts)

def start_oneshot_preset(name: str, override_minutes: Optional[int] = None) -> Optional[str]:
    nm = _norm_preset_name(name)
    d = r.hgetall(KEY_PRESET_HASH(nm))
    if not d or d.get("kind") != PRESET_KIND_ONESHOT: return None
    base_min = int(d.get("in_minutes","10"))
    mins = int(override_minutes) if (override_minutes is not None and int(override_minutes)>=1) else base_min
    due = now_ts() + mins*60
    return create_reminder_db(d.get("text",""), due)

# -------- OpenAI tools (function calling) --------
TOOLS = [
    # One-shot
    {
        "type": "function",
        "function": {
            "name": "create_reminder",
            "description": "Crea un recordatorio (hora Colombia). Usa 'when_iso' (YYYY-MM-DD HH:MM) o 'in_minutes'.",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "when_iso": {"type": "string", "nullable": True},
                    "in_minutes": {"type": "integer", "nullable": True, "minimum": 1}
                },
                "required": ["text"]
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "cancel_reminder",
            "description": "Cancela un recordatorio por id.",
            "parameters": {"type": "object","properties":{"id":{"type":"string"}},"required":["id"]}
        },
    },
    {
        "type": "function",
        "function": {
            "name": "cancel_reminder_by_text",
            "description": "Cancela un recordatorio buscando por fragmento de texto.",
            "parameters": {"type":"object","properties":{"text_like":{"type":"string"}},"required":["text_like"]}
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_pending_reminders",
            "description": "Lista recordatorios pendientes.",
            "parameters": {"type":"object","properties":{}}
        },
    },
    {
        "type": "function",
        "function": {
            "name": "mark_reminder_done",
            "description": "Marca un recordatorio como hecho por id.",
            "parameters": {"type":"object","properties":{"id":{"type":"string"}},"required":["id"]}
        },
    },
    {
        "type": "function",
        "function": {
            "name": "snooze_reminder",
            "description": "Pospone un recordatorio X minutos.",
            "parameters": {"type":"object","properties":{"id":{"type":"string"},"minutes":{"type":"integer","minimum":1}},"required":["id","minutes"]}
        },
    },
    {
        "type": "function",
        "function": {
            "name": "snooze_by_text",
            "description": "Pospone un recordatorio buscando por fragmento de texto.",
            "parameters": {"type":"object","properties":{"text_like":{"type":"string"},"minutes":{"type":"integer","minimum":1}},"required":["text_like","minutes"]}
        },
    },
    {
        "type": "function",
        "function": {
            "name": "reactivate_reminder",
            "description": "Reactiva un recordatorio por id; opcionalmente define nuevo in_minutes.",
            "parameters": {"type":"object","properties":{"id":{"type":"string"},"in_minutes":{"type":"integer","nullable":True,"minimum":1}},"required":["id"]}
        },
    },
    {
        "type": "function",
        "function": {
            "name": "mark_done_by_text",
            "description": "Marca hecho el recordatorio cuyo texto contenga el fragmento dado.",
            "parameters": {"type":"object","properties":{"text_like":{"type":"string"}},"required":["text_like"]}
        },
    },
    {
        "type": "function",
        "function": {
            "name": "reactivate_by_text",
            "description": "Reactiva un recordatorio buscando por fragmento de texto.",
            "parameters": {"type":"object","properties":{"text_like":{"type":"string"},"in_minutes":{"type":"integer","nullable":True,"minimum":1}},"required":["text_like"]}
        },
    },

    # Recurrentes (crear)
    {
        "type":"function",
        "function":{
            "name":"create_recurring",
            "description":"Crea un recordatorio recurrente por intervalo: 'cada N minutos/horas/días' con opcional 'hasta YYYY-MM-DD HH:MM'.",
            "parameters":{"type":"object","properties":{"text":{"type":"string"},"interval_minutes":{"type":"integer","minimum":5},"until_iso":{"type":"string","nullable":True}},"required":["text","interval_minutes"]}
        },
    },
    {
        "type":"function",
        "function":{
            "name":"create_recurring_daily_time",
            "description":"Crea un recordatorio recurrente diario a una hora fija HH:MM (24h).",
            "parameters":{"type":"object","properties":{"text":{"type":"string"},"hhmm":{"type":"string","pattern":"^\\d{2}:\\d{2}$"}},"required":["text","hhmm"]}
        },
    },

    # Recurrentes (gestión)
    {
        "type":"function",
        "function":{
            "name":"cancel_recurring",
            "description":"Cancela un recordatorio recurrente por id.",
            "parameters":{"type":"object","properties":{"id":{"type":"string"}},"required":["id"]}
        },
    },
    {
        "type":"function",
        "function":{
            "name":"pause_recurring",
            "description":"Pausa un recordatorio recurrente por id.",
            "parameters":{"type":"object","properties":{"id":{"type":"string"}},"required":["id"]}
        },
    },
    {
        "type":"function",
        "function":{
            "name":"resume_recurring",
            "description":"Reanuda un recordatorio recurrente por id.",
            "parameters":{"type":"object","properties":{"id":{"type":"string"}},"required":["id"]}
        },
    },
    {
        "type":"function",
        "function":{
            "name":"cancel_recurring_by_text",
            "description":"Cancela un recordatorio recurrente buscando por fragmento de texto.",
            "parameters":{"type":"object","properties":{"text_like":{"type":"string"}},"required":["text_like"]}
        },
    },
    {
        "type":"function",
        "function":{
            "name":"list_recurring",
            "description":"Lista los recordatorios recurrentes activos.",
            "parameters":{"type":"object","properties":{}}
        },
    },

    # Recurrentes (EDICIÓN)
    {
        "type":"function",
        "function":{
            "name":"edit_reminder",
            "description":"Edita un one-shot por id (texto y/o fecha).",
            "parameters":{"type":"object","properties":{
                "id":{"type":"string"},
                "new_text":{"type":"string","nullable":True},
                "when_iso":{"type":"string","nullable":True},
                "in_minutes":{"type":"integer","nullable":True,"minimum":1}
            },"required":["id"]}
        },
    },
    {
        "type":"function",
        "function":{
            "name":"edit_reminder_by_text",
            "description":"Edita un one-shot por fragmento de texto.",
            "parameters":{"type":"object","properties":{
                "text_like":{"type":"string"},
                "new_text":{"type":"string","nullable":True},
                "when_iso":{"type":"string","nullable":True},
                "in_minutes":{"type":"integer","nullable":True,"minimum":1}
            },"required":["text_like"]}
        },
    },
    {
        "type":"function",
        "function":{
            "name":"edit_recurring",
            "description":"Edita un recurrente por id: texto, intervalo, hasta, o daily_time HH:MM.",
            "parameters":{"type":"object","properties":{
                "id":{"type":"string"},
                "new_text":{"type":"string","nullable":True},
                "interval_minutes":{"type":"integer","nullable":True,"minimum":5},
                "until_iso":{"type":"string","nullable":True},
                "daily_time":{"type":"string","nullable":True,"pattern":"^\\d{2}:\\d{2}$"}
            },"required":["id"]}
        },
    },
    {
        "type":"function",
        "function":{
            "name":"edit_recurring_by_text",
            "description":"Edita un recurrente buscando por fragmento de texto.",
            "parameters":{"type":"object","properties":{
                "text_like":{"type":"string"},
                "new_text":{"type":"string","nullable":True},
                "interval_minutes":{"type":"integer","nullable":True,"minimum":5},
                "until_iso":{"type":"string","nullable":True},
                "daily_time":{"type":"string","nullable":True,"pattern":"^\\d{2}:\\d{2}$"}
            },"required":["text_like"]}
        },
    },

    # Pomodoro
    {
        "type":"function",
        "function":{
            "name":"create_pomodoro",
            "description":"Crea un pomodoro con parámetros opcionales.",
            "parameters":{"type":"object","properties":{"text":{"type":"string"},"work_min":{"type":"integer","minimum":1},"short_break_min":{"type":"integer","minimum":1},"long_break_min":{"type":"integer","minimum":1},"cycles_total":{"type":"integer","minimum":1},"long_every":{"type":"integer","minimum":1}},"required":["text"]}
        },
    },
    {
        "type":"function",
        "function":{
            "name":"cancel_pomodoro",
            "description":"Cancela un pomodoro por id.",
            "parameters":{"type":"object","properties":{"id":{"type":"string"}},"required":["id"]}
        },
    },
    {
        "type":"function",
        "function":{
            "name":"pause_pomodoro",
            "description":"Pausa un pomodoro por id.",
            "parameters":{"type":"object","properties":{"id":{"type":"string"}},"required":["id"]}
        },
    },
    {
        "type":"function",
        "function":{
            "name":"resume_pomodoro",
            "description":"Reanuda un pomodoro por id.",
            "parameters":{"type":"object","properties":{"id":{"type":"string"}},"required":["id"]}
        },
    },

    # Presets
    {
        "type":"function",
        "function":{
            "name":"save_pomodoro_preset",
            "description":"Guarda un preset de pomodoro con nombre.",
            "parameters":{"type":"object","properties":{
                "name":{"type":"string"},
                "text":{"type":"string"},
                "work_min":{"type":"integer","minimum":1},
                "short_break_min":{"type":"integer","minimum":1},
                "long_break_min":{"type":"integer","minimum":1},
                "cycles_total":{"type":"integer","minimum":1},
                "long_every":{"type":"integer","minimum":1}
            },"required":["name","text","work_min","short_break_min","long_break_min","cycles_total","long_every"]}
        },
    },
    {
        "type":"function",
        "function":{
            "name":"start_pomodoro_preset",
            "description":"Activa un pomodoro a partir de un preset por nombre.",
            "parameters":{"type":"object","properties":{"name":{"type":"string"}},"required":["name"]}
        },
    },
    {
        "type":"function",
        "function":{
            "name":"save_recurring_preset",
            "description":"Guarda un preset recurrente con nombre.",
            "parameters":{"type":"object","properties":{
                "name":{"type":"string"},
                "text":{"type":"string"},
                "interval_minutes":{"type":"integer","minimum":5},
                "until_iso":{"type":"string","nullable":True}
            },"required":["name","text","interval_minutes"]}
        },
    },
    {
        "type":"function",
        "function":{
            "name":"start_recurring_preset",
            "description":"Activa un recordatorio recurrente desde un preset por nombre.",
            "parameters":{"type":"object","properties":{"name":{"type":"string"}},"required":["name"]}
        },
    },
    {
        "type":"function",
        "function":{
            "name":"save_oneshot_preset",
            "description":"Guarda un preset one-shot con minutos por defecto.",
            "parameters":{"type":"object","properties":{
                "name":{"type":"string"},
                "text":{"type":"string"},
                "in_minutes":{"type":"integer","minimum":1}
            },"required":["name","text","in_minutes"]}
        },
    },
    {
        "type":"function",
        "function":{
            "name":"start_oneshot_preset",
            "description":"Activa un one-shot desde un preset por nombre; opcional override minutos.",
            "parameters":{"type":"object","properties":{
                "name":{"type":"string"},
                "override_minutes":{"type":"integer","nullable":True,"minimum":1}
            },"required":["name"]}
        },
    },
    {
        "type":"function",
        "function":{
            "name":"list_presets",
            "description":"Lista presets guardados.",
            "parameters":{"type":"object","properties":{}}
        },
    },
    {
        "type":"function",
        "function":{
            "name":"delete_preset",
            "description":"Elimina un preset por nombre.",
            "parameters":{"type":"object","properties":{"name":{"type":"string"}},"required":["name"]}
        },
    },

    # HISTÓRICO
    {
        "type":"function",
        "function":{
            "name":"list_all_reminders",
            "description":"Lista todo el historial de one-shot (todos los estados).",
            "parameters":{"type":"object","properties":{}}
        },
    },
    {
        "type":"function",
        "function":{
            "name":"list_done_today",
            "description":"Lista los one-shot hechos hoy.",
            "parameters":{"type":"object","properties":{}}
        },
    },
    {
        "type":"function",
        "function":{
            "name":"list_cancelled",
            "description":"Lista los one-shot cancelados.",
            "parameters":{"type":"object","properties":{}}
        },
    },
]

# -------- Worker --------
def reminder_worker():
    """
    Bucle cada 15s:
      (A) One-shot: avisos intermedios y confirmación + insistencia escalonada
      (B) Recurrencia: dispara en next_at, reprograma según modo (intervalo o diario fijo)
      (C) Pomodoro: fases y reprogramación por fase
    """
    while True:
        try:
            now = now_ts()

            # ----- (A) One-shot pings intermedios / confirmación / insistencia -----
            due_for_ping = r.zrangebyscore(KEY_REM_Z, -1, now)
            for rid in due_for_ping:
                h = KEY_REM_HASH(rid)
                data = r.hgetall(h)
                if not data:
                    r.zrem(KEY_REM_Z, rid); r.srem(KEY_REM_PENDING, rid); continue

                status = data.get("status","pending")
                text = data.get("text","").strip()
                due_at = float(data.get("due_at"))

                if status == "pending":
                    if now >= due_at:
                        confirm_msg = llm_rewrite("confirm", text)
                        try:
                            send_to_macrodroid(confirm_msg, require_response=True, require_sound=True)
                        except Exception:
                            retry_at = now + MIN_INTERVAL_SEC
                            r.hset(h, mapping={"next_ping_at": str(retry_at)})
                            r.zadd(KEY_REM_Z, {rid: retry_at})
                            continue

                        next_nag = now + schedule_next_nag(0)
                        r.hset(h, mapping={"status": "await_confirm", "nag_level": "0", "next_ping_at": str(next_nag)})
                        r.zadd(KEY_REM_Z, {rid: next_nag})
                        continue

                    mins_left = max(1, int((due_at - now) // 60))
                    msg = llm_rewrite("intermediate", text, mins_left=mins_left)
                    try:
                        send_to_macrodroid(msg, require_response=False, require_sound=True)
                    except Exception:
                        retry_at = now + MIN_INTERVAL_SEC
                        r.hset(h, mapping={"next_ping_at": str(retry_at)})
                        r.zadd(KEY_REM_Z, {rid: retry_at})
                        continue

                    nxt = schedule_next_ping(due_at)
                    if nxt:
                        r.hset(h, mapping={"next_ping_at": str(nxt)})
                        r.zadd(KEY_REM_Z, {rid: nxt})
                    else:
                        r.zrem(KEY_REM_Z, rid)

                elif status == "await_confirm":
                    nag_level = int(data.get("nag_level","0"))
                    msg = llm_rewrite("confirm_nag", text, nag_level=nag_level)
                    try:
                        send_to_macrodroid(msg, require_response=True, require_sound=True)
                    except Exception:
                        retry_at = now + MIN_INTERVAL_SEC
                        r.hset(h, mapping={"next_ping_at": str(retry_at)})
                        r.zadd(KEY_REM_Z, {rid: retry_at})
                        continue

                    next_gap = schedule_next_nag(nag_level + 1)
                    r.hset(h, mapping={"nag_level": str(nag_level + 1), "next_ping_at": str(now + next_gap)})
                    r.zadd(KEY_REM_Z, {rid: now + next_gap})

            # ----- (B) Recurrencia & Pomodoro -----
            due_rec = r.zrangebyscore(KEY_REC_Z, -1, now)
            for rid in due_rec:
                h = KEY_REC_HASH(rid)
                if not r.exists(h):
                    r.zrem(KEY_REC_Z, rid); r.srem(KEY_REC_SET, rid); continue
                d = r.hgetall(h)
                status = d.get("status")
                kind = d.get("kind")
                if status != "active":
                    r.zrem(KEY_REC_Z, rid); continue

                if kind == "recurring":
                    text = d.get("text","").strip()
                    mode = d.get("mode","interval")

                    # Emitir aviso
                    try:
                        send_to_macrodroid(llm_rewrite("recurring", text), require_response=False, require_sound=True)
                    except Exception:
                        retry_at = now + MIN_INTERVAL_SEC
                        r.hset(h, mapping={"next_at": str(retry_at)})
                        r.zadd(KEY_REC_Z, {rid: retry_at})
                        continue

                    # Programar siguiente
                    if mode == "interval":
                        until = float(d["until_ts"]) if d.get("until_ts") else None
                        interval = int(float(d.get("interval_sec","600")))
                        nxt = now + interval
                        if until and nxt > until:
                            cancel_recurring_db(rid)
                        else:
                            r.hset(h, mapping={"next_at": str(nxt)})
                            r.zadd(KEY_REC_Z, {rid: nxt})
                    else:
                        hh, mm = map(int, (d.get("hhmm","00:00").split(":")))
                        nxt = next_daily_time_epoch(hh, mm)
                        r.hset(h, mapping={"next_at": str(nxt)})
                        r.zadd(KEY_REC_Z, {rid: nxt})

                elif kind == "pomodoro":
                    phase = d.get("phase","work")
                    base_text = d.get("text","").strip()
                    cycle = int(d.get("cycle","1"))
                    cycles_total = int(d.get("cycles_total","4"))
                    work_min = int(d.get("work_min","25"))
                    short_min = int(d.get("short_break_min","5"))
                    long_min = int(d.get("long_break_min","15"))
                    long_every = int(d.get("long_every","4"))

                    try:
                        send_to_macrodroid(llm_rewrite("pomodoro", base_text, phase=phase), require_response=False, require_sound=True)
                    except Exception:
                        retry_at = now + MIN_INTERVAL_SEC
                        r.hset(h, mapping={"next_at": str(retry_at)})
                        r.zadd(KEY_REC_Z, {rid: retry_at})
                        continue

                    if phase == "work":
                        next_phase = "long_break" if (cycle % long_every == 0) else "short_break"
                        dur = long_min if next_phase == "long_break" else short_min
                        next_at = now + max(MIN_INTERVAL_SEC, dur * 60)
                        r.hset(h, mapping={"phase": next_phase, "next_at": str(next_at)})
                        r.zadd(KEY_REC_Z, {rid: next_at})
                    else:
                        cycle_next = cycle + 1
                        if cycle_next > cycles_total:
                            cancel_pomodoro_db(rid)
                            continue
                        next_phase = "work"
                        next_at = now + max(MIN_INTERVAL_SEC, work_min * 60)
                        r.hset(h, mapping={"phase": next_phase, "cycle": str(cycle_next), "next_at": str(next_at)})
                        r.zadd(KEY_REC_Z, {rid: next_at})

        except Exception:
            pass
        time.sleep(15)

# -------- OpenAI orquestador --------
def openai_chat_with_tools(user_text: str) -> str:
    # Atajos de servidor: PENDIENTES
    if infer_is_pending_request(user_text):
        return format_pending_plain(list_pending_db(), list_recurring_db())

    # One-shot implícito ("en X minutos/horas ...")
    sec = schedule_from_natural(user_text)
    if sec:
        create_reminder_db(user_text.strip(), now_ts() + sec)

    # Recurrente implícito (intervalo o diario)
    rec = parse_recurring(user_text)
    if rec:
        if rec["mode"] == "interval":
            create_recurring_db(user_text.strip(), rec["interval_sec"], rec["until_ts"])
        else:
            create_recurring_daily_time_db(user_text.strip(), rec["hh"], rec["mm"])

    # Atajos de servidor: HISTÓRICO (siempre desde Redis)
    hist_intent = infer_history_intent(user_text)
    if hist_intent == "done_today":
        items = list_done_today_db()
        return format_done_today_plain(items)
    if hist_intent == "cancelled":
        items = list_cancelled_db()
        return format_cancelled_plain(items)
    if hist_intent == "all":
        items = list_all_oneshot_db()
        return format_all_plain(items)

    # LLM + Tools (solo si no hubo atajo de servidor)
    messages = [{"role": "system", "content": SYSTEM_PROMPT}] + history_for_openai(20)
    messages.append({"role": "user", "content": user_text})

    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}
    payload = {"model": MODEL, "messages": messages, "temperature": 0.3, "tools": TOOLS, "tool_choice": "auto"}
    resp = requests.post(OPENAI_URL, json=payload, headers=headers, timeout=30)
    if not resp.ok:
        raise HTTPException(status_code=502, detail=f"OpenAI respondió {resp.status_code}")
    data = resp.json()
    msg = data["choices"][0]["message"]
    tool_calls = msg.get("tool_calls", [])

    while tool_calls:
        messages.append({"role": "assistant", "content": msg.get("content") or "", "tool_calls": tool_calls})
        for tc in tool_calls:
            name = tc["function"]["name"]
            args = json.loads(tc["function"]["arguments"] or "{}")

            # ----- One-shot -----
            if name == "create_reminder":
                text = (args.get("text") or "").strip()
                in_minutes = args.get("in_minutes")
                when_iso = args.get("when_iso")
                if in_minutes:
                    due = now_ts() + max(1, int(in_minutes)) * 60
                elif when_iso:
                    parsed = parse_iso_local(when_iso)
                    due = parsed if parsed else now_ts() + 10 * 60
                else:
                    due = now_ts() + 10 * 60
                rid = create_reminder_db(text, due)
                tool_output = {"ok": True, "id": rid, "due_at_col": ts_str_col(due)}

            elif name == "cancel_reminder":
                rid = str(args.get("id")); tool_output = {"ok": cancel_reminder_db(rid)}

            elif name == "cancel_reminder_by_text":
                tlike = str(args.get("text_like") or "").strip()
                rid = find_by_text_candidate(tlike) if tlike else None
                ok = cancel_reminder_db(rid) if rid else False
                tool_output = {"ok": ok, "id": rid}

            elif name == "list_pending_reminders":
                tool_output = {"ok": True, "items": list_pending_db()}

            elif name == "mark_reminder_done":
                rid = str(args.get("id")); tool_output = {"ok": mark_done_db(rid)}

            elif name == "snooze_reminder":
                rid = str(args.get("id")); minutes = int(args.get("minutes", 5))
                tool_output = {"ok": snooze_db(rid, minutes), "new_due_in_min": minutes}

            elif name == "snooze_by_text":
                tlike = str(args.get("text_like") or "").strip()
                minutes = int(args.get("minutes", 5))
                rid = find_by_text_candidate(tlike) if tlike else None
                ok = snooze_db(rid, minutes) if rid else False
                tool_output = {"ok": ok, "id": rid, "minutes": minutes}

            elif name == "reactivate_reminder":
                rid = str(args.get("id")); in_minutes = args.get("in_minutes")
                tool_output = {"ok": reactivate_by_id(rid, in_minutes)}

            elif name == "mark_done_by_text":
                tlike = str(args.get("text_like") or "").strip()
                rid = mark_done_by_text(tlike) if tlike else None
                tool_output = {"ok": bool(rid), "id": rid}

            elif name == "reactivate_by_text":
                tlike = str(args.get("text_like") or "").strip()
                in_minutes = args.get("in_minutes")
                rid = reactivate_by_text(tlike, in_minutes) if tlike else None
                tool_output = {"ok": bool(rid), "id": rid}

            # ----- Recurrentes (crear) -----
            elif name == "create_recurring":
                text = (args.get("text") or "").strip()
                interval_minutes = max(5, int(args.get("interval_minutes", 5)))
                until_iso = args.get("until_iso")
                until_ts = parse_iso_local(until_iso) if until_iso else None
                rid = create_recurring_db(text, interval_minutes * 60, until_ts)
                tool_output = {"ok": True, "id": rid}

            elif name == "create_recurring_daily_time":
                text = (args.get("text") or "").strip()
                hhmm = args.get("hhmm")
                hh, mm = map(int, hhmm.split(":"))
                rid = create_recurring_daily_time_db(text, hh, mm)
                tool_output = {"ok": True, "id": rid}

            # ----- Recurrentes (gestión) -----
            elif name == "cancel_recurring":
                rid = str(args.get("id")); tool_output = {"ok": cancel_recurring_db(rid)}

            elif name == "pause_recurring":
                rid = str(args.get("id")); tool_output = {"ok": pause_recurring_db(rid)}

            elif name == "resume_recurring":
                rid = str(args.get("id")); tool_output = {"ok": resume_recurring_db(rid)}

            elif name == "cancel_recurring_by_text":
                tlike = str(args.get("text_like") or "").strip()
                rid = find_recurring_by_text(tlike) if tlike else None
                ok = cancel_recurring_db(rid) if rid else False
                tool_output = {"ok": ok, "id": rid}

            elif name == "list_recurring":
                tool_output = {"ok": True, "items": list_recurring_db()}

            # ----- EDICIÓN -----
            elif name == "edit_reminder":
                rid = str(args.get("id"))
                ok = edit_reminder_db(
                    rid,
                    new_text=args.get("new_text"),
                    when_iso=args.get("when_iso"),
                    in_minutes=args.get("in_minutes"),
                )
                tool_output = {"ok": ok, "id": rid}

            elif name == "edit_reminder_by_text":
                rid = edit_reminder_by_text(
                    args["text_like"],
                    new_text=args.get("new_text"),
                    when_iso=args.get("when_iso"),
                    in_minutes=args.get("in_minutes"),
                )
                tool_output = {"ok": bool(rid), "id": rid}

            elif name == "edit_recurring":
                rid = str(args.get("id"))
                ok = edit_recurring_db(
                    rid,
                    new_text=args.get("new_text"),
                    interval_minutes=args.get("interval_minutes"),
                    until_iso=args.get("until_iso"),
                    daily_time=args.get("daily_time"),
                )
                tool_output = {"ok": ok, "id": rid}

            elif name == "edit_recurring_by_text":
                rid = edit_recurring_by_text(
                    args["text_like"],
                    new_text=args.get("new_text"),
                    interval_minutes=args.get("interval_minutes"),
                    until_iso=args.get("until_iso"),
                    daily_time=args.get("daily_time"),
                )
                tool_output = {"ok": bool(rid), "id": rid}

            # ----- Pomodoro -----
            elif name == "create_pomodoro":
                base = (args.get("text") or "").strip()
                rid = create_pomodoro_db(
                    base_text=base,
                    work_min=int(args.get("work_min", 25)),
                    short_break_min=int(args.get("short_break_min", 5)),
                    long_break_min=int(args.get("long_break_min", 15)),
                    cycles_total=int(args.get("cycles_total", 4)),
                    long_every=int(args.get("long_every", 4)),
                )
                tool_output = {"ok": True, "id": rid}

            elif name == "cancel_pomodoro":
                rid = str(args.get("id")); tool_output = {"ok": cancel_pomodoro_db(rid)}

            elif name == "pause_pomodoro":
                rid = str(args.get("id")); tool_output = {"ok": pause_pomodoro_db(rid)}

            elif name == "resume_pomodoro":
                rid = str(args.get("id")); tool_output = {"ok": resume_recurring_db(rid)}

            # ----- Presets -----
            elif name == "save_pomodoro_preset":
                ok = save_pomodoro_preset(
                    name=args["name"],
                    text=args["text"],
                    work_min=int(args["work_min"]),
                    short_break_min=int(args["short_break_min"]),
                    long_break_min=int(args["long_break_min"]),
                    cycles_total=int(args["cycles_total"]),
                    long_every=int(args["long_every"]),
                )
                tool_output = {"ok": ok}

            elif name == "start_pomodoro_preset":
                rid = start_pomodoro_preset(args["name"])
                tool_output = {"ok": bool(rid), "id": rid}

            elif name == "save_recurring_preset":
                ok = save_recurring_preset(
                    name=args["name"],
                    text=args["text"],
                    interval_minutes=int(args["interval_minutes"]),
                    until_iso=args.get("until_iso"),
                )
                tool_output = {"ok": ok}

            elif name == "start_recurring_preset":
                rid = start_recurring_preset(args["name"])
                tool_output = {"ok": bool(rid), "id": rid}

            elif name == "save_oneshot_preset":
                ok = save_oneshot_preset(
                    name=args["name"],
                    text=args["text"],
                    in_minutes=int(args["in_minutes"]),
                )
                tool_output = {"ok": ok}

            elif name == "start_oneshot_preset":
                rid = start_oneshot_preset(
                    name=args["name"],
                    override_minutes=args.get("override_minutes"),
                )
                tool_output = {"ok": bool(rid), "id": rid}

            elif name == "list_presets":
                tool_output = {"ok": True, "items": list_presets_db()}

            elif name == "delete_preset":
                tool_output = {"ok": delete_preset_db(args["name"])}

            # ----- HISTÓRICO -----
            elif name == "list_all_reminders":
                tool_output = {"ok": True, "items": list_all_oneshot_db()}

            elif name == "list_done_today":
                tool_output = {"ok": True, "items": list_done_today_db()}

            elif name == "list_cancelled":
                tool_output = {"ok": True, "items": list_cancelled_db()}

            else:
                tool_output = {"ok": False, "error": f"Tool desconocida: {name}"}

            messages.append({"role": "tool", "tool_call_id": tc["id"], "name": name, "content": json.dumps(tool_output)})

        resp = requests.post(OPENAI_URL, json={"model": MODEL, "messages": messages, "temperature": 0.3}, headers=headers, timeout=30)
        if not resp.ok:
            raise HTTPException(status_code=502, detail=f"OpenAI respondió {resp.status_code}")
        data = resp.json()
        msg = data["choices"][0]["message"]
        tool_calls = msg.get("tool_calls", [])

    final_text = msg.get("content", "").strip() or "listo"
    return sanitize_for_tts(final_text)

# -------- FastAPI --------
app = FastAPI(
    title="Chat con Function Calling y Redis (bonito + recurrentes + pomodoro + presets + require_response + confirmaciones + require_sound + histórico + edición)",
    version="3.0.0"
)

_worker_started = False
def start_worker_once():
    global _worker_started
    if _worker_started: return
    t = threading.Thread(target=reminder_worker, daemon=True)
    t.start()
    _worker_started = True

@app.on_event("startup")
def on_startup():
    try:
        r.ping()
    except Exception as e:
        raise RuntimeError(f"No se pudo conectar a Redis: {e}")
    start_worker_once()

@app.post("/task")
def task(task: str = Query(..., description="Texto a reenviar (ej: ?task=hola)")):
    """
    Chat único:
    - Guarda el mensaje, usa function calling para operar recordatorios, pomodoros, presets y edición.
    - Envía SIEMPRE la respuesta final a MacroDroid con: chat_entry y require_response=true/false.
    - require_sound=false en respuestas de chat; true en avisos lanzados por el worker (recordatorios).
    - Soporta:
        * One-shot: crear/listar/posponer/reactivar/cancelar/marcar hecho/EDITAR.
        * Recurrentes: crear intervalo o diario fijo, listar, pausar/reanudar/cancelar/EDITAR (texto/intervalo/hasta/hora fija).
        * Pomodoro: crear/pausar/reanudar/cancelar.
        * Presets: guardar/activar/listar/borrar.
        * Histórico: ver todo, ver hechos hoy, ver cancelados.
    - Confirmación al vencer: se pregunta si ya se hizo y se insiste con frecuencia creciente; ofrece aplazar o cancelar.
    """
    push_history("user", task)
    answer = openai_chat_with_tools(task)
    push_history("assistant", answer)
    require = infer_require_response(task, answer)
    send_to_macrodroid(answer, require_response=require, require_sound=False)
    return {"ok": True, "sent_to_macrodroid": True, "require_response": require}
