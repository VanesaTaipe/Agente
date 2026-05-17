import os
import json
import re
import tempfile
import time
import numpy as np
import streamlit as st
import faiss
from sentence_transformers import SentenceTransformer
from openai import OpenAI
from groq import Groq
from audio_recorder_streamlit import audio_recorder

# ==============================
# CONFIGURACIÓN INICIAL
# ==============================
st.set_page_config(page_title="Asistente NIC con RAG", page_icon="🩺", layout="wide")

# === API KEYS desde Streamlit Secrets ===
GROQ_API_KEY   = st.secrets.get("GROQ_API_KEY")
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY")

if not GROQ_API_KEY:
    st.error("⚠️ Falta GROQ_API_KEY en Streamlit Secrets")
    st.stop()
if not OPENAI_API_KEY:
    st.error("⚠️ Falta OPENAI_API_KEY en Streamlit Secrets")
    st.stop()

groq_client   = Groq(api_key=GROQ_API_KEY)
openai_client = OpenAI(api_key=OPENAI_API_KEY)

# ==============================
# CSS PERSONALIZADO
# ==============================
st.markdown("""
<style>
#MainMenu {visibility: hidden;}
footer    {visibility: hidden;}
header    {visibility: hidden;}
.main { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); }
.title-container {
    background: white; padding: 2rem; border-radius: 15px;
    box-shadow: 0 4px 6px rgba(0,0,0,0.1); margin-bottom: 2rem; text-align: center;
}
.title-container h1 { color: #667eea; margin: 0; font-size: 2.5rem; }
.title-container p  { color: #666; margin: 0.5rem 0 0 0; font-size: 1.1rem; }
[data-testid="stChatMessageContainer"] {
    background: white; border-radius: 15px; padding: 1.5rem;
    box-shadow: 0 4px 6px rgba(0,0,0,0.1); margin-bottom: 1rem;
    min-height: 500px; max-height: 600px; overflow-y: auto;
}
.stButton > button {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white; border: none; border-radius: 25px;
    padding: 0.5rem 2rem; font-weight: bold; transition: transform 0.2s;
}
.stButton > button:hover { transform: scale(1.05); }
.stTextInput > div > div > input { border-radius: 25px; border: 2px solid #667eea; }
</style>
""", unsafe_allow_html=True)

# ==============================
# CARGAR VECTORSTORE DESDE ARCHIVOS NIC
# Archivos esperados en el directorio raíz del proyecto:
#   embeddings_nic.npy
#   indice_nic.faiss
#   metadata_nic.json
# ==============================
@st.cache_resource(show_spinner=False)
def cargar_vectorstore():
    index     = faiss.read_index("indice_nic.faiss")
    emb_array = np.load("embeddings_nic.npy")

    with open("metadata_nic.json", "r", encoding="utf-8") as f:
        meta_data = json.load(f)

    # metadata_nic.json tiene {"mapping": {...}, "order": [...]}
    metadata_map   = meta_data["mapping"]   # {codigo: {codigo, nombre, definicion, ...}}
    codigos_orden  = meta_data["order"]     # lista de códigos en orden del índice

    # Textos completos para retrieval
    textos = []
    for cod in codigos_orden:
        m = metadata_map.get(cod, {})
        nombre     = m.get("nombre", "")
        definicion = m.get("definicion", "")
        activs     = m.get("actividades", [])
        if isinstance(activs, list):
            activs_txt = "; ".join(str(a) for a in activs)
        else:
            activs_txt = str(activs)
        texto = f"Código NIC: {cod}\n\nNombre de la intervención:\n{nombre}\n\nDefinición:\n{definicion}\n\nActividades:\n{activs_txt}"
        textos.append(texto)

    # Modelo de embeddings (mismo con el que se creó el índice)
    model = SentenceTransformer("intfloat/multilingual-e5-large")

    return index, codigos_orden, textos, metadata_map, model


# ==============================
# LLM: GROQ
# ==============================
def llamar_groq(prompt: str, system_prompt: str = "", temperature: float = 0.0, max_tokens: int = 700) -> str:
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    try:
        resp = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        st.warning(f"Error Groq: {e}")
        return ""


def limpiar_json(text: str) -> str:
    text = re.sub(r"```json|```", "", text).strip()
    m = re.search(r"(\{.*\}|\[.*\])", text, re.DOTALL)
    return m.group(1) if m else text


# ==============================
# BÚSQUEDA FAISS
# ==============================
def buscar(query: str, k: int = 8) -> list:
    index, codigos_orden, textos, metadata_map, model = cargar_vectorstore()
    emb = model.encode(["query: " + query], normalize_embeddings=True, convert_to_numpy=True).astype("float32")
    scores, indices = index.search(emb, k)
    resultados = []
    for score, idx in zip(scores[0], indices[0]):
        if idx < 0 or idx >= len(codigos_orden):
            continue
        resultados.append({
            "codigo": codigos_orden[idx],
            "texto":  textos[idx],
            "score":  float(score)
        })
    return resultados


def reciprocal_rank_fusion(rankings, k_rrf=60) -> list:
    from collections import defaultdict
    scores   = defaultdict(float)
    docs_map = {}
    for ranked_docs in rankings:
        for rank, doc in enumerate(ranked_docs):
            cod = doc["codigo"]
            docs_map[cod] = doc
            scores[cod]  += 1 / (k_rrf + rank + 1)
    resultados = []
    for cod, score in scores.items():
        d = docs_map[cod].copy()
        d["fusion_score"] = round(score, 5)
        d["score"] = d.get("score", score)
        resultados.append(d)
    resultados.sort(key=lambda x: x["fusion_score"], reverse=True)
    return resultados


def buscar_multi_query(query_original, query_opt, sub_queries, k=8) -> list:
    all_rankings = [buscar(query_original, k=k)]
    if query_opt:
        all_rankings.append(buscar(query_opt, k=k))
    for q in sub_queries:
        try:
            all_rankings.append(buscar(q, k=k))
        except Exception:
            continue
    return reciprocal_rank_fusion(all_rankings)[:k]


RESPIRATORY_PRIORITY_TERMS = [
    "respiratoria", "respiración", "vía aérea", "vias aereas",
    "oxigenación", "oxigenoterapia", "ventilación", "disnea",
    "tos", "secreciones", "monitorización", "bronquial", "pulmonar", "aspiración"
]


def clinical_reranker(docs, sintomas, keywords_nic, top_k=5) -> list:
    sintomas_l     = [s.lower() for s in sintomas]
    keywords_nic_l = [kw.lower() for kw in keywords_nic]
    reranked = []
    for d in docs:
        texto = d["texto"].lower()
        semantic_score    = d.get("fusion_score", d.get("score", 0))
        symptom_hits      = sum(1 for s in sintomas_l if s in texto)
        symptom_score     = symptom_hits * 0.12
        keyword_hits      = sum(1 for kw in keywords_nic_l if kw in texto)
        keyword_score     = keyword_hits * 0.08
        resp_hits         = sum(1 for t in RESPIRATORY_PRIORITY_TERMS if t in texto)
        resp_bonus        = resp_hits * 0.05
        final_score       = semantic_score + symptom_score + keyword_score + resp_bonus
        d_copy = d.copy()
        d_copy["rerank_score"] = round(final_score, 5)
        reranked.append(d_copy)
    reranked.sort(key=lambda x: x["rerank_score"], reverse=True)
    return reranked[:top_k]


# ==============================
# AGENTE 1 – CLINICAL REASONING / REFORMULACIÓN
# ==============================
def agente_1_reformular(pregunta: str) -> dict:
    system_prompt = """You are a Clinical Reasoning Agent specialized in respiratory nursing.
You reason BEFORE retrieval. You do not diagnose.
You analyze symptoms, prioritize nursing needs, and optimize retrieval planning."""

    prompt = f"""Analiza el caso clínico para optimizar la recuperación de intervenciones NIC.

CASO CLÍNICO:
\"\"\"{pregunta}\"\"\"

TAREA:
1. Identifica síntomas neutrales.
2. Define el problema principal.
3. Prioriza necesidad de enfermería.
4. Diseña plan de búsqueda semántico.
5. Genera query principal RAG.
6. Genera sub_queries específicas (máx. 3).
7. Extrae keywords NIC.
8. Estima confianza.

REGLAS:
- NO diagnosticar. NO medicamentos. Pensamiento orientado a NIC.
- Queries máximo 10 palabras. sub_queries deben mejorar retrieval.

RESPONDE SOLO JSON. FORMATO EXACTO:
{{
  "thinking": {{
      "problema_principal": "",
      "hallazgos": [],
      "prioridad_enfermeria": "",
      "plan_busqueda": []
  }},
  "sintomas": [],
  "query_rag": "",
  "sub_queries": [],
  "keywords_nic": [],
  "confidence": 0.0
}}"""

    resp = llamar_groq(prompt, system_prompt=system_prompt, temperature=0, max_tokens=450)
    defaults = {
        "thinking": {"problema_principal": "", "hallazgos": [], "prioridad_enfermeria": "", "plan_busqueda": []},
        "sintomas": [], "query_rag": pregunta, "sub_queries": [], "keywords_nic": [], "confidence": 0.5
    }
    try:
        data = json.loads(limpiar_json(resp))
        for k, v in defaults.items():
            data.setdefault(k, v)
        return data
    except Exception:
        return defaults


# ==============================
# AGENTE 2 – EVIDENCE GROUNDING
# ==============================
def agente_2_extraer_gold(docs: list) -> list:
    _, _, _, metadata_map, _ = cargar_vectorstore()
    contexto = ""
    for i, d in enumerate(docs):
        meta = metadata_map.get(d["codigo"], {})
        contexto += f"""
======= DOCUMENTO {i+1} =======
Código: {meta.get('codigo', d['codigo'])}
Nombre: {meta.get('nombre', '')}
Definición: {meta.get('definicion', '')}
"""

    prompt = f"""Selecciona únicamente las intervenciones NIC más relevantes.

CONTEXTO:
{contexto}

REGLAS ABSOLUTAS:
1. SOLO seleccionar intervenciones relevantes.
2. NO inventar actividades. 3. NO resumir. 4. Máximo 5 intervenciones.

RESPONDE SOLO JSON. FORMATO EXACTO:
[
    {{"codigo":""}}
]"""

    resp = llamar_groq(prompt, temperature=0, max_tokens=250)
    try:
        data = json.loads(limpiar_json(resp))
        salida = []
        for item in data:
            cod = str(item.get("codigo", "")).strip()
            if cod not in metadata_map:
                continue
            nic = metadata_map[cod]
            actividades = nic.get("actividades", [])
            if isinstance(actividades, list):
                actividades_txt = actividades
            else:
                actividades_txt = [str(actividades)]
            salida.append({
                "codigo":      nic.get("codigo", cod),
                "nombre":      nic.get("nombre", ""),
                "definicion":  nic.get("definicion", ""),
                "actividades": actividades_txt
            })
        return salida
    except Exception:
        # Fallback determinista
        _, _, _, metadata_map2, _ = cargar_vectorstore()
        fallback = []
        for d in docs[:3]:
            meta = metadata_map2.get(d["codigo"], {})
            if not meta:
                continue
            actividades = meta.get("actividades", [])
            if isinstance(actividades, list):
                actividades_txt = actividades[:3]
            else:
                actividades_txt = [str(actividades)][:3]
            fallback.append({
                "codigo":      d["codigo"],
                "nombre":      meta.get("nombre", ""),
                "definicion":  meta.get("definicion", ""),
                "actividades": actividades_txt
            })
        return fallback


# ==============================
# AGENTE 3 – CLINICAL SYNTHESIS
# ==============================
def agente_3_generar_respuesta(pregunta: str, lista_gold: list, thinking: dict) -> dict:
    if not lista_gold:
        return {"justificacion_clinica": "No existe evidencia NIC suficiente.", "plan_cuidados": []}

    evidencia_json = json.dumps(lista_gold, ensure_ascii=False, indent=2)
    thinking_json  = json.dumps(thinking,  ensure_ascii=False, indent=2)

    prompt = f"""Eres un Generador Clínico Grounded.

OBJETIVO: Generar recomendación SOLO usando evidencia recuperada.

CASO CLÍNICO:
\"\"\"{pregunta}\"\"\"

RAZONAMIENTO CLÍNICO:
{thinking_json}

EVIDENCIA NIC RECUPERADA:
{evidencia_json}

REGLAS ABSOLUTAS:
1. SOLO usar información presente en EVIDENCIA NIC.
2. NO conocimiento clínico externo. 3. NO inventar actividades.
4. NO mezclar actividades entre códigos. 5. Mantener nombres y códigos exactos.

FORMATO OBLIGATORIO:
{{
  "justificacion_clinica": "",
  "plan_cuidados": [
    {{
      "codigo": "",
      "nombre": "",
      "actividades": []
    }}
  ]
}}

RESPONDE SOLO JSON VÁLIDO."""

    resp = llamar_groq(prompt, temperature=0.1, max_tokens=900)
    try:
        return json.loads(limpiar_json(resp))
    except Exception:
        return {"justificacion_clinica": "No se pudo consolidar evidencia clínica.", "plan_cuidados": []}


# ==============================
# AGENTE 4 – HALLUCINATION VALIDATOR
# ==============================
def agente_4_validar(respuesta: dict, docs_contexto: list) -> dict:
    texto_fuente = "\n".join([d["texto"] for d in docs_contexto])
    respuesta_json = json.dumps(respuesta, ensure_ascii=False, indent=2)

    prompt = f"""Actúa como un Auditor de Seguridad Clínica. Detecta alucinaciones.

CONTEXTO ORIGINAL (LIBRO NIC):
{texto_fuente[:3000]}

RESPUESTA A VALIDAR:
{respuesta_json}

CRITERIOS:
1. ¿El código coincide exactamente con el texto fuente?
2. ¿Cada actividad aparece textualmente en el contexto?
3. ¿Se añadió alguna recomendación farmacológica no presente en el NIC?

REGLA DE ORO: Si hay error, corrígelo según texto fuente. No expliques cambios.
Devuelve la respuesta corregida en el mismo JSON."""

    resp = llamar_groq(prompt, temperature=0, max_tokens=900)
    try:
        corregida = json.loads(limpiar_json(resp))
        return corregida
    except Exception:
        return respuesta


# ==============================
# AGENTE 5 – HUMANIZER
# ==============================
def agente_5_humanizar(respuesta_validada: dict, pregunta: str) -> str:
    respuesta_json = json.dumps(respuesta_validada, ensure_ascii=False, indent=2)

    prompt = f"""Eres un Asistente Clínico Conversacional especializado en Enfermería.

Tu trabajo NO es crear un reporte. Tu trabajo es responder como un chatbot clínico profesional.

CONTEXTO DEL USUARIO:
{pregunta}

RESPUESTA CLÍNICA VALIDADA:
{respuesta_json}

OBJETIVO: Transforma la respuesta en una conversación clara, humana y rápida de leer para una enfermera.

ESTILO: Conversacional, profesional, empático, directo, fácil de leer en guardia.

IMPORTANTE:
1. NO cambies códigos NIC. 2. NO cambies nombres NIC. 3. NO cambies actividades.
4. NO agregues intervenciones nuevas. 5. NO inventes recomendaciones clínicas.
6. Puedes EXPLICAR brevemente por qué una intervención ayuda, pero SOLO basado en el texto validado.

TONO:
❌ NO sonar como informe médico.
✅ Sonar como una respuesta de chatbot profesional de enfermería.

RESPUESTA:"""

    resp = llamar_groq(prompt, temperature=0.1, max_tokens=700)
    return resp if resp else json.dumps(respuesta_validada, ensure_ascii=False, indent=2)


# ==============================
# PIPELINE COMPLETO (5 AGENTES)
# ==============================
def pipeline_rag(pregunta: str, k: int = 5) -> dict:
    """
    Pipeline multi-agente:
    Agente 1: Clinical Reasoning + Reformulación
    Retrieval: Multi-query con RRF + Clinical Reranker
    Agente 3: Reflection (en retrieval loop)
    Agente 2: Evidence Grounding
    Agente 3: Clinical Synthesis
    Agente 4: Hallucination Validator
    Agente 5: Humanizer
    """

    # --- Agente 1: Reformulación y reasoning ---
    a1 = agente_1_reformular(pregunta)

    # --- Retrieval multi-query ---
    docs = buscar_multi_query(
        query_original=pregunta,
        query_opt=a1.get("query_rag", pregunta),
        sub_queries=a1.get("sub_queries", []),
        k=k + 3
    )

    # --- Clinical Reranker ---
    docs_ranked = clinical_reranker(
        docs,
        sintomas=a1.get("sintomas", []),
        keywords_nic=a1.get("keywords_nic", []),
        top_k=k
    )

    # --- Agente 2: Extracción de evidencia gold ---
    lista_gold = agente_2_extraer_gold(docs_ranked[:k])

    # --- Agente 3: Síntesis clínica ---
    respuesta_raw = agente_3_generar_respuesta(pregunta, lista_gold, a1.get("thinking", {}))

    # --- Agente 4: Validación ---
    respuesta_validada = agente_4_validar(respuesta_raw, docs_ranked[:k])

    # --- Agente 5: Humanización ---
    respuesta_final = agente_5_humanizar(respuesta_validada, pregunta)

    # Contexto para mostrar en el expander
    contexto_mostrable = ""
    for i, d in enumerate(docs_ranked[:k], 1):
        score = d.get("rerank_score", d.get("fusion_score", d.get("score", 0)))
        contexto_mostrable += f"🔹 **Doc {i} — NIC {d['codigo']} (score={score:.4f})**\n{d['texto'][:300]}...\n\n"

    return {
        "thinking":    a1.get("thinking", {}),
        "query_rag":   a1.get("query_rag", ""),
        "intervenciones": [g["codigo"] for g in lista_gold],
        "respuesta":   respuesta_final,
        "contexto":    contexto_mostrable
    }


# ==============================
# TRANSCRIPCIÓN WHISPER
# ==============================
def transcribir_audio(audio_bytes: bytes) -> str:
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name
        with open(tmp_path, "rb") as audio_file:
            transcript = openai_client.audio.transcriptions.create(
                model="whisper-1", file=audio_file, language="es"
            )
        os.unlink(tmp_path)
        return transcript.text.strip()
    except Exception as e:
        st.error(f"❌ Error al transcribir con OpenAI Whisper: {e}")
        return ""


# ==============================
# ESTADO DE SESIÓN
# ==============================
if "messages" not in st.session_state:
    st.session_state.messages = [{
        "role": "assistant",
        "content": "👋 ¡Hola! Soy tu asistente NIC con pipeline multi-agente. Puedes escribir tu consulta o usar el micrófono 🎤"
    }]
if "audio_processed" not in st.session_state:
    st.session_state.audio_processed = None
if "pending_audio" not in st.session_state:
    st.session_state.pending_audio = None


# ==============================
# HEADER
# ==============================
st.markdown("""
<div class="title-container">
    <h1>🩺 Asistente NIC con RAG Multi-Agente</h1>
    <p>Pipeline de 5 agentes · Clinical Reasoning → Retrieval → Evidence Grounding → Synthesis → Validation → Humanizer</p>
</div>
""", unsafe_allow_html=True)

# ==============================
# CARGAR VECTORSTORE (warmup)
# ==============================
with st.spinner("⚙️ Cargando índice NIC y modelo de embeddings..."):
    cargar_vectorstore()

# ==============================
# ÁREA DE CHAT
# ==============================
with st.container():
    for msg in st.session_state.messages:
        avatar = "👤" if msg["role"] == "user" else "🩺"
        with st.chat_message(msg["role"], avatar=avatar):
            st.write(msg["content"])
            if "contexto" in msg:
                with st.expander("🔍 Ver contexto y agentes utilizados", expanded=False):
                    if "thinking" in msg and msg["thinking"]:
                        st.markdown("**🧠 Clinical Reasoning (Agente 1):**")
                        st.json(msg["thinking"])
                    if "query_rag" in msg and msg["query_rag"]:
                        st.markdown(f"**🔎 Query RAG optimizada:** `{msg['query_rag']}`")
                    if "intervenciones" in msg and msg["intervenciones"]:
                        st.markdown(f"**📋 NIC seleccionadas:** {', '.join(msg['intervenciones'])}")
                    st.markdown("**📚 Contexto recuperado:**")
                    st.markdown(msg["contexto"])

    # Procesar audio pendiente
    if st.session_state.pending_audio is not None:
        with st.spinner("🎤 Transcribiendo audio con OpenAI Whisper..."):
            transcribed = transcribir_audio(st.session_state.pending_audio)
        if transcribed:
            st.success("✅ Transcripción completada:")
            st.markdown(f"**Texto detectado:** {transcribed}")
            st.session_state.messages.append({"role": "user", "content": transcribed})
            st.session_state.pending_audio  = None
            st.session_state.audio_processed = None
        else:
            st.error("❌ No se pudo transcribir. Intenta nuevamente.")
            st.session_state.pending_audio = None

    # Generar respuesta si el último mensaje es del usuario
    if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
        user_query = st.session_state.messages[-1]["content"]

        with st.spinner("🤖 Ejecutando pipeline multi-agente (5 agentes)..."):
            resultado = pipeline_rag(user_query, k=5)

        st.session_state.messages.append({
            "role":          "assistant",
            "content":       resultado["respuesta"],
            "contexto":      resultado["contexto"],
            "thinking":      resultado["thinking"],
            "query_rag":     resultado["query_rag"],
            "intervenciones": resultado["intervenciones"]
        })
        st.rerun()


# ==============================
# INPUT DE TEXTO Y AUDIO
# ==============================
col1, col2 = st.columns([5, 1])

with col1:
    user_input = st.chat_input("💬 Escribe tu consulta aquí...")

with col2:
    audio_bytes = audio_recorder(
        text="",
        recording_color="#e74c3c",
        neutral_color="#667eea",
        icon_name="microphone",
        icon_size="2x",
        key=f"audio_{len(st.session_state.messages)}"
    )

if audio_bytes and audio_bytes != st.session_state.audio_processed:
    st.session_state.audio_processed = audio_bytes
    st.session_state.pending_audio   = audio_bytes
    st.rerun()

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.rerun()


# ==============================
# FOOTER
# ==============================
st.markdown("---")
st.caption("⚕️ Este sistema es solo de apoyo y no sustituye la valoración clínica profesional.")

if len(st.session_state.messages) > 1:
    if st.button("🗑️ Limpiar conversación", use_container_width=True):
        st.session_state.messages = [{
            "role": "assistant",
            "content": "👋 ¡Hola! Soy tu asistente NIC con pipeline multi-agente. Puedes escribir tu consulta o usar el micrófono 🎤"
        }]
        st.session_state.audio_processed = None
        st.session_state.pending_audio   = None
        st.rerun()
