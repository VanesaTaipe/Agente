import os
import json
import re
import tempfile
import numpy as np
import streamlit as st
import google.generativeai as genai
import faiss
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from audio_recorder_streamlit import audio_recorder

# Importación del módulo de agentes con prompts originales
from agentes import (
    agente_reformular_consulta,
    agente_criticar_recuperacion,
    agente_sintetizar_recomendacion,
    agente_humanizar_respuesta,
    NICRetriever
)

# ==============================
# CONFIGURACIÓN INICIAL
# ==============================
st.set_page_config(page_title="Asistente NIC Multi-Agente", page_icon="🩺", layout="wide")

# CONFIGURACIÓN DE TU MODELO PRIVADO DE HUGGING FACE
# El embedding E5 permanece fijo e invariable para mantener el rigor del experimento
MI_MODELO_PRIVADO_HF =  "vanesam123/Modelo-Funnintg"

# API Keys desde Secrets
GROQ_API_KEY = st.secrets.get("GROQ_API_KEY")
HF_TOKEN = st.secrets.get("HF_TOKEN")

if not all([GROQ_API_KEY, HF_TOKEN]):
    st.error("⚠️ Faltan API Keys requeridas en Streamlit Secrets (GROQ_API_KEY y HF_TOKEN).")
    st.stop()

# Configuración global de Variables de Entorno para los motores de inferencia
os.environ["GROQ_API_KEY"] = GROQ_API_KEY
os.environ["HF_TOKEN"] = HF_TOKEN

# Configuración opcional de Gemini (usado solo para el filtro de pertinencia
# y la sugerencia proactiva). Si no está presente, esas dos funciones
# simplemente no se ejecutan (ver los try/except más abajo).
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

# PARAMETRIZACIÓN DEL EXPERIMENTO (NVIDIA / MISTRAL POR DEFECTO PARA RAZONAMIENTO)
PROVIDER_EVALUADO = "groq"
MODELO_EVALUADO = "llama-3.3-70b-versatile"

# ==============================
# CSS PERSONALIZADO
# ==============================
st.markdown("""
<style>
#MainMenu {visibility: hidden;} footer {visibility: hidden;} header {visibility: hidden;}
.main { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); }
.title-container { background: white; padding: 2rem; border-radius: 15px; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1); margin-bottom: 2rem; text-align: center; }
.title-container h1 { color: #667eea; margin: 0; font-size: 2.5rem; }
.title-container p { color: #666; margin: 0.5rem 0 0 0; font-size: 1.1rem; }
.stChatFloatingInputContainer { background: white; border-radius: 15px; padding: 1.5rem; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1); }
</style>
""", unsafe_allow_html=True)

# ==============================
# CARGAR RETRIEVER EXPERIMENTAL (CORREGIDO)
# ==============================
@st.cache_resource(show_spinner=False)
def inicializar_retriever_experimental():
    """
    Inicializa el NICRetriever utilizando la lógica rigurosa de tu tesis,
    enlazando directamente el índice FAISS nativo y los archivos correspondientes.
    """
    PATH_INDICE = "rag_index (2).faiss"
    PATH_METADATA = "rag_metadata (2).json"

    # Se pasa el ID de Hugging Face del modelo privado tal como requiere SentenceTransformer internamente
    retriever_instancia = NICRetriever(
        index_path=PATH_INDICE,
        metadata_path=PATH_METADATA,
        model_path=MI_MODELO_PRIVADO_HF
    )
    return retriever_instancia

# Instanciamos el motor idéntico a tu pipeline de Kaggle
retriever = inicializar_retriever_experimental()

# ==============================
# FILTRO DE RELEVANCIA (AHORRO TOKENS)
# ==============================
def validar_pertinencia_clinica(consulta: str) -> bool:
    if not GEMINI_API_KEY:
        return True
    prompt_filtro = f"""
    Eres un validador estricto. Determina si la siguiente consulta tiene relación con enfermería, 
    diagnósticos, taxonomía NIC, medicina o cuidados de salud.
    Responde únicamente con 'SI' si guarda relación o 'NO' si es totalmente ajena.
    
    Consulta: "{consulta}"
    """
    try:
        model = genai.GenerativeModel("gemini-2.0-flash")
        resp = model.generate_content(prompt_filtro).text.strip().upper()
        return "SI" in resp
    except Exception:
        return True

# ==============================
# TRANSCRIPCIÓN CON OPENAI WHISPER
# ==============================
def transcribir_audio_openai(audio_bytes: bytes) -> str:
    try:
        openai_client = OpenAI() 
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
        st.error(f"❌ Error al transcribir con OpenAI Whisper: {str(e)}")
        return ""

# ==============================
# INICIALIZACIÓN DE SESIÓN (HISTORIAL)
# ==============================
if "messages" not in st.session_state:
    st.session_state.messages = [{
        "role": "assistant",
        "content": "👋 ¡Hola! Soy tu asistente NIC experto. Puedes escribir tu consulta o usar el micrófono."
    }]

if "audio_processed" not in st.session_state: st.session_state.audio_processed = None
if "pending_audio" not in st.session_state: st.session_state.pending_audio = None

# ==============================
# HEADER
# ==============================
st.markdown("""
<div class="title-container">
    <h1>🩺 Asistente NIC con Pipeline RAG</h1>
    <p>Consulta Inteligente con Control de Integridad y Verificación Metodológica de Agentes</p>
</div>
""", unsafe_allow_html=True)

# ==============================
# ÁREA DE CHAT (RESTRICCIÓN A ÚLTIMOS 5 MENSAJES)
# ==============================
mensajes_historial = st.session_state.messages[-5:]

with st.container():
    for msg in mensajes_historial:
        avatar = "👤" if msg["role"] == "user" else "🩺"
        with st.chat_message(msg["role"], avatar=avatar):
            st.write(msg["content"])
            if "context" in msg:
                with st.expander("🔍 Ver chunks y trazabilidad de evidencia utilizados", expanded=False):
                    st.markdown(f"```\n{msg['context']}\n```")

    if st.session_state.pending_audio is not None:
        with st.spinner("🎤 Transcribiendo audio con OpenAI Whisper..."):
            transcribed_text = transcribir_audio_openai(st.session_state.pending_audio)

        if transcribed_text:
            st.session_state.messages.append({"role": "user", "content": transcribed_text})
            st.session_state.pending_audio = None
            st.session_state.audio_processed = None
            st.rerun()
        else:
            st.session_state.pending_audio = None

    if len(st.session_state.messages) > 0 and st.session_state.messages[-1]["role"] == "user":
        user_query = st.session_state.messages[-1]["content"]

        if not validar_pertinencia_clinica(user_query):
            st.session_state.messages.append({
                "role": "assistant",
                "content": "❌ No estoy especializado para responder consultas ajenas al ámbito de la enfermería, medicina o la taxonomía de intervenciones NIC."
            })
            st.rerun()

        with st.spinner("🧠 Ejecutando Agente Reformulador..."):
            reformulacion = agente_reformular_consulta(user_query, provider=PROVIDER_EVALUADO, model=MODELO_EVALUADO)
            query_rag = reformulacion.get("query_rag_final", user_query)
            datos_faltantes = reformulacion.get("datos_faltantes", [])

        if datos_faltantes:
            vicios_texto = ", ".join(datos_faltantes)
            st.session_state.messages.append({
                "role": "assistant",
                "content": f"🩺 Para estructurar adecuadamente las intervenciones NIC, requiero información adicional. ¿Podrías especificar detalles sobre: **{vicios_texto}**?"
            })
            st.rerun()

        with st.spinner("🔍 Realizando match vectorial en base de conocimiento..."):
            pool_resultados = retriever.buscar(query_rag, k=10) # Ajustado a k=10 para máxima profundidad semántica como en tu test
            if not pool_resultados:
                pool_resultados = retriever.buscar(user_query, k=10)

        with st.spinner("⚖️ Ejecutando Agente Crítico de Integridad..."):
            informe_critico = agente_criticar_recuperacion(user_query, pool_resultados, provider=PROVIDER_EVALUADO, model=MODELO_EVALUADO)
            chunks_aprobados_raw = informe_critico.get("chunks_aprobados", [])
            necesita_mas_busqueda = informe_critico.get("necesita_mas_busqueda", False)

        # Bucle iterativo adaptado del script de evaluación
        if necesita_mas_busqueda and informe_critico.get("sugerencia_mejora"):
            sugerencia = informe_critico.get("sugerencia_mejora", "").strip()
            if sugerencia and sugerencia.lower() != query_rag.lower():
                with st.spinner("↪️ Solicitando rescate semántico complementario..."):
                    resultados_extra = retriever.buscar(sugerencia, k=5)

                    chunks_existentes = {r.get("chunk_id") for r in pool_resultados}
                    for r in resultados_extra:
                        if r.get("chunk_id") not in chunks_existentes:
                            pool_resultados.append(r)

                    informe_critico = agente_criticar_recuperacion(user_query, pool_resultados, provider=PROVIDER_EVALUADO, model=MODELO_EVALUADO)
                    chunks_aprobados_raw = informe_critico.get("chunks_aprobados", [])

        # Sanitización estricta de regex para limpiar los chunk_ids seleccionados
        validos = []
        for item in chunks_aprobados_raw:
            item_str = str(item).strip()
            match = re.search(r'\d{4}_\d+', item_str)
            if match:
                validos.append(match.group())
            else:
                clean = re.sub(r'[^\d_]', '', item_str)
                if clean:
                    validos.append(clean)

        contexto_filtrado = [
            r for r in pool_resultados 
            if r.get('chunk_id') in validos or r.get('codigo') in validos
        ]

        if not contexto_filtrado:
            contexto_filtrado = sorted(pool_resultados, key=lambda x: x['score'])[:3]

        with st.spinner("✍️ Ejecutando Agente Sintetizador..."):
            plan_tecnico = agente_sintetizar_recomendacion(user_query, contexto_filtrado, provider=PROVIDER_EVALUADO, model=MODELO_EVALUADO)

        with st.spinner("🎭 Ejecutando Agente Humanizador..."):
            respuesta_humanizada = agente_humanizar_respuesta(plan_tecnico, provider=PROVIDER_EVALUADO)

        if GEMINI_API_KEY:
            with st.spinner("💭 Formulando sugerencia proactiva..."):
                prompt_proactivo = f"""
                Basándote en el siguiente plan de cuidados clínico: {plan_tecnico[:800]}
                Escribe una sola pregunta corta y complementaria para proponerle al usuario si desea información adicional sobre un aspecto crítico del cuidado omitido o relacionado.
                """
                try:
                    pregunta_interactiva = genai.GenerativeModel("gemini-2.0-flash").generate_content(prompt_proactivo).text.strip()
                    respuesta_humanizada += f"\n\n---\n💡 **¿Deseas profundizar más?** {pregunta_interactiva}"
                except Exception:
                    pass

        contexto_mostrable = ""
        for idx, r in enumerate(contexto_filtrado, start=1):
            contexto_mostrable += f"🔹 Fragmento {idx} | Código: {r['codigo']} | Intervención: {r['nombre']} | Chunk ID: {r.get('chunk_id')} (Score={r['score']})\n{r['texto_completo']}\n\n"

        st.session_state.messages.append({
            "role": "assistant",
            "content": respuesta_humanizada,
            "context": contexto_mostrable
        })
        st.rerun()

# ==============================
# INPUTS DE CONTROL DE USUARIO
# ==============================
col1, col2 = st.columns([5, 1])
with col1:
    user_input = st.chat_input("💬 Escribe tu consulta aquí...")
with col2:
    audio_bytes = audio_recorder(
        text="", recording_color="#e74c3c", neutral_color="#667eea", icon_name="microphone", icon_size="2x",
        key=f"audio_recorder_{len(st.session_state.messages)}"
    )

if audio_bytes and audio_bytes != st.session_state.audio_processed:
    st.session_state.audio_processed = audio_bytes
    st.session_state.pending_audio = audio_bytes
    st.rerun()

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.rerun()

st.markdown("---")
st.caption("⚕️ Este sistema es solo de apoyo metodológico y no sustituye la valoración clínica profesional.")

if len(st.session_state.messages) > 1:
    if st.button("🗑️ Limpiar e iniciar conversación", use_container_width=True):
        st.session_state.messages = [{
            "role": "assistant",
            "content": "👋 ¡Hola! Soy tu asistente NIC experto. Puedes escribir tu consulta o usar el micrófono."
        }]
        st.session_state.audio_processed = None
        st.session_state.pending_audio = None
        st.rerun()
