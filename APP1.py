import os
import json
import pickle
import tempfile
import numpy as np
import streamlit as st
import google.generativeai as genai
import faiss
from openai import OpenAI
from langchain.docstore.document import Document
from langchain.docstore import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
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

# API Keys desde Secrets
GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY")
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY")
GROQ_API_KEY = st.secrets.get("GROQ_API_KEY")

if not all([GEMINI_API_KEY, OPENAI_API_KEY, GROQ_API_KEY]):
    st.error("⚠️ Faltan API Keys requeridas en Streamlit Secrets (GEMINI_API_KEY, OPENAI_API_KEY y GROQ_API_KEY).")
    st.stop()

# Configuración global de Variables de Entorno para el motor
os.environ["GROQ_API_KEY"] = GROQ_API_KEY
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
os.environ["GEMINI_API_KEY"] = GEMINI_API_KEY

genai.configure(api_key=GEMINI_API_KEY)
openai_client = OpenAI(api_key=OPENAI_API_KEY)

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
# CARGAR VECTORSTORE COMPLETO
# ==============================
@st.cache_resource(show_spinner=False)
def cargar_vectorstore_desde_archivos():
    index = faiss.read_index("indice_faiss.index")
    with open("metadata.pkl", "rb") as f:
        metadata = pickle.load(f)
    with open("chunks_con_headers.pkl", "rb") as f:
        textos = pickle.load(f)

    embedding_model = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-large")
     
    documentos = []
    for i, t in enumerate(textos):
        contenido = f"[{t.get('seccion', 'Sin sección')}] {t.get('texto', '')}" if isinstance(t, dict) else str(t)
        meta = metadata[i] if i < len(metadata) else {}
        documentos.append(Document(page_content=contenido, metadata=meta))

    docstore_items = {}
    index_to_docstore_id = {}
    for i, doc in enumerate(documentos):
        doc_id = f"doc_{i}"
        docstore_items[doc_id] = doc
        index_to_docstore_id[i] = doc_id
    
    docstore = InMemoryDocstore(docstore_items)
    vectorstore = FAISS(
        embedding_function=embedding_model.embed_query,
        index=index,
        docstore=docstore,
        index_to_docstore_id=index_to_docstore_id
    )
    return vectorstore

vectorstore = cargar_vectorstore_desde_archivos()
retriever = NICRetriever(vectorstore.index, vectorstore.docstore, vectorstore)

# ==============================
# FILTRO DE RELEVANCIA (AHORRO TOKENS)
# ==============================
def validar_pertinencia_clinica(consulta: str) -> bool:
    """
    Evalúa de forma rápida si la consulta pertenece al área de enfermería o medicina,
    evitando que se llame a la cadena entera de agentes si el usuario pregunta algo no relacionado.
    """
    prompt_filtro = f"""
    Eres un validador estricto. Determina si la siguiente consulta tiene relación con enfermería, 
    diagnósticos, taxonomía NIC, medicina o casos de cuidados de salud.
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
# Obtenemos únicamente los últimos 5 mensajes de la sesión para limitar la ventana de contexto
mensajes_historial = st.session_state.messages[-5:]

with st.container():
    for msg in mensajes_historial:
        avatar = "👤" if msg["role"] == "user" else "🩺"
        with st.chat_message(msg["role"], avatar=avatar):
            st.write(msg["content"])
            if "context" in msg:
                with st.expander("🔍 Ver chunks y trazabilidad de evidencia utilizados", expanded=False):
                    st.markdown(f"```\n{msg['context']}\n```")

    # Procesar entrada por micrófono
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

    # Si el último mensaje es del usuario, ejecutar el pipeline completo
    if len(st.session_state.messages) > 0 and st.session_state.messages[-1]["role"] == "user":
        user_query = st.session_state.messages[-1]["content"]

        # GUARDARAIL: Validar pertinencia clínica inmediata para ahorro estricto de tokens
        if not validar_pertinencia_clinica(user_query):
            st.session_state.messages.append({
                "role": "assistant",
                "content": "❌ No estoy especializado para responder consultas ajenas al ámbito de la enfermería, medicina o la taxonomía de intervenciones NIC."
            })
            st.rerun()

        # FASE 1: Agente Reformulador
        with st.spinner("🧠 Ejecutando Agente Reformulador..."):
            reformulacion = agente_reformular_consulta(user_query, provider=PROVIDER_EVALUADO, model=MODELO_EVALUADO)
            query_rag = reformulacion.get("query_rag_final", user_query)
            datos_faltantes = reformulacion.get("datos_faltantes", [])

        # BUCLE DE PREGUNTA: Si faltan datos clínicos para evaluar la taxonomía, se le pregunta al usuario
        if datos_faltantes:
            vicios_texto = ", ".join(datos_faltantes)
            st.session_state.messages.append({
                "role": "assistant",
                "content": f"🩺 Para estructurar adecuadamente las intervenciones NIC, requiero información adicional. ¿Podrías especificar detalles sobre: **{vicios_texto}**?"
            })
            st.rerun()

        # FASE 2: Búsqueda Vectorial (Retriever Recall k=5)
        with st.spinner("🔍 Realizando match vectorial en base de conocimiento..."):
            pool_resultados = retriever.buscar(query_rag, k=5)

        # FASE 3: Agente Crítico de Integridad
        with st.spinner("⚖️ Ejecutando Agente Crítico de Integridad..."):
            informe_critico = agente_criticar_recuperacion(user_query, pool_resultados, provider=PROVIDER_EVALUADO, model=MODELO_EVALUADO)
            chunks_aprobados = informe_critico.get("chunks_aprobados", [])
            necesita_mas_busqueda = informe_critico.get("necesita_mas_busqueda", False)

        # Bucle de re-búsqueda conceptual si el crítico lo solicita
        if necesita_mas_busqueda and informe_critico.get("sugerencia_mejora"):
            with st.spinner("↪️ Solicitando rescate semántico complementario..."):
                resultados_extra = retriever.buscar(informe_critico["sugerencia_mejora"], k=3)
                pool_resultados.extend(resultados_extra)
                # Re-evaluar pool ampliado
                informe_critico = agente_criticar_recuperacion(user_query, pool_resultados, provider=PROVIDER_EVALUADO, model=MODELO_EVALUADO)
                chunks_aprobados = informe_critico.get("chunks_aprobados", [])

        # Filtrado basado estrictamente en las decisiones del Crítico
        contexto_filtrado = [r for r in pool_resultados if r.get('chunk_id') in chunks_aprobados]
        if not contexto_filtrado:
            contexto_filtrado = sorted(pool_resultados, key=lambda x: x['score'])[:3]

        # FASE 4: Agente Sintetizador (Plan Técnico Grounded)
        with st.spinner("✍️ Ejecutando Agente Sintetizador..."):
            plan_tecnico = agente_sintetizar_recomendacion(user_query, contexto_filtrado, provider=PROVIDER_EVALUADO, model=MODELO_EVALUADO)

        # FASE 5: Capa de Narrativa (Humanizador)
        with st.spinner("🎭 Ejecutando Agente Humanizador..."):
            respuesta_humanizada = agente_humanizar_respuesta(plan_tecnico, provider=PROVIDER_EVALUADO)

        # PREGUNTAR INFORMACIÓN ADICIONAL PROACTIVA (De acuerdo al tema que preguntó)
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

        # Formatear traza mostrable
        contexto_mostrable = ""
        for idx, r in enumerate(contexto_filtrado, start=1):
            contexto_mostrable += f"🔹 Fragmento {idx} | Código: {r['codigo']} | Intervención: {r['nombre']} (Score={r['score']})\n{r['texto_completo']}\n\n"

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

# Botón inferior para reiniciar la conversación
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
