import os
import json
import re
import time
import sys
import requests
from typing import List, Dict, Any, Optional

# ==========================================================
# 0. VERIFICACIÓN PREVIA DE DEPENDENCIAS CLAVE
# ==========================================================
def verificar_dependencias():
    librerias_faltantes = []
    try:
        import groq
    except ImportError:
        librerias_faltantes.append("groq")
    try:
        import faiss
    except ImportError:
        librerias_faltantes.append("faiss-cpu")
    try:
        import sentence_transformers
    except ImportError:
        librerias_faltantes.append("sentence-transformers")
        
    if librerias_faltantes:
        print("\n" + "="*80)
        print("🚨 [ERROR DE DEPENDENCIAS] Faltan librerías críticas en tu entorno de ejecución.")
        print("="*80)
        print("👉 Por favor, ejecuta la siguiente celda en tu Notebook (Kaggle o Colab):")
        print(f"   !pip install {' '.join(librerias_faltantes)} -q")
        print("="*80 + "\n")
        sys.exit(1)

verificar_dependencias()

# ==========================================================
# 1. MOTOR DE LLAMADA MULTI-MODELO CON CONTROL DE TRÁFICO (429)
# ==========================================================
def call_llm(prompt_sistema: str, prompt_usuario: str, provider: str = "groq", model_name: Optional[str] = None) -> str:
    temperature = 0.0
    
    if provider == "groq":
        from groq import Groq
        client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        model = model_name or "llama-3.3-70b-versatile"
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": prompt_sistema},
                {"role": "user", "content": prompt_usuario}
            ],
            temperature=temperature
        )
        return resp.choices[0].message.content.strip()

    raise ValueError(f"Proveedor no soportado: {provider}")


def call_llm_with_retry(prompt_sistema: str, prompt_usuario: str, provider: str = "groq", model_name: Optional[str] = None, max_retries: int = 3) -> str:
    base_delay = 5
    for attempt in range(max_retries):
        try:
            return call_llm(prompt_sistema, prompt_usuario, provider=provider, model_name=model_name)
        except Exception as e:
            err_msg = str(e)
            if "429" in err_msg or "rate_limit" in err_msg.lower() or "limit reached" in err_msg.lower():
                wait_time = base_delay * (2 ** attempt)
                time_match = re.search(r'try again in (?:(\d+)h)?\s*(?:(\d+)m)?\s*(\d+(?:\.\d+)?)s', err_msg, re.IGNORECASE)
                if time_match:
                    hours = int(time_match.group(1)) if time_match.group(1) else 0
                    minutes = int(time_match.group(2)) if time_match.group(2) else 0
                    seconds = float(time_match.group(3)) if time_match.group(3) else 0.0
                    extracted_wait = (hours * 3600) + (minutes * 60) + seconds
                    if extracted_wait > 0:
                        wait_time = int(extracted_wait) + 3
                
                print(f"   ⏳ [Rate Limit 429] Límite en {provider}. Esperando {wait_time}s antes de reintentar...")
                time.sleep(wait_time)
            else:
                raise e
    raise Exception("Límite de reintentos agotado por Rate Limit en la API del LLM.")


def parse_json_robust(texto_crudo: str) -> Dict[str, Any]:
    texto_limpio = texto_crudo.strip()
    if "{" in texto_limpio:
        texto_limpio = texto_limpio[texto_limpio.find("{"):texto_limpio.rfind("}")+1]
    
    try:
        return json.loads(texto_limpio)
    except Exception:
        try:
            reparado = re.sub(r',\s*([}\]])', r'\1', texto_limpio)
            return json.loads(reparado)
        except Exception as e:
            print(f"   ⚠️ Error crítico de parseo JSON. Estructura no reparable. Detalle: {e}")
            return {}


# ==========================================================
# 2. AGENTE 1: REFORMATOR Y AUDITOR METODOLÓGICO (NEUTRO)
# ==========================================================
PROMPT_SISTEMA_REFORMULADOR = """
Eres un agente especializado en reformulación de consultas clínicas para sistemas de Recuperación Aumentada por Generación (Retrieval-Augmented Generation, RAG).

Tu única función es transformar una consulta escrita en lenguaje natural en una representation clínica más estructurada y consistente que facilite la búsqueda semántica mediante embeddings.

IMPORTANTE:

- No respondas la consulta.
- No proporciones recomendaciones clínicas.
- No proporciones diagnósticos.
- No sugieras intervenciones NIC.
- No inventes información.
- No completes datos faltantes.
- No utilices conocimiento externo.
- Conserva únicamente la información presente en la consulta.

OBJETIVOS

1. Identificar el tipo de consulta:
   - Caso clínico
   - Consulta técnica
   - Consulta simple

2. Extraer únicamente la información explícita contenida en el texto.

Debes identificar cuando existan:

• signos y síntomas
• enfermedades mencionadas
• procedimientos
• medicamentos
• resultados de laboratorio
• constantes fisiológicas
• edad
• sexo
• población (niño, adulto, adulto mayor, gestante, neonato, etc.)
• contexto clínico
• dispositivos o tratamientos ya existentes

3. Reformular la consulta eliminando únicamente:

• redundancias
• conectores innecesarios
• lenguaje conversacional
• expresiones ambiguas

sin modificar el significado clínico.

4. La consulta reformulada debe conservar:

- todos los conceptos clínicos presentes
- edades
- grupos etarios
- enfermedades
- signos y síntomas
- valores numéricos
- unidades de medida
- resultados de laboratorio
- contexto temporal cuando exista
- procedimientos mencionados

REGLA DE MEJORA METODOLÓGICA (NORMALIZACIÓN LÉXICA):
- Puedes normalizar variantes léxicas o coloquiales ampliamente aceptadas (ej. 'calentura' a 'fiebre' o 'falta de aire' a 'disnea') siempre que no modifiques ni alteres de ninguna forma el significado clínico de la consulta original.

No agregues conceptos nuevos.

No reemplaces términos por otros más técnicos si estos no aparecen explícitamente en el texto original (excepto para la normalización de lenguaje coloquial descrita arriba).

No generes sinónimos clínicos especulativos o que reorienten la búsqueda.

No expandas abreviaturas que no estén presentes.

La salida debe ser únicamente un JSON válido con la siguiente estructura:

{{
  "tipo_consulta": "",
  "hallazgos_clinicos": [],
  "query_rag": "",
  "terminos_clave": [],
  "datos_faltantes": []
}}

Donde:

- tipo_consulta:
  Clasificación general de la entrada.

- hallazgos_clinicos:
  Lista de todos los hallazgos explícitos encontrados.

- query_rag:
  Consulta clínica reformulada manteniendo únicamente la información existente y organizada para favorecer la recuperación semántica.

- terminos_clave:
  Conceptos clínicos relevantes presentes en la consulta.

- datos_faltantes:
  Información clínica que no aparece en la consulta pero cuya ausencia podría limitar la interpretación. Esta información es únicamente descriptiva y nunca debe utilizarse para modificar la query_rag.
"""

def agente_reformular_consulta(query_usuario: str, provider: str = "groq", model: Optional[str] = None) -> Dict[str, Any]:
    print(f"🧠 [Agente Reformulador - {provider}] Reformulando para optimizar búsqueda RAG...")
    respuesta_raw = call_llm_with_retry(
        prompt_sistema=PROMPT_SISTEMA_REFORMULADOR,
        prompt_usuario=f"Consulta:\n{query_usuario}",
        provider=provider,
        model_name=model
    )
    resultado = parse_json_robust(respuesta_raw)
    
    query_text = resultado.get("query_rag", query_usuario)
    if not query_text.lower().startswith("query: "):
        resultado["query_rag_final"] = f"query: {query_text}"
    else:
        resultado["query_rag_final"] = query_text
        
    resultado["_raw_response"] = respuesta_raw
    return resultado


# ==========================================================
# 3. MOTOR DE RECUPERACIÓN VECTORIAL (RETRIEVER)
# ==========================================================
class NICRetriever:
    def __init__(self, index, metadata, embedding_model):
        self.index = index
        self.metadata = metadata
        self.embedding_model = embedding_model

    def buscar(self, query_rag: str, k: int = 5) -> List[Dict]:
        q_text = query_rag if query_rag.lower().startswith("query: ") else f"query: {query_rag}"
        
        # Interrogación a través del objeto vectorstore de LangChain
        resultados = self.embedding_model.similarity_search_with_score(q_text, k=k)
        pool = []
        
        for i, (doc, score) in enumerate(resultados):
            pool.append({
                "codigo": doc.metadata.get("codigo", "0000"),
                "nombre": doc.metadata.get("nombre", "Intervención NIC"),
                "chunk_num": doc.metadata.get("chunk_num", i),
                "chunk_id": doc.metadata.get("chunk_id", f"{doc.metadata.get('codigo', '0000')}_{i}"),
                "texto_completo": doc.page_content,
                "score": round(float(score), 4)
            })
        return pool


# ==========================================================
# 4. AGENTE 2: CRÍTICO DE INTEGRIDAD CLÍNICA (NEUTRO)
# ==========================================================
PROMPT_CRITICO = """
Eres un Auditor de Integridad de Evidencia experto en taxonomía de enfermería NIC.
Tu única función es evaluar de manera estrictamente objetiva si los fragmentos específicos (chunks) recuperados de la NIC contain el contenido clínico y de actividades necesario para responder de forma completa y adecuada a la consulta clínica del usuario.

DIRECTRICES DE AUDITORÍA CONCEPTUAL (DEEP THINKING):
1. EVALUACIÓN DE COBERTURA TAXONÓMICA: No evalúes si faltan datos en la historia clínica del paciente (como edad, laboratorios o antecedentes). En su lugar, evalúa de manera crítica si el contenido y las actividades de los fragmentos (chunks) recuperados cubren los conceptos de cuidado necesarios para responder la consulta mediante intervenciones NIC (ej. si la consulta describe fiebre, debe haber actividades sobre temperatura).
2. DETERMINACIÓN DE SUFICIENCIA (MEJORA DE UMBRAL): Si existen fragmentos o intervenciones NIC recuperadas que ya cubren de forma directa y sustancial la necesidad o síntoma clínico principal de la consulta (ej. si la consulta describe fiebre alta, y ya hay fragmentos de control térmico o termorregulación), responde obligatoriamente con "necesita_mas_busqueda": false, aunque existan aspectos demográficos o clínicos secundarios no cubiertos. Prioriza la resolución del problema central para evitar un comportamiento hiper-estricto e innecesario.
3. FILTRADO DE COMPONENTES AJENOS Y RUIDO: Identifica y excluye de los fragmentos válidos aquellos chunks individuales que correspondan a ruido, colisión vectorial o que contengan actividades no pertinentes o contradictorias para la consulta del usuario.
4. REGLA ESTRICTA DE SELECCIÓN DE CHUNKS: En el campo 'chunks_aprobados' de la salida JSON, debes escribir ÚNICAMENTE los identificadores de fragmento (chunk_id) válidos (ej: ["3900_01", "3740_02"]) de los chunks provistos en la lista. Queda strictly prohibido incluir nombres de intervención, texto explicativo o códigos generales de 4 dígitos. Solo los chunk_id específicos que realmente aportan contenido útil y seguro.
5. SUGERENCIA DE BÚSQUEDA CONCEPTUAL: Si es necesaria una búsqueda complementaria, genera una consulta simplificada ('sugerencia_mejora') que combine ÚNICAMENTE conceptos explícitamente omitidos presentes en la consulta original (Ej. 'fiebre' o 'hipertermia'). No agregues códigos ni nombres de intervenciones complejas.

La salida debe ser únicamente un JSON válido con la siguiente estructura:
{{
    "analisis_auditoria": "Desglose técnico del contenido de los chunks y su pertinencia directa para responder la consulta.",
    "conceptos_cubiertos": ["lista", "de", "conceptos", "NIC", "cubiertos"],
    "conceptos_faltantes": ["lista", "de", "conceptos", "NIC", "faltantes"],
    "chunks_aprobados": ["3900_01", "3740_02"],
    "necesita_mas_busqueda": true/false,
    "sugerencia_mejora": "Nueva consulta sumamente simplificada utilizando conceptos explícitos ausentes en los resultados. No agregues códigos ni nombres de intervenciones."
}}
"""

def agente_criticar_recuperacion(query_original: str, resultados: List[Dict], provider: str = "groq", model: Optional[str] = None) -> Dict:
    if not resultados:
        return {
            "necesita_mas_busqueda": True, 
            "chunks_aprobados": [], 
            "conceptos_cubiertos": [], 
            "conceptos_faltantes": [query_original],
            "sugerencia_mejora": query_original
        }
    print(f"⚖️ [Agente Crítico - {provider}] Evaluando equilibrio e integridad del contexto...")
    
    contexto_str = ""
    for r in resultados:
        contexto_str += f"--- Chunk ID: {r.get('chunk_id')} ---\n"
        contexto_str += f"Intervención: {r.get('nombre')}\n"
        contexto_str += f"Contenido del Chunk:\n{r.get('texto_completo')[:1200]}\n\n"
        
    user_prompt = f"CASO ORIGINAL / PREGUNTA: {query_original}\n\nFRAGMENTOS (CHUNKS) RECUPERADOS EN LA BASE VECTORIAL:\n{contexto_str}"
    
    respuesta_raw = call_llm_with_retry(
        prompt_sistema=PROMPT_CRITICO,
        prompt_usuario=user_prompt,
        provider=provider,
        model_name=model
    )
    resultado = parse_json_robust(respuesta_raw)
    resultado["_raw_response"] = respuesta_raw
    return resultado


# ==========================================================
# 5. AGENTE 3: SINTETIZADOR CLÍNICO (ESTRICTAMENTE GROUNDED)
# ==========================================================
PROMPT_SINTESIS = """
Eres un especialista en Metodología del Cuidado y Taxonomía de Intervenciones. 
Tu tarea es construir una respuesta o plan estructurado basándote EXCLUSIVAMENTE en la evidencia recuperada y validada.

REGLAS DE CONSTRUCCIÓN DE RESPUESTA (ESTRICTO GROUNDEDNESS):
1. JUSTIFICACIÓN METODOLÓGICA: Explica de manera estricta y únicamente utilizando la información contenida en el Contexto Fiel cómo los fragmentos seleccionados responden de forma directa a la necesidad de la consulta. No utilices conocimiento clínico general, no inventes explicaciones biológicas ni uses justificaciones externas para fundamentar la selección. Queda terminantemente prohibido incluir introducciones narrativas o comentarios de metadatos sobre la consulta del usuario (como 'La solicitud original se refiere a...'). Comienza directamente con la justificación técnica del plan de cuidados.
2. PRESENTACIÓN DE INTERVENCIONES: Muestra las intervenciones estructuradas con su código identificador, nombre literal y objetivo.
3. EXTRACTO DE ACCIONES OPERATIVAS: Selecciona las actividades más representativas dentro del contexto proporcionado.
      REGLA DE GROUNDEDNESS CRÍTICA: Nunca resumas las actividades, nunca las combines ni inventes su redacción. Copia literalmente las actividades relevantes que figuren de manera explímitamente literal en los documentos del contexto recuperado. Bajo ningún concepto inventes o añadas intervenciones o soportes que no estén en el texto (ej: no agregues oxigenoterapia si no está en la evidencia literal).
      FILTRADO ESTRICTO DE POBLACIÓN: Evalúa críticamente el grupo de edad, población o estado del paciente descrito en la consulta original (ej: si es un niño, un recién nacido, una gestante o un adulto mayor). Si un fragmento aprobado contiene actividades diseñadas exclusivamente para otra población incompatible (ej. actividades neonatales o de 'recién nacido' para un niño de 5 años, o viceversa), ignora y excluye estrictamente esas actividades no pertinentes de tu síntesis final.
4. ALERTAS DE SEGURIDAD Y CONTROL: Identifica y resalta en negrita cualquier contraindicación, límite de seguridad, advertencia o pauta de control que esté expresada de manera explímitamente literal en los documentos del contexto recuperado que aplique directamente al perfil del paciente. Si no existen advertencias literales en los documentos, no agregues ninguna.

MANDATO DE FIDELIDAD ABSOLUTA: 
Queda totalmente prohibido usar conocimiento externo o añadir actividades que no figuren en el Contexto Fiel. Si la evidencia es insuficiente para algún aspecto de la solicitud, descríbelo explícitamente como un vacío de información disponible en la base de datos.

La respuesta debe ser en formato Markdown limpio y altamente profesional.
"""

def agente_sintetizar_recomendacion(query_original: str, contexto_depurado: List[Dict], provider: str = "groq", model: Optional[str] = None) -> str:
    if not contexto_depurado:
        return "Error: No se dispone de evidencia documental validada para procesar la respuesta."
    print(f"✍️ [Agente Sintetizador - {provider}] Consolidando plan técnico fundamentado...")
    contexto_str = ""
    for doc in contexto_depurado:
        contexto_str += f"--- CÓDIGO {doc['codigo']}\nChunk {doc.get('chunk_num')}\nIntervención {doc['nombre']}\n---\n\n{doc['texto_completo']}\n\n"
        
    user_prompt = f"SOLICITUD ORIGINAL DEL USUARIO:\n{query_original}\n\nEVIDENCIA TAXONÓMICA DISPONIBLE (CONTEXTO FIEL):\n{contexto_str}"
    
    return call_llm_with_retry(
        prompt_sistema=PROMPT_SINTESIS,
        prompt_usuario=user_prompt,
        provider=provider,
        model_name=model
    )


# ==========================================================
# 6. AGENTE 4: HUMANIZADOR ENFERMERO DE GUARDIA (CAPA DE NARRATIVA)
# ==========================================================
PROMPT_HUMANIZADOR = """
Actúa como un Enfermero Supervisor transmitiendo pautas operativas e inmediatas a un colega en turno.
Tu única tarea es adaptar el plan técnico de entrada para que sea asimilado en menos de 30 segundos.

REGLAS DE FORMATO Y CONTRALOR:
- Estilo directo, ejecutivo, estructurado por puntos clave de acción.
- Resalta alertas críticas de seguridad en negrita.
- Conserva de manera estricta los códigos taxonómicos de las intervenciones.
- NO ABSTRAER NI GENERALIZAR: No resumas ni traduzcas las acciones técnicas de enfermería a abstracciones vagas (ej: nunca cambies 'controlar presión arterial, pulso, respiración' por 'vigilar signos vitales' ni generalices pautas específicas). Copia y conserva las acciones operativas de forma literal, tal como provienen del plan técnico.
- No inventes elements operativos ni actividades que no existan en el plan técnico de entrada.
"""

def agente_humanizar_respuesta(respuesta_tecnica: str, provider: str = "groq") -> str:
    print(f"🎭 [Agente Humanizador - {provider}] Adaptando narrativa al perfil: enfermero_guardia...")
    
    user_prompt = f"PLAN TÉCNICO DE ENTRADA:\n{respuesta_tecnica}"
    
    return call_llm_with_retry(
        prompt_sistema=PROMPT_HUMANIZADOR,
        prompt_usuario=user_prompt,
        provider=provider
    )
