import os
import requests
import streamlit as st
from google import genai
from google.genai import types

# Configuració de la pàgina
st.set_page_config(page_title="SOD Pedagògic", page_icon="🎓", layout="centered")

st.title("🎓 SOD Pedagògic Engine")
st.caption("Sistema Operatiu de Deliberació Pedagògica connectat a Render")

# Clau d'API i Endpoint de Render
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
RENDER_ENDPOINT = "https://sod-engine.onrender.com/evaluate-hypothesis"

if not GEMINI_API_KEY:
    st.error("⚠️ La variable GEMINI_API_KEY no està configurada als Secrets de Streamlit.")
    st.stop()

# Inicialitzar el client de GenAI
client = genai.Client(api_key=GEMINI_API_KEY)

# Definició de la funció per al Function Calling
evaluate_func = types.FunctionDeclaration(
    name="evaluateHypothesis",
    description="Envia les dades d'una hipòtesi pedagògica al backend Python a Render per avaluar la matriu de confiança.",
    parameters={
        "type": "OBJECT",
        "properties": {
            "context": {"type": "STRING", "description": "Context de la situació pedagògica"},
            "observations": {"type": "ARRAY", "items": {"type": "STRING"}, "description": "Llista d'observacions fetes"},
            "evidences": {"type": "ARRAY", "items": {"type": "STRING"}, "description": "Llista d'evidències o dades"},
            "hypothesis": {"type": "STRING", "description": "Hipòtesi pedagògica a validar"}
        },
        "required": ["context", "observations", "evidences", "hypothesis"]
    }
)

tools_config = [types.Tool(function_declarations=[evaluate_func])]

SYSTEM_INSTRUCTION = """
Ets el SOD (Sistema Operatiu de Deliberació Pedagògica).
Acompanya el docent en la presa de decisions pedagògiques rigoroses.

REGLES DE RESPOSTA:
1. Respon SEMPRE en català.
2. Separa clarament fets, observacions, normativa, evidències i inferències.
3. Classifica les intervencions en la jerarquia: MACRO, MESO o MICRO.
4. Quan el docent plantegi una hipòtesi o situació per avaluar, crida la funció `evaluateHypothesis`.
"""

# Inicialitzar l'historial de missatges
if "messages" not in st.session_state:
    st.session_state.messages = []

# Mostrar l'historial del xat
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Entrada de text de l'usuari
if user_input := st.chat_input("Escriu la situació pedagògica o hipòtesi..."):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Deliberant amb Gemini..."):
            try:
                # Utilitzem gemini-3.1-flash-lite com a model actiu per defecte
                response = client.models.generate_content(
    model='gemini-3.1-flash-lite',
    contents=user_input,
    config=types.GenerateContentConfig(
        system_instruction=SYSTEM_INSTRUCTION,
        tools=tools_config,
        temperature=0.7
    )
)

                # Comprovar si Gemini ha decidit executar la funció
                if response.function_calls:
                    for call in response.function_calls:
                        if call.name == "evaluateHypothesis":
                            st.info("🔄 Connectant amb el backend de Render (`sod-engine.onrender.com`)...")
                            payload = dict(call.args)
                            
                            # Petició al backend de Render
                            backend_res = requests.post(RENDER_ENDPOINT, json=payload)
                            
                            if backend_res.status_code == 200:
                                res_data = backend_res.json()
                                confidence = res_data.get("confidence_level", "DESCONEGUT")
                                math_matrix = res_data.get("matrix", {})
                                
                                final_reply = f"### 📊 Resultats del Motor de Càlcul (Render)\n"
                                final_reply += f"- **Nivell de Confiança:** `{confidence}`\n"
                                if math_matrix:
                                    final_reply += f"- **Matriu:** {math_matrix}\n"
                                final_reply += f"\n### 💬 Anàlisi del SOD\n"
                                final_reply += f"Segons les evidències i el motor de càlcul, la hipòtesi presenta un nivell de confiança **{confidence}**."
                            else:
                                final_reply = f"⚠️ Error en la connexió amb Render (Codi HTTP {backend_res.status_code})."
                            
                            st.markdown(final_reply)
                            st.session_state.messages.append({"role": "assistant", "content": final_reply})
                else:
                    # Resposta de text directe de Gemini
                    st.markdown(response.text)
                    st.session_state.messages.append({"role": "assistant", "content": response.text})

            except Exception as e:
                st.error(f"Error en el processament: {e}")
