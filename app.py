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

# URL configurada per coincidir amb @app.post("/evaluate-hypothesis") de main.py
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
        with st.spinner("Deliberant amb el SOD Pedagògic..."):
            try:
                # 1. Primera crida a Gemini per analitzar l'entrada i detectar si cal cridar el backend
                response = client.models.generate_content(
                    model='gemini-3.1-flash-lite',
                    contents=user_input,
                    config=types.GenerateContentConfig(
                        system_instruction=SYSTEM_INSTRUCTION,
                        tools=tools_config,
                        temperature=0.7
                    )
                )

                # Comprovar si Gemini ha decidit executar la funció d'avaluació
                if response.function_calls:
                    for call in response.function_calls:
                        if call.name == "evaluateHypothesis":
                            st.info("🔄 Processant dades amb el motor de càlcul a Render...")
                            payload = dict(call.args)
                            
                            # 2. Petició HTTP POST al backend de Render
                            backend_res = requests.post(RENDER_ENDPOINT, json=payload)
                            
                            if backend_res.status_code == 200:
                                res_data = backend_res.json()
                                
                                # 3. Re-enviem els resultats de Render a Gemini per a la deliberació final
                                prompt_de_deliberacio = f"""
                                L'usuari ha plantejat la següent situació pedagògica:
                                "{user_input}"

                                El motor de càlcul de Render ha retornat aquests resultats tècnics:
                                {res_data}

                                Com a SOD (Sistema Operatiu de Deliberació Pedagògica), redacta una resposta deliberativa completa, 
                                rigorosa i ben estructurada per al docent. 
                                Incorporeu les dades del backend (nivell de confiança, recomanació i mètriques) dins d'un raonament pedagògic natural, 
                                respectant les vostres instruccions del sistema (separar fets, observacions, evidències i inferències, 
                                i classificar el nivell d'intervenció en MACRO, MESO o MICRO).
                                """
                                
                                final_analysis = client.models.generate_content(
                                    model='gemini-3.1-flash-lite',
                                    contents=prompt_de_deliberacio,
                                    config=types.GenerateContentConfig(
                                        system_instruction=SYSTEM_INSTRUCTION,
                                        temperature=0.7
                                    )
                                )
                                
                                st.markdown(final_analysis.text)
                                st.session_state.messages.append({"role": "assistant", "content": final_analysis.text})
                            else:
                                error_msg = f"⚠️ Error en la connexió amb Render (Codi HTTP {backend_res.status_code})."
                                st.error(error_msg)
                else:
                    # Resposta de text directe si no s'ha requerit l'avaluació d'hipòtesi
                    st.markdown(response.text)
                    st.session_state.messages.append({"role": "assistant", "content": response.text})

            except Exception as e:
                st.error(f"Error en el processament: {e}")
