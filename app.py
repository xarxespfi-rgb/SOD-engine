import os
import requests
import streamlit as st
from google import genai
from google.genai import types

st.set_page_config(page_title="SOD Pedagògic", page_icon="🎓", layout="centered")

st.title("🎓 SOD Pedagògic Engine")
st.caption("Sistema Operatiu de Deliberació Pedagògica connectat a Render")

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
RENDER_ENDPOINT = "https://sod-engine.onrender.com/evaluate"

if not GEMINI_API_KEY:
    st.error("⚠️ La variable GEMINI_API_KEY no està configurada.")
    st.stop()

client = genai.Client(api_key=GEMINI_API_KEY)

# Definició correcta de la funció utilitzant types.FunctionDeclaration
evaluate_func = types.FunctionDeclaration(
    name="evaluateHypothesis",
    description="Envia les dades d'una hipòtesi pedagògica al backend Python a Render.",
    parameters={
        "type": "OBJECT",
        "properties": {
            "context": {"type": "STRING"},
            "observations": {"type": "ARRAY", "items": {"type": "STRING"}},
            "evidences": {"type": "ARRAY", "items": {"type": "STRING"}},
            "hypothesis": {"type": "STRING"}
        },
        "required": ["context", "observations", "evidences", "hypothesis"]
    }
)

tools_config = [types.Tool(function_declarations=[evaluate_func])]

SYSTEM_INSTRUCTION = """
Ets el SOD (Sistema Operatiu de Deliberació Pedagògica).
Respon SEMPRE en català. Separa clarament fets, observacions, evidències i inferències.
Classifica les intervencions en MACRO, MESO o MICRO.
Quan el docent plantegi una hipòtesi, crida la funció `evaluateHypothesis`.
"""

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if user_input := st.chat_input("Escriu la situació pedagògica o hipòtesi..."):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Deliberant pedagògicament..."):
            try:
                response = client.models.generate_content(
                    model='gemini-2.5-flash',
                    contents=user_input,
                    config=types.GenerateContentConfig(
                        system_instruction=SYSTEM_INSTRUCTION,
                        tools=tools_config,
                        temperature=0.7
                    )
                )

                if response.function_calls:
                    for call in response.function_calls:
                        if call.name == "evaluateHypothesis":
                            st.info("🔄 Connectant amb el backend de Render (`sod-engine.onrender.com`)...")
                            payload = dict(call.args)
                            
                            backend_res = requests.post(RENDER_ENDPOINT, json=payload)
                            
                            if backend_res.status_code == 200:
                                res_data = backend_res.json()
                                confidence = res_data.get("confidence_level", "DESCONEGUT")
                                
                                final_reply = f"### 📊 Resultats del Motor de Càlcul (Render)\n"
                                final_reply += f"- **Nivell de Confiança:** `{confidence}`\n\n"
                                final_reply += f"### 💬 Anàlisi del SOD\n"
                                final_reply += f"Segons les evidències analitzades, la hipòtesi presenta un nivell de confiança **{confidence}**."
                            else:
                                final_reply = f"⚠️ Error en la connexió amb Render (Codi {backend_res.status_code})."
                            
                            st.markdown(final_reply)
                            st.session_state.messages.append({"role": "assistant", "content": final_reply})
                else:
                    st.markdown(response.text)
                    st.session_state.messages.append({"role": "assistant", "content": response.text})

            except Exception as e:
                st.error(f"Error: {e}")
