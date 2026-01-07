import streamlit as st
import os
from dotenv import load_dotenv
from rag_integrado import RAGIntegrado

load_dotenv()

st.set_page_config(page_title="RAG Híbrido", page_icon="�", layout="wide")

# CSS personalizado para el diseño del chat
st.markdown("""
<style>
            
    /* Estilo para mensajes del usuario */
    .stChatMessage[data-testid="user-message"] {
        background-color: #5B8FD8 !important;
        border-radius: 20px;
        padding: 15px 20px;
        margin: 10px 0;
    }
    
    .stChatMessage[data-testid="user-message"] p {
        color: white !important;
        font-size: 15px;
    }
    
    /* Estilo para mensajes del asistente */
    .stChatMessage[data-testid="assistant-message"] {
        background-color: #ffffff !important;
        border-radius: 15px;
        padding: 15px 20px;
        margin: 10px 0;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    }
    
    .stChatMessage[data-testid="assistant-message"] p {
        color: #1f1f1f !important;
        font-size: 15px;
    }
    
    /* Estilo del input de chat */
    .stChatInputContainer {
        background-color: #ffffff;
        border-radius: 25px;
        padding: 5px;
    }
    
    /* Sidebar con degradado azul marino a negro */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a3a52 0%, #0d1b2a 100%) !important;
    }
    
    section[data-testid="stSidebar"] * {
        color: #ecf0f1 !important;
    }
    
    /* Botones del sidebar */
    section[data-testid="stSidebar"] button {
        background-color: #2c5f7f !important;
        color: white !important;
        border-radius: 8px;
    }
    
    section[data-testid="stSidebar"] button:hover {
        background-color: #3d7a9e !important;
    }
    
    /* Header del chat */
    h2 {
        color: #2c3e50 !important;
    }
</style>
""", unsafe_allow_html=True)

if 'rag_system' not in st.session_state:
    st.session_state.rag_system = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'last_file_ids' not in st.session_state:
    st.session_state.last_file_ids = []
if 'last_mode' not in st.session_state:
    st.session_state.last_mode = None

st.title("ChatPDF utilizando RAG")
st.divider()

with st.sidebar:
    st.header("Configuración")
    
    api_key = os.getenv("GROQ_API_KEY", "")
    if not api_key:
        api_key = st.text_input("Introduce tu Groq API Key", type="password")
    
    uploaded_files = st.file_uploader("Subir documento(s) (PDF)", type=['pdf'], accept_multiple_files=True)
    
    st.subheader("Modo de Análisis")
    
    # Radio buttons con descripciones integradas
    analysis_mode = st.radio(
        "Seleccionar modo:",
        ["Rápido", "Académico", "Exhaustivo"],
        captions=[
            "Chunks pequeños (400 chars) • Búsqueda básica",
            "Chunks medianos (800 chars) • Búsqueda híbrida + Multi-hop",
            "Chunks grandes (1200 chars) • Búsqueda híbrida + Multi-hop"
        ]
    )
    
    # Configuración de chunks según el modo seleccionado
    mode_configs = {
        "Rápido": {"chunk_size": 400, "chunk_overlap": 50, "search_mode": "basic"},
        "Académico": {"chunk_size": 800, "chunk_overlap": 150, "search_mode": "advanced"},
        "Exhaustivo": {"chunk_size": 1200, "chunk_overlap": 250, "search_mode": "advanced"}
    }
    
    current_config = mode_configs[analysis_mode]
    chunk_size = current_config["chunk_size"]
    chunk_overlap = current_config["chunk_overlap"]
    search_mode = current_config["search_mode"]
    
    # Procesar automáticamente cuando cambian los archivos o el modo
    if uploaded_files:
        file_ids = [f.file_id for f in uploaded_files]
        
        # Detectar si cambiaron los archivos o el modo
        if (file_ids != st.session_state.last_file_ids or 
            analysis_mode != st.session_state.last_mode):
            
            if not api_key:
                st.error("Por favor configura tu API Key en el archivo .env o ingrésala arriba")
            else:
                with st.spinner(f"Indexando {len(uploaded_files)} documento(s) con BM25 y FAISS..."):
                    try:
                        # Inicializar sistema RAG integrado
                        st.session_state.rag_system = RAGIntegrado(
                            api_key=api_key,
                            chunk_size=chunk_size,
                            chunk_overlap=chunk_overlap
                        )
                        
                        num_chunks = st.session_state.rag_system.process_documents(uploaded_files)
                        
                        # Actualizar estados
                        st.session_state.last_file_ids = file_ids
                        st.session_state.last_mode = analysis_mode
                        st.session_state.chat_history = []
                        
                        st.success(f"{len(uploaded_files)} documento(s) procesado(s) exitosamente!")
                        st.info(f"{num_chunks} fragmentos indexados con búsqueda híbrida")
                    except Exception as e:
                        st.error(f"Error al procesar: {str(e)}")
    
    if st.button("Limpiar Chat"):
        st.session_state.chat_history = []
        st.rerun()

for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        if message["role"] == "assistant":
            st.markdown(f"**DocuQ&A:** {message['content']}")
        else:
            st.markdown(message["content"])

if prompt := st.chat_input("Escribe tu pregunta aquí..."):
    if st.session_state.rag_system is None:
        st.warning("Por favor procesa un documento primero")
    else:
        # Agregar pregunta del usuario
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generar respuesta usando el modo configurado
        with st.chat_message("assistant"):
            with st.spinner("Analizando documento..."):
                try:
                    response = st.session_state.rag_system.get_answer(prompt, mode=search_mode)
                    st.markdown(f"**DocuQ&A:** {response}")
                    st.session_state.chat_history.append({"role": "assistant", "content": response})
                except Exception as e:
                    st.error(f"Error al generar respuesta: {str(e)}")
