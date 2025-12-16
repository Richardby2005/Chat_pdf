import streamlit as st
import os
from dotenv import load_dotenv
from rag_integrado import RAGIntegrado

load_dotenv()

st.set_page_config(page_title="🧬 RAG Híbrido", page_icon="🧬", layout="wide")

if 'rag_system' not in st.session_state:
    st.session_state.rag_system = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

st.title("🧬 RAG Híbrido: BM25 + FAISS")
st.markdown("*Fusionando la precisión de búsqueda por keywords con embeddings multilingües*")

with st.sidebar:
    st.header("⚙️ Configuración")
    
    api_key = os.getenv("GROQ_API_KEY", "")
    if not api_key:
        api_key = st.text_input("Introduce tu Groq API Key", type="password")
    
    uploaded_file = st.file_uploader("📄 Subir documento (PDF)", type=['pdf'])
    
    st.subheader("Configuración de Chunks")
    chunk_size = st.slider("Tamaño del chunk", 200, 1000, 500, 25)
    chunk_overlap = st.slider("Superposición", 0, 300, 100, 25)
    
    st.subheader("Nivel de Análisis")
    search_mode = st.radio("Seleccionar modo:", 
                          ["Básico (Solo FAISS)", "Mejorado (Híbrido BM25+FAISS + Multi-hop)"],
                          help="**Básico**: Búsqueda vectorial simple + prompt estándar\n**Mejorado**: Búsqueda híbrida + expansión de conceptos + prompt forense")
    
    if st.button("🚀 Procesar Documento"):
        if not api_key:
            st.error("❌ Por favor configura tu API Key en el archivo .env o ingrésala arriba")
        elif not uploaded_file:
            st.error("❌ Por favor sube un documento PDF")
        else:
            with st.spinner("🔄 Indexando con BM25 y FAISS..."):
                try:
                    # Inicializar sistema RAG integrado
                    st.session_state.rag_system = RAGIntegrado(
                        api_key=api_key,
                        chunk_size=chunk_size,
                        chunk_overlap=chunk_overlap
                    )
                    
                    num_chunks = st.session_state.rag_system.process_document(uploaded_file)
                    
                    st.success(f"✅ Documento procesado exitosamente!")
                    st.info(f"📊 {num_chunks} fragmentos indexados con búsqueda híbrida")
                    st.session_state.chat_history = []
                except Exception as e:
                    st.error(f"❌ Error al procesar: {str(e)}")
    
    if st.button("🗑️ Limpiar Chat"):
        st.session_state.chat_history = []
        st.rerun()

st.divider()
st.header("💬 Chat Forense")

for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Haz tu consulta forense aquí..."):
    if st.session_state.rag_system is None:
        st.warning("⚠️ Por favor procesa un documento primero")
    else:
        # Agregar pregunta del usuario
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generar respuesta
        mode_key = "advanced" if "Mejorado" in search_mode else "basic"
        
        with st.chat_message("assistant"):
            with st.spinner("🕵️ Analizando documento..."):
                try:
                    response = st.session_state.rag_system.get_answer(prompt, mode=mode_key)
                    st.markdown(response)
                    st.session_state.chat_history.append({"role": "assistant", "content": response})
                except Exception as e:
                    st.error(f"❌ Error al generar respuesta: {str(e)}")
