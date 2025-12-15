import streamlit as st
import os
from dotenv import load_dotenv
from rag_basico import RAGBasico
from rag_avanzado import RAGAvanzado

load_dotenv()

st.set_page_config(page_title="Chatbot RAG", page_icon="🤖", layout="wide")

if 'rag_system' not in st.session_state:
    st.session_state.rag_system = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'current_mode' not in st.session_state:
    st.session_state.current_mode = None

st.title("🤖 Chatbot RAG con Groq")

with st.sidebar:
    st.header("⚙️ Configuración")
    
    api_key = os.getenv("GROQ_API_KEY", "")
    
    uploaded_file = st.file_uploader("📄 Subir documento", 
                                    type=['txt', 'pdf'])
    
    st.subheader("Configuración de Chunks")
    chunk_size = st.slider("Tamaño del chunk", 100, 2000, 500, 50)
    chunk_overlap = st.slider("Superposición", 0, 500, 50, 10)
    
    st.subheader("Modo de Búsqueda")
    search_mode = st.radio("Seleccionar modo:", 
                          ["Básico", "Avanzado (Multi-hop)"])
    
    if st.session_state.current_mode and st.session_state.current_mode != search_mode:
        st.warning("⚠️ Has cambiado de modo. Debes volver a procesar el documento.")
        st.session_state.rag_system = None
    
    if st.button("🚀 Procesar Documento"):
        if not api_key:
            st.error("Por favor configura tu API Key en el archivo .env")
        elif not uploaded_file:
            st.error("Por favor sube un documento")
        else:
            with st.spinner("Procesando documento..."):
                try:
                    is_advanced = search_mode == "Avanzado (Multi-hop)"
                    if is_advanced:
                        st.session_state.rag_system = RAGAvanzado(
                            api_key=api_key,
                            chunk_size=chunk_size,
                            chunk_overlap=chunk_overlap
                        )
                    else:
                        st.session_state.rag_system = RAGBasico(
                            api_key=api_key,
                            chunk_size=chunk_size,
                            chunk_overlap=chunk_overlap
                        )
                    st.session_state.rag_system.process_document(uploaded_file)
                    st.session_state.current_mode = search_mode
                    st.success("✅ Documento procesado exitosamente!")
                    st.session_state.chat_history = []
                except Exception as e:
                    st.error(f"Error: {str(e)}")
    
    if st.button("🗑️ Limpiar Chat"):
        st.session_state.chat_history = []
        st.rerun()

st.header("💬 Chat")

for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Escribe tu pregunta aquí..."):
    if st.session_state.rag_system is None:
        st.error("⚠️ Por favor procesa un documento primero")
    else:
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        with st.chat_message("assistant"):
            with st.spinner("Pensando..."):
                response = st.session_state.rag_system.query(prompt)
                st.markdown(response)
        
        st.session_state.chat_history.append({"role": "assistant", "content": response})
