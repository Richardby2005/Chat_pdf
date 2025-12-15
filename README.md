# 🤖 Chatbot RAG con Groq

Sistema de Retrieval-Augmented Generation (RAG) con interfaz Streamlit y dos modos de búsqueda.

## 🌟 Características

- **Subida de documentos**: Soporta archivos .txt y .pdf
- **Configuración flexible**: Ajusta tamaño de chunks y superposición
- **Dos modos de búsqueda**:
  - **Básico** (`rag_basico.py`): Respuestas directas del documento
  - **Avanzado** (`rag_avanzado.py`): Conecta información de múltiples partes del documento (Multi-hop)
- **IA potente**: Usa modelos de Groq (Llama 3.3)
- **Embeddings TF-IDF**: Sin dependencias pesadas

## 📋 Requisitos

- Python 3.8+
- API Key de Groq (gratuita en https://console.groq.com/)

## 🚀 Instalación

```bash
# Crear entorno virtual
python3 -m venv venv
source venv/bin/activate

# Instalar dependencias
pip install -r requirements.txt

# Configurar API Key
echo "GROQ_API_KEY=tu_api_key_aqui" > .env

# Ejecutar
streamlit run app.py
```

## 📖 Ejemplo de Uso Multi-hop

**Documento:**
- Párrafo 1: "El Proyecto X utiliza la sustancia Z en su composición."
- Párrafo 5: "La sustancia Z es altamente volátil y explota a 150°C."

**Pregunta:** "¿A qué temperatura explota el Proyecto X?"

**Modo Básico:** "No hay suficiente información..."

**Modo Avanzado:** "El Proyecto X explota a 150°C, ya que utiliza la sustancia Z, la cual es volátil y explota a esa temperatura."

## 📁 Estructura

- `app.py` - Interfaz Streamlit
- `rag_basico.py` - Sistema RAG básico
- `rag_avanzado.py` - Sistema RAG con multi-hop reasoning
- `requirements.txt` - Dependencias
- `.env` - Configuración de API Key

## 📝 Licencia

MIT
# Chat_pdf
# Chat_pdf
# Chat_pdf
