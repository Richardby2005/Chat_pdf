# Chatbot RAG con Groq

Sistema de Retrieval-Augmented Generation (RAG) con interfaz Streamlit, búsqueda híbrida BM25+FAISS y dos modos de operación.

## Características

- **Subida de documentos PDF**: Procesamiento inteligente con extracción de metadatos de páginas
- **Configuración flexible**: Ajusta tamaño de chunks (200-1000) y superposición (0-300)
- **Dos modos de búsqueda**:
  - **Básico**: FAISS vectorial puro + prompt simple (3 chunks)
  - **Mejorado**: BM25+FAISS híbrido (40%/60%) + Multi-hop reasoning + prompt forense (5-10 chunks)
- **LLM potente**: Llama 3.1-8b-instant vía Groq API
- **Embeddings multilingües**: HuggingFace Sentence Transformers (paraphrase-multilingual-MiniLM-L12-v2)
- **Citación de fuentes**: Referencias automáticas a páginas del documento original

## Stack Tecnológico

- **Framework**: LangChain (document loaders, text splitters, retrievers, chains)
- **UI**: Streamlit
- **Vector Store**: FAISS (Facebook AI Similarity Search)
- **Keyword Search**: BM25 (rank-bm25)
- **Embeddings**: HuggingFace Sentence Transformers (384-dimensional, multilingual)
- **LLM**: Groq API (Llama 3.1-8b-instant)
- **PDF Processing**: PyPDFLoader con extracción de metadata

## Requisitos

- Python 3.8+
- API Key de Groq (gratuita en https://console.groq.com/)
- ~2GB de espacio para modelos de embeddings

## Instalación

```bash
# Clonar repositorio
git clone https://github.com/Richardby2005/Chat_pdf.git
cd Chat_pdf

# Crear entorno virtual
python3 -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate

# Instalar dependencias
pip install -r requirements.txt

# Configurar API Key
echo "GROQ_API_KEY=tu_api_key_aqui" > .env

# Ejecutar aplicación
streamlit run app.py
```

## Uso

1. **Cargar documento**: Sube un archivo PDF en la barra lateral
2. **Configurar parámetros**:
   - Chunk Size: Tamaño de cada fragmento de texto (recomendado: 800)
   - Chunk Overlap: Superposición entre chunks (recomendado: 100)
3. **Seleccionar modo**:
   - **Básico**: Para preguntas directas con información localizada
   - **Mejorado**: Para preguntas que requieren conectar información dispersa
4. **Preguntar**: Escribe tu consulta en el chat

## Ejemplo de Multi-hop Reasoning

**Documento:**
- Página 2: "El Proyecto X utiliza la sustancia Z en su composición."
- Página 7: "La sustancia Z es altamente volátil y explota a 150°C."

**Pregunta:** "¿A qué temperatura explota el Proyecto X?"

**Modo Básico:** "No hay suficiente información en el documento para responder esta pregunta."

**Modo Mejorado:** "El Proyecto X explota a 150°C. Esto se debe a que utiliza la sustancia Z en su composición (según la página 2), y la sustancia Z es altamente volátil, explotando a esa temperatura específica (según la página 7)."

**Fuentes consultadas:** p. 2, p. 7

## Arquitectura

```
Usuario → Streamlit UI (app.py)
           ↓
       RAGIntegrado (rag_integrado.py)
           ↓
    ┌──────┴──────┐
    │             │
  BÁSICO      MEJORADO
    │             │
FAISS (k=3)   SimpleEnsembleRetriever
              ↓
         BM25 (40%) + FAISS (60%)
              ↓
         Multi-hop Reasoning
              ↓
         Prompt Forense
```

## Estructura del Proyecto

```
Chat_pdf/
├── app.py                 # Interfaz Streamlit
├── rag_integrado.py       # Sistema RAG unificado con ambos modos
├── requirements.txt       # Dependencias del proyecto
├── .env                   # API Key (no incluido en repo)
├── .gitignore            # Exclusiones de git
└── README.md             # Este archivo
```

## Componentes Clave

- **SimpleEnsembleRetriever**: Combina retrievers BM25 y FAISS con ponderación personalizada
- **_format_docs()**: Agrega números de página a cada fragmento usando metadata de LangChain
- **_advanced_mode()**: Implementa razonamiento multi-hop con expansión de conceptos
- **_basic_mode()**: Búsqueda vectorial simple con prompt directo

## Limitaciones

- **Modelo LLM**: Llama 3.1-8b-instant tiene límite de 100K tokens/día en tier gratuito
- **Idioma optimizado**: Embeddings multilingües funcionan mejor con español e inglés
- **Tipo de archivo**: Solo PDF soportado actualmente

## Licencia

MIT
