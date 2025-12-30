import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
import tempfile

class SimpleEnsembleRetriever:
    """Implementación simple de Ensemble Retriever que combina BM25 y FAISS"""
    def __init__(self, retrievers, weights):
        self.retrievers = retrievers
        self.weights = weights
    
    def invoke(self, query):
        """Combina resultados de múltiples retrievers con pesos"""
        all_docs = []
        
        for retriever, weight in zip(self.retrievers, self.weights):
            docs = retriever.invoke(query)
            # Agregar peso como metadata para scoring
            for doc in docs:
                doc.metadata['ensemble_score'] = weight
                all_docs.append(doc)
        
        # Eliminar duplicados manteniendo el de mayor score
        seen = {}
        for doc in all_docs:
            content = doc.page_content
            if content not in seen or doc.metadata.get('ensemble_score', 0) > seen[content].metadata.get('ensemble_score', 0):
                seen[content] = doc
        
        return list(seen.values())[:10]  # Retornar top 10

class RAGIntegrado:
    """
    Sistema RAG Unificado con dos modos:
    - BÁSICO: Solo FAISS + prompt simple (comparación con sistema original)
    - MEJORADO: BM25+FAISS híbrido + Multi-hop + Prompt forense
    """
    
    def __init__(self, api_key, chunk_size=1000, chunk_overlap=200):
        self.api_key = api_key
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Retrievers (se crearán al procesar documento)
        self.faiss_retriever = None  # Para modo básico
        self.ensemble_retriever = None  # Para modo mejorado
        self.splits = []  # Guardar splits con metadata de páginas
        
        # Embeddings multilingües
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        )
        
        # LLM configurado
        self.llm = ChatGroq(
            temperature=0,
            model_name="llama-3.1-8b-instant",  #llama-3.3-70b-versatile (modelo con limites)
            api_key=self.api_key
        )

    def process_documents(self, uploaded_files):
        """Procesa múltiples documentos creando AMBOS retrievers (básico y mejorado)"""
        
        all_docs = []
        
        # Procesar cada PDF
        for uploaded_file in uploaded_files:
            # Guardar archivo temporalmente para PyPDFLoader
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.read())
                tmp_path = tmp_file.name

            try:
                # Cargar PDF con metadata de páginas (LangChain lo hace automáticamente)
                loader = PyPDFLoader(tmp_path)
                docs = loader.load()
                
                # Agregar nombre del archivo fuente a la metadata
                for doc in docs:
                    doc.metadata['source_file'] = uploaded_file.name
                
                all_docs.extend(docs)
                
            finally:
                os.unlink(tmp_path)
        
        # Chunking con configuración dinámica sobre TODOS los documentos
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", ".", " ", ""]
        )
        self.splits = text_splitter.split_documents(all_docs)
        
        # CREAR RETRIEVER BÁSICO (solo FAISS)
        vectorstore = FAISS.from_documents(self.splits, self.embeddings)
        self.faiss_retriever = vectorstore.as_retriever(search_kwargs={"k": 6})
        
        # CREAR RETRIEVER MEJORADO (BM25 + FAISS)
        faiss_retriever_advanced = vectorstore.as_retriever(search_kwargs={"k": 5})
        bm25_retriever = BM25Retriever.from_documents(self.splits)
        bm25_retriever.k = 5
        
        self.ensemble_retriever = SimpleEnsembleRetriever(
            retrievers=[bm25_retriever, faiss_retriever_advanced],
            weights=[0.4, 0.6]
        )
        
        return len(self.splits)

    def get_answer(self, query, mode="basic"):
        """
        Genera respuesta usando el modo seleccionado
        mode='basic': FAISS simple + prompt básico
        mode='advanced': BM25+FAISS + Multi-hop + prompt forense
        """
        if not self.faiss_retriever:
            return "Por favor, procesa un documento primero."

        if mode == "advanced":
            return self._advanced_mode(query)
        else:
            return self._basic_mode(query)

    def _basic_mode(self, query):
        """MODO BÁSICO: Solo FAISS + Prompt Simple"""
        
        # Recuperar documentos con FAISS únicamente
        docs = self.faiss_retriever.invoke(query)
        
        if not docs:
            return "No hay suficiente información en el documento para responder esta pregunta."
        
        context_text = self._format_docs(docs)
        pages_used = self._extract_pages(docs)
        
        # PROMPT BÁSICO mejorado para preguntas generales
        template = """Basándote en el siguiente contexto del documento, responde la pregunta de manera clara y concisa.

INSTRUCCIONES:
- Si la pregunta es general (ej: "de qué trata", "tema principal", "resumen"), proporciona un resumen basado en el contexto disponible.
- Si la pregunta es específica y la información exacta no está en el contexto, indica que no se encontró esa información específica.
- Siempre basa tu respuesta en el contexto proporcionado.

Contexto:
{context}

Pregunta: {question}

Respuesta:"""

        prompt = ChatPromptTemplate.from_template(template)
        
        chain = (
            {"context": lambda x: context_text, "question": RunnablePassthrough()}
            | prompt
            | self.llm
            | StrOutputParser()
        )
        
        response = chain.invoke(query)
        
        # Agregar referencias de archivos y páginas
        if pages_used:
            sources_list = sorted(list(pages_used), key=lambda x: (x[0], x[1]))
            sources_str = ", ".join([f"{file} p.{page}" for file, page in sources_list])
            return f"{response}\n\n*Fuentes: {sources_str}*"
        
        return response

    def _advanced_mode(self, query):
        """MODO MEJORADO: BM25+FAISS + Multi-hop + Prompt Forense"""
        
        # Paso 1: Búsqueda inicial con retriever híbrido
        initial_docs = self.ensemble_retriever.invoke(query)
        initial_context = self._format_docs(initial_docs)
        pages_initial = self._extract_pages(initial_docs)
        
        # Paso 2: Expansión de conceptos (Multi-hop)
        extraction_prompt = f"""Analiza la pregunta y el contexto inicial. Identifica hasta 3 conceptos clave adicionales para investigar más a fondo.

Pregunta: {query}
Contexto: {initial_context[:1000]}...

Lista SOLO los conceptos separados por comas (máximo 3):"""
        
        try:
            concepts = self.llm.invoke(extraction_prompt).content
            additional_docs = []
            
            # Paso 3: Buscar información sobre cada concepto usando el retriever híbrido
            for concept in concepts.split(',')[:3]:
                concept = concept.strip()
                if concept:
                    additional_docs.extend(self.ensemble_retriever.invoke(concept))
            
            # Combinar y eliminar duplicados
            all_docs = initial_docs + additional_docs
            unique_docs = self._deduplicate_docs(all_docs)
            
            final_docs = unique_docs[:10]
            context_text = self._format_docs(final_docs)
            pages_all = self._extract_pages(final_docs)
            
        except Exception as e:
            # Fallback a búsqueda básica si falla la expansión
            context_text = initial_context
            pages_all = pages_initial
        
        # PROMPT FORENSE/ACADÉMICO mejorado
        template = """Eres un asistente académico experto especializado en análisis de documentos.

INSTRUCCIONES:
1. Para preguntas generales (ej: "de qué trata", "tema principal"): Proporciona un análisis comprehensivo basado en el contexto.
2. Para preguntas específicas: Conecta hechos dispersos usando razonamiento multi-hop.
3. Usa el contexto proporcionado y cita las páginas de origen.
4. Si conectas información de múltiples páginas, explica la conexión lógica.
5. Solo indica "información insuficiente" si realmente no hay datos relevantes para una pregunta específica.

CONTEXTO RECUPERADO:
{context}

PREGUNTA DEL USUARIO: {question}

RESPUESTA RAZONADA (incluye citas de páginas):"""

        prompt = ChatPromptTemplate.from_template(template)
        
        chain = (
            {"context": lambda x: context_text, "question": RunnablePassthrough()}
            | prompt
            | self.llm
            | StrOutputParser()
        )
        
        response = chain.invoke(query)
        
        # Agregar referencias de archivos y páginas al final
        if pages_all:
            sources_list = sorted(list(pages_all), key=lambda x: (x[0], x[1]))
            sources_str = ", ".join([f"{file} p.{page}" for file, page in sources_list])
            return f"{response}\n\n**Fuentes consultadas:** {sources_str}"
        
        return response

    def _deduplicate_docs(self, docs):
        """Elimina documentos duplicados por contenido"""
        seen = set()
        unique = []
        for doc in docs:
            if doc.page_content not in seen:
                unique.append(doc)
                seen.add(doc.page_content)
        return unique

    def _format_docs(self, docs):
        """Formatea documentos con información de archivo y página"""
        formatted = []
        for doc in docs:
            # PyPDFLoader guarda el número de página en metadata['page']
            page_num = doc.metadata.get('page', 0) + 1  # PyPDFLoader usa índice 0
            source_file = doc.metadata.get('source_file', 'documento')
            formatted.append(f"[{source_file} - PÁG. {page_num}]:\n{doc.page_content}")
        return "\n\n".join(formatted)

    def _extract_pages(self, docs):
        """Extrae información de archivo y página únicos de los documentos"""
        sources = set()
        for doc in docs:
            page_num = doc.metadata.get('page', 0) + 1
            source_file = doc.metadata.get('source_file', 'documento')
            sources.add((source_file, page_num))
        return sources