from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter
import tempfile
import os
from pypdf import PdfReader
import time
from embeddings import HuggingFaceEmbeddings
import faiss
import numpy as np

class RAGAvanzado:
    def __init__(self, api_key, chunk_size=500, chunk_overlap=50):
        self.api_key = api_key
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.chunks = []
        self.chunk_pages = []  # Páginas asociadas a cada chunk
        self.embeddings_list = []
        self.faiss_index = None
        self.llm = ChatGroq(
            api_key=api_key,
            model_name="llama-3.3-70b-versatile",
            temperature=0
        )
        self.embeddings = HuggingFaceEmbeddings()
    
    def process_document(self, uploaded_file):
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len
        )
        
        if uploaded_file.type == "application/pdf":
            # Extraer texto por página
            pages_text = self._extract_pdf_text(uploaded_file)
            # Crear chunks manteniendo track de páginas
            self.chunks, self.chunk_pages = self._create_chunks_with_pages(pages_text, text_splitter)
        else:
            # Para archivos de texto plano
            text = uploaded_file.read().decode('utf-8')
            self.chunks = text_splitter.split_text(text)
            self.chunk_pages = [None] * len(self.chunks)
        
        # Generar embeddings
        self.embeddings_list = self.embeddings.embed_documents(self.chunks)
        
        # Crear índice FAISS
        embeddings_array = np.array(self.embeddings_list).astype('float32')
        dimension = embeddings_array.shape[1]
        self.faiss_index = faiss.IndexFlatL2(dimension)
        self.faiss_index.add(embeddings_array)
    
    def _extract_pdf_text(self, pdf_file):
        """Extrae texto del PDF manteniendo track de páginas"""
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_file.write(pdf_file.read())
            tmp_path = tmp_file.name
        
        try:
            reader = PdfReader(tmp_path)
            pages_text = []  # Lista de (texto, número_página)
            
            for page_num, page in enumerate(reader.pages, start=1):
                page_text = page.extract_text()
                if page_text.strip():  # Solo agregar si hay texto
                    pages_text.append((page_text, page_num))
            
            return pages_text
        finally:
            os.unlink(tmp_path)
    
    def _create_chunks_with_pages(self, pages_text, text_splitter):
        """Crea chunks manteniendo la referencia a la página original"""
        all_chunks = []
        all_pages = []
        
        for page_text, page_num in pages_text:
            # Dividir el texto de esta página en chunks
            page_chunks = text_splitter.split_text(page_text)
            
            # Cada chunk de esta página se asocia con el número de página
            for chunk in page_chunks:
                all_chunks.append(chunk)
                all_pages.append(page_num)
        
        return all_chunks, all_pages
    
    def _similarity_search(self, query, k=3):
        """Búsqueda vectorial optimizada con FAISS"""
        query_embedding = self.embeddings.embed_query(query)
        query_vector = np.array([query_embedding]).astype('float32')
        
        # FAISS busca los k vecinos más cercanos (instantáneo)
        distances, indices = self.faiss_index.search(query_vector, k)
        
        # Retornar chunks con sus páginas
        results = []
        for idx in indices[0]:
            results.append({
                'content': self.chunks[idx],
                'page': self.chunk_pages[idx]
            })
        return results
    
    def query(self, question):
        if not self.chunks:
            return "Error: No hay documento procesado"
        
        try:
            docs_step1 = self._similarity_search(question, k=5)
            
            # Construir contexto del paso 1
            context_parts_step1 = []
            for doc in docs_step1:
                if doc['page']:
                    context_parts_step1.append(f"[Página {doc['page']}]: {doc['content']}")
                else:
                    context_parts_step1.append(doc['content'])
            context_step1 = "\n\n".join(context_parts_step1)
            
            extraction_prompt = f"""Analiza la siguiente pregunta y el contexto. Identifica conceptos clave que podrían estar relacionados indirectamente.

Pregunta: {question}

Contexto:
{context_step1}

Lista SOLO los conceptos clave relevantes (máximo 3), separados por comas:"""
            
            max_retries = 3
            key_concepts = None
            for attempt in range(max_retries):
                try:
                    time.sleep(1)
                    key_concepts = self.llm.invoke(extraction_prompt).content.strip()
                    break
                except Exception as e:
                    if attempt == max_retries - 1:
                        return self._basic_fallback(question, docs_step1)
                    time.sleep(2)
            
            additional_docs = []
            for concept in key_concepts.split(',')[:3]:
                concept = concept.strip()
                if concept:
                    docs = self._similarity_search(concept, k=2)
                    additional_docs.extend(docs)
            
            all_docs = docs_step1 + additional_docs
            
            # Construir contexto combinado con páginas
            context_parts = []
            pages_used = set()
            for doc in all_docs:
                if doc['page']:
                    context_parts.append(f"[Página {doc['page']}]: {doc['content']}")
                    pages_used.add(doc['page'])
                else:
                    context_parts.append(doc['content'])
            
            combined_context = "\n\n".join(context_parts)
            
            final_prompt = f"""Usando el siguiente contexto que puede contener información relacionada indirectamente, responde la pregunta.
Conecta la información de diferentes partes del contexto si es necesario.
Si no puedes encontrar suficiente información incluso después de analizar todo el contexto, responde: "No hay suficiente información en el documento para responder esta pregunta."

Contexto:
{combined_context}

Pregunta: {question}

Respuesta detallada (menciona las páginas consultadas):"""
            
            for attempt in range(max_retries):
                try:
                    time.sleep(1)
                    response = self.llm.invoke(final_prompt)
                    
                    # Agregar referencias de páginas al final
                    if pages_used:
                        pages_list = sorted(list(pages_used))
                        pages_str = ", ".join([f"p. {p}" for p in pages_list])
                        return f"{response.content}\n\n📄 *Fuentes: {pages_str}*"
                    
                    return response.content
                except Exception as e:
                    if attempt == max_retries - 1:
                        return "Error al procesar la consulta avanzada. Intenta con el modo básico o revisa tu conexión a internet."
                    time.sleep(2)
        
        except Exception as e:
            return f"Error en modo avanzado: {str(e)}. Intenta con el modo básico."
    
    def _basic_fallback(self, question, docs):
        # Construir contexto con páginas
        context_parts = []
        pages_used = set()
        for doc in docs:
            if doc['page']:
                context_parts.append(f"[Página {doc['page']}]: {doc['content']}")
                pages_used.add(doc['page'])
            else:
                context_parts.append(doc['content'])
        
        context = "\n\n".join(context_parts)
        
        prompt = f"""Basándote ÚNICAMENTE en el siguiente contexto, responde la pregunta.
Si la información no está en el contexto, responde: "No hay suficiente información en el documento para responder esta pregunta."

Contexto:
{context}

Pregunta: {question}

Respuesta (menciona las páginas consultadas):"""
        
        response = self.llm.invoke(prompt)
        
        # Agregar referencias de páginas al final
        if pages_used:
            pages_list = sorted(list(pages_used))
            pages_str = ", ".join([f"p. {p}" for p in pages_list])
            return f"{response.content}\n\n📄 *Fuentes: {pages_str}*"
        
        return response.content
