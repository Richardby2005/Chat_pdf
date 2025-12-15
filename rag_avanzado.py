from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter
import tempfile
import os
from pypdf import PdfReader
import time
from embeddings import SimpleEmbeddings

class RAGAvanzado:
    def __init__(self, api_key, chunk_size=500, chunk_overlap=50):
        self.api_key = api_key
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.chunks = []
        self.embeddings_list = []
        self.llm = ChatGroq(
            api_key=api_key,
            model_name="llama-3.3-70b-versatile",
            temperature=0
        )
        self.embeddings = SimpleEmbeddings()
    
    def process_document(self, uploaded_file):
        if uploaded_file.type == "application/pdf":
            text = self._extract_pdf_text(uploaded_file)
        else:
            text = uploaded_file.read().decode('utf-8')
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len
        )
        self.chunks = text_splitter.split_text(text)
        self.embeddings_list = self.embeddings.embed_documents(self.chunks)
    
    def _extract_pdf_text(self, pdf_file):
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_file.write(pdf_file.read())
            tmp_path = tmp_file.name
        
        try:
            reader = PdfReader(tmp_path)
            text = ""
            for page in reader.pages:
                text += page.extract_text() + "\n"
            return text
        finally:
            os.unlink(tmp_path)
    
    def _cosine_similarity(self, vec1, vec2):
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        mag1 = sum(a * a for a in vec1) ** 0.5
        mag2 = sum(b * b for b in vec2) ** 0.5
        if mag1 == 0 or mag2 == 0:
            return 0
        return dot_product / (mag1 * mag2)
    
    def _similarity_search(self, query, k=3):
        query_embedding = self.embeddings.embed_query(query)
        similarities = []
        
        for idx, chunk_embedding in enumerate(self.embeddings_list):
            sim = self._cosine_similarity(query_embedding, chunk_embedding)
            similarities.append((idx, sim))
        
        similarities.sort(key=lambda x: x[1], reverse=True)
        return [self.chunks[idx] for idx, _ in similarities[:k]]
    
    def query(self, question):
        if not self.chunks:
            return "Error: No hay documento procesado"
        
        try:
            docs_step1 = self._similarity_search(question, k=5)
            context_step1 = "\n\n".join(docs_step1)
            
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
            combined_context = "\n\n".join(all_docs)
            
            final_prompt = f"""Usando el siguiente contexto que puede contener información relacionada indirectamente, responde la pregunta.
Conecta la información de diferentes partes del contexto si es necesario.
Si no puedes encontrar suficiente información incluso después de analizar todo el contexto, responde: "No hay suficiente información en el documento para responder esta pregunta."

Contexto:
{combined_context}

Pregunta: {question}

Respuesta detallada:"""
            
            for attempt in range(max_retries):
                try:
                    time.sleep(1)
                    response = self.llm.invoke(final_prompt)
                    return response.content
                except Exception as e:
                    if attempt == max_retries - 1:
                        return "Error al procesar la consulta avanzada. Intenta con el modo básico o revisa tu conexión a internet."
                    time.sleep(2)
        
        except Exception as e:
            return f"Error en modo avanzado: {str(e)}. Intenta con el modo básico."
    
    def _basic_fallback(self, question, docs):
        context = "\n\n".join(docs)
        
        prompt = f"""Basándote ÚNICAMENTE en el siguiente contexto, responde la pregunta.
Si la información no está en el contexto, responde: "No hay suficiente información en el documento para responder esta pregunta."

Contexto:
{context}

Pregunta: {question}

Respuesta:"""
        
        response = self.llm.invoke(prompt)
        return response.content
