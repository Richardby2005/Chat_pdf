from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter
import tempfile
import os
from pypdf import PdfReader
from embeddings import SimpleEmbeddings

class RAGBasico:
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
        
        docs = self._similarity_search(question, k=3)
        
        if not docs:
            return "No hay suficiente información en el documento para responder esta pregunta."
        
        context = "\n\n".join(docs)
        
        prompt = f"""Basándote ÚNICAMENTE en el siguiente contexto, responde la pregunta.
Si la información no está en el contexto, responde: "No hay suficiente información en el documento para responder esta pregunta."

Contexto:
{context}

Pregunta: {question}

Respuesta:"""
        
        response = self.llm.invoke(prompt)
        return response.content
