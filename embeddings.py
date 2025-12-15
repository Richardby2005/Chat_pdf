from sentence_transformers import SentenceTransformer

class HuggingFaceEmbeddings:
    """
    Embeddings profesionales usando modelo multilingüe de HuggingFace.
    Modelo: paraphrase-multilingual-MiniLM-L12-v2
    - Soporta más de 50 idiomas
    - Embeddings de 384 dimensiones
    - Mucho mejor calidad que TF-IDF
    """
    def __init__(self, model_name="paraphrase-multilingual-MiniLM-L12-v2"):
        self.model = SentenceTransformer(model_name)
    
    def embed_documents(self, texts):
        """Genera embeddings para una lista de documentos"""
        embeddings = self.model.encode(texts, convert_to_tensor=False, show_progress_bar=False)
        return embeddings.tolist()
    
    def embed_query(self, text):
        """Genera embedding para una consulta"""
        embedding = self.model.encode(text, convert_to_tensor=False, show_progress_bar=False)
        return embedding.tolist()
