import re
from collections import Counter

try:
    import numpy as np
except ImportError:
    class np:
        @staticmethod
        def log(x):
            import math
            return math.log(x)

class SimpleEmbeddings:
    def __init__(self):
        self.vocabulary = {}
        self.idf = {}
    
    def _tokenize(self, text):
        return re.findall(r'\w+', text.lower())
    
    def _build_vocabulary(self, texts):
        all_words = set()
        for text in texts:
            all_words.update(self._tokenize(text))
        self.vocabulary = {word: idx for idx, word in enumerate(sorted(all_words))}
        
        doc_count = len(texts)
        word_doc_count = Counter()
        for text in texts:
            unique_words = set(self._tokenize(text))
            for word in unique_words:
                word_doc_count[word] += 1
        
        self.idf = {word: np.log(doc_count / (count + 1)) 
                   for word, count in word_doc_count.items()}
    
    def _vectorize(self, text):
        words = self._tokenize(text)
        vector = [0.0] * len(self.vocabulary)
        word_freq = Counter(words)
        
        for word, freq in word_freq.items():
            if word in self.vocabulary:
                idx = self.vocabulary[word]
                tf = freq / len(words) if words else 0
                idf = self.idf.get(word, 0)
                vector[idx] = tf * idf
        
        return vector
    
    def embed_documents(self, texts):
        self._build_vocabulary(texts)
        return [self._vectorize(text) for text in texts]
    
    def embed_query(self, text):
        return self._vectorize(text)
