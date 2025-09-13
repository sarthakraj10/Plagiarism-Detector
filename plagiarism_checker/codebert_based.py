from transformers import AutoTokenizer, AutoModel
import torch
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

class CodeBERTChecker:
    def __init__(self):
        self.model_name = "microsoft/codebert-base"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name)
    
    def get_embeddings(self, code):
        """Get CodeBERT embeddings for code snippet"""
        inputs = self.tokenizer(
            code, 
            return_tensors="pt", 
            truncation=True, 
            padding=True, 
            max_length=512
        )
        
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Use the [CLS] token representation as the embedding
        embeddings = outputs.last_hidden_state[:, 0, :].numpy()
        return embeddings
    
    def compare(self, code1, code2):
        """Compare two code snippets using CodeBERT embeddings"""
        emb1 = self.get_embeddings(code1)
        emb2 = self.get_embeddings(code2)
        
        # Calculate cosine similarity
        similarity = cosine_similarity(emb1, emb2)[0][0]
        return similarity