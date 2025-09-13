import ast
import tokenize
from io import StringIO
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re

class TokenBasedChecker:
    def __init__(self):
        self.vectorizer = TfidfVectorizer()
    
    def preprocess_code(self, code):
        """Remove comments and whitespace from code"""
        # Remove single-line comments
        code = re.sub(r'#.*', '', code)
        # Remove multi-line comments
        code = re.sub(r'\"\"\"[\s\S]*?\"\"\"', '', code)
        code = re.sub(r"\'\'\'[\s\S]*?\'\'\'", '', code)
        # Remove extra whitespace
        code = re.sub(r'\s+', ' ', code)
        return code.strip()
    
    def compare(self, code1, code2):
        """Compare two code snippets and return similarity score"""
        processed1 = self.preprocess_code(code1)
        processed2 = self.preprocess_code(code2)
        
        # Handle empty code after preprocessing
        if not processed1 or not processed2:
            return 0.0
        
        # Create TF-IDF vectors
        tfidf_matrix = self.vectorizer.fit_transform([processed1, processed2])
        
        # Calculate cosine similarity
        similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        
        return similarity