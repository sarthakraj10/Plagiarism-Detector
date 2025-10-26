import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class TokenBasedChecker:
    def __init__(self):
        self.vectorizer = TfidfVectorizer()

    def preprocess_code(self, code):
        # Remove comments
        code = re.sub(r'#.*', '', code)
        code = re.sub(r'\"\"\"[\s\S]*?\"\"\"', '', code)
        code = re.sub(r"\'\'\'[\s\S]*?\'\'\'", '', code)
        # Collapse whitespace
        code = re.sub(r'\s+', ' ', code)
        return code.strip()

    def compare(self, code1, code2):
        processed1 = self.preprocess_code(code1)
        processed2 = self.preprocess_code(code2)

        if not processed1 or not processed2:
            return 0.0

        tfidf_matrix = self.vectorizer.fit_transform([processed1, processed2])
        similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        return similarity
