import ast
from .token_based import TokenBasedChecker

class ASTBasedChecker:
    def __init__(self):
        self.token_checker = TokenBasedChecker()
    
    def normalize_code(self, code):
        """Normalize code by parsing to AST and converting back to string"""
        try:
            tree = ast.parse(code)
            # Normalize by converting AST back to code
            # This standardizes formatting and removes comments
            normalized_code = ast.unparse(tree)
            return normalized_code
        except:
            # If AST parsing fails, fall back to token-based preprocessing
            return self.token_checker.preprocess_code(code)
    
    def compare(self, code1, code2):
        """Compare two code snippets using AST normalization"""
        normalized1 = self.normalize_code(code1)
        normalized2 = self.normalize_code(code2)
        
        # Use token-based comparison on normalized code
        return self.token_checker.compare(normalized1, normalized2)