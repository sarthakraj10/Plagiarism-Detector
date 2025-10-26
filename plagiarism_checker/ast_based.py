import ast
from .token_based import TokenBasedChecker

class ASTBasedChecker:
    def __init__(self):
        self.token_checker = TokenBasedChecker()
    
    def normalize_code(self, code):
        """Parses code into an AST and converts it back to a standardized string form"""
        try:
            tree = ast.parse(code)
            # Convert AST back to code to make formatting consistent and remove comments
            normalized_code = ast.unparse(tree)
            return normalized_code
        except:
            # If AST parsing fails, use token-based preprocessing as a fallback
            return self.token_checker.preprocess_code(code)
    
    def compare(self, code1, code2):
        """Checks how similar two code snippets are after AST-based normalization"""
        normalized1 = self.normalize_code(code1)
        normalized2 = self.normalize_code(code2)
        
        # Compare the normalized versions using the token-based method
        return self.token_checker.compare(normalized1, normalized2)
