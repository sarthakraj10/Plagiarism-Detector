import ast
from collections import Counter

class ASTBasedChecker:
    def __init__(self):
        pass

    def ast_to_features(self, node):
        """Convert AST to feature list (node types)"""
        features = []
        for n in ast.walk(node):
            features.append(type(n).__name__)
        return features

    def compare(self, code1, code2):
        try:
            tree1 = ast.parse(code1)
            tree2 = ast.parse(code2)
        except:
            # If parsing fails, treat as completely different
            return 0.0

        features1 = Counter(self.ast_to_features(tree1))
        features2 = Counter(self.ast_to_features(tree2))

        # Cosine similarity on AST node type counts
        all_nodes = set(features1.keys()) | set(features2.keys())
        vec1 = [features1.get(n, 0) for n in all_nodes]
        vec2 = [features2.get(n, 0) for n in all_nodes]

        # Cosine similarity formula
        dot = sum(a*b for a, b in zip(vec1, vec2))
        norm1 = sum(a*a for a in vec1) ** 0.5
        norm2 = sum(b*b for b in vec2) ** 0.5
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return dot / (norm1 * norm2)
