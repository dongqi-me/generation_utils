import spacy

# 加载spaCy模型
nlp = spacy.load("en_core_web_sm")

text = "Your sample text here."

# 使用spaCy处理文本
doc = nlp(text)

# Type-Token Ratio (TTR)
types = set(token.text for token in doc)
tokens = [token.text for token in doc]
ttr = len(types) / len(tokens)

# Lexical Density
content_tags = ["NOUN", "VERB", "ADJ", "ADV"]  # 主要是名词、动词、形容词、副词
content_word_count = sum(1 for token in doc if token.pos_ in content_tags)
lexical_density = content_word_count / len(tokens)

print(f"Type-Token Ratio (TTR): {ttr}")
print(f"Lexical Density: {lexical_density}")
