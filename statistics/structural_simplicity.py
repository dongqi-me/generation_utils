import spacy

# 加载spaCy模型
nlp = spacy.load("en_core_web_sm")

text = """This is the first sentence. And this is another one.
And yet another sentence. This is the start of a new paragraph. Another sentence here."""

# 使用spaCy处理文本
doc = nlp(text)

# Average Sentence Length (by words)
average_sentence_length_by_words = sum(len(sent) for sent in doc.sents) / len(list(doc.sents))

# Average Sentence Length (by characters)
average_sentence_length_by_chars = sum(len(sent.text) for sent in doc.sents) / len(list(doc.sents))

# Sentence Count
sentence_count = len(list(doc.sents))

# Paragraph Length
# Assuming paragraphs are separated by '\n\n'
paragraphs = text.split('\n\n')
average_paragraph_length = sum(len(nlp(para).sents) for para in paragraphs) / len(paragraphs)

print(f"Average Sentence Length (by words): {average_sentence_length_by_words}")
print(f"Average Sentence Length (by characters): {average_sentence_length_by_chars}")
print(f"Sentence Count: {sentence_count}")
print(f"Average Paragraph Length: {average_paragraph_length}")
