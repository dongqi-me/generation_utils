import spacy

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

text = "Your sample text here."
doc = nlp(text)

# Subordination Index
subordinating_conjunctions = ["because", "although", "if", "while", ...]  # Include other subordinating conjunctions
subordinate_clauses = sum(1 for token in doc if token.dep_ == 'mark' and token.text in subordinating_conjunctions)
subordination_index = subordinate_clauses / len(list(doc.sents))

# Coordination Index
coordinating_conjunctions = ["and", "but", "or", "so", ...]  # Include other coordinating conjunctions
coordinations = sum(1 for token in doc if token.dep_ == 'cc' and token.text in coordinating_conjunctions)
coordination_index = coordinations / len(tokens)

# Average Number of Modifiers per Noun Phrase
modifiers_count = sum(len([child for child in noun.children if child.dep_ in ["amod", "compound"]]) for noun in doc if noun.pos_ == "NOUN")
noun_count = sum(1 for token in doc if token.pos_ == "NOUN")
average_modifiers_per_np = modifiers_count / noun_count

# Average Depth of Parse Tree
def get_depth(token):
    if not list(token.children):
        return 0
    return 1 + max(get_depth(child) for child in token.children)
depths = [get_depth(sent.root) for sent in doc.sents]
average_depth = sum(depths) / len(depths)

# Passive Constructions
passive_verb_tokens = [token for token in doc if "pass" in token.tag_]
passive_constructions = len(passive_verb_tokens)

print(f"Subordination Index: {subordination_index}")
print(f"Coordination Index: {coordination_index}")
print(f"Average Number of Modifiers per Noun Phrase: {average_modifiers_per_np}")
print(f"Average Depth of Parse Tree: {average_depth}")
print(f"Passive Constructions: {passive_constructions}")
