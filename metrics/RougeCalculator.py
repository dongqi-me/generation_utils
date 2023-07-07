import json
import os
import spacy
import numpy as np
from datasets import load_metric

class RougeCalculator:
    def __init__(self):
        self.nlp = spacy.load('en_core_web_sm')
        self.rouge = load_metric('rouge')

    def add_newlines_to_sentences(self, text):
        doc = self.nlp(text)
        return '\n'.join([sent.text for sent in doc.sents])

    def compute_rouge(self, input_data):
        if not input_data:
            raise ValueError("Input data cannot be empty.")

        results = []
        for pair in input_data:
            try:
                reference = pair['reference'].strip()
                prediction = pair['prediction'].strip()
            except KeyError:
                raise ValueError("Each dictionary in input data should contain 'reference' and 'prediction' keys.")
                
            reference = self.add_newlines_to_sentences(reference)
            prediction = self.add_newlines_to_sentences(prediction)
            scores = self.rouge.compute(predictions=[prediction], references=[reference], use_stemmer=True)

            # Round the results to 4 decimal places
            scores = {k: round(v.mid.fmeasure * 100, 4) for k, v in scores.items()}
            results.append(scores)

        average_scores = {k: round(np.mean([res[k] for res in results]), 4) for k in results[0]}
        results.append({'corpus_level': average_scores})
        return results

    def compute_rouge_from_file(self, input_json_path, output_json_path):
        if not os.path.isfile(input_json_path):
            raise FileNotFoundError(f"{input_json_path} does not exist.")

        with open(input_json_path, 'r') as f:
            data = json.load(f)
        results = self.compute_rouge(data)

        with open(output_json_path, 'w') as f:
            json.dump(results, f, indent=4)

if __name__ == '__main__':
    rouge_calculator = RougeCalculator()

    input_data = [
        {"reference": "The cat sat on the mat. It was happy.", "prediction": "A cat was sitting on the mat. The cat seemed pleased."},
        {"reference": "The quick brown fox jumps over the lazy dog. The dog did not react.", "prediction": "A fast brown fox jumps above the lazy dog. The dog seemed indifferent."}
    ]

    input_json_path = 'input.json'
    with open(input_json_path, 'w') as f:
        json.dump(input_data, f)

    output_json_path = 'output.json'
    rouge_calculator.compute_rouge_from_file(input_json_path, output_json_path)
