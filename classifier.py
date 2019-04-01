import os
import re
from math import log10
import numpy as np
from tokenizer import generate_tokens, no_filter
from constants import *


class Classifier:
    def __init__(self, training_dir, model_file):
        self.data_type = np.dtype([(INDEX, int), (WORD, 'U10'), (HAM_PROB, np.float64), (S_HAM_PROB, np.float64),
                                  (SPAM_PROB, np.float64), (S_SPAM_PROB, np.float64)])
        self.model, self.word_index = self.process_model(model_file)
        self.prob_spam, self.prob_ham = self.compute_probs(training_dir)

    def process_model(self, model_file):
        data = np.genfromtxt(model_file, dtype=self.data_type)
        # Dictionary mapping a word to its corresponding index in the model
        word_index_dict = dict([(v[WORD], i) for i, v in enumerate(data)])
        return data, word_index_dict

    def compute_probs(self, training_dir):
        spam_count = 0
        ham_count = 0
        for file in os.listdir(training_dir):
            label = self.get_label_from_file(file)
            if label == SPAM:
                spam_count += 1
            elif label == HAM:
                ham_count += 1
        total = spam_count + ham_count
        return spam_count / total, ham_count / total

    def classify(self, test_path, filter_func, output_file="baseline-result.txt"):
        line_counter = 1
        results = []
        for f in os.listdir(test_path):
            tokens = generate_tokens(os.path.join(test_path, f), filter_func)
            spam_score = self.score_spam(tokens)
            ham_score = self.score_ham(tokens)
            test_class = SPAM if spam_score > ham_score else HAM
            correct_class = self.get_label_from_file(f)
            results.append(self.compile_output_line(line_counter, f, test_class, ham_score, spam_score, correct_class))
            line_counter += 1
        with open(output_file, 'w') as f_to_write:
            f_to_write.write('\n'.join(results))
        return True

    def get_label_from_file(self, filename):
        return re.search(r'[test | train]-\s*(\S+)-', filename).group(1)

    def score_spam(self, tokens):
        return log10(self.prob_spam) + sum([self.get_spam_prob(token) for token in tokens])

    def score_ham(self, tokens):
        return log10(self.prob_ham) + sum([self.get_ham_prob(token) for token in tokens])

    def get_spam_prob(self, word):
        if word in self.word_index:
            return log10(self.model[self.word_index[word]][S_SPAM_PROB])
        return 0

    def get_ham_prob(self, word):
        if word in self.word_index:
            return log10(self.model[self.word_index[word]][S_HAM_PROB])
        return 0

    def compile_output_line(self, line_counter, test_file, test_class, ham_score, spam_score, correct_class):
        right_or_wrong = 'right' if test_class == correct_class else 'wrong'
        return '  '.join([str(line_counter), test_file, test_class, str(ham_score), str(spam_score), correct_class, right_or_wrong])

# Example usage
# classifier = Classifier('./Project2-Train/train', './model.txt')
# classifier.classify('./Project2-Test/test', no_filter)
