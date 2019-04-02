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

    def classify(self, test_path, filter_func, output_file="baseline-result-2.txt"):
        line_counter = 1
        ham_correct_count = 0
        spam_correct_count = 0
        ham_test_count = 0
        spam_test_count = 0
        ham_precision_recall_count = 0
        spam_precision_recall_count = 0
        right_total = 0
        wrong_total = 0

        results = []
        for f in os.listdir(test_path):
            tokens = generate_tokens(os.path.join(test_path, f), filter_func)
            spam_score = self.score_spam(tokens)
            ham_score = self.score_ham(tokens)

            # Conditionals to calculate precision and recall probabilities
            test_class = SPAM if spam_score > ham_score else HAM
            if test_class == HAM:
                ham_test_count += 1
            elif test_class == SPAM:
                spam_test_count += 1

            correct_class = self.get_label_from_file(f)
            if correct_class == HAM:
                ham_correct_count += 1
            elif correct_class == SPAM:
                spam_correct_count += 1

            right_or_wrong = 'right' if test_class == correct_class else 'wrong'
            if right_or_wrong == 'right' and test_class == HAM:
                ham_precision_recall_count += 1
            elif right_or_wrong == 'right' and test_class == SPAM:
                spam_precision_recall_count += 1

            if right_or_wrong == 'right':
                right_total += 1
            elif right_or_wrong == 'wrong':
                wrong_total += 1

            results.append(self.compile_output_line(line_counter, f, test_class, ham_score, spam_score, correct_class))
            line_counter += 1
        with open(output_file, 'w') as f_to_write:
            f_to_write.write('\n'.join(results))

        print("ham_correct_count: " + str(ham_correct_count)
                + "\nspam_correct_count: " + str(spam_correct_count)
                + "\nham_test_count: " + str(ham_test_count)
                + "\nspam_test_count: " + str(spam_test_count)
                + "\nham_precision_recall_count: " + str(ham_precision_recall_count)
                + "\nspam_precision_recall_count: " + str(spam_precision_recall_count))

        accuracy = right_total / (right_total + wrong_total)

        ham_precision = ham_precision_recall_count / ham_test_count
        spam_precision = spam_precision_recall_count / spam_test_count

        ham_recall = ham_precision_recall_count / ham_correct_count
        spam_recall = spam_precision_recall_count / spam_correct_count

        ham_f1_measure = (2 * ham_precision * ham_recall) / (ham_precision + ham_recall)
        spam_f1_measure = (2 * spam_precision * spam_recall) / (spam_precision + spam_recall)

        print("accuracy: " + str(accuracy)
                + "\nham_precision: " + str(ham_precision)
                + "\nspam_precision: " + str(spam_precision)
                + "\nham_recall: " + str(ham_recall)
                + "\nspam_recall: " + str(spam_recall)
                + "\nham_f1_measure: " + str(ham_f1_measure)
                + "\nspam_f1_measure: " + str(spam_f1_measure))

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
# classifier = Classifier('./train', './model.txt')
# classifier.classify('./test', no_filter)
