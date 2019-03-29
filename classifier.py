import os
import sys
from math import log
import numpy as np

INDEX = 'index'
WORD = 'word'
HAM_PROB = 'ham_prob'
S_HAM_PROB = 'smoothed_ham_prob'
SPAM_PROB = 'spam_prob'
S_SPAM_PROB = 'smoothed_spam_prob'

class Classifier:
    def __init__(self, model_file):
        self.data_type = np.dtype([(INDEX, int), (WORD, 'U10'), (HAM_PROB, np.float64), (S_HAM_PROB, np.float64),\
            (SPAM_PROB, np.float64), (S_SPAM_PROB, np.float64)])
        selfpass.model, self.word_index = self.process_model(model_file)
        self.prob_spam = 0
        self.prob_ham = 0

    def process_model(self, model_file):
        data = np.genfromtxt(model_file, dtype=self.data_type)
        # Dictionary mapping a word to its corresponding index in the model
        word_index_dict = dict([(v['word'], i) for i,v in enumerate(data)])
        return data, word_index_dict

    # Your basic, no-frills, naive bayes classifcation
    def classify(self, test_path, filter_func, output_file="baseline-result.txt"):
        for file in os.listdir(test_path):
            pass
            
    def score_spam(self, filename):
        pass
    
    def score_ham(self, filename):
        pass

    def compile_output_file(self, results):
        pass
