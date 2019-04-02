import matplotlib
import numpy as np
from constants import *

def analyze_output(filename):
    dt = np.dtype([(LINE_COUNTER, int), (FILE_NAME, 'U32'), (PREDICTED_CLASS, 'U32'),\
        (HAM_SCORE, np.float64), (SPAM_SCORE, np.float64), (ACTUAL_CLASS, 'U32'), (CORRECT_CLASS, 'U10')])
    data = np.genfromtxt(filename, dtype=dt)
    tp = 0 
    fp = 0
    fn = 0
    tn = 0
    # in this case, positive means that a result has tested positive for being SPAM
    for result in data:
        if result[PREDICTED_CLASS] == SPAM:
            if result[CORRECT_CLASS] == 'right':
                # Tested positive for being spam
                tp += 1
            elif result[CORRECT_CLASS] == 'wrong':
                # False positive--incorrectly labeled spam
                fp += 1
        elif result[PREDICTED_CLASS] == HAM:
            if result[CORRECT_CLASS] == 'right':
                # Not spam and correctly labeled as such
                tn += 1
            elif result[CORRECT_CLASS] == 'wrong':
                # Labeled as ham but actually spam
                fn += 1
    print('TP: {} \nFP: {} \nFN: {} \nTN {} '.format(tp, fp, fn, tn))

def generate_confusion_matrix(filename):
    pass
        

# analyze_output('./baseline-result.txt')
# analyze_output('./stopword-result.txt)
# analyze_output('./wordlength-result.txt)