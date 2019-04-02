import matplotlib.pyplot as plt
import numpy as np
from constants import *

def analyze_output(filename):
    dt = np.dtype([(LINE_COUNTER, int), (FILE_NAME, 'U32'), (PREDICTED_CLASS, 'U10'),\
        (HAM_SCORE, np.float64), (SPAM_SCORE, np.float64), (ACTUAL_CLASS, 'U10'), (CORRECT_CLASS, 'U10')])
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
    return tp, fp, fn, tn

def generate_confusion_matrix(file_to_analyze, title='Confusion Matrix'):
    tp, fp, fn, tn = analyze_output(file_to_analyze)

    data = np.asarray([[tp, fp], [fn, tn]])
    cols = ('Predicted: SPAM', 'Predicted: HAM')
    rows = ['Actual: SPAM', 'Actual: HAM']

    fig, ax = plt.subplots()
    im = ax.imshow(data, interpolation='nearest', cmap=plt.cm.Greens)
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(data.shape[1]),
           yticks=np.arange(data.shape[0]),
           xticklabels=[SPAM, HAM], yticklabels=[SPAM, HAM],
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")
    fmt = 'd'
    threshold = data.max() / 2.
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            ax.text(j, i, format(data[i, j], fmt),
                    ha="center", va="center", fontsize=22,
                    color="white" if data[i, j] > threshold else "black")
    fig.tight_layout()
    plt.show()

# generate_confusion_matrix('./baseline-result.txt', 'Confusion Matrix')
# generate_confusion_matrix('./stopword-result.txt', 'Stop-Word Confusion Matrix')
# generate_confusion_matrix('./wordlength-result.txt', 'Word Length Confusion Matrix')