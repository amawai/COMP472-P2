# C-O-M-P-4-7-2: Naive Bayes Classification 
ダブルカード C ○ M P 4 7 2 ●

This is written in Python 3 and makes use of the following libraries **NumPy, math, re, sys
and Matplotlib.**

![naive_bayes](https://imgur.com/EpTaPNK.png)

## Developers

| Name          | GitHub Handle | 
| ------------------ | ------------- |
| Amanda Wai |  _amawai_ |
| Chen Jie Lu | _SunXP_ |
| Wai Lun Lau | _Wai-Lau_ |
| Jad Malek | _jadmalek_ |

### How to build model
In lines 104 and 105 in the `tokenizer.py` file, enter either `no_filter` (baseline experiment), `stop_filter` (stop-word filtering experiment) or `word_len_filter` (word-length filtering experiment) as the filter parameter in both the `get_spam_tokens` and `get_ham_tokens` function calls. Thereafter, run:

`python3 tokenizer.py`

### How to run the classifier
In lines 74 and 75 in the `classifier.py` file, enter `baseline-model.txt`, `no_filter` and `baseline-result.txt`(baseline experiment), `stopword-model.txt`, `no_filter` and `stopword-result.txt`(stop-word filtering experiment), or `wordlength-model.txt`, `word_len_filter` and `wordlength-result.txt`(stop-word filtering experiment) as indicated by the example usage in said lines. Thereafter, run:

`python3 classifier.py`

### How to evaluate the classifier
Uncomment any of the commented out function calls in the last lines of the file, depending on which experiment you are trying to carry out and are interested in evaluating, and run:

`python3 classifier_analysis.py`
