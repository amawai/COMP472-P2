import re
import os
from constants import *

def no_filter(word):
    return word != ''

def stop_word_filter(word, stop_words):
    return word not in stop_words

def word_len_filter(word):
    return len(word) > 2 and len(word) < 9

# Tokenizes file, outputs array of valid words based on passed-in filter
def generate_tokens(filename, filter_func, filter_words=None):
    with open(filename, 'r', encoding='latin-1') as f:
        file_content = f.readlines()
    if filter_words:
        vocab_list = [[word for word in re.split('[^a-zA-Z]', line.lower().strip()) if filter_func(word, filter_words)] for line in file_content]
    else:
        vocab_list = [[word for word in re.split('[^a-zA-Z]', line.lower().strip()) if filter_func(word)] for line in file_content]
    flattened = [token for sublist in vocab_list for token in sublist]
    return flattened

def get_label_from_file(filename):
    return re.search(r'[test | train]-\s*(\S+)-', filename).group(1)

def get_tokens(training_dir, ham_or_spam, filter_func, filter_words=None):
    token_list = []
    for f in os.listdir(training_dir):
        file_path = os.path.join(training_dir, f)
        if get_label_from_file(file_path) == ham_or_spam:
            token_list += generate_tokens(file_path, filter_func, filter_words)
    return token_list

def get_stop_words(filename):
    with open(filename, 'r') as f:
        file_content = f.readlines()
    return list(map(lambda x: x.lower().strip(), file_content))

def frequency(token_list):
    dict = {}
    for word in token_list:
        if word in dict:
            dict[word] += 1
        else:
            dict[word] = 1
    return dict


def calculate_conditional(token_freq, token_total, vocab_size):
    return (token_freq + 0.5) / (token_total + (0.5 * vocab_size))

# Returns all unique keys of both classes
def get_all_keys(ham_freq, spam_freq):
    return set().union(ham_freq, spam_freq)


def build_model(ham_tokens, spam_tokens, filename):
    f = open(filename, 'w+')
    line_counter = 1

    # If word doesn't exist in one of the dicts, set value to 0
    ham_freq = frequency(ham_tokens)
    spam_freq = frequency(spam_tokens)
    all_keys = get_all_keys(ham_freq, spam_freq)
    for word in all_keys:
        if word not in ham_freq:
            ham_freq[word] = 0
        if word not in spam_freq:
            spam_freq[word] = 0

    # For cond. prob calculations
    ham_token_total = len(ham_tokens)
    spam_token_total = len(spam_tokens)
    ham_vocab_size = len(ham_freq.items())
    spam_vocab_size = len(spam_freq.items())

    for k, v in sorted(ham_freq.items()):
        f.write('  '.join([str(line_counter), k, str(v), str(calculate_conditional(v, ham_token_total, ham_vocab_size)),\
            str(spam_freq[k]), str(calculate_conditional(spam_freq[k], spam_token_total, spam_vocab_size))]))
        f.write('\n')
        line_counter += 1
    f.close()

def build_model_wrapper(training_dir, output_file, filter_func, filter_words=None):
    ham_tokens = get_tokens(training_dir, HAM, filter_func, filter_words)
    spam_tokens = get_tokens(training_dir, SPAM, filter_func, filter_words)
    build_model(ham_tokens, spam_tokens, output_file)

