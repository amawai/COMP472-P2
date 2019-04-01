import re
import glob
import math


path_ham = "./train_ham/*.txt"
path_spam = "./train_spam/*.txt"

ham_files = glob.glob(path_ham)
spam_files = glob.glob(path_spam)


def no_filter(word):
    return word != ''


def stop_word_filter(word):
    pass


def word_len_filter(word):
    pass


# Tokenizes file, outputs array of valid words based on passed-in filter
def generate_tokens(filename, filter_func=no_filter):
    with open(filename, 'r', encoding='latin-1') as f:
        file_content = f.readlines()
    vocab_list = [[word for word in re.split('[^a-zA-Z]', line.lower().strip()) if filter_func(word)] for line in file_content]
    flattened = [token for sublist in vocab_list for token in sublist]
    return flattened


def get_ham_tokens(filter_func=no_filter):
    ham_list = []
    for ham in ham_files:
        with open(ham, 'r', encoding='latin-1') as f:
            file_content = f.readlines()
        vocab_list = [[word for word in re.split('[^a-zA-Z]', line.lower().strip()) if filter_func(word)] for line in file_content]
        flattened = [token for sublist in vocab_list for token in sublist]
        ham_list.extend(flattened)
    return ham_list


def get_spam_tokens(filter_func=no_filter):
    spam_list = []
    for spam in spam_files:
        with open(spam, 'r', encoding='latin-1') as f:
            file_content = f.readlines()
        vocab_list = [[word for word in re.split('[^a-zA-Z]', line.lower().strip()) if filter_func(word)] for line in file_content]
        flattened = [token for sublist in vocab_list for token in sublist]
        spam_list.extend(flattened)
    return spam_list


def frequency(token_list):
    dict = {}
    for word in token_list:
        if word in dict:
            dict[word] += 1
        else:
            dict[word] = 1
    return dict


def calculate_conditional(token_freq, token_total, vocab_size):
    return math.log10((token_freq + 0.5) / (token_total + (0.5 * vocab_size)))


# returns all unique keys of both classes
def get_all_keys(ham_freq, spam_freq):
    return set().union(ham_freq, spam_freq)


def build_model(ham_freq, spam_freq):
    f = open('model.txt', 'w+')
    line_counter = 1

    # if word doesn't exist in one of the dicts, set value to 0
    all_keys = get_all_keys(ham_freq, spam_freq)
    for word in all_keys:
        if word not in ham_freq:
            ham_freq[word] = 0
        if word not in spam_freq:
            spam_freq[word] = 0

    # for cond. prob calculations
    ham_token_total = len(get_ham_tokens(no_filter))
    spam_token_total = len(get_spam_tokens(no_filter))
    ham_vocab_size = len(ham_freq.items())
    spam_vocab_size = len(spam_freq.items())

    for k, v in sorted(ham_freq.items()):
        f.write(str(line_counter) + '  ' + k + '  ' + str(v) + '  ' + str(calculate_conditional(v, ham_token_total, ham_vocab_size)) + '  ' + str(spam_freq[k]) + '  ' + str(calculate_conditional(spam_freq[k], spam_token_total, spam_vocab_size)) + '\n')
        line_counter += 1
    f.close()


# Example usage
ham_tokens = get_ham_tokens(no_filter)
spam_tokens = get_spam_tokens(no_filter)

ham_freq = frequency(ham_tokens)
spam_freq = frequency(spam_tokens)

build_model(ham_freq, spam_freq)
