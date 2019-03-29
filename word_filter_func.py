import re

def no_filter(word):
    return word != ''

def stop_word_filter(word):
    pass

def word_len_filter(word):
    pass

# Tokenizes file, outputs array of valid words based on passed-in filter
def filter(filename, filter_func=no_filter):
    with open(filename) as f:
        file_content = f.readlines()
    vocab_list = [[word for word in re.split('[^a-zA-Z]', line.lower().strip()) if filter_func(word)] for line in file_content]
    flattened = [token for sublist in vocab_list for token in sublist]
    return flattened

# Example usage
# print(filter('./Project2-Test/test/test-ham-00002.txt', no_filter))