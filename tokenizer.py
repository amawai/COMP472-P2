import re

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

def frequency(token_list):
    dict = {}
    for word in token_list:
        if word in dict:
            dict[word] += 1
        else:
            dict[word] = 1
    return dict

def build_model(freq):
    f = open('model.txt', 'w+')
    line_counter = 1
    for k, v in sorted(freq.items()):
        f.write(str(line_counter) + '  ' + k + '  ' + str(v) + '\n')
        line_counter += 1
    f.close()


# Example usage
tokens = generate_tokens('./train-ham-00002.txt', no_filter)

freq = (frequency(tokens))
build_model(freq)