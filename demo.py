import os
from classifier import Classifier
from tokenizer import build_model_wrapper, no_filter,\
    stop_word_filter, word_len_filter, get_stop_words

STOP_WORDS_FILE = './English-Stop-Words.txt'
stop_words = get_stop_words(STOP_WORDS_FILE)

def choose_experiment(experiments):
    experiment = ""
    while experiment not in experiments:
        experiment = input(
            "Input one of the following experiments, [%s] > " % ", ".join(experiments))
    return experiment

def validate_directory(directory):
    return os.path.isdir(directory)

def input_directory(directory_type):
    input_dir = input('Input {} set location > '.format(directory_type))
    while not validate_directory(input_dir):
        input_dir = input('Please input a valid, existing directory > ')
    return input_dir

experiment = choose_experiment(["1", "2", "3"])
training_dir = input_directory('training')
test_dir = input_directory('test')

if experiment == '1':
    model_file = 'demo-model-base.txt'
    result_file = 'demo-result-base.txt'
    filter_func = no_filter
    word_filter = None
elif experiment == '2':
    model_file = 'demo-model-exp2.txt'
    result_file = 'demo-result-exp2.txt'
    filter_func = stop_word_filter
    word_filter = stop_words
elif experiment == '3':
    model_file = 'demo-model-exp3.txt'
    result_file = 'demo-result-exp3.txt'
    filter_func = word_len_filter
    word_filter = None

print('Building model...')
build_model_wrapper(training_dir, model_file, filter_func, word_filter)
print('Model complete! Classifying...')
Classifier(training_dir, model_file).classify(test_dir, result_file, filter_func, word_filter)
print('Classification complete. Check out {} and {}.'.format(model_file, result_file))
