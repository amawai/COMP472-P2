from classifier import Classifier
from tokenizer import build_model_wrapper, no_filter,\
    stop_word_filter, word_len_filter, get_stop_words

TRAINING_DIRECTORY = './train'
TESTING_DIRECTORY = './test'
STOP_WORDS_FILE = './English-Stop-Words.txt'

stop_words = get_stop_words(STOP_WORDS_FILE)
# Building the models
build_model_wrapper(TRAINING_DIRECTORY, './models/model.txt', no_filter)
build_model_wrapper(TRAINING_DIRECTORY, './models/stopword-model.txt', stop_word_filter, stop_words)
build_model_wrapper(TRAINING_DIRECTORY, './models/wordlength-model.txt', word_len_filter)

# Running the classifiers on models
vanilla_classifier = Classifier(TRAINING_DIRECTORY, './models/model.txt')
vanilla_classifier.classify(TESTING_DIRECTORY, './results/baseline-result.txt', no_filter)

stop_word_classifier = Classifier(TRAINING_DIRECTORY, './models/stopword-model.txt')
stop_word_classifier.classify(TESTING_DIRECTORY, './results/stopword-result.txt', stop_word_filter, stop_words)

word_len_classifier = Classifier(TRAINING_DIRECTORY, './models/wordlength-model.txt')
word_len_classifier.classify(TESTING_DIRECTORY, './results/wordlength-result.txt', word_len_filter)