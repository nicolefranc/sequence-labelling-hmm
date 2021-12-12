import numpy as np
from tqdm import tqdm

# DATASET
# train - labelled training set, train
# dev.in - unlabelled dev set, test
# dev.out - labelled dev set, validate on our own end, for submission


def get_labelled_data(lang: str, filename: str):
    x, y = [], []
    f = open(f'./{lang.upper()}/{filename}', encoding='utf-8')
    sentences = f.read().strip().split(sep='\n\n')

    for sentence in sentences:
        words_labels = sentence.splitlines()
        words, labels = [], []
        for word_label in words_labels:
            word_label_arr = word_label.rsplit(' ', 1)
            # if len(word_label_arr) > 2:
            #     continue
            words.append(word_label_arr[0])
            labels.append(convert_label(word_label_arr[1]))
        x.append(words)
        y.append(labels)

    return x, y


def get_unlabelled_data(lang: str, filename: str):
    x = []

    f = open(f'./{lang.upper()}/{filename}', encoding='utf-8')
    sentences = f.read().strip().split(sep='\n\n')

    for sentence in sentences:
        words = sentence.splitlines()
        x.append(words)

    return x


def conversions():
    return {"O": 0, "B-positive": 1, "B-negative": 2, "B-neutral": 3, "I-positive": 4, "I-negative": 5, "I-neutral": 6, "STOP": 998, "START": 999}


def convert_label(label: str, toInt: bool = True):
    labels = conversions()
    if toInt and (label in labels.keys()):
        # print(labels[label])
        return labels[label]
    # convert the value to the key
    for key, value in labels.items():
        if label == value:
            return key


def export_predictions_from_list(x_val: list, predictions: list, lang: str, part: int):
    f = open(f'./{lang.upper()}/{part}/dev.prediction', 'w', encoding='utf-8')
    out = ''

    for sentence, prediction in zip(x_val, predictions):
        for word, label in zip(sentence, prediction):
            out += f'{word} {convert_label(label, False)}\n'
        out += '\n'

    f.write(out)
    # print(out)
    f.close()

# LABEL CONVERSION USAGE
### intLabel = convert_label('B-positive')
### strLabel = convert_label(1, False)
# print(intLabel)
# print(strLabel)


# DATA EXTRACTION USAGE
### x, y = get_labelled_data(lang="es", filename="train")
### x = get_unlabelled_data(lang="es", filename="dev.in")
# print(x)

# ########################## #
# DATA PREPROCESSING FOR RNN
# ########################## #

def rnn_preprocess(x: list):
    ''' Extract unique words '''
    # words = list(set([word for sentence in x for _, word in enumerate(sentence)]))
    all_words = []
    for sentence in x:
        for idx, word in enumerate(sentence):
            all_words.append(word)

    words = list(set(all_words))
    num_of_unique_words = len(words)
    # print('Number of unique words:', num_of_unique_words)

    word_to_idx = {}
    idx_to_word = {}
    inputs = []

    for idx, word in enumerate(words):
        ''' Assign an id to word '''
        word_to_idx[word] = idx
        idx_to_word[idx] = word

    # print(word_to_idx['disfrutemos'])
    # print(idx_to_word[0])
    # print(word_to_idx.items())
    return word_to_idx, idx_to_word, num_of_unique_words


def rnn_inputs(sentence: str, word_to_idx: dict, num_of_unique_words: int):
    inputs = []
    for idx, word in enumerate(sentence):
        # print(f'word {idx} - {word}')
        '''
        Create a vector of shape (num_of_unique_words, 1)
        - the value at position word_to_idx[word] in the vector should be 1
        '''
        vector = np.zeros((num_of_unique_words, 1))
        # get the index of the word, make the value of the vector at that index to 1
        vector[word_to_idx[word]] = 1
        inputs.append(vector)
        # print(f'sentence {sentence}', sentence)
        # print(f'vector {idx}:', vector.shape)

    return np.asarray(inputs)


def prep_y(Y: list):
    encodedy = []
    for y in Y:
        one_hot_char = np.zeros((7))
        one_hot_char[y] = 1
        encodedy.append(one_hot_char)

    return encodedy


def softmax(xs):
    return np.exp(xs) / sum(np.exp(xs))
