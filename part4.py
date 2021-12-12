from data import get_labelled_data, rnn_inputs, rnn_preprocess
import numpy as np
from rnn import RNN


if __name__ == '__main__':
    x, y = get_labelled_data(lang='es', filename='train')
    word_to_idx, idx_to_word, num_of_unique_words = rnn_preprocess(x=x)

    # for idx, sentence in enumerate(x):
    # inputs = rnn_inputs(x, word_to_idx, num_of_unique_words)

    rnn = RNN(num_of_unique_words, 7, word_to_idx,
              idx_to_word, num_of_unique_words)
    # # out, h = rnn.forward(inputs)
    # # probs = out
    # rnn.train(x, y, 2)

    # print(probs)
    # print(x[0])

    rnn.test(x)
