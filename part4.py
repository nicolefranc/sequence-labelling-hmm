from data import export_predictions_from_list, get_labelled_data, get_unlabelled_data, rnn_inputs, rnn_preprocess
import numpy as np
from rnn import RNN


if __name__ == '__main__':
    ES = 'es'
    RU = 'ru'

    # Español Training
    x_es, y_es = get_labelled_data(lang=ES, filename='train')
    word_to_idx, idx_to_word, num_of_unique_words = rnn_preprocess(x=x_es)

    rnn_es = RNN(num_of_unique_words, 7, word_to_idx,
                 idx_to_word, num_of_unique_words)

    print("[ES] Training model...")
    rnn_es.train(x_es, y_es, 10)

    # Russian Training

    x_ru, y_ru = get_labelled_data(lang=RU, filename='train')
    word_to_idx, idx_to_word, num_of_unique_words = rnn_preprocess(x=x_ru)

    rnn_ru = RNN(num_of_unique_words, 7, word_to_idx,
                 idx_to_word, num_of_unique_words)

    print("[RU] Training model...")
    rnn_ru.train(x_ru, y_ru, 100)

    # Dev Test
    # Español
    x_dev_test_es = get_unlabelled_data(lang=ES, filename='dev.in')
    predictions = rnn_es.test(x_dev_test_es)
    export_predictions_from_list(
        x_val=x_dev_test_es, predictions=predictions, lang=ES, part=4, filename='dev.p4.out')
    print('Exported dev predictions for Español.')

    # Russian
    x_dev_test_ru = get_unlabelled_data(lang=RU, filename='dev.in')
    predictions = rnn_ru.test(x_dev_test_ru)
    export_predictions_from_list(
        x_val=x_dev_test_ru, predictions=predictions, lang=RU, part=4, filename='dev.p4.out')
    print('Exported dev predictions for Russian.')

    # Actual Test Sets
    # Español
    x_test_es = get_unlabelled_data(lang=ES, filename='test.in')
    predictions = rnn_es.test(x_test_es)
    export_predictions_from_list(
        x_val=x_test_es, predictions=predictions, lang=ES, part=4, filename='test.p4.out')
    print('Exported predictions for Español.')

    # Russian
    x_test_ru = get_unlabelled_data(lang=RU, filename='test.in')
    predictions = rnn_ru.test(x_test_ru)
    export_predictions_from_list(
        x_val=x_test_ru, predictions=predictions, lang=RU, part=4, filename='test.p4.out')
    print('Exported predictions for Russian.')
