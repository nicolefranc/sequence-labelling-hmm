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
            word_label_arr = word_label.split(' ')
            if len(word_label_arr) > 2:
                continue
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


def export_predictions(x_val: list, predictions: list, lang: str):
    f = open(f'./{lang.upper()}/dev.prediction', 'w', encoding='utf-8')
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
