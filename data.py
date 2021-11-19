## DATASET
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


def convert_label(y: str):
    if y == "O": y = 0
    elif y == "B-positive": y = 1
    elif y == "B-negative": y = 2
    elif y == "B-neutral": y = 3
    elif y == "I-positive": y = 4
    elif y == "I-negative": y = 5
    elif y == "I-neutral": y = 6
    return y


# USAGE
### x, y = get_labelled_data(lang="es", filename="train")
### x = get_unlabelled_data(lang="es", filename="dev.in")
### print(x)

def export_predictions(x_val: list, predictions: list, lang: str):
    f = open(f'./{lang.upper()}/dev.prediction', 'w')
    out = ''

    for sentence, prediction in zip(x_val, predictions):
        for word, label in zip(sentence, prediction):
            out += f'{word} {label}\n'
        out += '\n'
    
    f.write(out)