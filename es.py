## DATASET
# train - labelled training set, train
# dev.in - unlabelled dev set, test
# dev.out - labelled dev set, validate

class Data():

    def get_data(lang: str, filename: str):
        x, y = [], []
        f = open(f'./{lang.upper()}/{filename}', encoding='utf-8')
        sentences = f.read().split(sep='\n\n')
        
        for sentence in sentences:
            words_labels = sentence.splitlines()
            words, labels = [], []
            for word_label in words_labels:
                word_label_arr = word_label.split(' ')
                words.append(word_label_arr[0])
                labels.append(word_label_arr[1])
            x.append(words)
            y.append(labels)

        print(y[0])

Data.get_data(lang="es", filename="train")