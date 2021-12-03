from data import *


class part1:

    def __init__(self, lang: str) -> None:
        self.x_train, self.y_train = get_labelled_data(
            lang=lang, filename="train")
        self.x_val = get_unlabelled_data(lang=lang, filename="dev.in")
        self.emission_x_given_y = {}

    def get_x_val(self):
        return self.x_val

    def get_emission(self, emission_x_given_y, x, y):
        # print(x,y)
        
        if x in emission_x_given_y[y].keys():
            return emission_x_given_y[y][x]/sum(emission_x_given_y[y].values())
        else:
            return emission_x_given_y[y]["#UNK#"]/sum(emission_x_given_y[y].values())

    def get_argmax_emission_x_given_y(self, emission_x_given_y, x):
        max_val = 0
        key = ""
        for i in emission_x_given_y.keys():
            val = self.get_emission(emission_x_given_y, x, i)
            if val > max_val:
                max_val = val
                key = i
            else:
                continue

        return key

    def emission_training(self):
        # generate emission parameters based on training data

        # dictionary format: {label:{word:frequency}}
        # this is to build the dictionary
        # then you need a function to retrieve the values

        for i in range(len(self.y_train)):
            for j in range(len(self.y_train[i])):
                label = self.y_train[i][j]
                word = self.x_train[i][j]

                if label in self.emission_x_given_y.keys():
                    if word in self.emission_x_given_y[label].keys():
                        self.emission_x_given_y[label][word] += 1
                    else:
                        self.emission_x_given_y[label][word] = 1
                else:
                    self.emission_x_given_y[label] = {word: 1}

        # you have one occurence of K, for any given y
        for i in self.emission_x_given_y.keys():
            self.emission_x_given_y[i]["#UNK#"] = 1

        return self.emission_x_given_y


if __name__ == "__main__":
    lang = "ru"
    part1 = part1(lang)
    emission_x_given_y = part1.emission_training()
    listofpredictions = []
    for sentence in part1.get_x_val():
        y_i = []
        for word in sentence:
            y_i.append(part1.get_argmax_emission_x_given_y(
                emission_x_given_y, word))
        listofpredictions.append(y_i)

    # print(listofpredictions)
    # ## Export predictions and print
    export_predictions_from_list(part1.get_x_val(), listofpredictions, lang, part=1)
