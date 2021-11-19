from data import *

class part2:

    def __init__(self, lang) -> None:
        self.x_train, self.y_train = get_labelled_data(lang=lang, filename="train")
        self.x_val = get_unlabelled_data(lang=lang, filename="dev.in")
        self.transition_x_given_y = {}

    def transition_training(self):
        # dictionary format: {yi-1:{(yi-1,yi):frequency}

        #arbitrary numbers for start and stop
        start = 999
        stop = 998

        for i in range(len(self.y_train)):
            for j in range(len(self.y_train[i])-1):

                if (j+2) != len(self.y_train[i]):
                    yi_1 = self.y_train[i][j]
                    yi_1_yi = (yi_1,self.y_train[i][j+1])

                #add stop
                if (j+2) == len(self.y_train[i]):
                    yi_1 = self.y_train[i][j]
                    yi_1_yi = (yi_1,stop)

                

                #add start
                if j == 0:
                    yi_1 = start
                    yi_1_yi = (yi_1,self.y_train[i][j])

                    

                if yi_1 in self.transition_x_given_y.keys():
                    if yi_1_yi in self.transition_x_given_y[yi_1].keys():
                        self.transition_x_given_y[yi_1][yi_1_yi] += 1
                    else:
                        self.transition_x_given_y[yi_1][yi_1_yi] = 1
                else:
                    print(yi_1)
                    self.transition_x_given_y[yi_1] = {yi_1_yi:1}

        return self.transition_x_given_y



if __name__ == "__main__":
    part2 = part2("es")
    transition_x_given_y = part2.transition_training()
    print(transition_x_given_y)

           
