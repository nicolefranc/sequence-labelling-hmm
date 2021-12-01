from data import *
from part1 import *

class part2:

    def __init__(self, lang) -> None:
        self.x_train, self.y_train = get_labelled_data(lang=lang, filename="train")
        self.x_val = get_unlabelled_data(lang=lang, filename="dev.in")
        self.transition_x_given_y = {}
        self.states_dict = conversions()

    def get_transmission(self,transmission_x_given_y, x, y):
        if x in transmission_x_given_y[y].keys():
            return transmission_x_given_y[y][x]/sum(transmission_x_given_y[y].values())
        else:
            print("No such state")

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

    def pi_k_v(self, prev_pi, x_k, u_v, emission_class, emission_dict, transition_dict):
        #initialise a path prob matrix
        emission = emission_class.get_emission(emission_dict, u_v[0], x_k)
        transmission = self.get_transmission(transition_dict,u_v, u_v[0])

        return prev_pi*emission*transmission




if __name__ == "__main__":
    part2 = part2("es")
    transition_x_given_y = part2.transition_training()
    print(transition_x_given_y)

           
