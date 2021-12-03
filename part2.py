from data import *
from part1 import *

class part2:

    def __init__(self, lang) -> None:
        self.x_train, self.y_train = get_labelled_data(lang=lang, filename="train")
        self.x_val = get_unlabelled_data(lang=lang, filename="dev.in")
        self.transition_x_given_y = {}
        self.states_dict = conversions()

    def get_x_val(self):
        return self.x_val

    def get_transmission(self,transmission_x_given_y, x, y):

        if x in transmission_x_given_y[y].keys():
            return transmission_x_given_y[y][x]/sum(transmission_x_given_y[y].values())
        else:
            # print("No such transition: ", x)
            return 0

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
                    self.transition_x_given_y[yi_1] = {yi_1_yi:1}

        return self.transition_x_given_y

    def pi_k_v(self, prev_pi, x_k, u_v, emission_class, emission_dict, transition_dict):
        #initialise a path prob matrix
        emission = emission_class.get_emission(emission_dict, x_k, u_v[1])
        transmission = self.get_transmission(transition_dict, u_v, u_v[0])

        return prev_pi*emission*transmission

    def argmax(self,listt):
        #ToDo: do later
        argmax = max(listt)
        index = listt.index(argmax)
        return argmax, index

    def viterbi_per_sentence(self, emission_class, sentence):
        # initialise a matrix that is of no.of states (row) x len of input sequence (col)
        viterbi_lookup = [ [0]*len(self.states_dict) for i in range(len(sentence)+1)]

        #perform training
        emission_dict = emission_class.emission_training()
        transition_dict = self.transition_training()

        prev_pi = None
        prev_state = [] #ToDo: change this to a list
        path = []

        # print('Sentence', sentence)

        for j in range(len(sentence)):
            for u in range(len(self.states_dict)-2):
                if j == 0:
                    #we dont care about 0 -> START
                    #we just begin at the 1st input, START -> u and consider for every state there is
                    #in this case the prev_pi is 1
                    prev_pi = 1
                    x_k = sentence[j]
                    u_v = (self.states_dict["START"], u)

                    pi_k_v = self.pi_k_v(prev_pi, x_k, u_v, emission_class, emission_dict, transition_dict)

                    viterbi_lookup[j][u] = pi_k_v

                elif j == len(sentence)-1:
                    #here we care about the transition to stop
                    # x_k = sentence[j]
                    u_v = (prev_state[j-1], self.states_dict["STOP"])
                    # print(u_v)
                    pi_k_v = self.get_transmission(transition_dict, u_v, u_v[0])
                    viterbi_lookup[j][u] = pi_k_v
                
                else:
                    x_k = sentence[j]
                    u_v = (prev_state[j-1], u)
                    pi_k_v = self.pi_k_v(prev_pi, x_k, u_v, emission_class, emission_dict, transition_dict)
                    viterbi_lookup[j][u] = pi_k_v

            argmax_pi, state = self.argmax(viterbi_lookup[j])
            prev_pi = argmax_pi
            prev_state.append(state)
            # print("hero",j,prev_state)


        return prev_state

    def viterbi(self,emission_class):
        predictions = []
        for i in range(len(self.x_val)):
            preds_list = self.viterbi_per_sentence(emission_class, self.x_val[i])
            predictions.append(preds_list)
            # if i == 2:
                # break
        # print(predictions)
        return predictions


if __name__ == "__main__":
    LANG = "ru"
    part2 = part2(LANG)
    emission_class = part1(LANG)
    transition_x_given_y = part2.transition_training()
    # print(transition_x_given_y)

    states = part2.viterbi(emission_class)
    # print(states)

    export_predictions_from_list(part2.get_x_val(), predictions=states, lang=LANG, part=2)
