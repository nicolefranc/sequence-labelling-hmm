from data import *
from part1 import *
from pprint import pprint

class part3:

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

    def kthSmallest(self, arr, k):
  
        # Sort the given array 
        arr.sort()
        return arr[k-1]
    def arg5th(self,listt):
        #ToDo: do later
        arg5th = self.kthSmallest(listt, 5)
        index = listt.index(arg5th)
        return arg5th, index

    def viterbi_per_sentence(self,emission_class, sentence):

        viterbi_lookup = [ [0]*(len(self.states_dict)-2) for i in range(len(sentence))]

        #perform training
        emission_dict = emission_class.emission_training()
        transition_dict = self.transition_training()

        #at every node you store the highest achieveable pi that passes through that node
        #and you store the path thus far
        
        for layer in range(len(sentence)):

            for state in range(len(self.states_dict)-2):
                print("layer: ", layer,"state: ", state)
                if layer == 0:
                    #we dont care about 0 -> START
                    #we just begin at the 1st input, START -> node and consider for every state there is
                    #in this case the prev_pi is 1
                    prev_pi = 1
                    word = sentence[layer]
                    u_v = (self.states_dict["START"], state)

                    pi_k_v = self.pi_k_v(prev_pi, word, u_v, emission_class, emission_dict, transition_dict)
                    
                    #([start,state], pi(start,state))
                    viterbi_lookup[layer][state] = (list(u_v) ,pi_k_v)
                    # pprint(viterbi_lookup)

                elif layer == len(sentence)-1:
                    #here we care about the transition to stop
                    u_v = (state, self.states_dict["STOP"])
                    pi_k_v = self.get_transmission(transition_dict, u_v, u_v[0])
                    best_path_so_far = viterbi_lookup[layer-1][state][0]+[state,self.states_dict["STOP"]]
                    viterbi_lookup[layer][state] = (best_path_so_far, pi_k_v)
                    pprint(viterbi_lookup)

                else:
                    temp_list = []
                    for prev_state in range(len(viterbi_lookup[layer-1])):
                        #([start,state], pi(start,state))
                        prev_pi = viterbi_lookup[layer-1][prev_state][1]
                        word = sentence[layer]
                        u_v = (prev_state, state)

                        pi_k_v = self.pi_k_v(prev_pi, word, u_v, emission_class, emission_dict, transition_dict)
                        temp_list.append(pi_k_v)
                    #change to max
                    max_pi, best_prev_state = self.argmax(temp_list)
                    # print("here:")
                    # print(state)
                    # pprint(viterbi_lookup[layer-1][best_prev_state][0]+[state])
                    # print("end")
                    best_path_so_far = viterbi_lookup[layer-1][best_prev_state][0] + [state]
                    viterbi_lookup[layer][state] = (best_path_so_far, max_pi)
                    # pprint(viterbi_lookup)
        
        return viterbi_lookup


    def viterbi(self,emission_class):
        predictions = []
        for i in range(len(self.x_val)):
            viterbi_lookup = self.viterbi_per_sentence(emission_class, self.x_val[i])

            final_pi = []
            for i in range(len(self.states_dict)-2):
                final_pi.append(viterbi_lookup[-1][i][1])
            
            max_pi, index = self.arg5th(final_pi)

            best_path = viterbi_lookup[-1][index][0]

            predictions.append(best_path)
            print("predictions:",predictions)
            print("end")
        return predictions


if __name__ == "__main__":
    LANG = "ru"
    part3 = part3(LANG)
    emission_class = part1(LANG)
    transition_x_given_y = part3.transition_training()
    # print(transition_x_given_y)

    states = part3.viterbi(emission_class)
    # print(states)

    export_predictions_from_list(part3.get_x_val(), predictions=states, lang=LANG, part=2)
