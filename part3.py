from re import U
from data import *
from part1 import *
from pprint import pprint
from tqdm import tqdm


class viterbi_node:
    def __init__(self, layer, state_path, pi_value) -> None:
        self.layer = layer
        # this can be a list of state instead of a single thing, depends
        self.state = state_path
        self.pi_value = pi_value


class part3:

    def __init__(self, lang) -> None:
        self.x_train, self.y_train = get_labelled_data(
            lang=lang, filename="train")
        self.x_val = get_unlabelled_data(lang=lang, filename="dev.in")
        self.transition_x_given_y = {}
        self.states_dict = conversions()

    def get_x_val(self):
        return self.x_val

    def get_transmission(self, transmission_x_given_y, x, y):

        if x in transmission_x_given_y[y].keys():
            return transmission_x_given_y[y][x]/sum(transmission_x_given_y[y].values())
        else:
            # print("No such transition: ", x)
            return 0

    def transition_training(self):
        # dictionary format: {yi-1:{(yi-1,yi):frequency}

        # arbitrary numbers for start and stop
        start = 999
        stop = 998

        for i in range(len(self.y_train)):
            for j in range(len(self.y_train[i])-1):

                if (j+2) != len(self.y_train[i]):
                    yi_1 = self.y_train[i][j]
                    yi_1_yi = (yi_1, self.y_train[i][j+1])

                # add stop
                if (j+2) == len(self.y_train[i]):
                    yi_1 = self.y_train[i][j]
                    yi_1_yi = (yi_1, stop)

                # add start
                if j == 0:
                    yi_1 = start
                    yi_1_yi = (yi_1, self.y_train[i][j])

                if yi_1 in self.transition_x_given_y.keys():
                    if yi_1_yi in self.transition_x_given_y[yi_1].keys():
                        self.transition_x_given_y[yi_1][yi_1_yi] += 1
                    else:
                        self.transition_x_given_y[yi_1][yi_1_yi] = 1
                else:
                    self.transition_x_given_y[yi_1] = {yi_1_yi: 1}

        return self.transition_x_given_y

    def pi_k_v(self, prev_pi, x_k, u_v, emission_class, emission_dict, transition_dict):
        # initialise a path prob matrix
        emission = emission_class.get_emission(emission_dict, x_k, u_v[1])
        transmission = self.get_transmission(transition_dict, u_v, u_v[0])

        return prev_pi*emission*transmission

    def argmax(self, listt):
        # ToDo: do later
        argmax = max(listt)
        index = listt.index(argmax)
        return argmax, index

    def kthLargest(self, arr, k):

        # Sort the given array
        arr.sort(reverse=True)
        return arr[k-1]

    def kLargest(self, arr, k):
        # Sort the given array arr in reverse
        # order.
        arr.sort(reverse=True)
        # Print the first kth largest elements
        return arr[:k]

    def arg5(self, listt):
        # ToDo: do later
        arg5th = self.kLargest(listt, 5)
        index5 = []
        for i in arg5th:
            index = listt.index(i)
            index5.append(index)
        return arg5th, index5

    def arg5th(self, listt):
        # ToDo: do later
        arg5th = self.kthLargest(listt, 5)
        index = listt.index(arg5th)
        return arg5th, index

    # in every slot of the list instead of storing the values directly you store it as a node,so you can have different things in the stupid thing

    def viterbi_per_sentence(self, emission_class, sentence):

        viterbi_lookup = [[0]*(len(self.states_dict)-2)
                          for i in range(len(sentence))]

        # perform training
        emission_dict = emission_class.emission_training()
        transition_dict = self.transition_training()

        # at every node you store the highest achieveable pi that passes through that node
        # and you store the path thus far
        # print(sentence)
        for layer in range(len(sentence)):
            list_of_pi_values = []
            list_of_states = []

            for state in range(len(self.states_dict)-2):
                # print("layer: ", layer,"state: ", state)
                if layer == 0:
                    # we dont care about 0 -> START
                    # we just begin at the 1st input, START -> node and consider for every state there is
                    # in this case the prev_pi is 1
                    prev_pi = 1
                    word = sentence[layer]
                    u_v = (self.states_dict["START"], state)

                    pi_k_v = self.pi_k_v(
                        prev_pi, word, u_v, emission_class, emission_dict, transition_dict)

                    list_of_pi_values.append(pi_k_v)
                    list_of_states.append(list(u_v))

                    #([start,state], pi(start,state))
                    viterbi_lookup[layer][state] = viterbi_node(
                        layer, list(u_v), pi_k_v)
                    # pprint(viterbi_lookup)

                elif layer == len(sentence)-1 and len(sentence) > 2:

                    list_of_pi_values = []
                    list_of_states = []

                    for prev_node in range(len(viterbi_lookup[layer-1])):
                        #pi(state,state), pi(state,state), ...
                        pi_values = viterbi_lookup[layer-1][prev_node].pi_value
                        #[start,state,state], [start,state,state], ...
                        prev_states = viterbi_lookup[layer-1][prev_node].state
                        for i in range(5):
                            pi_value = pi_values[i]
                            u_v = (prev_states[i][-1],
                                   self.states_dict["STOP"])
                            pi_k_v = pi_value * \
                                self.get_transmission(
                                    transition_dict, u_v, u_v[0])
                            list_of_pi_values.append(pi_k_v)
                            list_of_states.append(
                                prev_states[i]+[self.states_dict["STOP"]])

                elif layer == len(sentence)-1:
                    list_of_pi_values = []
                    list_of_states = []

                    for prev_node in range(len(viterbi_lookup[layer-1])):
                        #pi(state,state), pi(state,state), ...
                        pi_value = viterbi_lookup[layer-1][prev_node].pi_value
                        #[start,state,state], [start,state,state], ...
                        prev_state = viterbi_lookup[layer-1][prev_node].state

                        u_v = (prev_state[-1], self.states_dict["STOP"])
                        pi_k_v = pi_value * \
                            self.get_transmission(transition_dict, u_v, u_v[0])
                        list_of_pi_values.append(pi_k_v)
                        list_of_states.append(
                            prev_state+[self.states_dict["STOP"]])

                else:
                    # in subsequent layers, for each node you have to look for the top 5 values. Hence you have to append 5 list in the pi_v and the states
                    if layer == 1:
                        # the nodes here wont have more than one values, so its just a int
                        #([start,state], pi(start,state))
                        list_of_pi_values = []
                        list_of_states = []
                        for prev_node in range(len(viterbi_lookup[layer-1])):
                            # pi(start,state)
                            pi_value = viterbi_lookup[layer -
                                                      1][prev_node].pi_value
                            # [start,state]
                            prev_state = viterbi_lookup[layer -
                                                        1][prev_node].state[-1]
                            u_v = (prev_state, state)

                            pi_k_v = self.pi_k_v(
                                pi_value, word, u_v, emission_class, emission_dict, transition_dict)
                            list_of_pi_values.append(pi_k_v)
                            list_of_states.append(
                                viterbi_lookup[layer-1][prev_node].state+[state])

                        pis5, indexes5 = self.arg5(list_of_pi_values)
                        list_of_5_states = [list_of_states[indexes5[0]]] + [list_of_states[indexes5[1]]] + [
                            list_of_states[indexes5[2]]] + [list_of_states[indexes5[3]]] + [list_of_states[indexes5[4]]]
                        viterbi_lookup[layer][state] = viterbi_node(
                            layer, list_of_5_states, pis5)

                    else:
                        list_of_pi_values = []
                        list_of_states = []
                        for prev_node in range(len(viterbi_lookup[layer-1])):
                            #pi(state,state), pi(state,state), ...
                            pi_values = viterbi_lookup[layer -
                                                       1][prev_node].pi_value
                            #[start,state,state], [start,state,state], ...
                            prev_states = viterbi_lookup[layer -
                                                         1][prev_node].state
                            for i in range(5):
                                pi_value = pi_values[i]
                                u_v = (prev_states[i][-1], state)
                                pi_k_v = self.pi_k_v(
                                    pi_value, word, u_v, emission_class, emission_dict, transition_dict)
                                list_of_pi_values.append(pi_k_v)
                                list_of_states.append(prev_states[i]+[state])

                        pis5, indexes5 = self.arg5(list_of_pi_values)
                        list_of_5_states = [list_of_states[indexes5[0]]] + [list_of_states[indexes5[1]]] + [
                            list_of_states[indexes5[2]]] + [list_of_states[indexes5[3]]] + [list_of_states[indexes5[4]]]
                        viterbi_lookup[layer][state] = viterbi_node(
                            layer, list_of_5_states, pis5)

        # pprint(viterbi_lookup)
        # print(len(list_of_pi_values))
        # pprint(list_of_pi_values)
        # pprint(list_of_states)

        pi5th, index5th = self.arg5th(list_of_pi_values)
        path5th = list_of_states[index5th]

        return pi5th, path5th

    def viterbi(self, emission_class):
        predictions = []

        for i in tqdm(range(len(self.x_val))):
            pi5th, path5th = self.viterbi_per_sentence(
                emission_class, self.x_val[i])

            # predictions.append(path5th)
            predictions.append(path5th[1:len(path5th)-1])
            # print(pi5th, predictions)
            # print("end")

        return predictions


if __name__ == "__main__":
    LANG = "es"
    part3 = part3(LANG)
    emission_class = part1(LANG)
    transition_x_given_y = part3.transition_training()
    # print(transition_x_given_y)

    states = part3.viterbi(emission_class)
    # print(states)

    export_predictions_from_list(
        part3.get_x_val(), predictions=states, lang=LANG, part=3)
