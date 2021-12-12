import numpy as np
from numpy.random import randn
from data import convert_label, get_labelled_data, prep_y, rnn_inputs, softmax
import math


class RNN:

    def __init__(self, input_size, output_size, word_to_idx, idx_to_word, num_of_unique_words, hidden_size=64):
        # Weights
        self.Whh = randn(hidden_size, hidden_size) / 1000
        self.Wxh = randn(hidden_size, input_size) / 1000
        self.Why = randn(output_size, hidden_size) / 1000

        # Biases
        self.bh = np.zeros((hidden_size, 1))
        self.by = np.zeros((output_size, 1))

        # Data
        self.word_to_idx, self.idx_to_word, self.num_of_unique_words = word_to_idx, idx_to_word, num_of_unique_words

    def forward(self, inputs):
        # print('inputs:', inputs.shape)
        h = np.zeros((self.Whh.shape[0], 1))
        self.prev_inputs = inputs
        self.prev_h = {0: h}
        self.y_pred = {}

        # Perform each step of the RNN
        for i, x in enumerate(inputs):
            h = np.tanh(np.dot(self.Wxh, x) + np.dot(self.Whh, h) + self.bh)
            self.prev_h[i + 1] = h

            # Compute the output
            y_pred_i = np.dot(self.Why, h) + self.by
            self.y_pred[i] = softmax(y_pred_i)

        return self.y_pred, h

    def backward(self, y, y_pred, lr=2e-2):
        n = len(self.prev_inputs)
        # print('n', n)
        # print('y', y)
        # print('ypred', y_pred)
        # print('prev inputs', self.prev_inputs)

        # Initialize dL/dWhh, dL/dWxh, dL/dbh to zero
        dWhh = np.zeros(self.Whh.shape)
        dWxh = np.zeros(self.Wxh.shape)
        dbh = np.zeros(self.bh.shape)

        '''
        Backpropagation through time (BPTT)
        '''
        for t in reversed(range(n)):
            # print('n', n)
            # print('t', t)
            # print('yt', y[t])
            # print('ypredt', y_pred[t])
            dy = np.copy(y_pred[t])
            # print('ytmax', np.argmax(y))  # -- get the index of the max value
            dy[np.argmax(y[t])] -= 1
            # print('subtracted dy', dy)
            # dy[0][np.argmax(y[t])] -= 1  # predicted y - actual y

            dWhy = np.dot(dy, self.prev_h[n].T)  # dL/dWhy
            dby = dy    # dL/dby

            dh_prev = np.dot(self.Why.T, dy)    # dL/dh

            dhidden = ((1 - self.prev_h[t + 1] ** 2) * dh_prev)

            # Formula: dL/db = dL/dh * (1 - h^2)
            dbh += dhidden

            # Formula: dL/dWhh = dL/dh * (1 - h^2) * h_{t-1}
            dWhh += np.dot(dhidden, self.prev_h[t].T)

            # Formula: dL/dWxh = dL/dh * (1 - h^2) * x
            dWxh += np.dot(dhidden, self.prev_inputs[t].T)

            # Formula for next dL/dh = dL/dh * (1 - h^2) * Whh
            dh = np.dot(self.Whh, dhidden)

        # Clip to avoid exploding gradient
        for d in [dWxh, dWhh, dWhy, dbh, dby]:
            np.clip(d, -1, 1, out=d)

        self.Whh -= lr * dWhh
        self.Wxh -= lr * dWxh
        self.Why -= lr * dWhy
        self.bh -= lr * dbh
        self.by -= lr * dby

    def train(self, X, Y, epochs: int):
        for epoch in range(1, epochs + 1):

            for _, (x, y) in enumerate(zip(X, Y)):
                # print(f'idx: {idx}, x: {x}, y: {y}')
                inputs = rnn_inputs(x, self.word_to_idx,
                                    self.num_of_unique_words)
                target = prep_y(y)
                # print('prepped y', target)

                # Forward
                y_pred, _ = self.forward(inputs)

                # Calculate loss
                loss = 0
                loss_list = []
                for word_idx in range(len(x)):
                    for i in range(len(y_pred[word_idx][0])):
                        # print('predwordidx', y_pred[word_idx])

                        # Formula: loss = y_i * log(yhat_i)
                        loss -= target[word_idx][i] * \
                            np.log(y_pred[word_idx][i][0])
                    # loss *= -1
                    loss_list.append(loss)

                # Calculate loss / accuracy
                # print('ypred', y_pred)
                # print('argmax', np.argmax(y_pred[idx]))
                # loss += np.log(y[idx][0, np.argmax(y_pred[idx])])

                # num_correct += int(np.argmax(out) == target)
                # print(idx, x, y)
                # print('\n\n=====\n\n', y_pred)
                # dLdy = y_pred[idx]
                # dLdy[0][np.argmax(target[idx])] -= 1

                self.backward(y=target, y_pred=y_pred)

            print(f'Epoch: {epoch}\tLoss: {loss}')

    def test(self, X):
        Y = []
        for idx, sentence in enumerate(X):
            # if idx > 2:
            #     break

            # print('sentence', sentence)

            y_preds = []
            encodedX = []
            word_idxs = []
            for _, word in enumerate(sentence):
                # Initialize input vector, x
                x = np.zeros((self.num_of_unique_words, 1))

                # One hot encoded the word
                if word in self.word_to_idx.keys():
                    word_idx = self.word_to_idx[word]
                else:
                    word_idx = np.random.randint(0, 7)
                x[word_idx] = 1
                word_idxs.append(word_idx)
                encodedX.append(x)

                # print('one hot encoded x', x)

            # Iterate through the encoded X and predict
            for _, word in enumerate(np.asarray(encodedX)):
                h = np.zeros((self.Whh.shape[0], 1))

                # Forward
                h = np.tanh(np.dot(self.Wxh, x) +
                            np.dot(self.Whh, h) + self.bh)
                y_pred_i = np.dot(self.Why, h) + self.by
                y_pred = softmax(y_pred_i)

                # print('y-pred', y_pred)
                # print('y-pred ravel', y_pred.ravel())

                # Get random index from the probability distribution of y
                index = np.random.choice(
                    range(7), p=y_pred.ravel())

                # print('index', index)

                # Re-initialize input vector, x
                # x = np.zeros((self.num_of_unique_words, 1))
                # encodedX[idx]
                # label = convert_label(index)
                y_preds.append(index)

            Y.append(y_preds)

        # print(Y)
        return Y
