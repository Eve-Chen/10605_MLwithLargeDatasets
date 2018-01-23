"""
Long Short Term Memory for character level entity classification
"""
import sys
import argparse
import numpy as np
from xman import *
from utils import *
from autograd import *
from copy import deepcopy

np.random.seed(0)


class LSTM(object):
    """
    Long Short Term Memory + Feedforward layer
    Accepts maximum length of sequence, input size, number of hidden units and output size
    """

    def __init__(self, max_len, in_size, num_hid, out_size):
        self.my_xman = self._build(max_len, in_size, num_hid, out_size)  # DO NOT REMOVE THIS LINE. Store the output of xman.setup() in this variable
        print(self.my_xman.operationSequence(self.my_xman.loss))

    def _build(self, max_len, in_size, num_hid, out_size):
        print "INITIAZLIZING model inputs..."
        self.params = {}

        a_W = np.sqrt(6. / (in_size + num_hid))
        a_U = np.sqrt(6. / (num_hid + num_hid))
        a2 = np.sqrt(6. / (num_hid + out_size))
        for char in ['i', 'f', 'o', 'c']:
            self.params['W' + char] = f.param(name='W' + char,
                                              default=a_W * np.random.uniform(low=-1., high=1., size=(in_size, num_hid)))
            self.params['U' + char] = f.param(name='U' + char,
                                              default=a_U * np.random.uniform(low=-1, high=1, size=(num_hid, num_hid)))
            self.params['b' + char] = f.param(name='b' + char,
                                              default=0.1 * np.random.uniform(low=-1., high=1., size=(num_hid,)))
        self.params['W2'] = f.param(name="W2", default=a2 * np.random.uniform(-1., 1., (num_hid, out_size)))
        self.params['b2'] = f.param(name="b2", default=0.1 * np.random.uniform(-1., 1., (out_size,)))

        self.inputs = {}
        self.inputs['y'] = f.input(name='y', default=np.eye(1, out_size))
        self.inputs['h'] = f.input(name='h', default=np.zeros((1, num_hid)))
        self.inputs['c'] = f.input(name='c', default=np.zeros((1, num_hid)))
        for i in range(1, max_len + 1):
            self.inputs['x' + str(i)] = f.input(name='x' + str(i), default=np.ones((1, in_size)))

        x = XMan()
        # TODO: define your model here
        for t in range(1, max_len + 1):
            it = f.sigmoid(f.mul(self.inputs['x' + str(t)], self.params['Wi']) + f.mul(self.inputs['h'], self.params['Ui']) + self.params['bi'])
            ft = f.sigmoid(f.mul(self.inputs['x' + str(t)], self.params['Wf']) + f.mul(self.inputs['h'], self.params['Uf']) + self.params['bf'])
            ot = f.sigmoid(f.mul(self.inputs['x' + str(t)], self.params['Wo']) + f.mul(self.inputs['h'], self.params['Uo']) + self.params['bo'])
            candidate = f.tanh(f.mul(self.inputs['x' + str(t)], self.params['Wc']) + f.mul(self.inputs['h'], self.params['Uc']) + self.params['bc'])

            self.inputs['c'] = f.hadamard(ft, self.inputs['c']) + f.hadamard(it, candidate)
            self.inputs['h'] = f.hadamard(ot, f.tanh(self.inputs['c']))
        x.o2 = f.relu(f.mul(self.inputs['h'], self.params['W2']) + self.params['b2'])
        x.p = f.softMax(x.o2)
        x.loss = f.mean(f.crossEnt(x.p, self.inputs['y']))
        return x.setup()


def prepareInputs(value_dict, e, l, num_hid):
    N, M, V = e.shape
    # print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~e.shape:', e.shape)
    for t in range(1, M + 1):
        value_dict['x' + str(t)] = e[:, M - t, :].reshape(e.shape[0], -1)
    # print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~x20.shape:', value_dict['x20'].shape)
    value_dict['y'] = l
    value_dict['h'] = np.zeros((N, num_hid))
    value_dict['c'] = np.zeros((N, num_hid))
    return value_dict


def accuracy(probs, targets):
    preds = np.argmax(probs, axis=1)
    targ = np.argmax(targets, axis=1)
    return float((preds == targ).sum()) / preds.shape[0]


def main(params):
    epochs = params['epochs']
    max_len = params['max_len']
    num_hid = params['num_hid']
    batch_size = params['batch_size']
    dataset = params['dataset']
    init_lr = params['init_lr']
    output_file = params['output_file']
    train_loss_file = params['train_loss_file']

    # load data and preprocess
    dp = DataPreprocessor()
    data = dp.preprocess('%s.train' % dataset, '%s.valid' % dataset, '%s.test' % dataset)
    # minibatches
    mb_train = MinibatchLoader(data.training, batch_size, max_len,
                               len(data.chardict), len(data.labeldict))
    mb_valid = MinibatchLoader(data.validation, len(data.validation), max_len,
                               len(data.chardict), len(data.labeldict), shuffle=False)
    mb_test = MinibatchLoader(data.test, len(data.test), max_len,
                              len(data.chardict), len(data.labeldict), shuffle=False)

    # build
    print "building lstm..."
    lstm = LSTM(max_len, mb_train.num_chars, num_hid, mb_train.num_labels)
    # OPTIONAL: CHECK GRADIENTS HERE

    print "done"

    # train
    print "training..."
    # get default data and params

    value_dict = lstm.my_xman.inputDict()
    lr = init_lr
    train_loss = np.ndarray([0])
    ad = Autograd(lstm.my_xman)
    min_valid_loss = np.inf
    wengert_list = lstm.my_xman.operationSequence(lstm.my_xman.loss)

    for i in range(epochs):
        for (idxs, e, l) in mb_train:
            # TODO prepare the input and do a fwd-bckwd pass over it and update the weights
            value_dict = prepareInputs(value_dict, e, l, num_hid)
            value_dict = ad.eval(wengert_list, value_dict)
            # save the train loss
            train_loss = np.append(train_loss, value_dict['loss'])
            gradients = ad.bprop(wengert_list, value_dict, loss=np.float_(1.))
            for rname in gradients:
                if lstm.my_xman.isParam(rname):
                    value_dict[rname] -= lr * gradients[rname]
        # validate
        for (idxs, e, l) in mb_valid:
            # TODO prepare the input and do a fwd pass over it to compute the loss
            value_dict = prepareInputs(value_dict, e, l, num_hid)
            value_dict = ad.eval(wengert_list, value_dict)
            print("%sth epoch validation loss" % i, value_dict['loss'])
            # TODO compare current validation loss to minimum validation loss
            # and store params if needed
            if value_dict['loss'] < min_valid_loss:
                min_valid_loss = value_dict['loss']
                best_value_dict = deepcopy(value_dict)
    print('trianing loss:', train_loss)

    print "done"
    # write out the train loss
    np.save(train_loss_file, train_loss)

    for (idxs, e, l) in mb_test:
        # prepare input and do a fwd pass over it to compute the output probs
        prepareInputs(best_value_dict, e, l, num_hid)
        best_value_dict = ad.eval(wengert_list, best_value_dict)
        # TODO save probabilities on test set
        # ensure that these are in the same order as the test input
        output_probabilities = best_value_dict['p']
        np.save(output_file, output_probabilities)
        acc = accuracy(np.vstack(output_probabilities), np.vstack(l))
        c_loss = best_value_dict['loss']
        print "done, test loss = %.3f acc = %.3f" % (c_loss, acc)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_len', dest='max_len', type=int, default=10)
    parser.add_argument('--num_hid', dest='num_hid', type=int, default=50)
    parser.add_argument('--batch_size', dest='batch_size', type=int, default=64)
    parser.add_argument('--dataset', dest='dataset', type=str, default='tiny')
    parser.add_argument('--epochs', dest='epochs', type=int, default=15)
    parser.add_argument('--init_lr', dest='init_lr', type=float, default=0.5)
    parser.add_argument('--output_file', dest='output_file', type=str, default='output')
    parser.add_argument('--train_loss_file', dest='train_loss_file', type=str, default='train_loss')
    params = vars(parser.parse_args())
    main(params)
