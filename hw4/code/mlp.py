"""
Multilayer Perceptron for character level entity classification
"""
import argparse
import numpy as np
from xman import *
from utils import *
from autograd import *
from copy import deepcopy
import time

np.random.seed(0)


class MLP(object):
    """
    Multilayer Perceptron
    Accepts list of layer sizes [in_size, hid_size1, hid_size2, ..., out_size]
    """

    def __init__(self, layer_sizes):
        self.layer_sizes = layer_sizes
        self.my_xman = self._build()  # DO NOT REMOVE THIS LINE. Store the output of xman.setup() in this variable

    def _build(self):
        x = XMan()
        # declare registers
        d_in = self.layer_sizes[0]
        d_hid = self.layer_sizes[1]
        d_out = self.layer_sizes[2]
        x.x = f.input(name='x', default=np.ones((1, d_in)))
        x.y = f.input(name='y', default=np.eye(1, d_out))
        a1 = (6.0 / (d_in + d_hid)) ** 0.5
        x.W1 = f.param(name="W1", default=a1 * np.random.uniform(-1., 1., (d_in, d_hid)))
        x.b1 = f.param(name="b1", default=0.1 * np.random.uniform(-1., 1., (d_hid,)))
        a2 = (6.0 / (d_hid + d_hid)) ** 0.5
        x.W2 = f.param(name="W2", default=a2 * np.random.uniform(-1., 1., (d_hid, d_out)))
        x.b2 = f.param(name="b2", default=0.1 * np.random.uniform(-1., 1., (d_out,)))
        # TODO define your model here
        x.o1 = f.relu(f.mul(x.x, x.W1) + x.b1)
        x.o2 = f.relu(f.mul(x.o1, x.W2) + x.b2)
        x.p = f.softMax(x.o2)
        x.loss = f.mean(f.crossEnt(x.p, x.y))
        return x.setup()


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
    # mb_valid = MinibatchLoader(data.validation, len(data.validation), max_len,
    #                            len(data.chardict), len(data.labeldict), shuffle=False)
    # mb_test = MinibatchLoader(data.test, len(data.test), max_len,
    #                           len(data.chardict), len(data.labeldict), shuffle=False)

    # build
    print("building mlp...")
    mlp = MLP([max_len * mb_train.num_chars, num_hid, mb_train.num_labels])
    # TODO CHECK GRADIENTS HERE
    print("checking gradients...")
    # my_xman = mlp.my_xman
    # ad = Autograd(my_xman)
    # wengert_list = my_xman.operationSequence(my_xman.loss)
    # value_dict = my_xman.inputDict()
    # value_dict = ad.eval(wengert_list, value_dict)
    # gradients = ad.bprop(wengert_list, value_dict, loss=np.float_(1.))
    # epsilon = 1e-4
    # for rname in value_dict:
    #     if my_xman.isParam(rname):
    #         gradient = gradients[rname]
    #         for ind, value in np.ndenumerate(value_dict[rname]):
    #             value_dict[rname][ind] += epsilon
    #             J_upper = ad.eval(wengert_list, value_dict)['loss']
    #             value_dict[rname][ind] -= (2 * epsilon)
    #             J_lower = ad.eval(wengert_list, value_dict)['loss']
    #             value_dict[rname][ind] += epsilon
    #             gradient_approx = (J_upper - J_lower) / (2 * epsilon)
    #             np.testing.assert_almost_equal(gradient[ind], gradient_approx, decimal=3, err_msg='%s Did not pass grad check' % rname)
    print "done"

    # train
    print "training..."
    # get default data and params
    my_xman = mlp.my_xman
    value_dict = my_xman.inputDict()
    lr = init_lr
    train_loss = np.ndarray([0])
    min_valid_loss = np.inf
    ad = Autograd(my_xman)
    wengert_list = my_xman.operationSequence(my_xman.loss)

    training_time = []
    for i in range(epochs):
        start_time = time.time()
        for (idxs, e, l) in mb_train:
            # TODO prepare the input and do a fwd-bckwd pass over it and update the weights
            # e = np.array(e)
            e = np.reshape(e, (e.shape[0], -1))
            value_dict['x'] = e
            value_dict['y'] = l
            value_dict = ad.eval(wengert_list, value_dict)
            # save the train loss
            train_loss = np.append(train_loss, value_dict['loss'])
            gradients = ad.bprop(wengert_list, value_dict, loss=np.float_(1.))
            for rname in gradients:
                if my_xman.isParam(rname):
                    value_dict[rname] -= lr * gradients[rname]
            # write out the train loss
            np.save(train_loss_file, train_loss)
        end_time = time.time()
        training_time.append(end_time - start_time)

        # validate
        # for (idxs, e, l) in mb_valid:
        #     # TODO prepare the input and do a fwd pass over it to compute the loss
        #     e = np.reshape(e, (e.shape[0], -1))
        #     value_dict['x'] = e
        #     value_dict['y'] = l
        #     value_dict = ad.eval(wengert_list, value_dict)
        #     print("%sth epoch validation loss" % i, value_dict['loss'])
        #     # TODO compare current validation loss to minimum validation loss
        #     # and store params if needed
        #     if value_dict['loss'] < min_valid_loss:
        #         min_valid_loss = value_dict['loss']
        #         best_value_dict = deepcopy(value_dict)

    print "done"
    print("Average training time: ", np.mean(training_time))

    # for (idxs, e, l) in mb_test:
    #     # prepare input and do a fwd pass over it to compute the output probs
    #     e = np.reshape(e, (e.shape[0], -1))
    #     best_value_dict['x'] = e
    #     best_value_dict['y'] = l
    #     best_value_dict = ad.eval(wengert_list, best_value_dict)
    #     print("test loss")
    #     # TODO save probabilities on test set
    #     # ensure that these are in the same order as the test input
    #     output_probabilities = best_value_dict['p']
    #     np.save(output_file, output_probabilities)


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
