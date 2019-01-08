# Leaky integrator model of Echo State Network
import numpy as np
import networkx as nx
import seaborn as sns
from matplotlib import pyplot as plt

import argparse

import make_modular_networks
import make_layered_networks
import generate_networks

N_NODES = 200
SPECT_RADIUS = 0.9

a = 1
time_scale = np.ones(N_NODES) * a
trainlen = 5000
future = 1000
buffer = 100


def correct_dimensions(s, targetlength):
    if s is not None:
        s = np.array(s)
        if s.ndim == 0:
            s = np.array([s] * targetlength)
        elif s.ndim == 1:
            if not len(s) == targetlength:
                raise ValueError("arg must have length " + str(targetlength))
        else:
            raise ValueError("Invalid argument")
    return s


def identity(x):
    return x


def step_function(x):
    if x > 0.5:
        return 1
    else:
        return 0


def sigmoid(x):
    return 1 / (1 + np.exp(-10 * x + 1))


def memory_capacity(L, buffer, data, output_data):
    MC = 0
    for k in range(L):
        cov_matrix = np.cov(
            np.array([data[trainlen + buffer - (k + 1): trainlen + buffer - (k + 1) + 1000], output_data.T[k]]))
        MC_k = cov_matrix[0][1] ** 2
        MC_k = MC_k / (np.var(data[trainlen + buffer:]) * np.var(output_data.T[k]))
        MC += MC_k
    return MC


class LI_ESN_internal:

    def __init__(self, n_inputs, n_outputs, n_reservoir=200, W=None, W_in=None,
                 noise=0.001, input_shift=None,
                 input_scaling=None, feedback_scaling=None,
                 teacher_scaling=None, teacher_shift=None,
                 out_activation=identity, inverse_out_activation=identity,
                 random_state=None, time_scale=None):
        # check for proper dimensionality of all arguments and write them down.
        self.n_inputs = n_inputs
        self.n_reservoir = n_reservoir
        self.n_outputs = n_outputs
        self.noise = noise
        self.input_shift = correct_dimensions(input_shift, n_inputs)
        self.input_scaling = correct_dimensions(input_scaling, n_inputs)

        self.teacher_scaling = teacher_scaling
        self.teacher_shift = teacher_shift

        self.out_activation = out_activation
        self.inverse_out_activation = inverse_out_activation
        self.random_state = random_state
        self.time_scale = time_scale
        self.W = W
        self.W_in = W_in

        # the given random_state might be either an actual RandomState object,
        # a seed or None (in which case we use numpy's builtin RandomState)
        if isinstance(random_state, np.random.RandomState):
            self.random_state_ = random_state
        elif random_state:
            try:
                self.random_state_ = np.random.RandomState(random_state)
            except TypeError as e:
                raise Exception("Invalid seed: " + str(e))
        else:
            self.random_state_ = np.random.mtrand._rand

    def _scale_inputs(self, inputs):
        """for each input dimension j: multiplies by the j'th entry in the
        input_scaling argument, then adds the j'th entry of the input_shift
        argument."""
        if self.input_scaling is not None:
            inputs = np.dot(inputs, np.diag(self.input_scaling))
        if self.input_shift is not None:
            inputs = inputs + self.input_shift
        return inputs

    def _update(self, state, input_pattern):
        # leaky integrator model:
        # it can adjust timescales for each neurons.
        preactivation = (np.dot(self.W, state) + np.dot(self.W_in, input_pattern))
        state = (1 - self.time_scale) * state + self.time_scale * np.tanh(preactivation)
        # state = (1 - self.time_scale) * state + self.time_scale * sigmoid(preactivation)
        return state + self.noise * self.time_scale * (self.random_state_.rand(self.n_reservoir) - 0.5)

    def calc_lyapunov_exp(self, inputs, initial_distance, n):
        if inputs.ndim < 2:
            inputs = np.reshape(inputs, (len(inputs), -1))
        states1 = np.zeros((inputs.shape[0], self.n_reservoir))
        states2 = np.zeros((inputs.shape[0], self.n_reservoir))
        transient = min(int(inputs.shape[0] / 10), 100)
        for i in range(1, transient):
            states1[i, :] = self._update(states1[i - 1], inputs[i, :])
        states2[transient - 1, :] = states1[transient - 1, :]
        states2[transient - 1, n] = states2[transient - 1, n] + initial_distance
        gamma_k_list = []
        for k in range(transient, inputs.shape[0]):
            states1[k, :] = self._update(states1[k - 1], inputs[k, :])
            states2[k, :] = self._update(states2[k - 1], inputs[k, :])
            gamma_k = np.linalg.norm(states2[k, :] - states1[k, :])
            gamma_k_list.append(gamma_k / initial_distance)
            states2[k, :] = states1[k, :] + (initial_distance / gamma_k) * (states2[k, :] - states1[k, :])
        lyapunov_exp = np.mean(np.log(gamma_k_list))
        return lyapunov_exp

    def fit(self, inputs, outputs):
        if inputs.ndim < 2:
            inputs = np.reshape(inputs, (len(inputs), -1))
        if outputs.ndim < 2:
            outputs = np.reshape(outputs, (len(outputs), -1))
        inputs_scaled = self._scale_inputs(inputs)
        teachers_scaled = outputs

        # step the reservoir through the given input,output pairs:
        states = np.zeros((inputs.shape[0], self.n_reservoir))
        for n in range(1, inputs.shape[0]):
            states[n, :] = self._update(states[n - 1], inputs_scaled[n, :])
        transient = min(int(inputs.shape[0] / 10), 100)
        extended_states = np.hstack((states, inputs_scaled))

        self.W_out = np.dot(np.linalg.pinv(extended_states[transient:, :]), teachers_scaled[transient:, :]).T
        # print(self.W_out.shape)

        # remember the last state for later:
        self.laststate = states[-1, :]
        self.lastinput = inputs[-1, :]
        self.lastoutput = teachers_scaled[-1, :]

        # apply learned weights to the collected states:
        pred_train = np.dot(extended_states, self.W_out.T)
        return pred_train

    def predict(self, inputs, continuation=True):
        if inputs.ndim < 2:
            inputs = np.reshape(inputs, (len(inputs), -1))
        n_samples = inputs.shape[0]

        if continuation:
            laststate = self.laststate
            lastinput = self.lastinput
            lastoutput = self.lastoutput
        else:
            laststate = np.zeros(self.n_reservoir)
            lastinput = np.zeros(self.n_inputs)
            lastoutput = np.zeros(self.n_outputs)

        inputs = np.vstack([lastinput, inputs])
        states = np.vstack(
            [laststate, np.zeros((n_samples, self.n_reservoir))])
        outputs = np.vstack(
            [lastoutput, np.zeros((n_samples, self.n_outputs))])

        for n in range(n_samples):
            states[n + 1, :] = self._update(states[n, :], inputs[n + 1, :])
            outputs[n + 1, :] = np.dot(self.W_out, np.concatenate([states[n + 1, :], inputs[n + 1, :]]))

        return self.out_activation(outputs[1:])
        # print(outputs[1:])
        # return np.heaviside(outputs[1:]-0.5, 0)*0.3


def calculate_memory_capacity(mu, r_sig, average_degree, num_community, is_layered):
    memory_capacity_list = []
    for k in range(30):
        if is_layered:
            W = make_layered_networks.make_layered_network(N_NODES, average_degree, num_community, mu)
        else:
            W = make_modular_networks.make_modular_network(N_NODES, average_degree, num_community, mu)
        W_IN = (np.random.rand(N_NODES, 1) * 2 - 1) * 0.1
        W_IN[int(N_NODES / num_community):] = 0
        radius = np.max(np.abs(np.linalg.eigvals(W)))
        spectral_radius = SPECT_RADIUS
        W = W * (spectral_radius / radius)

        # delayed signal
        L = int(N_NODES * 2)
        buffer = L
        total_len = future + trainlen + buffer
        data = [0 if np.random.rand() < 0.5 else 1 for i in range(total_len)]
        data = np.array(data) * r_sig
        # print(data)

        esn = LI_ESN_internal(n_inputs=1,
                              n_outputs=L,
                              n_reservoir=N_NODES,
                              W=W,
                              W_in=W_IN,
                              noise=0,
                              time_scale=time_scale)

        target = np.zeros((total_len - buffer, L))
        for i in range(L):
            target.T[i][:] = data[buffer - (i + 1):-(i + 1)]

        esn.fit(data[buffer:trainlen + buffer], target[:trainlen])

        prediction = esn.predict(data[trainlen + buffer:])
        memory_capacity_result = memory_capacity(L, buffer, data, prediction)
        memory_capacity_list.append(memory_capacity_result)
    return np.mean(memory_capacity_list), np.std(memory_capacity_list)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--average_degree', type=int, default=10)
    parser.add_argument('--r_sig', type=float, default=0.8)
    parser.add_argument('--num_community', type=float, default=5)
    parser.add_argument('--layered', type=bool, default=False)
    args = parser.parse_args()

    print(args)

    mu_list = np.arange(0, 0.10, 0.001)
    mc_mean_list = []
    mc_std_list = []
    for mu in mu_list:
        mc_mean, mc_std = calculate_memory_capacity(mu=mu, r_sig=args.r_sig,
                                                    average_degree=args.average_degree,
                                                    num_community=args.num_community,
                                                    is_layered=args.layered)
        print(mu, mc_mean, mc_std)
        mc_mean_list.append(mc_mean)
        mc_std_list.append(mc_std)
    mc_mean_list = np.array(mc_mean_list)
    mc_std_list = np.array(mc_std_list)
    if args.layered:
        np.savetxt('./mc-result/result_rsig_{}_averagedegree_{}_ncom_{}_layered.out'.
                   format(args.r_sig, args.average_degree, args.num_community), (mu_list, mc_mean_list, mc_std_list))
    else:
        np.savetxt('./mc-result/result_rsig_{}_averagedegree_{}_ncom_{}.out'.
                   format(args.r_sig, args.average_degree, args.num_community), (mu_list, mc_mean_list, mc_std_list))
