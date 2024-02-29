import numpy as np
import scipy
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader



def bin_spikes(spike_times, bin_width, trial_dur):

    """
    :param spike_times: nested list of spike times (relative to start of trial)
    :param bin_width: width of the bins (in sec)
    :param trial_dur: duration of each trial (in seconds)
    :return: spike counts in each bin (of size int(trialDuration/binWidth) x 1)
    """

    # define our bin edges
    bin_edges = np.arange(0, trial_dur+0.001, bin_width)

    return [[np.histogram(x, bin_edges)[0] for x in y] for y in spike_times]


def calculate_acceleration(velocity):

    """
    :param velocity: list of hand velocities (x and y).
    :return: filtered hand acceleration. Padded to be the same length as velocity
    """

    # take temporal derivative (assuming a 1000 Hz sampling rate)
    acceleration = [np.diff(x, 1, axis=0)/0.001 for x in velocity]

    # pad acceleration
    acceleration = [np.concatenate((x[0, :][np.newaxis, :], x), axis=0) for x in acceleration]

    # filter acceleration to get rid of high frequency noise
    # hard-coding filter parameters just for convenience (we're not going to be playing around with this
    sos = scipy.signal.butter(2, 100, btype='low', output='sos', fs=1000)

    return [scipy.signal.sosfiltfilt(sos, x, axis=0) for x in acceleration]


def calculate_bin_centers(bin_width, trial_duration, ts):

    """
    :param bin_width: duration of each bin (in seconds)
    :param trial_duration: trial duration (in seconds)
    :param ts: sampling interval (in seconds)
    :return: bin_idx: indexes for the center of each bin; avg_window: samples over which to average
    """

    # first, define the bin centers (in seconds)
    bin_centers = np.arange(bin_width / 2, trial_duration, bin_width)

    # find the indices for these times
    time = np.arange(0, trial_duration, ts)
    bin_idx = np.nonzero(np.isin((time * 1000).astype('int'), (bin_centers * 1000).astype('int')))[0]

    # window for calculating average kinematics
    avg_window = np.arange(-bin_width / 2 / ts, bin_width / 2 / ts).astype('int')

    return bin_idx, avg_window


def bin_kinematics(kinematics, bin_centers, bin_window):

    """
    :param kinematics: vector (possibly 2d) of kinematics
    :param bin_centers: vector of indices
    :param bin_window: width of averaging window (in samples)
    :return: list of averages
    """

    kin_avg = []
    for ii in bin_centers:
        kin_avg.append(np.mean(kinematics[ii+bin_window, :], axis=0))
    return np.array(kin_avg)


def spiketimes_to_spiketrains(spiketimes,trial_duration):

    """
    :param spiketimes: list of np array containing spike times (in seconds) relative to the start of the trial
    :param trial_duration: duration of each trial (in seconds)
    :return: N x T array of spikes
    """

    # initialize a N x T matrix
    n = len(spiketimes)
    t = np.round(trial_duration*1000)
    spikes = np.zeros((int(n), int(t)+1))

    # convert the spike times to indices
    s_idx = [np.unique((np.round(x * 1000)).astype('int')) for x in spiketimes]

    # cycle through neurons, adding spikes where necessary
    for ii in range(n):
        spikes[ii, s_idx[ii]] = 1

    # trim the final ms
    return spikes[:, :-1]


def bin_spikes_SP(spikes, bin_size):
    """
    Bin spikes in time.

    Inputs
    ------
    spikes: numpy array of spikes (neurons x time)

    bin_size: number of time points to pool into a time bin

    Outputs
    -------
    S: numpy array of spike counts (neurons x bins)

    """

    # Get some useful constants.
    [N, n_time_samples] = spikes.shape
    K = int(n_time_samples / bin_size)  # number of time bins

    # Count spikes in bins.
    S = np.empty([N, K])
    for k in range(K):
        S[:, k] = np.sum(spikes[:, k * bin_size:(k + 1) * bin_size], axis=1)

    return S


def calculate_r2(y, y_hat):

    """
    :param y: T x K matrix of target values
    :param y_hat: T x K matrix of predicted values
    :return:
    """

    return 1 - ((np.linalg.norm(y - y_hat, ord='fro')**2) / (np.linalg.norm(y, ord='fro')**2))


def zero_order_hold_upsample(x, bin_width):

    """
    Performs upsampling via a zero-order hold. (samples are head constant between updates)
    Note that (depending on the bin width), x_us will likely be longer than the original data.
    Just truncate the final samples as necessary
    :param x: T x K array of downsampled data
    :param bin_width: duration of each bin (in samples)
    :return: (T*bin_width) x K array of upsampled data
    """

    # number of variables
    if np.ndim(x) == 1:
        num_vars = 1
    else:
        num_vars = x.shape[1]

    # perform interpolation
    x_us = np.tile(x.reshape(-1, 1, order='F'), (1, bin_width)).reshape(-1, 1).reshape(-1, num_vars, order='F')

    return x_us


class FF_dataset(Dataset):

    """
    custom dataset class for training the feedforward network
    Parameters:
        data: list of T x (N x lagBins) arrays containing spike counts
        targets: list of T x K arrays containing kinematic data
    """

    def __init__(self, data, targets):
        self.data    = torch.Tensor(np.array(data))
        self.targets = torch.Tensor(np.array(targets))

    def __getitem__(self, index):
        x = self.data[index]
        y = self.targets[index]

        return x, y

    def __len__(self):
        return len(self.data)


class FF_network(nn.Module):

    """
    FF network model. Model is a feedforward network composed of ReLU units.
    Model is trained with dropout. No other regularization

    Parameteters:
        input_dim: number of input units
        num_hidden_layers: number of layers between input and output layers
        num_units_per_layer: self explanatory
        output_dim: number of readout units (linear)
        dropout_p: fraction of units within a given layer that are 'turned off' for one batch.
        This layer is inactivated (i.e., there is no dropout) during model evaluation.
    """

    def __init__(self, input_dim, num_hidden_layers, num_units_per_layer, output_dim, dropout_p):
        super().__init__()

        # define the nonlinearity
        self.nonLinearity = nn.ReLU()

        # define the input layer
        self.fc1 = nn.Linear(input_dim, num_units_per_layer)

        # define the hidden layers
        self.fc2 = nn.ModuleList()
        for ii in range(num_hidden_layers):
            self.fc2.append(nn.Linear(num_units_per_layer, num_units_per_layer))

        # define the output layer
        self.fc3 = nn.Linear(num_units_per_layer, output_dim)

        # define our dropout layer that will sit before each hidden layer
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, x):

        # input layer
        out = self.nonLinearity(self.fc1(x))

        # hidden layers
        for fc_hidden in self.fc2:
            out = self.dropout(out)
            out = self.nonLinearity(fc_hidden(out))

        # output layer
        out = self.fc3(out)

        return out










