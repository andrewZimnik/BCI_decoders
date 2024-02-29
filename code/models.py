import numpy as np
import utils
import torch
from torch.utils.data import DataLoader


class Kalman(object):

    """
    Class for the Kalman filter object
    Hyperparameters:

    binWidth: duration of bins (in seconds)
    """

    def __init__(self, hyperparameters):
        self.bin_width = hyperparameters['bin_width']
        # hard-coding a sampling rate of 1000 Hz
        self.Ts = 0.001

    def fit(self, z, position, velocity):

        """
        Train the (standard) kalman filter
        :param z: list of spike times. Top level is trials, second level corresponds to neurons
        :param position: list of hand positions. Top level is trials, second level is a T x 2 matrix (x any y position)
        :param velocity: list of hand velocities. Top level is trials, second level is a T x 2 matrix (x any y velocity)
        :return: nothing. Stores learned parameters in self

        Learned Parameters
        A: state transition matrix. Determines (linear) dynamics of state variables
        H: observation matrix. Determines relationship between neural activity (z_t) and state (x_t)
        W: covariance matrix of state transition noise.
        Q: covariance matrix of observation noise
        P0: initial state covariance (assumed here to be 0).

        note: here, we initialize our estimate of the state (x_hat) with the true state at t=0.
        """

        # unpack some things
        bin_width = self.bin_width
        ts = self.Ts

        # grab some useful values from the inputs
        num_trials  = len(position)
        num_neurons = len(z[0])
        trial_duration = position[0].shape[0]*ts

        # calculate acceleration (from velocity)
        # the relevant function also does a bit of smoothing
        acceleration = utils.calculate_acceleration(velocity)

        # bin the spikes for each trial
        z = utils.bin_spikes(z, bin_width, trial_duration)

        # calculate the average kinematics within each bin
        bin_idx, avg_window = utils.calculate_bin_centers(bin_width, trial_duration, ts)
        binned_kinematics = [[utils.bin_kinematics(x, bin_idx, avg_window) for x in y] for y in [position, velocity, acceleration]]

        # shift around the training data so that 'trials' are the top element, and each entry is a K x T matrix
        kin_training = list(np.array(binned_kinematics).transpose(1, 0, 2, 3))
        kin_training = [np.array(x).transpose(1, 2, 0).reshape(-1, 6, order='F').T for x in kin_training]

        # add a row of ones to each trial to allow a constant offset for each neuron
        # number of bins per trial
        numBins = kin_training[0].shape[1]
        kin_training = [np.concatenate((x, np.ones((1, numBins))), axis=0) for x in kin_training]

        # convert spikes to large N x T matrix
        Z = np.array(z).transpose(2, 0, 1).reshape(-1, num_neurons, order='F').T

        # get a large array of kinematics (7 x T)
        X = np.array(kin_training).transpose(2, 0, 1).reshape(-1, 7, order='F').T

        # for each trial, calculate an x1 (:-1) and an x2 (1:)
        x1 = [x[:, :-1] for x in kin_training]
        x2 = [x[:, 1:] for x in kin_training]

        # make into giant arrays
        X1 = np.array(x1).transpose(2, 0, 1).reshape(-1, 7, order='F').T
        X2 = np.array(x2).transpose(2, 0, 1).reshape(-1, 7, order='F').T

        # # estimate A (transition matrix) and H (observation matrix) via simple regression
        A = X2 @ X1.T @ np.linalg.inv(X1 @ X1.T)
        H = Z @ X.T @ np.linalg.inv(X @ X.T)

        # estimate noise covariance matrices
        T1 = X1.shape[1]
        W = ((X2 - A @ X1) @ (X2 - A @ X1).T) / T1
        T2 = Z.shape[1]
        Q = ((Z - H @ X) @ (Z - H @ X).T) / T2

        # the initial (a priori) error covariance is just 0
        P0 = np.zeros((7, 7))

        # store all of the learned parameters
        self.A  = A
        self.H  = H
        self.W  = W
        self.Q  = Q
        self.P0 = P0


    def predict(self, z, position, velocity):

        """
        Use the trained filter parameters to predict kinematics from neural activity
        :param z: list of spike times. Top level is trials, second level corresponds to neurons
        :param position: list of hand positions. Top level is trials, second level is a T x 2 matrix (x any y position)
        :param velocity: list of hand velocities. Top level is trials, second level is a T x 2 matrix (x any y velocity)
        :return: X_hat: list of predicted kinematics; R2: list of R2 values for each trial (velocity)
        """

        # unpack stuff
        A  = self.A
        H  = self.H
        W  = self.W
        Q  = self.Q
        P0 =self.P0
        bin_width = self.bin_width
        ts = self.Ts

        # grab some useful values from the inputs
        num_trials  = len(position)
        num_neurons = len(z[0])
        trial_duration = position[0].shape[0]*ts

        # calculate acceleration (from velocity)
        # the relevant function also does a bit of smoothing
        acceleration = utils.calculate_acceleration(velocity)

        # bin the spikes for each trial
        Z = utils.bin_spikes(z, bin_width, trial_duration)

        # calculate the average kinematics within each bin
        bin_idx, avg_window = utils.calculate_bin_centers(bin_width, trial_duration, ts)
        binned_kinematics = [[utils.bin_kinematics(x, bin_idx, avg_window) for x in y] for y in [position, velocity, acceleration]]

        # shift around the training data so that 'trials' are the top element, and each entry is a K x T matrix
        kin_testing = list(np.array(binned_kinematics).transpose(1, 0, 2, 3))
        kin_testing = [np.array(x).transpose(1, 2, 0).reshape(-1, 6, order='F').T for x in kin_testing]

        # add a row of ones to each trial to allow a constant offset for each neuron
        # number of bins per trial
        numBins = kin_testing[0].shape[1]
        kin_testing = [np.concatenate((x, np.ones((1, numBins))), axis=0) for x in kin_testing]

        # initialize a list to hold estimated states
        X_hat = []

        # cycle through trials
        for trl in range(len(kin_testing)):

            # pull out the neural data for this trial
            z = np.array(Z[trl])

            # vector for a posteriori state estimate
            state_estimate = np.zeros((7, numBins)) + np.nan

            # initialize with the true state
            x_hat = np.copy(kin_testing[trl][:, 0][:, np.newaxis])
            P = np.copy(P0)
            state_estimate[:, 0] = x_hat[:, 0]

            # cycle through time steps
            for ii in range(1, numBins):

                # time update
                xHat_minus = A @ x_hat
                P_minus = A @ P @ A.T + W

                # measurement update
                K = P_minus @ H.T @ np.linalg.pinv(H @ P_minus @ H.T + Q)
                x_hat = xHat_minus + K @ (z[:, ii][:, np.newaxis] - H @ xHat_minus)
                state_estimate[:, ii] = np.copy(x_hat[:, 0])
                P = (np.eye(7) - K @ H) @ P_minus

            # add to list
            X_hat.append(state_estimate)

        # calculate R2 between actual and predicted velocity
        r2 = [utils.calculate_r2(kin_testing[ii][2:4, :].T, X_hat[ii][2:4, :].T) for ii in range(num_trials)]

        # return estimates
        return X_hat, r2


class Wiener(object):

    """
    class object for the wiener filter

    Hyperparameters:

    bin_width: duration of bins (in seconds)
    num_lagging_bins: number of preceding spike bins to include
    ridge_lambda: magnitude of ridge penalty

    """

    def __init__(self, hyperparameters):

        self.bin_width = hyperparameters['bin_width']
        self.num_lagging_bins = hyperparameters['num_lagging_bins']
        self.ridge_lam = hyperparameters['ridge_lambda']

        # hard-coding a sampling rate of 1000 Hz
        self.Ts = 0.001

    def fit(self, z, kinematics):

        """
        Train the wiener filter weights
        :param z: list of spike times. Top level is trials, second level corresponds to neurons. A row of ones is added
        to allow a constant offset for each neuron
        :param kinematics: list of hand kinematics. Top level is trials, second level is a K x T matrix of kinematics
        :return: nothing. Stores learned parameters in self

        Learned Parameters
        B: matrix of K+1 x N weights that predict kinematics from spiking activity.
        """

        # unpack some things
        bin_width = self.bin_width
        ts = self.Ts
        num_lagging_bins = self.num_lagging_bins
        ridge_lam = self.ridge_lam

        # grab some useful values from the inputs
        num_trials  = len(kinematics)
        num_neurons = len(z[0])
        trial_duration = kinematics[0].shape[0]*ts

        # bin the spikes for each trial
        binned_spikes = utils.bin_spikes(z, bin_width, trial_duration)

        # list of arrays (N x T)
        Z = [np.array(x) for x in binned_spikes]

        # number of bins
        num_bins = binned_spikes[0][0].shape[0]

        # calculate the average kinematics within each bin
        bin_idx, avg_window = utils.calculate_bin_centers(bin_width, trial_duration, ts)
        binned_kinematics = [utils.bin_kinematics(x, bin_idx, avg_window) for x in kinematics]
        binned_kinematics = [x.T for x in binned_kinematics]

        # concatenate the velocities
        Y = np.concatenate([y[:, num_lagging_bins - 1:] for y in binned_kinematics], axis=1)

        # pull out neural data
        X = [[z[:, num_lagging_bins - x:(num_bins + 1 - x)] for x in np.arange(1, num_lagging_bins + 1)] for z in Z]

        # reshape to proper size
        X = np.array(X).transpose(3, 0, 2, 1).reshape(-1, num_neurons * num_lagging_bins, order='F').T

        # add a row of ones to allow a constant offset
        X = np.concatenate((np.ones((1, X.shape[1])), X), axis=0)

        # regress
        B = Y @ X.T @ np.linalg.pinv(X @ X.T + np.eye(X.shape[0]) * ridge_lam)

        # save result
        self.B = B


    def predict(self, z, kinematics):

        """
        Predict kinematics from neural activity
        :param z: list of spike times
        :param kinematics: list of K x T arrays of kinematics
        :return: Y_hat: list of K x T arrays of predicted kinematics
        """

        # unpack some things
        bin_width = self.bin_width
        ts = self.Ts
        num_lagging_bins = self.num_lagging_bins

        B = self.B


        # grab some useful values from the inputs
        num_trials  = len(kinematics)
        num_kins    = kinematics[0].shape[0]
        num_neurons = len(z[0])
        trial_duration = kinematics[0].shape[0]*ts

        # bin the spikes for each trial
        binned_spikes = utils.bin_spikes(z, bin_width, trial_duration)

        # list of arrays (N x T)
        Z = [np.array(x) for x in binned_spikes]

        # number of bins
        num_bins = binned_spikes[0][0].shape[0]

        # calculate the average kinematics within each bin
        bin_idx, avg_window = utils.calculate_bin_centers(bin_width, trial_duration, ts)
        binned_kinematics = [utils.bin_kinematics(x, bin_idx, avg_window) for x in kinematics]
        binned_kinematics = [x.T for x in binned_kinematics]

        # concatenate the velocities
        Y = np.concatenate([y[:, num_lagging_bins - 1:] for y in binned_kinematics], axis=1)

        # pull out neural data
        X = [[z[:, num_lagging_bins - x:(num_bins + 1 - x)] for x in np.arange(1, num_lagging_bins + 1)] for z in Z]

        # reshape to proper size
        X = np.array(X).transpose(3, 0, 2, 1).reshape(-1, num_neurons * num_lagging_bins, order='F').T

        # add a row of ones to allow a constant offset
        X = np.concatenate((np.ones((1, X.shape[1])), X), axis=0)

        # predict
        Y_hat = B @ X

        # reshape prediction into a T x K x trials array and pack into lists
        Y_hat_rs = Y_hat.T.reshape(-1, num_trials, 2, order='F').transpose(0, 2, 1)
        Y_Hat_list = []
        for ii in range(num_trials):
            Y_Hat_list.append(Y_hat_rs[:, :, ii])

        # calculate R2
        r2 = [utils.calculate_r2(binned_kinematics[ii][:, num_lagging_bins-1:].T, Y_Hat_list[ii]) for ii in range(num_trials)]

        # return values
        return Y_Hat_list, r2

class FF_network(object):

    """
    class object for the feed forward network.
    The network is composed of ReLU units, and is trained with dropout.

    Hyperparameters:

    bin_width: duration of bins (in seconds)
    num_lagging_bins: number of preceding spike bins to include
    num_hidden_layers: number of layers between input and output layers
    num_units_per_layer: self-explanatory

    """

    def __init__(self, hyperparameters):
        self.bin_width = hyperparameters['bin_width']
        self.num_lagging_bins = hyperparameters['num_lagging_bins']
        self.num_hidden_layers = hyperparameters['num_hidden_layers']
        self.num_units_per_layer = hyperparameters['num_units_per_layer']
        self.allLoss = []
        self.net = None

        # hard-coding a sampling rate of 1000 Hz
        self.Ts = 0.001

    def fit(self, z, kinematics, fit_parameters):

        """
        Train the wiener filter weights
        :param z: list of spike times. Top level is trials, second level corresponds to neurons. A row of ones is added
        to allow a constant offset for each neuron
        :param kinematics: list of hand kinematics. Top level is trials, second level is a K x T matrix of kinematics
        :param fit_parameters: dictionary containing  model training parameters:
            batch_size: number of trials to include in the traning batch
            num_epochs: number of training epochs
            dropout_p: fraction of neurons (within a layer) that are 'turned off' during that batch.
        :return: loss during training. Stores learned parameters in self

        Learned Parameters
        trained network is saved in self
        """

        # unpack some things
        bin_width = self.bin_width
        ts = self.Ts
        num_lagging_bins = self.num_lagging_bins
        num_hidden_layers = self.num_hidden_layers
        num_units_per_layer = self.num_units_per_layer
        batch_size = fit_parameters['batch_size']
        num_epochs = fit_parameters['num_epochs']
        dropout_p  = fit_parameters['dropout_p']

        # grab some useful values from the inputs
        num_trials = len(kinematics)
        num_neurons = len(z[0])
        trial_duration = kinematics[0].shape[0] * ts
        input_dim = num_neurons * num_lagging_bins
        output_dim = kinematics[0].shape[1]

        # bin the spikes for each trial
        binned_spikes = utils.bin_spikes(z, bin_width, trial_duration)

        # list of arrays (N x T)
        Z_list = [np.array(x) for x in binned_spikes]

        # number of bins
        num_bins = binned_spikes[0][0].shape[0]

        # calculate the average kinematics within each bin
        bin_idx, avg_window = utils.calculate_bin_centers(bin_width, trial_duration, ts)
        binned_kinematics = [utils.bin_kinematics(x, bin_idx, avg_window) for x in kinematics]
        binned_kinematics = [x.T for x in binned_kinematics]

        # pull out the kinematics (starting with the earliest time bin possible, given the need to lag the neural activity)
        Y = [x[:, num_lagging_bins - 1:].T for x in binned_kinematics]

        # pull out neural data
        X = [np.array([z[:, num_lagging_bins - x:(num_bins + 1 - x)] for x in np.arange(1, num_lagging_bins + 1)])
             .transpose(2, 1, 0).reshape(-1, num_neurons * num_lagging_bins, order='F') for z in Z_list]

        # define the datatset and the data loader
        dataset = utils.FF_dataset(data=X, targets=Y)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # initialize the network
        net = utils.FF_network(input_dim=input_dim, num_hidden_layers=num_hidden_layers,
                               num_units_per_layer=num_units_per_layer, output_dim=output_dim, dropout_p=dropout_p)

        # define an optimizer (using default ADAM parameters)
        # get a list of the named parameters
        optimizer = torch.optim.Adam(net.parameters())

        # use a simple MSE loss function
        criterion = torch.nn.MSELoss()

        # train the network
        allLoss = []

        for epoch in range(num_epochs):

            for ii, (x, y) in enumerate(dataloader):

                # make sure we're in training mode (the drop-out layers are active)
                net.train()

                # clear the gradient
                optimizer.zero_grad()

                # forward pass
                out = net(x)

                # calculate loss
                loss = criterion(out, y)
                allLoss.append(loss.detach().numpy())

                # calculate gradient
                loss.backward()

                # update parameters
                optimizer.step()

        # save the loss
        self.allLoss = allLoss

        # and the network
        self.net = net

    def predict(self, z, kinematics):

        """
        Train the wiener filter weights
        :param z: list of spike times. Top level is trials, second level corresponds to neurons. A row of ones is added
        to allow a constant offset for each neuron
        :param kinematics: list of hand kinematics. Top level is trials, second level is a K x T matrix of kinematics
        :return: predicted output and R2

        Learned Parameters
        trained network is saved in self
        """

        # unpack some things
        bin_width = self.bin_width
        ts = self.Ts
        num_lagging_bins = self.num_lagging_bins
        net = self.net

        # grab some useful values from the inputs
        num_trials = len(kinematics)
        num_neurons = len(z[0])
        trial_duration = kinematics[0].shape[0] * ts

        # bin the spikes for each trial
        binned_spikes = utils.bin_spikes(z, bin_width, trial_duration)

        # list of arrays (N x T)
        Z_list = [np.array(x) for x in binned_spikes]

        # number of bins
        num_bins = binned_spikes[0][0].shape[0]

        # calculate the average kinematics within each bin
        bin_idx, avg_window = utils.calculate_bin_centers(bin_width, trial_duration, ts)
        binned_kinematics = [utils.bin_kinematics(x, bin_idx, avg_window) for x in kinematics]
        binned_kinematics = [x.T for x in binned_kinematics]

        # pull out the kinematics (starting with the earliest time bin possible, given the need to lag the neural activity)
        Y = [x[:, num_lagging_bins - 1:].T for x in binned_kinematics]

        # pull out neural data
        X = [np.array([z[:, num_lagging_bins - x:(num_bins + 1 - x)] for x in np.arange(1, num_lagging_bins + 1)])
             .transpose(2, 1, 0).reshape(-1, num_neurons * num_lagging_bins, order='F') for z in Z_list]

        # put the neural data in a tensor
        X_test_tensor = torch.Tensor(np.array(X))

        # push testing spiking data thorugh the network
        net.eval()
        out = net(X_test_tensor)
        out = out.detach().numpy()

        # place output in list
        out_list = [out[ii, :, :] for ii in range(out.shape[0])]

        # calculate R2
        from utils import calculate_r2
        r2 = [calculate_r2(Y[ii], out[ii]) for ii in range(len(out_list))]

        return out_list, r2















