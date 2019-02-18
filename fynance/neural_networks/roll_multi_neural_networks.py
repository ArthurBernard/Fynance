#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Built-in packages

# External packages
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

# Internal packages
from fynance.backtest.dynamic_plot_backtest import DynaPlotBackTest
from fynance.neural_networks.roll_neural_network import RollNeuralNet

# Set plot style
plt.style.use('seaborn')


class RollMultiNeuralNet(RollNeuralNet):
    """ Rolling Multi Neural Networks object allow you to train several 
    neural networks along training periods (from t - n to t) and predict 
    along testing periods (from t to t + s) and roll along this time axis.

    Attributes
    ----------
    y : np.ndarray[np.float32, ndim=2] with shape=(T, 1)
        Target to estimate or predict.
    X : np.ndarray[np.float32, ndim=2] with shape=(T, N)
        Features (inputs).
    NN : list of keras.Model
        Neural network models to train and predict.
    y_train : np.ndarray[np.float64, ndim=1]
        Prediction on training set.
    y_estim : np.ndarray[np.float64, ndim=1]
        Prediction on estimating set.


    Methods
    -------
    run(y, X, NN, plot_loss=True, plot_perf=True, x_axis=None)
        Train several rolling neural networks along pre-specified training 
        period and predict along test period. Display loss and performance 
        if specified.
    __call__(y, X, NN, start=0, end=1e8, x_axis=None)
        Callable method to set target and features data, neural network 
        object (Keras object is prefered).
    __iter__()
        Train and predict along time axis from day number n to last day 
        number T and by step of size s period. 
    plot_loss(self, f, ax)
        Plot loss function
    plot_perf(self, f, ax)
        Plot perfomances.

    See Also
    --------
    RollNeuralNet, RollAggrMultiNeuralNet, RollMultiRollNeuralNet

    """
    def __call__(self, y, X, NN, start=0, end=1e8, x_axis=None):
        """ Callable method to set terget and features data, neural network 
        object (Keras object is prefered).

        Parameters
        ----------
        y : np.ndarray[ndim=1, dtype=np.float32]
            Target to predict.
        X : np.ndarray[ndim=2, dtype=np.float32]
            Features data.
        NN : list of keras.engine.training.Model
            Neural network models.
        start : int, optional
            Starting observation, default is first one.
        end : int, optional
            Ending observation, default is last one.
        x_axis : np.ndarray[ndim=1], optional
            X-Axis to use for the backtest.

        Returns
        -------
        rmnn : RollMultiNeuralNet

        """
        RollNeuralNet.__call__(
            self, y, X, NN, start=start, end=end, x_axis=x_axis
        )
        # Set neural network model 
        if isinstance(NN, list):
            self.NN = NN
            self.n_NN = len(NN)
        else:
            self.NN = [NN]
            self.n_NN = 1
        
        return self
        
    def __iter__(self):
        """ Set iterative method """
        self.y_train = np.zeros([self.T, self.n_NN])
        self.y_estim = np.zeros([self.T, self.n_NN])
        k = self.params['epochs'] * (self.T - self.t) // self.s
        self.loss_train = np.ones([k, self.n_NN])
        self.loss_estim = np.ones([k, self.n_NN])
        return self
    
    def __next__(self):
        """ Incrementing method """
        # Incremant time
        self.t += self.s
        t = self.t
        if self.t >= self.T:
            raise StopIteration
        
        # Splitting
        subtrain_X = self.X[t - self.n: t, :]
        subtrain_y = self.f(self.y[t - self.n: t, :])
        subestim_X = self.X[t: t + self.s, :]
        subestim_y = self.f(self.y[t: t + self.s, :])
        
        # TODO : asynchronize this loop
        for i in range(self.n_NN):
            # Training
            self.y_train[t - self.s: t, i] = self._train(
                y=subtrain_y, X=subtrain_X, i=i, 
                val_set=(subestim_X, subestim_y)
            )
            
            # Estimating
            self.y_estim[t: t + self.s, i] = self.NN[i].predict(
                subestim_X
            ).flatten()
        
        return self.y_train[t - self.s: t, :], self.y_estim[t: t + self.s, :]
    
    def _train(self, y, X, i, val_set=None):
        """ Train method and return prediction on training set """
        # Set parameters
        k = self.params['epochs'] * ((self.t - self.n) // self.s - 1)
        k_1 = self.params['epochs'] * (self.t - self.n) // self.s
        
        # Fit loss
        hist = self.NN[i].fit(
            x=X, y=y, validation_data=val_set, **self.params
        )
        self.loss_train[k: k_1, i] = hist.history['loss']
        self.loss_estim[k: k_1, i] = hist.history['val_loss']
        
        # Return prediction
        return self.NN[i].predict(
            X[-self.s: ], verbose=self.params['verbose']
        ).flatten()
        
    def run(self, y, X, NN, plot_loss=True, plot_perf=True, x_axis=None):
        """ Train several rolling neural networks along pre-specified train 
        period and predict along test period. Display loss and performance 
        if specified.
        
        Parameters
        ----------
        y : np.ndarray[np.float32, ndim=2] with shape=(T, 1)
            Time series of target to estimate or predict.
        X : np.ndarray[np.float32, ndim=2] with shape=(T, N)
            Several time series of features.
        NN : keras.Model or list of keras.Model
            Neural networks to train and predict.
        plot_loss : bool, optional
            If true dynamic plot of loss function, default is True.
        plot_perf : bool, optional
            If true dynamic plot of strategy performance, default is True.
        x_axis : list or array, optional
            x-axis to plot (e.g. list of dates).

        Returns
        -------
        rmnn : RollMultiNeuralNet

        """
        if isinstance(NN, list):
            self.n_NN = len(NN)
        else:
            self.n_NN = 1

        # Set perf and loss arrays
        self.perf_train = self.V0 * np.ones([y.size, self.n_NN])
        self.perf_estim = self.V0 * np.ones([y.size, self.n_NN])

        # Set axes and figure
        f, ax_loss, ax_perf = self._set_figure(plot_loss, plot_perf)

        # Start Rolling Neural Network
        for pred_train, pred_estim in self(y, X, NN, x_axis=x_axis):
            t, s = self.t, self.s
            
            # Set performances of training period
            returns = np.sign(pred_train) * y[t - s: t]
            cum_ret = np.exp(np.cumsum(returns, axis=0))
            self.perf_train[t - s: t] = self.perf_train[t - s - 1] * cum_ret

            # Set performances of estimated period
            returns = np.sign(pred_estim) * y[t: t + s]
            cum_ret = np.exp(np.cumsum(returns, axis=0))
            self.perf_estim[t: t + s] = self.perf_estim[t - 1] * cum_ret

            # Plot loss and perf
            self._dynamic_plot(f, ax_loss=ax_loss, ax_perf=ax_perf)

        return self