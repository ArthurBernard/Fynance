#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Built-in packages

# External packages
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

# Internal packages

# Set plot style
plt.style.use('seaborn')


class RollNeuralNet:
    """
    Rolling Neural Network object allow you to train your neural network along
    training periods (from t - n to t) and predict along testing periods (from
    t to t + s) and roll along this time axis.

    Attribute
    ---------
    :y: np.ndarray[np.float32, ndim=2] with shape=(T, 1)
        Target to estimate or predict.
    :X: np.ndarray[np.float32, ndim=2] with shape=(T, N)
        Features (inputs).
    :NN: keras.Model
        Neural network to train and predict.
    :y_train: np.ndarray[np.float64, ndim=1]
        Prediction on training set.
    :y_estim: np.ndarray[np.float64, ndim=1]
        Prediction on estimating set.


    Methods
    -------
    :run: Train the rolling neural network along pre-specified train period 
    and predict along test period. Display loss and performance if specified.
    :__iter__: Train and predict along time axis from day number n to last day 
    number T and by step of size s period. 

    TODO:
    3 - Manager of models (cross val to aggregate or choose signal's model).
    """
    def __init__(
            self, train_period=252, estim_period=63, value_init=100, 
            dummify_target=True, params=None, init_params=None
        ):
        """ Init method """
        self.n = train_period
        self.s = estim_period
        self.V0 = value_init
        self._set_parameters(params, init_params)
        if dummify_target:
            self.f = np.sign
        else:
            self.f = lambda x: x
    
    def __call__(self, y, X, NN, start=0, end=1e6, x_axis=None):
        """ Callable method """
        # Set target and features
        self.y = y
        self.X = X

        # Set neural network
        self.NN = NN

        # Set bound of period
        self.t = max(self.n, start)
        self.T = min(y.size, end)
        if x_axis is None:
            self.x_axis = range(self.T)
        else:
            self.x_axis = x_axis

        return self
        
    def __iter__(self):
        """ Set iterative method """
        self.y_train = np.zeros([self.T])
        self.y_estim = np.zeros([self.T])
        self.hist = None
        return self
    
    def __next__(self):
        """ Incrementing method """
        # Incremant time
        self.t += self.s
        t = self.t
        if self.t > self.T:
            raise StopIteration
        
        # Splitting
        subtrain_X = self.X[t - self.n: t, :]
        subtrain_y = self.f(self.y[t - self.n: t, :])
        subestim_X = self.X[t: t + self.s, :]
        subestim_y = self.f(self.y[t: t + self.s, :])
        
        # Training
        self.y_train[t - self.s: t] = self._train(
            y=subtrain_y, X=subtrain_X, val_set=(subestim_X, subestim_y)
        )
        
        # Estimating
        self.y_estim[t: t + self.s] = self.NN.predict(subestim_X).flatten()
        
        return self.y_train[t - self.s: t], self.y_estim[t: t + self.s]
    
    def _train(self, y, X, val_set=None):
        """ Train method and return prediction on training set """
        if self.hist is None:
            self.hist = self.NN.fit(
                x=X, y=y, validation_data=val_set, **self.init_params
            )
        else:
            hist = self.NN.fit(
                x=X, y=y, validation_data=val_set, **self.params
            )
            for key, arg in hist.history.items():
                self.hist.history[key] += arg
        return self.NN.predict(
            X[-self.s: ], verbose=self.params['verbose']
        ).flatten()
    
    def _set_parameters(self, params, init_params=None): 
        """ 
        Setting parameters for fit method of neural network. If is None set as 
        default parameters: batch_size=train_period, epochs=1, shuffle=False
        and no verbosity.

        Parameters
        ----------
        :params: dict
        :init_params: dict 
        """
        self.params = {
            'batch_size': self.n, 'epochs': 1, 'shuffle': False, 'verbose': 0
        } if params is None else params
        self.init_params = self.params if init_params is None else init_params
        return self
        
    def run(self, y, X, NN, plot_loss=True, plot_perf=True, x_axis=None):
        """ 
        Train the rolling neural network along pre-specified train period and 
        predict along test period. Display loss and performance if specified.
        
        Parameters
        ----------
        :y: np.ndarray[np.float32, ndim=2] with shape=(T, 1)
            Time series of target to estimate or predict.
        :X: np.ndarray[np.float32, ndim=2] with shape=(T, N)
            Several time series of features.
        :NN: keras.Model
            Neural network to train and predict.
        :plot_loss: bool
            If true dynamic plot of loss function.
        :plot_perf: bool
            If true dynamic plot of strategy performance.
        :x_axis: list or array
            x-axis to plot (e.g. list of dates).

        Returns
        -------
        :self: RollNeuralNet (object)
        """
        self.perf_train = [self.V0]
        self.perf_estim = [self.V0]

        # Set axes and figure
        f, ax = plt.subplots(plot_loss + plot_perf, 1, figsize=(10, 6))
        plt.ion()
        if plot_loss and not plot_perf:
            ax_loss, ax_perf = ax, None
        elif not plot_loss and plot_perf:
            ax_loss, ax_perf = None, ax
        elif plot_loss and plot_perf:
            ax_loss, ax_perf = ax
        else: 
            ax_loss, ax_perf = None, None

        # Start Rolling Neural Network
        for pred_train, pred_estim in self(y, X, NN, x_axis=x_axis):
            # Set performance
            self.perf_train += list(self.perf_train[-1] * np.exp(np.cumsum(
                np.sign(pred_train) * self.y[self.t - self.s: self.t, 0]
            )))
            self.perf_estim += list(self.perf_estim[-1] * np.exp(np.cumsum(
                np.sign(pred_estim) * self.y[self.t: self.t + self.s, 0]
            )))
            # Plot perf and loss
            self._dynamic_plot(f, ax_loss=ax_loss, ax_perf=ax_perf)

        return self
            
    def _dynamic_plot(self, f, ax_loss=None, ax_perf=None):
        """ 
        Dynamic plot 
        """
        # Plot progress of loss
        if ax_loss is not None:
            ax_loss.clear()
            ax_loss.plot(
                self.hist.history['loss'], 
                color=sns.xkcd_rgb["pumpkin"], 
                LineWidth=2.
            )
            ax_loss.plot(
                self.hist.history['val_loss'], 
                color=sns.xkcd_rgb["brownish green"], 
                LineWidth=2.
            )
            ax_loss.set_title('Model loss')
            ax_loss.set_ylabel('Loss')
            ax_loss.set_xlabel('Epoch', x=0.9)
            ax_loss.legend(['Train', 'Estim'])
            ax_loss.set_yscale('log')

        # Plot progress of performance
        if ax_perf is not None:
            ax_perf.clear()
            ax_perf.plot(
                self.x_axis[self.n - self.s - 1: self.t - self.s],
                self.perf_train, 
                color=sns.xkcd_rgb["pale red"], 
                LineWidth=2.
            )
            ax_perf.plot(
                self.x_axis[self.n - 1: min(self.t, self.T - self.s)],
                self.perf_estim, 
                color=sns.xkcd_rgb["denim blue"], 
                LineWidth=2.
            )
            ax_perf.set_title('Model performance')
            ax_perf.set_ylabel('Perf.')
            ax_perf.set_xlabel('Rolling period', x=0.9)
            ax_perf.set_yscale('log')
            ax_perf.legend(['Training set', 'Estimation set'])
        f.canvas.draw()