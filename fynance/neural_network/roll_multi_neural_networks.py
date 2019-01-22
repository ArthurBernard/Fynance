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

# TODO: Merge with RollNeuralNet object

class RollMultiNeuralNet:
    """
    Rolling Multi Neural Networks object allow you to train several neural 
    networks along training periods (from t - n to t) and predict along 
    testing periods (from t to t + s) and roll along this time axis.

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
    :run: Train several rolling neural networks along pre-specified training 
        period and predict along test period. Display loss and performance if 
        specified.
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
        self.set_params_dynamic_plot()
    
    def __call__(self, y, X, NN, start=0, end=1e6, x_axis=None):
        """ Callable method """
        # Set target and features
        self.y = y
        self.X = X
        
        # Set neural network model 
        if isinstance(NN, list):
            self.NN = NN
            self.n_NN = len(NN)
        else:
            self.NN = [NN]
            self.n_NN = 1
        
        # Set periodicity
        self.t = max(self.n, start)
        self.T = min(y.size, end)
        if x_axis is None:
            self.x_axis = range(self.T)
        else:
            self.x_axis = x_axis
        
        return self
        
    def __iter__(self):
        """ Set iterative method """
        self.y_train = np.zeros([self.T, self.n_NN])
        self.y_estim = np.zeros([self.T, self.n_NN])
        self.hist = {i: None for i in range(self.n_NN)}
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
        
        return self.y_train[t - self.s: t], self.y_estim[t: t + self.s]
    
    def _train(self, y, X, i, val_set=None):
        """ Train method and return prediction on training set """
        if self.hist[i] is None:
            self.hist[i] = self.NN[i].fit(
                x=X, y=y, validation_data=val_set, **self.init_params
            )
        else:
            hist = self.NN[i].fit(
                x=X, y=y, validation_data=val_set, **self.params
            )
            for key, arg in hist.history.items():
                self.hist[i].history[key] += arg
        return self.NN[i].predict(
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
        Train several rolling neural networks along pre-specified train 
        period and predict along test period. Display loss and performance 
        if specified.
        
        Parameters
        ----------
        :y: np.ndarray[np.float32, ndim=2] with shape=(T, 1)
            Time series of target to estimate or predict.
        :X: np.ndarray[np.float32, ndim=2] with shape=(T, N)
            Several time series of features.
        :NN: keras.Model or list of keras.Model
            Neural networks to train and predict.
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
        if isinstance(NN, list):
            n = len(NN)
        self.perf_train = {i: [self.V0] for i in range(n)}
        self.perf_estim = {i: [self.V0] for i in range(n)}

        # Set axes and figure
        f, ax = plt.subplots(plot_loss + plot_perf, 1, figsize=(9, 6))
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
            for i in range(n):
                # Set performances of each Neural Network
                self.perf_train[i] += list(
                    self.perf_train[i][-1] * np.exp(np.cumsum(np.sign(
                        pred_train[:, i]
                    ) * self.y[self.t - self.s: self.t, 0]))
                )
                self.perf_estim[i] += list(
                    self.perf_estim[i][-1] * np.exp(np.cumsum(np.sign(
                        pred_estim[:, i]
                    ) * self.y[self.t: self.t + self.s, 0]))
                )
            # Plot loss and perf
            self._dynamic_plot(f, ax_loss=ax_loss, ax_perf=ax_perf)

        return self
    
    def _dynamic_plot(self, f, ax_loss=None, ax_perf=None):
        """ 
        Dynamic plot 
        """
        # Plot progress of loss
        if ax_loss is not None:
            legend = []
            ax_loss.clear()
            for i in range(self.n_NN):
                ax_loss.plot(
                    self.hist[i].history['val_loss'], 
                    color=sns.color_palette('BuGn', self.n_NN)[i],
                    LineWidth=2.,
                    label='Estim {}: {:.2f}'.format(
                        self.NN[i].name, self.hist[i].history['val_loss'][-1]
                    )
                )
                #legend += ['Estim NN {}'.format(i)]
            for i in range(self.n_NN):
                ax_loss.plot(
                    self.hist[i].history['loss'], 
                    color=sns.color_palette('YlOrBr', self.n_NN)[i],
                    LineWidth=1.5,
                    label='Train {}: {:.2f}'.format(
                        self.NN[i].name, self.hist[i].history['loss'][-1]
                    )
                )
                #legend += ['Train NN {}'.format(i)]
            ax_loss.set_title('Model loss')
            ax_loss.set_ylabel('Loss')
            ax_loss.set_xlabel('Epoch', x=0.9)
            ax_loss.legend(
                loc='upper right', ncol=2, fontsize=10, 
                handlelength=0.8, columnspacing=0.5, frameon=True
            )
            ax_loss.set_yscale(self.loss_scale)
            ax_loss.tick_params(axis='x', labelsize=10)

        # Plot progress of performance
        if ax_perf is not None:
            legend = []
            ax_perf.clear()
            for i in range(self.n_NN):
                ax_perf.plot(
                    self.x_axis[self.n - 1: min(self.t, self.T - self.s)],
                    self.perf_estim[i], 
                    color=sns.color_palette('GnBu', self.n_NN)[i],
                    LineWidth=1.7,
                    label='Estim {}: {:.0f} %'.format(
                        self.NN[i].name, self.perf_estim[i][-1] - 100.
                    )
                )
                #legend += ['Estim NN {}'.format(i)]
            for i in range(self.n_NN):
                ax_perf.plot(
                    self.x_axis[self.n - self.s - 1: self.t - self.s],
                    self.perf_train[i], 
                    color=sns.color_palette('OrRd', self.n_NN)[i],
                    LineWidth=1.2,
                    label='Train {}: {:.0f} %'.format(
                        self.NN[i].name, self.perf_train[i][-1] - 100.
                    )
                )
                #legend += ['Train NN {}'.format(i)]
            ax_perf.set_title('Model performance')
            ax_perf.set_ylabel('Perf.')
            ax_perf.set_xlabel('Date', x=0.9)
            ax_perf.set_yscale(self.perf_scale)
            ax_perf.legend(
                loc='upper left', ncol=2, fontsize=10, 
                handlelength=0.8, columnspacing=0.5, frameon=True
            )
            ax_perf.tick_params(axis='x', rotation=30, labelsize=10)
        f.canvas.draw()

    def set_params_dynamic_plot(self, loss_scale='log', perf_scale='log', period=252):
        self.loss_scale = loss_scale
        self.perf_scale = perf_scale
        self.period = period
        return self