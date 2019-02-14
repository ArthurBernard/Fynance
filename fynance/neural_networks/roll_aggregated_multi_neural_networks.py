#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Built-in packages

# External packages
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

# Internal packages
from fynance.backtest.dynamic_plot_backtest import DynaPlotBackTest
from fynance.neural_networks.roll_multi_neural_networks import RollMultiNeuralNet

# Set plot style
plt.style.use('seaborn')

# TODO: Aggregated method

class RollAggrMultiNeuralNet(RollMultiNeuralNet):
    """ Rolling Aggregated Multi Neural Networks object allow you to train 
    several neural networks along training periods (from t - n to t), 
    predict along testing periods (from t to t + s) and aggregate prediction 
    following a specified rule and roll along this time axis.

    Attributes
    ----------
    y : np.ndarray[np.float32, ndim=2] with shape=(T, 1)
        Target to estimate or predict.
    X : np.ndarray[np.float32, ndim=2] with shape=(T, N)
        Features (inputs).
    NN : list of keras.Model
        Neural network to train and predict.
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
    aggregate(mat_pred, y, t=0, t_s=-1)
        Method to aggregate predictions from several neural networks.
    set_aggregate(*args)
        Set your own aggregation method.
    plot_loss(self, f, ax)
        Plot loss function
    plot_perf(self, f, ax)
        Plot perfomances.

    """
    def __init__(self, *args, agg_fun='mean', **kwargs):
        RollMultiNeuralNet.__init__(self, *args, **kwargs)
        self.agg_fun = agg_fun


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
            Neural network model.
        start : int, optional
            Starting observation, default is 0.
        end : int, optional
            Ending observation, default is end.
        x_axis : np.ndarray[ndim=1], optional
            X-Axis to use for the backtest.

        Returns
        -------
        ramnn : RollAggrMultiNeuralNet

        """
        RollMultiNeuralNet.__call__(
            self, y, X, NN, start=start, end=end, x_axis=x_axis
        )
        self.agg_y = np.zeros([self.T, 1])
        
        return self
        
    
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
        ramnn : RollAggrMultiNeuralNet

        """
        if isinstance(NN, list):
            self.n_NN = len(NN)
        else:
            self.n_NN = 1

        # Set perf and loss arrays
        self.perf_train = self.V0 * np.ones([y.size, self.n_NN])
        self.perf_estim = self.V0 * np.ones([y.size, self.n_NN])
        self.perf_agg = self.V0 * np.ones([y.size, 1])

        # Set axes and figure
        f, ax_loss, ax_perf = self._set_figure(plot_loss, plot_perf)

        # Start Rolling Neural Network
        for pred_train, pred_estim in self(y, X, NN, x_axis=x_axis):
            t, s, t_s = self.t, self.s, min(self.t + self.s, self.T)
            
            # Set performances of training period
            returns = np.sign(pred_train) * y[t - s: t]
            cum_ret = np.exp(np.cumsum(returns, axis=0))
            self.perf_train[t - s: t] = self.perf_train[t - s - 1] * cum_ret

            # Set performances of estimated period
            returns = np.sign(pred_estim) * y[t: t_s]
            cum_ret = np.exp(np.cumsum(returns, axis=0))
            self.perf_estim[t: t_s] = self.perf_estim[t - 1] * cum_ret
            
            # Aggregate prediction
            self.aggregate(pred_estim, y[t: t_s], t=t, t_s=t_s)
            returns = np.sign(self.agg_y[t: t_s]) * y[t: t_s]
            cum_ret = np.exp(np.cumsum(returns, axis=0))
            self.perf_agg[t: t_s] = self.perf_agg[t - 1] * cum_ret

            # Plot loss and perf
            self._dynamic_plot(f, ax_loss=ax_loss, ax_perf=ax_perf)

        return self
    
    def aggregate(self, mat_pred, y, t=0, t_s=-1):
        """ Method to aggregate predictions from several neural networks.

        Parameters
        ----------
        mat_pred : np.ndarray[np.float32, ndim=2] with shape=(T, n_NN)
            Several time series of neural networks predictions.
        y : np.ndarray[np.float32, ndim=2] with shape=(T, 1)
            Time series of target to estimate or predict.
        t : int, optional
            First observation, default is first one.
        t_s : int, optional
            Last observation, default is last one.
        
        Returns
        -------
        ramnn : RollAggrMultiNeuralNet

        """
        self.agg_y[t: t_s, 0] = self._aggregate(mat_pred, y)
        return self

    def _aggregate(self, mat_pred, y):
        """ """
        # TODO : find a better aggregation method
        if self.agg_fun == 'mean':
            return np.mean(mat_pred, axis=1)
        elif self.agg_fun == 'sum':
            return np.sum(mat_pred, axis=1)
        elif self.agg_fun == 'best':
            i = np.argmax(self.perf_estim[self.t])
            return mat_pred[:, i]
        elif self.agg_fun == 'bests':
            perfs = self.perf_estim[self.t]
            perf_list = []
            arg_list = []
            for i in range(self.n_NN): 
                if len(perf_list) < 3:
                    perf_list += [perfs[i]]
                    arg_list += [i]
                elif perfs[i] > min(perf_list):
                    j = np.argmin(perf_list)
                    perf_list[j] = perfs[i]
                    arg_list[j] = i
                else: 
                    pass
            y = mat_pred[:, arg_list[0]]
            y += mat_pred[:, arg_list[1]]
            y += mat_pred[:, arg_list[2]]
            y /= 3
            return y

    
    # TODO : Make method to customize aggregation function
    def set_aggregate(self, *args):
        """ Set your own aggregation method. 

        Parameters
        ----------
        args : tuple of function
            Any function such that the final value is a numpy array.

        Returns
        -------
        ramnn : RollAggrMultiNeuralNet

        """
        self._aggregate = lambda x: x
        for arg in args:
            self._aggregate = lambda x: arg(self._aggregate(x))
        return self

    def plot_perf(self, f, ax):
        """ Plot performances method 
        
        Parameters
        ----------
        fig : matplotlib.figure.Figure
            Figure to display backtest.
        ax : matplotlib.axes
            Axe(s) to display a part of backtest.

        Returns
        -------
        ramnn : RollAggrMultiNeuralNet

        """
        t, t_s = self.t, min(self.t + self.s, self.T)
        
        dpbt = DynaPlotBackTest(
            fig=f, ax=ax, title='Model performance', ylabel='Perf.', 
            xlabel='Date', yscale='log',
            tick_params={'axis': 'x', 'rotation': 30, 'labelsize': 10}
        )
        
        # Set graphs
        dpbt.plot(
            self.perf_estim[: t_s], x=self.x_axis[: t_s],  
            names='Estim NN', col='GnBu', lw=1.7, unit='perf',
        )
        dpbt.plot(
            self.perf_agg[: t_s], x=self.x_axis[: t_s], 
            names='Aggr NN', col='Reds', lw=2., unit='perf'
        )
        dpbt.plot(
            self.perf_train[: t], x=self.x_axis[: t], 
            names='Train NN', col='OrRd', lw=1.2, unit='perf'
        )
        ax.legend(loc='upper left', ncol=2, fontsize=10, 
            handlelength=0.8, columnspacing=0.5, frameon=True)

        return self