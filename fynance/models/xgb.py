#!/usr/bin/env python3
# coding: utf-8

# Built-in packages

# External packages
import xgboost as xgb

# Internal packages


__all__ = ['XGB', 'XGBData']


class XGB:
    # TODO : train method, predict method
    def __init__(self, X, y, **kwargs):
        """ Setting data to XGBoot model.

        Parameters
        ----------
        X, y : np.ndarray[ndim=2, dtype=np.float64]
            Respectively features with shape `(T, N)` and target with shape
            `(T, 1)` of the model.
        kwargs : dict, optional
            Parameters of DMatrix object, cf XGBoost documentation [1]_.

        References
        ----------
        .. [1] https://xgboost.readthedocs.io/en/latest/python/python_api.html

        """
        self.data = XGBData(X, label=y, **kwargs)

    def run(self, n, s, **params):
        # TODO : to remove
        train = self.data[:-n]
        estim = self.data[: s]
        bst = xgb.train(params, train)
        return bst.predict(estim)


class XGBData(xgb.DMatrix):
    def __getitem__(self, key):
        """ Slice the DMatrix and return a new DMatrix that only contains `key`.

        Parameters
        ----------
        key : slice
            Slice to be selected.

        Returns
        -------
        res : DMatrix
            A new DMatrix containing only selected indices.

        """
        start = 0 if key.start is None else key.start
        step = 1 if key.step is None else key.step
        stop = self.num_row() if key.stop is None else key.stop

        if step < 0:
            stop, start = start - 1, stop + 1

        if stop < 0:
            stop += self.num_row() + 1

        return self.slice(list(range(start, stop, step)))
