#!/usr/bin/env python3
# coding: utf-8

# Built-in packages

# External packages
from keras.models import Model
from keras.layers import Input, Dense, Dropout
from keras.optimizers import Adam
from keras import regularizers, initializers, constraints

# Internal packages


__all__ = ['incr_seed', 'set_layer', 'set_nn_model']


def incr_seed(SEED, incr=1):
    """ Increment seed """
    if SEED is None:
        return None
    return SEED + incr


def set_layer(nn, n_neurons, dropout=None, SEED=None, **kwargs):
    """ Set `Dense` layers 
    
    Parameters
    ----------
    :nn: keras.Model
        An initilized neural network (cf `Input` keras documation).
    :n_neurons: int
        Number of neurons to set in this layer.
    :dropout: float
        At each iteration an part of variables is dropout. cf keras doc.
    :SEED: int
        A number to set the random weights.
    :kwargs: any parameters of `Dense` 
        cf keras documentation.
        
    Returns
    -------
    :nn: keras.Model
        Neural network with one more layer.
    
    """
    if dropout is None:
        return Dense(n_neurons, **kwargs)(nn)
    else:
        nn = Dense(n_neurons, **kwargs)(nn)
        SEED = incr_seed(SEED, incr=1)
        return Dropout(dropout, seed=SEED)(nn)
    
    
def set_nn_model(
        X, SEED=None, l1=0.01, l2=0.01, dropout=None, activation='tanh', 
        lr=0.01, b_1=0.99, b_2=0.999, decay=0.0, name=None, use_bias=True,
        loss='mse', metrics=['accuracy'], l1_bias=0.01, l2_bias=0.01,
        l1_acti=0.01, l2_acti=0.01, m=0.0, std=1.,
    ):
    """ Set a very basic neural network with `Dense` layers.
    
    Parameters
    ----------
    :X: np.ndarray[ndim=2, dtype=np.float32]
        Matrix of features of shape (T, N) with T is the number of 
        observations and N the number of features.
    :dropout: float
        At each iteration an part of variables is dropout. cf keras doc.
    :SEED: int
        A number to set the random weights.
    For other parameters cf Keras documentation.
    
    Returns
    -------
    :model: Keras.model
         A Neural Network ready to be train !
    
    """
    T, N = X.shape
    
    # Set constant parameters for each layer
    USE_BIAS = use_bias
    KERN_REG = regularizers.l1_l2(l1=l1, l2=l2)
    BIAS_REG = regularizers.l1_l2(l1=l1_bias, l2=l2_bias)
    ACTIV_REG = regularizers.l1_l2(l1=l1_acti, l2=l2_acti)
    KERN_CONS = constraints.MinMaxNorm(
        min_value=-2., max_value=2.0, rate=1.0, axis=0
    )
    BIAS_CONS = constraints.MinMaxNorm(
        min_value=-2., max_value=2.0, rate=1.0, axis=0
    )
    
    # Set input layer
    inputs = Input(shape=(N,), sparse=False)
    
    # FIRST # INIT WITH IDENTITY MATRIX ? 
    kern_init = initializers.RandomNormal(mean=m, stddev=std, seed=SEED)
    SEED = incr_seed(SEED, incr=1)
    
    nn = set_layer(
        inputs, 256, dropout=dropout, activation=activation, 
        use_bias=USE_BIAS, kernel_regularizer=KERN_REG, SEED=SEED,
        bias_regularizer=BIAS_REG, activity_regularizer=ACTIV_REG,
        kernel_initializer=kern_init, #initializers.Identity(gain=1.0),
        kernel_constraint=KERN_CONS, bias_constraint=BIAS_CONS,
    )
    
    # SECOND
    kern_init = initializers.RandomNormal(mean=m, stddev=std, seed=SEED)
    SEED = incr_seed(SEED, incr=1)
    
    nn = set_layer(
        nn, 512, dropout=dropout, activation=activation, 
        use_bias=USE_BIAS, kernel_regularizer=KERN_REG,
        bias_regularizer=BIAS_REG,activity_regularizer=ACTIV_REG,
        kernel_initializer=kern_init, SEED=SEED,
        kernel_constraint=KERN_CONS, bias_constraint=BIAS_CONS,
    )
    
    # THIRD
    kern_init = initializers.RandomNormal(mean=m, stddev=std, seed=SEED)
    SEED = incr_seed(SEED, incr=1)
    
    nn = set_layer(
        nn, 512, dropout=dropout, activation=activation, 
        use_bias=USE_BIAS, kernel_regularizer=KERN_REG,
        bias_regularizer=BIAS_REG, activity_regularizer=ACTIV_REG,
        kernel_initializer=kern_init, SEED=SEED,
        kernel_constraint=KERN_CONS, bias_constraint=BIAS_CONS,
    )
    
    # FOURTH
    kern_init = initializers.RandomNormal(mean=m, stddev=std, seed=SEED)
    SEED = incr_seed(SEED, incr=1)
    
    nn = set_layer(
        nn, 256, dropout=dropout, activation=activation, 
        use_bias=USE_BIAS, kernel_regularizer=KERN_REG,
        bias_regularizer=BIAS_REG,activity_regularizer=ACTIV_REG, 
        kernel_initializer=kern_init, SEED=SEED,
        kernel_constraint=KERN_CONS, bias_constraint=BIAS_CONS,
    )
    
    # FIVE
    kern_init = initializers.RandomNormal(mean=m, stddev=std, seed=SEED)
    SEED = incr_seed(SEED, incr=1)
    
    nn = set_layer(
        nn, 64, dropout=dropout, activation=activation, 
        use_bias=USE_BIAS, kernel_regularizer=KERN_REG,
        bias_regularizer=BIAS_REG, activity_regularizer=ACTIV_REG,
        kernel_initializer=kern_init, SEED=SEED,
        kernel_constraint=KERN_CONS, bias_constraint=BIAS_CONS,
    )
    
    # LAST ONE
    kern_init = initializers.RandomNormal(mean=m, stddev=std, seed=SEED)
    SEED = incr_seed(SEED, incr=1)
    
    outputs = set_layer(
        nn, 1, dropout=dropout, activation=activation, 
        use_bias=USE_BIAS, kernel_regularizer=KERN_REG,
        bias_regularizer=BIAS_REG, activity_regularizer=ACTIV_REG,
        kernel_initializer=kern_init, SEED=SEED,
        kernel_constraint=KERN_CONS, bias_constraint=BIAS_CONS,
    )
    
    # Compile model
    model = Model(inputs=inputs, outputs=outputs)
    model.name = name
    model.compile(
        optimizer=Adam(
            lr=lr, 
            beta_1=b_1, 
            beta_2=b_2, 
            decay=decay, 
            amsgrad=True
        ),
        loss=loss, 
        metrics=metrics
    )
    return model