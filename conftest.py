import numpy as np

# NumPy 2.0 changed scalar repr: np.float64(0.8) instead of 0.8.
# Restore legacy repr so existing doctests don't need mass-updating.
np.set_printoptions(legacy='1.25')
