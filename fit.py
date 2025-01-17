import numpy as np

def fit_coords(left, right, top, bottom):
    coords = np.array([left, right, top, bottom])
    targets = np.array([
        [-1, 0, 1],
        [1, 0, 1],
        [0, 1, 1],
        [0, -1, 1],
    ])


    params = np.linalg.lstsq(targets, coords)[0]
    return params[:2], params[2]
