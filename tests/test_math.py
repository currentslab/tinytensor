import numpy as np
from tinytensor.math_utils import sigmoid, softmax



def test_sigmoid():
    space = np.linspace(-5, 5).reshape(1, -1)
    points = sigmoid(space)[0]
    assert np.min(points) > 0
    assert points[-1] > 0.9
    assert points[0] < 0.1



def test_softmax():
    # shape of 3 x 4
    scores2D = np.array([[1, 2, 3, 6],
                     [2, 4, 5, 6],
                     [3, 8, 7, 6]])
    
    softmax_result = softmax(scores2D)    

    for prob in softmax_result:
        assert sum(prob) == 1.0

