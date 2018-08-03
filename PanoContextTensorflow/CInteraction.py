from itertools import islice
from random import random
from time import perf_counter
import numpy as np

from PythonCallC import fast_tanh
from PythonCallC import segmentGraphEdge

COUNT = 500000  # Change this value depending on the speed of your computer
DATA = list(islice(iter(lambda: (random() - 0.5) * 3.0, None), COUNT))

e = 2.7182818284590452353602874713527

def sinh(x):
    return (1 - (e ** (-2 * x))) / (2 * (e ** -x))

def cosh(x):
    return (1 + (e ** (-2 * x))) / (2 * (e ** -x))

def tanh(x):
    tanh_x = sinh(x) / cosh(x)
    return tanh_x

def sequence_tanh(data):
    '''Applies the hyperbolic tangent function to map all values in
    the sequence to a value between -1.0 and 1.0.
    '''
    result = []
    for x in data:
        result.append(tanh(x))
    return result

def test(fn, name):
    start = perf_counter()
    result = fn(DATA)
    duration = perf_counter() - start
    print('{} took {:.3f} seconds\n\n'.format(name, duration))

    for d in result:
        assert -1 <= d <= 1, " incorrect values"

if __name__ == "__main__":
    print('Running benchmarks with COUNT = {}'.format(COUNT))

    #test(sequence_tanh, 'sequence_tanh')

    #test(lambda d: [tanh(x) for x in d], '[tanh(x) for x in d]')

    
    func = lambda d: [fast_tanh(x) for x in d]
    test(func, '[fast_tanh(x) for x in d]')

    x = [np.pi / 4 , -np.pi/4]

    for v in x:
        y = fast_tanh(v)
        print(y)
         

    maxID = 512.0
    num = 9
    edge = np.array([[123,232,3],[456,5,6]])
    edge = np.arange(27)
    edge = np.reshape(edge,[3,9])

    k = 5.0
    minSz = 1.0
    ret = segmentGraphEdge((maxID,num,edge,k,minSz))
    print(ret)
    input("Press Enter to continue...")
