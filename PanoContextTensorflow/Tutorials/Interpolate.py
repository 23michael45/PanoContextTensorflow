import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import numpy as np

### 1d example of interpolation ###

in_data_x = np.array([1., 2., 3., 4., 5., 6.])
in_data_y = np.array([1.5, 2., 2.5, 3.,  3.5,  4.])  # y = .5 x - 1
f = interp1d(in_data_x, in_data_y, kind='linear')

print(f)
# f in all of the points of the grid (in_data_x): output coincides with in_data_y


print( f(1), f(1.), f(1.5), f(2.), f(2.5), f(3.))
# f in a point outside the grid:
print( f(1.8))
# this is equal to y = .5 x - 1 for x = 1.8, up to some point.
assert( np.round(0.5 * 1.8 + 1, decimals=10) == np.round(f(1.8), decimals=10))

# plot up to this point
xnew = np.arange(1, 6, 0.1)
ynew = f(xnew)
plt.plot(in_data_x, in_data_y, 'o', xnew, ynew, '-')
# close the image to move forward.
#plt.show()

### another 1d example of interpolation ###

in_data_x = np.array([1., 2., 3., 4., 5., 6.])
in_data_y = np.array([-1.8, -1.2, -0.2, 1.2, 3., 5.2])  # y = .2 x**2 - 2
f = interp1d(in_data_x, in_data_y, kind='cubic')

print (f)
# f in all of the points of the grid (in_data_x): output coincides with in_data_y
print (f(1), f(1.), f(1.5), f(2.), f(2.5), f(3.))
# f in a point outside the grid:
print( f(1.8))
# this is equal to y = .2 x**2 - 2 for x = 1.8, up to some precision.
assert(np.round(0.2 * 1.8 ** 2 - 2, decimals=10) == np.round(f(1.8), decimals=10))

# plot up to this point
xnew = np.arange(1, 6, 0.1)
ynew = f(xnew)
plt.plot(in_data_x, in_data_y, 'o', xnew, ynew, '-')
#plt.show()



from scipy.ndimage.interpolation import map_coordinates
import numpy as np


in_data = np.array([[0., -1., 2.],
                    [2., 1., 0.],
                    [4., 3., 2.]])  # z = 2.*x - 1.*y

# want the second argument as a column vector (or a transposed row)
# see on some points of the grid:
print( 'at the point 0, 0 of the grid the function z is: ')
print( map_coordinates(in_data, np.array([[0., 0.]]).T, order=1))
print( 'at the point 0, 1 of the grid the function z is: ')
print( map_coordinates(in_data, np.array([[0., 1.]]).T, order=1))
print( 'at the point 0, 2 of the grid the function z is: ')
print( map_coordinates(in_data, np.array([[0., 2.]]).T, order=1))

# see some points outside the grid
print()
print( 'at the point 0.2, 0.2 of the grid, with linear interpolation z is:')
print( map_coordinates(in_data, np.array([[.2, .2]]).T, order=1))
print( 'and it coincides with 2.*.2 - .2')
print()
print( 'at the point 0.2, 0.2 of the grid, with cubic interpolation z is:')
print( map_coordinates(in_data, np.array([[0.2, .2]]).T, order=3))