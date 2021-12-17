import numpy
import input
import numba
from numba import jit

dark_side_value: float = 0.0
# coeff is defined here to make numba work correctly, cannot define the array with a list of lists otherwise within the function
coeff = numpy.array([[0.0, 1.0, 0.0, 0.0],
                        [-1.0 / 3.0, -1.0 / 2.0, 1.0, -1.0 / 6.0],
                        [1.0 / 2.0, -1.0, 1.0 / 2.0, 0.0],
                        [-1.0 / 6.0, 1.0 / 2.0, -1.0 / 2.0, 1.0 / 6.0]], dtype=float)

#Called by bi_cubic_interp as a part of the bi cubic interpolation
@numba.jit(nopython=True, debug=True)
def cubic_interp(ref_grid: numba.types.float64[:,:], x: float, j_0: int, k_i_0: int):

    sum: float = 0.0
    x_power: float = 1.0

    for i in range(4):
        for j in range(4):
            sum: float = sum + ref_grid[j+j_0,k_i_0] * coeff[i,j] * x_power
        x_power = x_power*x
    return sum

#Performs part of the bi cubic interpololation and calls on cubic_interp to complete during a loop and with interp_grid
@numba.jit(nopython=True, debug=True)
def bi_cubic_interp(x: float, y: float, data_in:numba.types.float64[:,:], nyCL: float, nxCL: float):
    #Shaped this way for the last call to cubic_interp with numba compliance, same 2D shape as the first calls
    interp_grid: numpy.ndarray = numpy.zeros((4,4))
    i_1 = int(numpy.floor(y))
    i_0 = i_1 - 1
    i_2 = i_1 + 1
    i_3 = i_2 + 1
    j_1 = int(numpy.floor(x))

    j_0 = j_1 - 1
    j_2 = j_1 + 1
    j_3 = j_2 + 1

    if (i_1 < 0 or j_1 < 0 or i_2 > nyCL - 1 or j_2 > nxCL - 1):
        print("Error while interpolating, grid size did not match up")
        return 0

    if (i_3 > nyCL - 1):
        i_0 = i_0-1
        i_1 = i_1-1
        i_2 = i_2-1
        i_3 = i_3-1

    if (j_3 > nxCL - 1):
        j_0 = j_0-1
        j_1 = j_1-1
        j_2 = j_2-1
        j_3 = j_3-1

    if (i_0 < 0):
        i_0 = i_0+1
        i_1 = i_1+1
        i_2 = i_2+1
        i_3 = i_3+1

    if (j_0 < 0):
        j_0 = j_0+1
        j_1 = j_1+1
        j_2 = j_2+1
        j_3 = j_3+1

    alpha_x = x - j_1
    alpha_y = y - i_1

    for k in range(4):
        temp: int = int(j_0) + int(k)
        interp_grid[k][0] = float(cubic_interp(data_in, alpha_x, i_0, temp))
    temp = cubic_interp(interp_grid, alpha_y, 0, 0)
    return temp

