import numpy.linalg
import numpy as np 
import time

test_array1 = np.array([ [ 4,  1, -2,  2],
                         [ 1,  2,  0,  1],
                         [-2,  0,  3, -2],
                         [ 2,  1, -2, -1] ])   #creating symmetric 5 by 5 array




test_array2 = np.array([[ 3, -5,  1,  2, -9],
                        [-5,  4, -2,  7,  8],
                        [ 1, -2, -1,  0,  3],
                        [ 2,  7,  0, -6,  5],
                        [-9,  8,  3,  5, -3] ])   #creating symmetric 5 by 5 array


def calculate_alpha(iteration, test_array):

    alpha = 0
    m,n = test_array.shape
    for i in range(m - iteration):
        #print(test_array[i + iteration][iteration - 1])
        alpha += test_array[i + iteration][iteration - 1] ** 2
    
    if test_array[iteration][iteration -1] >= 0:
        alpha = -1 * (alpha ** (0.5))
    else:
        alpha = 1 * (alpha ** (0.5))

    return alpha

def calculate_r(iteration,test_array):
    alpha = calculate_alpha(iteration, test_array)
    r = (0.5*( alpha**2 - test_array[iteration][iteration -1]*alpha  ))** (0.5)
    return r

def generate_household_vector(iteration,test_array):
    m,n = test_array.shape
    alpha = calculate_alpha(iteration, test_array)
    r = calculate_r(iteration,test_array)
    vector = []
    for i in range(m):
        if i < iteration:
            vector.append(0)
        elif i == iteration:
            vector.append( (test_array[iteration][iteration -1] - alpha)/(2*r) )
        else:
            vector.append( (test_array[i][iteration - 1])/ (2 * r))
    
    return vector

        
        


print(generate_household_vector(1,test_array2))
