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
    for i in range(m):
        if i - iteration  < 0:
            alpha += 0
        else:
            alpha += test_array[i][iteration - 1] ** 2
    
    if test_array[iteration][iteration -1] >= 0:
        alpha = -1 * (alpha ** (0.5))
    else:
        alpha = 1 * (alpha ** (0.5))

    return alpha

def calculate_r(iteration,test_array):
    alpha = calculate_alpha(iteration, test_array)
    r = (0.5*( alpha**2 - test_array[iteration][iteration -1]*alpha  ))** (0.5)
    #print(r)
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
    
    #print(vector)
    return np.array(vector)

        
        
def generate_householder_matrix(iteration, test_array):
    m,n = test_array.shape
    identity = np.identity(m)
    vector = generate_household_vector(iteration, test_array)
    vect_tran = np.transpose(vector)
    v_vt = np.outer(vector,vect_tran)
    #print(v_vt)
    return identity - 2 * v_vt
    
    

house_1 = generate_householder_matrix(1,test_array1)
#print(house_1)
a1_matrix = house_1 @ test_array1 @ house_1
print("After first iteration of Householder Method:\n",a1_matrix,"\n")

house_2  = generate_householder_matrix(2,a1_matrix)
a2_matrix = house_2 @ a1_matrix @ house_2
print("After second iteration of Householder Method:\n",a2_matrix,"\n")


