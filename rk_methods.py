import numpy as np

def euler(inputs, diffeq, h, *params):
    """
    solves one step of eulers method for a system of differential equations

    inputs:
        inputs  - current values of the system
        diffeq  - function that returns the derivatives
        h       - step size
        *params - extra parameters for the differential equation

    returns:
        outputs - next step values
    """

    inputs = np.array(inputs, dtype=float)

    derivative = np.array(diffeq(inputs, *params))

    outputs = inputs + derivative * h

    return outputs

#k1 = f(current point)
#predictor = current point + h*k1
#k2 = f(predictor)
#new point = current point + h*(k1+k2)/2

def explicit_rk2(inputs, diffeq, h, *params):

    inputs = np.array(inputs, dtype=float)
    k1 = np.array(diffeq(inputs, *params))
    eulerOutputs = inputs + k1 * h
    k2 = np.array(diffeq(eulerOutputs, *params))
    outputs = inputs + (k1 + k2) * h / 2

    return outputs

# 1. use current xyz guess
# 2. build midpoint
# 3. compute slope at midpoint
# 4. use slope to compute better xyz
# 5. repeat

def implicit_rk2(inputs, diffeq, h, *params, maxIter=10, tol=1e-6):
    #future point = current point + h*f(midpoint)
    #midpoint = (current point + future point)/2
    inputs = np.array(inputs, dtype=float)
    k1 = np.array(diffeq(inputs, *params))
    initialVals = inputs + k1 * h #guess for future point
    for i in range(maxIter):
        midpoints = (inputs + initialVals) / 2

        k2 = np.array(diffeq(midpoints, *params))

        newVals = inputs + k2 * h
        #if the new point is close enough to the old point, we can stop iterating
        for j in range(len(inputs)):
            if np.linalg.norm(newVals[j] - initialVals[j]) < tol:
                initialVals = newVals
                break
            else:
                initialVals = newVals

    return initialVals


def explicit_rk4(inputs, diffeq, h, *params):
    inputs = np.array(inputs, dtype=float)
    k1 = np.array(diffeq(inputs, *params))
    euler1 = inputs + k1 * h/2
    k2 = np.array(diffeq(euler1, *params))
    euler2 = inputs + k2 * h/2
    k3 = np.array(diffeq(euler2, *params))
    euler3 = inputs + k3 * h
    k4 = np.array(diffeq(euler3, *params))
    outputs = inputs + (k1 + 2*k2 + 2*k3 + k4) * h / 6

    return outputs