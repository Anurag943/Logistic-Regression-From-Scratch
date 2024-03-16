import math
import numpy as np
from act_cost_pred import sigmoid, compute_cost

def compute_gradient(X, y, w, b, *argv):
    """ 
    
    Compute the gradient for logistic regression
    
    Args:
        X : (ndarray Shape (m,n)) data, m examples by n features
        y : (ndarray Shape(m,)) target value
        w : (ndarray Shape(n,)) values of parameters of the model
        b : (scalar) values of bias parameter of the model
        *argv : unused, for comparibility with regularized version below

    Returns:
        dj_dw : (ndarray Shape(n,)) The gradien of the cost with respect to the parameters w.
        dj_db : (scalar) The gradient of the cost with respect to the parameter b.
    """

    m, n = X.shape
    dj_dw = np.zeros(w.shape)
    dj_db = 0. 

    for i in range(m):
        z_wb = 0
        for  j in range(n):
            z_wb += w[j] * X[i,j]
        z_wb += b

        f_wb = sigmoid(z_wb)

        dj_db_i = f_wb - y[i]
        dj_db += dj_db_i

        for j in range(n):
            dj_dw[j] += (f_wb - y[i])*X[i][j]

    dj_dw = dj_dw / m
    dj_db = dj_db / m

    return dj_db, dj_dw

def gradient_descent(X, y, w_in, b_in, cost_function, gradient_function, alpha, num_iters, lambda_):
    """ 
    Perform batch gradient descent to learn thete. Updatas thetaby taking 
    num_iters gradient steps wiht learning rate alpha

    Args: 
        X : (ndarray Shape(m,n)) data, m examples by n features
        y : (ndarray Shape(m,)) target value
        w_in : (ndarray Shape (n,)) Initial values of parameters of the model
        b_in : (Scalar) Tnitial values of parameters of the model
        cost_function : function to vompute cost
        gradient_function : function to compute gradient 
        alpha : (float) Learning Rate
        num_iters(int) : number og iterations to run gradient descent
        lamdba_ : (scalar, float) regularization constant

    Returns:
        w : (ndarray Shape(n,)) Updated values of parameters of the model arter gradeint descent
        b : (scalar) Updated values of parameters of the model after gradient descent

    """
    # number of training examples
    m = len(X)

    # An array to store cost J and w's at each iteration primarily fr graphing later
    J_history = []
    w_history = []

    for i in range(num_iters):

        # Calculate the gradient and update the parameters
        dj_db, dj_dw = gradient_function(X, y, w_in, b_in, lambda_)

        # Update parameters using w, b alpha and gradient
        w_in = w_in - alpha * dj_dw
        b_in = b_in - alpha * dj_db

        # Save cost J at each iteration
        cost = cost_function(X, y, w_in, b_in)
        J_history.append(cost)
        
        # Print cost every at intervals 10 times or as many iterations if <10
        if  i%math.ceil(num_iters/10) == 0 or i == (num_iters-1):
            w_history.append(w_in)
            print(f"Iteration {i:4} : Cost {float(J_history[-1]):8.2f} ")
                
    return w_in, b_in, J_history, w_history