import numpy as np

def sigmoid(z):
    """ 
    Compute the sigmoid of z

    Args :
        z (ndarray) : A scalar, numpy array of any size

    Returns :
        g (ndarray) : sigmiod(z), with the same shape as z.
    """

    g = 1/ (1 + np.exp(-z))

    return g

def compute_cost(X, y, w, b, *args):
    """ 
    Compute the cost over all examples

    Args:
        X (ndarray Shape(m, n)) : data , m examples by n features
        y (ndarray Shape(m,))   : target value
        w (ndarray Shape(n,))   : values of parameters of the model 
        b (scalar)              : values of bias parameter of the model
        * args                  : unsused, for compatibility with regularized version
    
    Returns :
        total_cost (scalar) : cost

    """
    m, n = X.shape
    loss_sum = 0 

    for i in range(m):
        z_wb =0
        
        for j in range(n):
            z_wb_ij = w[j] * X[i,j]
            z_wb += z_wb_ij
        z_wb += b

        f_wb = sigmoid(z_wb)
        loss = -y[i] * np.log(f_wb) - (1 - y[i]) * np.log(1 - f_wb)

        loss_sum += loss
    total_cost = loss_sum/ m

    return total_cost


def predict(X, w, b):
    """ 
    Predict whether the label is 0 or 1 using learnied logistic 
    regression parameters w and b

    Args : 
        X : (ndarray Shape(m,n)) data, m examples by n features
        w : (ndarray Shape(n,)) values of parameters of the model 
        b : (scalar) values of bias parameter of the model

    Returns :
        p : (ndarray (m,)) The predictions for X using a threshold at 0.5
    """
    # number of training examples
    m,n = X.shape
    p = np.zeros(m)

    # loop over each example
    for i in range(m):
        z_wb = 0 
        # loop over each feature
        for j in range(n):
            # Add the corresponding terms to z_wb
            z_wb += w[j] * X[i][j]
        
        # Add the bias term
        z_wb += b

        # Calculate the prediction for this exampl
        f_wb = sigmoid(z_wb)

        # Apply the threshold
        p[i] = f_wb >= 0.5

    return p

