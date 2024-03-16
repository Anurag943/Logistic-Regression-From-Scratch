import numpy as np
from act_cost_pred import compute_cost
from gradient_descent import compute_gradient

def compute_cost_reg(X, y, w, b,lambda_ = 1):
    """ 
    Compute the cost over all examples
    Args:
        X :(ndarray Shape(m,n)) data, m examples by n features
        y : (ndarray Shape(m,)) target value
        w : (ndarray Shape(n,)) values of parameters of the model
        b : (scalar) : value of bias parameter of the model
        lambda_ : (scalar, float) Controls amount of regularization

    Returns:
        total_cost : (scalar) cost

    """

    m, n = X.shape

    cost_without_reg = compute_cost(X, y, w, b)

    reg_cost = 0.

    for j in range(n):
        reg_cost_j = w[j]**2
        reg_cost = reg_cost + reg_cost_j
    reg_cost = (lambda_/(2 * m)) * reg_cost

    # Add the regularization cost ot fet the total cost
    total_cost = cost_without_reg + reg_cost

    return total_cost

     
def compute_gradient_reg(X, y, w, b, lambda_=1):
    """ 
    Compute the gradient for logistic regression with regularization

    Args:
        X : (ndarray Shape (m,n)) data, m examples by n features
        y : (ndarray Shape (m,)) target values
        w : (ndarray Shape (n,)) values of parameters of the model
        b : (scalar) values of bias parameter of the model
        lamdba_ : (scalar) Regualization constant

    Returns:
        dj_db : (scalar) The gradient of the cost with respect to the parameter b.
        dj_dw : (ndarray Shape(n,)) The gradient of the cost with respect to the parameter w.
    """

    m, n = X.shape
    
    dj_db, dj_dw = compute_gradient(X, y, w, b)

    for j in range(n) : 
        dj_dw_j_reg = (lambda_ / m) * w[j]
        dj_dw[j] += dj_dw_j_reg
    
    return dj_db, dj_dw