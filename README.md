## Logistic Regression Implementation From Scratch

## Introduction

This project implements Logistic Regression from scratch in Python using L2 Regularization. Logistic Regression is a supervised machine learning algorithm used for binary classification problems. It estimated the probablity of an observation belonging to a specific class based on its features.

## Code Structure

- `LogisticRegression.ipynb` : Contains the core implementatin of the Logistic Regression and used various modules to do so.
- `act_cost_pred.py` : Contains function implementing sigmoid activation along with functions for computing non-regularized cost and to predict the individual data point.
- `gradient_descent.py` : It contains only 2 fuunction one of which is `compute_gradient` which calculated all the derivatives required for performing gradient descent.
Another function is `gradient_descent` which is an implementation of optimization algorithm, which is responsible for reducing cost and finding the best solution.
- `Regualized_cost_gradient.py` : It contains only 2 function one of which is `compute_cost_reg` which calculates l2 regularized cost using the `compute_cost` function in `act_cost_pred.py` module. Another function calculates gradient of the regularised cost function.
- `visualization.py` : It implements all the visualization functions for simplicity.

## Dependencies
* NumPy : for all the numerical computations
* Matplotlib : for data visualization

# Logistic Regression Details

## Mathematical Formulation

Logistic regression uses the sigmoid function to map a linear combination of features(z) to a probability between 0 and 1.

        sigmoid(z) = 1 / (1 + exp(-z))

        where ,z   - wx + b
               exp - exponential function
               w   - weight
               b   - bias

The model aims to minimize the cost function, which measusre the difference between predicted probabilities and actual class labels. A common cost function for logistic regression is cross-entropy.

The cost function for logistic regression is given by 
$$ J(\mathbf{w},b) = \frac{1}{m}\sum_{i=0}^{m-1} \left[ loss(f_{\mathbf{w},b}(\mathbf{x}^{(i)}), y^{(i)}) \right] \$$

where
* m is the number of training examples in the dataset


* $loss(f_{\mathbf{w},b}(\mathbf{x}^{(i)}), y^{(i)})$ is the cost for a single data point, which is also know as loss - 

    $$loss(f_{\mathbf{w},b}(\mathbf{x}^{(i)}), y^{(i)}) = (-y^{(i)} \log\left(f_{\mathbf{w},b}\left( \mathbf{x}^{(i)} \right) \right) - \left( 1 - y^{(i)}\right) \log \left( 1 - f_{\mathbf{w},b}\left( \mathbf{x}^{(i)} \right) \right) \tag{2}$$
    
    
*  $f_{\mathbf{w},b}(\mathbf{x}^{(i)})$ is the model's prediction, while $y^{(i)}$, which is the actual label

*  $f_{\mathbf{w},b}(\mathbf{x}^{(i)}) = g(\mathbf{w} \cdot \mathbf{x^{(i)}} + b)$ where function $g$ is the sigmoid function.
    * It might be helpful to first calculate an intermediate variable $z_{\mathbf{w},b}(\mathbf{x}^{(i)}) = \mathbf{w} \cdot \mathbf{x^{(i)}} + b = w_0x^{(i)}_0 + ... + w_{n-1}x^{(i)}_{n-1} + b$ where $n$ is the number of features, before calculating $f_{\mathbf{w},b}(\mathbf{x}^{(i)}) = g(z_{\mathbf{w},b}(\mathbf{x}^{(i)}))$


The gradient descent algorithm iteratively updated the model's weights(w) and bias(b) to minimize the cost function.
The mathematical regresentation og Gradient Descent algorithm is as follows:
$$\begin{align*}& \text{repeat until convergence:} \; \lbrace \newline \; & b := b -  \alpha \frac{\partial J(\mathbf{w},b)}{\partial b} \newline       \; & w_j := w_j -  \alpha \frac{\partial J(\mathbf{w},b)}{\partial w_j} \tag{1}  \; & \text{for j := 0..n-1}\newline & \rbrace\end{align*}$$

where, parameters $b$, $w_j$ are all updated simultaniously

 the `compute_gradient` function to compute $\frac{\partial J(\mathbf{w},b)}{\partial w}$, $\frac{\partial J(\mathbf{w},b)}{\partial b}$ from equations (2) and (3) below.

$$
\frac{\partial J(\mathbf{w},b)}{\partial b}  = \frac{1}{m} \sum\limits_{i = 0}^{m-1} (f_{\mathbf{w},b}(\mathbf{x}^{(i)}) - \mathbf{y}^{(i)}) \tag{2}
$$
$$
\frac{\partial J(\mathbf{w},b)}{\partial w_j}  = \frac{1}{m} \sum\limits_{i = 0}^{m-1} (f_{\mathbf{w},b}(\mathbf{x}^{(i)}) - \mathbf{y}^{(i)})x_{j}^{(i)} \tag{3}
$$
* m is the number of training examples in the dataset

    
*  $f_{\mathbf{w},b}(x^{(i)})$ is the model's prediction, while $y^{(i)}$ is the actual label


## L2 Regularization
Regularization is the term added to the cost function for preventing the model from overfitting.

The regularized cost function is given by 
$$J(\mathbf{w},b) = \frac{1}{m}  \sum_{i=0}^{m-1} \left[ -y^{(i)} \log\left(f_{\mathbf{w},b}\left( \mathbf{x}^{(i)} \right) \right) - \left( 1 - y^{(i)}\right) \log \left( 1 - f_{\mathbf{w},b}\left( \mathbf{x}^{(i)} \right) \right) \right] + \frac{\lambda}{2m}  \sum_{j=0}^{n-1} w_j^2$$


The difference is the regularization term, which is added to the end of the cost function to prevent model from overfitting $$\frac{\lambda}{2m}  \sum_{j=0}^{n-1} w_j^2$$ 

where 
* $\lambda$ is the regularization parameter
* $m$: Number of training examples
* $n$: Number of features (including the bias term)
* $w_j$: Weight associated with feature $j$

## Feature Mapping

To enhace model performance, feature mapping is employed. This involves creating additional features from existing ones. The `map_feature` function mapsfeatures into all polynomial terms of $x_1$ and $x_2$ up to the sixth power.

$$\mathrm{map\_feature}(x) = 
\left[\begin{array}{c}
x_1\\
x_2\\
x_1^2\\
x_1 x_2\\
x_2^2\\
x_1^3\\
\vdots\\
x_1 x_2^5\\
x_2^6\end{array}\right]$$

As a result of this mapping, our vector of two features has been transformed into a 27-dimensional vector. 

- A logistic regression classifier trained on this higher-dimension feature vector will have a more complex decision boundary and will be nonlinear when drawn in our 2-dimensional plot


This documentation provides a basic framework for understanding and using the Logistic Regression implementation. Feel free to explore the code further and experiment with different hyperparameters for your specific use case.