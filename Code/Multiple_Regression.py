import numpy as np

def compute_cost(X, y, w, b):
    """
    Compute cost function (Mutiple Variable Regression)

    X (ndarray (m,n))       -> Feature matrix
    y (ndarray (m,))        -> Labels
    w (ndarray (n,))        -> Weight vector
    b (scalar)              -> Bias
    
    cost (float) -> The mean squared error cost
    """
    m = X.shape[0]                             # Numbere of rows
    f_wb = np.dot(X, w) + b                    # Prediction for each row # TODO
    cost = np.sum((f_wb - y) ** 2) / (2 * m)   # Mean of sqaured error 
    return cost

def compute_gradient(X, y, w, b, lambda_=0.001): 
    """
    Compute gradient (Mutiple Variable Regression)
    
    X (ndarray (m,n)) -> Feature matrix
    y (ndarray (m,))  -> Labels
    w (ndarray (n,))  -> Weight vector
    b (scalar)        -> Bias

    Returns:
    dj_dw (ndarray (n,)) -> Gradient with respect to w
    dj_db (scalar)       -> Gradient with respect to b
    """
    m = X.shape[0]              # Number of rows
    f_wb = np.dot(X, w) + b     # Prediction for each row
    error = f_wb - y            # Error in predictions (derivative)

    dj_dw = (1/m) * np.dot(X.T, error) + (lambda_ * w / m) # Gradient for weights (derivative) + Regularization
    dj_db = (1/m) * np.sum(error)                          # Gradient for bias (derivative)
    
    return dj_dw, dj_db

def gradient_descent(X, y, w, b, alpha=0.0001, lambda_=0.001, iters=1000, cost_func=compute_cost, gradient_func=compute_gradient, history_print=False): 
    """
    Compute gradient descent (Mutiple Variable Regression)
    Performs gradient descent to optimize the parameters w and b. 
    
    
    X (ndarray (m,n))    -> Feature data with m examples
    y (ndarray (m,))     -> Labels
    w (ndarray)          -> Initial weights
    b (scalar)           -> Initial bias
    alpha (float)        -> Learning rate
    num_iters (int)      -> Number of iterations
    cost_function        -> Function to compute cost
    gradient_function    -> Function to compute gradients
    history_print (bool) -> Whether to print updates and store history

    w (scalar)               -> Updated weight
    b (scalar)               -> Updated bias
    If history_print is True:
      cost_history (list)    -> List of cost values
      weights_history (list) -> List of [w, b] pairs
    """
    cost_history = []
    weights_history = []

    for i in range(iters):
        dj_dw, dj_db = gradient_func(X, y, w, b, lambda_)
        
        w -= alpha * dj_dw  # Update weight
        b -= alpha * dj_db  # Update bias

        # Store history if requested
        if history_print and (i % (iters // 10) == 0 or i == iters - 1):
            cost = cost_func(X, y, w, b)
            cost_history.append(cost)
            weights_history.append((w.copy(), b))
            print(f"Iteration {i}: Cost {cost:.2e}, w: {[round(x, 3) for x in w]}, b: {b:.3e}")

    return (w, b) if not history_print else (w, b, cost_history, weights_history)

def zscore_normalization(X):
    
    mean = X.mean(axis=0)
    std = X.std(axis=0)

    X_normalized = (X - mean) / std

    return X_normalized





