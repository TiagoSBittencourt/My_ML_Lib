import numpy as np

def compute_cost(x, y, w, b):
    """
    Compute cost funcion (LINEAR REGRESSION)

    x (ndarray (m,)) -> Feature
    y (ndarray (m,)) -> Labels
    w,b (scalar)     -> Weight and Bias

    cost (int)       -> Loss
    """
    m = x.shape[0] # Number of data rows
    cost = 0
    
    for i in range(m):
        f_wb = w * x[i] + b              # ŷ
        cost += (f_wb - y[i])**2         # (ŷ - y)^2
    cost /= (2 * m)                      # AVG() / (2 * m)

    return cost

def compute_gradient(x, y, w, b): 
    """
    Compute gradient function (LINEAR REGRESSION)
    
    x (ndarray (m,)) -> Feature
    y (ndarray (m,)) -> Labels
    w,b (scalar)     -> Weight and Bias 

    dj_dw (scalar)   -> Gradient (parameter w)
    dj_db (scalar)   -> Gradient (parameter b)  
    """
    
    m = x.shape[0] # Number data rows
    dj_dw = 0
    dj_db = 0
    
    for i in range(m):  
        f_wb = w * x[i] + b             # ŷ
        dj_dw += (f_wb - y[i]) * x[i]   # d((ŷ - y)^2)/dw
        dj_db += f_wb - y[i]            # d((ŷ - y)^2)/db
    dj_dw /= m                          # AVG() / (2 * m)
    dj_db /= m 
        
    return dj_dw, dj_db

def gradient_descent(x, y, w, b, alpha, num_iters, cost_function, gradient_function, history_print = False): 
    """
    Compute gradient descent (LINEAR REGRESSION)
    Performs gradient descent to optimize the parameters w and b. 
    
    
    x (ndarray (m,))     -> Feature data with m examples
    y (ndarray (m,))     -> Labels
    w (scalar)           -> Initial weight
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

    for i in range(num_iters):
        dj_dw, dj_db = gradient_function(x, y, w, b)
        
        w -= alpha * dj_dw  # Update weight
        b -= alpha * dj_db  # Update bias

        # Store history if requested
        if history_print and (i % (num_iters // 10) == 0 or i == num_iters - 1):
            cost = cost_function(x, y, w, b)
            cost_history.append(cost)
            weights_history.append((w, b))
            print(f"Iteration {i}: Cost {cost:.2e}, w: {w:.3e}, b: {b:.3e}")

    return (w, b) if not history_print else (w, b, cost_history, weights_history)