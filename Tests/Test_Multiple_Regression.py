import unittest
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import os
"""
This line adds the parent directory of the current script to the Python path
It helps Python find modules in that directory when importing them (Code.Multiple_Regression)
'os.path.dirname(__file__)' gets the directory of the current script
'os.path.join(..., '..')' navigates to the parent directory
and 'os.path.abspath(...)' ensures the path is absolute
"""
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Code.Multiple_Regression import compute_cost, compute_gradient, gradient_descent, zscore_normalization

class TestMultipleRegression(unittest.TestCase):
    """ Unittest class """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        """
        This line reads the "Student_Performance.csv" file from the "Test_Data" directory. 
        It uses `os.path.join()` to create the correct file path, ensuring compatibility across 
        different operating systems. `os.path.dirname(__file__)` gets the directory of the current script, 
        and then "Test_Data" and "Student_Performance.csv" are added to form the complete path. 
        """
        self.data = pd.read_csv(os.path.join(os.path.dirname(__file__), "Test_Data", "Student_Performance.csv"))

        # Extract features (self.X) and target (self.y) variable from the DataFrame
        m_Xy= int(self.data.shape[0] * 0.75)

        self.X_train = self.data[["Hours Studied", "Previous Scores", "Extracurricular Activities", "Sleep Hours", "Sample Question Papers Practiced"]].values[:m_Xy]
        self.X_test = self.data[["Hours Studied", "Previous Scores", "Extracurricular Activities", "Sleep Hours", "Sample Question Papers Practiced"]].values[m_Xy:]

        self.y_train = self.data["Performance"].values[:m_Xy]
        self.y_test = self.data["Performance"].values[m_Xy:]



    def setUp(self):
        """Set up the initial parameters for the tests"""
        self.w_in = np.array([0.1, 0.2, 0.3, 0.4, 0.5])  
        self.b_in = 0.1
        self.alpha = 0.01
        self.num_iters = 1000
        self.X_train_normalize = zscore_normalization(self.X_train)
        self.X_test_normalize = zscore_normalization(self.X_test)


    def test_data(self):
        """Test the data to ensure the data is right loaded"""
        self.assertIsNotNone(self.X_train)
        self.assertIsNotNone(self.y_train)
        self.assertIsNotNone(self.X_test)
        self.assertIsNotNone(self.y_test)

    def test_gradient_descent(self):
        """Test the gradient_descent function for expected output."""
        
        w_best = [2.854, 1.018, 0.654, 0.472, 0.193]
        b_best = -3.406e+01
        alpha, lambda_, iters = 0.0003, 1, 10000
        test_w, test_b, J_history, wb_history = gradient_descent(self.X_train, self.y_train, self.w_in, self.b_in, alpha=alpha, lambda_=lambda_, iters=iters, history_print=True)

        # Predictions for all test examples using computed w and b
        predictions = np.dot(self.X_test, test_w) + test_b

        # Expected predictions using w_best and b_best
        expected_predictions = np.dot(self.X_test, w_best) + b_best

        # Parameters print
        print(f"\n| Test Gradient Descent\n| Paramenters: w_in: {self.w_in} | b_in: {self.b_in}\n|\t       alpha: {alpha} | lambda: {lambda_} | iterations: {iters}")

        n = self.y_test.shape[0]

        # Mean Absolute Error
        tol = 8
        mean_absolute_error = sum(abs(self.y_test - predictions)) / n
        assert mean_absolute_error < tol, "The Mean Absolute Error is higher than the tolarance ({tol})"
        print(f"|-> Mean Absolute Error: {mean_absolute_error}")
        
        # Mean Squared Error
        tol = 30
        mse = sum((self.y_test[i] - predictions[i]) ** 2 for i in range(n)) / n
        assert mse < tol, "Thee Mean Squared Error is higher than the tolarance ({tol})"
        print(f"|-> Mean Squared Error (MSE): {mse}")

    
        # Plot the data for analysis
        if J_history:
            plot_cost_history(J_history)
        plot_predictions_vs_actuals(self.y_test, predictions)
        feature_names = ["Hours Studied","Previous Scores","Extracurricular Activities","Sleep Hours","Sample Question Papers Practiced"]
        plot_weights(test_w, feature_names)
        

    def test_zscore_normalization(self):
        """Test the zscore_normalization funtion for expected output"""
        # Compare the first row of the normalizated version
        answ_normalization_0 = [ 0.77518771,  1.70417565,  1.01045465,  1.45620461, -1.24975394]
        #self.assertTrue(np.allclose(self.X_normalize[0], answ_normalization_0), "Normalized values do not match expected values.")

def plot_cost_history(J_history):
    fig, (ax1, ax2) = plt.subplots(1, 2, constrained_layout=True, figsize=(12, 4))
    
    # Plotting all the cost history
    ax1.plot(J_history)
    ax1.set_title("Cost vs. Iteration")
    ax1.set_ylabel('Cost')
    ax1.set_xlabel('Iteration (1000)')

    # Plotting the tail of the cost history
    tail = len(J_history)//10
    print(tail)
    ax2.plot(tail + np.arange(len(J_history[tail:])), J_history[tail:])
    ax2.set_title("Cost vs. Iteration (Tail)")
    ax2.set_ylabel('Cost')
    ax2.set_xlabel('Iteration (1000)')
    
    plt.show()

def plot_predictions_vs_actuals(y_test, predictions):
    plt.figure(figsize=(10, 6))
    plt.plot(y_test, label='Actual Values', marker='o', color='green', linestyle='None')
    plt.plot(predictions, label='Predictions', marker='x', color='red', linestyle='None')
    plt.xlabel('Examples')
    plt.ylabel('Performance')
    plt.title('Predictions vs. Actual Values')
    plt.legend()
    plt.grid(True) 
    plt.show()

def plot_weights(weights, feature_names):
    plt.figure(figsize=(10, 6))
    plt.bar(feature_names, weights, color='purple')
    plt.xlabel('Features')
    plt.ylabel('Weights')
    plt.title('Feature Weights after Gradient Descent')
    plt.xticks(rotation=45) 
    plt.grid(axis='y')  
    plt.tight_layout()  
    plt.show()


if __name__ == '__main__':
    unittest.main()