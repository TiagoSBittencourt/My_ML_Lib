import unittest
import numpy as np
import pandas as pd
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
        self.X = self.data[["Hours Studied", "Previous Scores", "Extracurricular Activities", "Sleep Hours", "Sample Question Papers Practiced"]].values
        self.y = self.data["Performance"].values


    def setUp(self):
        """Set up the initial parameters for the tests"""
        self.w = np.array([0.1, 0.2, 0.3, 0.4, 0.5])  
        self.b = 0.1
        self.alpha = 0.01
        self.num_iters = 1000
        self.X_normalize = zscore_normalization(self.X)

    def test_data(self):
        """Test the data to ensure the data is right loaded"""
        self.assertIsNotNone(self.X)
        self.assertIsNotNone(self.y)

    def test_compute_cost(self):
        """Test the compute_cost function for expected output"""
        self.assertEqual(compute_cost(self.X, self.y, self.w, self.b), 763.8617009999999)

    def test_compute_gradient(self):
        """Test the compute_gradient function for expected output"""
        answ_dj_dw, answ_dj_db = np.array([ -196.12941, -2722.6363 ,   -17.80053,  -233.41586,  -161.7188 ]), -35.68404
        test_dj_dw, test_dj_db = compute_gradient(self.X, self.y, self.w, self.b)
        self.assertTrue(np.allclose(answ_dj_dw, test_dj_dw, rtol=1e-5, atol=1e-5))
        self.assertAlmostEqual(answ_dj_db, test_dj_db)

    def test_gradient_descent(self):
        """Test the gradient_descent function for expected output"""
        answ_w, answ_b = np.array([ 2.23879558,  0.83110804, -0.45998607, -1.37926589, -0.23638972]), -2.537177157886497 # For alpha=0.003 and iters=10000
        test_w, test_b, J_history, wb_history = gradient_descent(self.X, self.y, self.w, self.b, alpha=0.0003, iters=10000, history_print=True)
        self.assertTrue(np.allclose(answ_w, test_w, rtol=1e-5, atol=1e-5))
        self.assertAlmostEqual(answ_b, test_b)
        f_wb = np.dot(test_w, self.X[0]) + test_b # AI prediction for the first line
        print("\nAlpha = 0.0003 | Iterations = 10000")
        print("My AI Guess: ", f_wb, "\nANSW: ", self.y[0]) 
    
    def test_zscore_normalization(self):
        """Test the zscore_normalization funtion for expected output"""
        # Compare the first row of the normalizated version
        answ_normalization_0 = [ 0.77518771,  1.70417565,  1.01045465,  1.45620461, -1.24975394]
        self.assertTrue(np.allclose(self.X_normalize[0], answ_normalization_0), "Normalized values do not match expected values.")

    def test_gradient_descent_normalize(self):
        """Test the gradient_descent function for expected output"""
        test_w, test_b, J_history, wb_history = gradient_descent(self.X_normalize, self.y, self.w, self.b, alpha=0.0003, iters=10000, history_print=True)
        f_wb = np.dot(test_w, self.X_normalize[0]) + test_b # AI prediction for the first line
        print("\nAlpha = 0.0003 | Iterations = 10000")
        print("My AI Guess: ", f_wb, "\nANSW: ", self.y[0]) 


if __name__ == '__main__':
    unittest.main()