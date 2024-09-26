import unittest
import numpy as np
import pandas as pd
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Code.Mutiple_Regression import compute_cost, compute_gradient, gradient_descent

class TestMultipleRegression(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.data = pd.read_csv(os.path.join(os.path.dirname(__file__), 'Test_Data', 'Student_Performance.csv'))
        self.X = self.data[["Hours Studied", "Previous Scores", "Extracurricular Activities", "Sleep Hours", "Sample Question Papers Practiced"]].values
        self.y = self.data["Performance"].values


    def setUp(self):
        self.w = np.array([0.1, 0.2, 0.3, 0.4, 0.5])  
        self.b = 0.1
        self.alpha = 0.01
        self.num_iters = 1000

    def test_data(self):
        self.assertIsNotNone(self.X)
        self.assertIsNotNone(self.y)

    def test_compute_cost(self):
        self.assertEqual(compute_cost(self.X, self.y, self.w, self.b), 763.8617009999999)

    def test_compute_gradient(self):
        answ_dj_dw, answ_dj_db = np.array([ -196.12941, -2722.6363 ,   -17.80053,  -233.41586,  -161.7188 ]), -35.68404
        test_dj_dw, test_dj_db = compute_gradient(self.X, self.y, self.w, self.b)
        self.assertTrue(np.allclose(answ_dj_dw, test_dj_dw, rtol=1e-5, atol=1e-5))
        self.assertAlmostEqual(answ_dj_db, test_dj_db)

    def test_gradient_descent(self):
        answ_w, answ_b = np.array([ 2.23879558,  0.83110804, -0.45998607, -1.37926589, -0.23638972]), -2.537177157886497 # For alpha=0.003 and iters=10000
        test_w, test_b, J_history, wb_history = gradient_descent(self.X, self.y, self.w, self.b, alpha=0.0003, iters=10000, history_print=True)
        self.assertTrue(np.allclose(answ_w, test_w, rtol=1e-5, atol=1e-5))
        self.assertAlmostEqual(answ_b, test_b)
        f_wb = np.dot(test_w, self.X[0]) + test_b
        print("\nAlpha = 0.0003 | Iterations = 10000")
        print("My AI Guess: ", f_wb, "\nANSW: ", self.y[0])


if __name__ == '__main__':
    unittest.main()