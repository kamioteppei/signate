import unittest
import sys, os
path = os.path.join(os.path.dirname(__file__), '..')
sys.path.append(path)
#print(path)

import numpy as np
import pandas as pd
# The taget of tests
from my_encoder import NumericValueEncoder

class TestNumericValueEncoder(unittest.TestCase):
    
    def __init__(self):
        counts = [0, 1, 2, 4, 8, 16, 32]
        self.data = pd.Series(counts, name = 'counts')

    def test_normalize_1(self):
        encoder = NumericValueEncoder(self.data)
        encoded = encoder.normalize()
        print("test_normalize_1")
        print(type(encoded))
        print(encoded)

if __name__ == "__main__":
    
    TestNumericValueEncoder().test_normalize_1()