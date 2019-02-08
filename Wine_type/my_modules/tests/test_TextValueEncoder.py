import unittest
import sys, os
path = os.path.join(os.path.dirname(__file__), '..')
sys.path.append(path)
#print(path)

import numpy as np
import pandas as pd
# The taget of tests
from my_encoder import TextValueEncoder

class TestTextValueEncoder(unittest.TestCase):
    
    def __init__(self):
        fluits = ["apple & banana apple","maron","melon","grape & apple","lemon","melon & maron","orange","queie","orange"]
        self.data = pd.Series(fluits, name = 'fruits_name')

    def test_to_bow_encoding_1(self):
        encoder = TextValueEncoder(self.data)
        encoded = encoder.to_bow_encoding()
        print("test_to_bow_encoding_1")
        print(type(encoded))
        print(encoded)

if __name__ == "__main__":
    
    TestTextValueEncoder().test_to_bow_encoding_1()