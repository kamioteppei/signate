import unittest
import sys, os
path = os.path.join(os.path.dirname(__file__), '..')
sys.path.append(path)
#print(path)

import numpy as np
import pandas as pd
# The taget of tests
from my_encoder import CategoryValueEncoder

class TestCategoryValueEncoder(unittest.TestCase):

    def __init__(self):
        self.data = pd.Series(["apple","amazon","google","facebook","microsoft","apple","apple","amazon","amazon","google", None], name = 'company_name')

    def test_to_label_encoding_1(self):
        encoder = CategoryValueEncoder(self.data)
        encoded = encoder.to_label_encoding()
        print("test_to_label_encoding_1")
        print(type(encoded))
        print(encoded)

    def test_to_one_hot_encoding_1(self):
        encoder = CategoryValueEncoder(self.data)
        encoded = encoder.to_one_hot_encoding()
        print("test_to_one_hot_encoding_1")
        print(type(encoded))
        print(encoded)        

    def test_to_binary_encoding_1(self):
        encoder = CategoryValueEncoder(self.data)
        encoded = encoder.to_binary_encoding()
        print("test_to_binary_encoding_1")
        print(type(encoded))
        print(encoded)        

    def test_to_binary_encoding_2(self):
        encoder = CategoryValueEncoder(self.data)
        encoded = encoder.to_binary_encoding(2)
        print("test_to_binary_encoding_2")
        print(type(encoded))
        print(encoded)        

    def test_to_binary_encoding_3(self):
        encoder = CategoryValueEncoder(self.data)
        encoded = encoder.to_binary_encoding(3)
        print("test_to_binary_encoding_3")
        print(type(encoded))
        print(encoded)        

    def test_to_binary_encoding_4(self):
        encoder = CategoryValueEncoder(self.data)
        encoded = encoder.to_binary_encoding(4)
        print("test_to_binary_encoding_4")
        print(type(encoded))
        print(encoded)        
        
if __name__ == "__main__":
    
    TestCategoryValueEncoder().test_to_label_encoding_1()
    TestCategoryValueEncoder().test_to_one_hot_encoding_1()
    TestCategoryValueEncoder().test_to_binary_encoding_1()
    TestCategoryValueEncoder().test_to_binary_encoding_2()
    TestCategoryValueEncoder().test_to_binary_encoding_3()
    TestCategoryValueEncoder().test_to_binary_encoding_4()