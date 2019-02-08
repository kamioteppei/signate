import numpy as np
import pandas as pd
import warnings

class CategoryValueEncoder(object):
    
    __slots__ = ['series', 'series_name', 'unique_data', 'char_to_int', 'int_to_char']

    def __init__(self,series):
        self.series = series
        self.series_name = series.name
        self.unique_data = pd.unique(series)
        self.char_to_int = dict((c, i) for i, c in enumerate(self.unique_data))
        self.int_to_char = dict((i, c) for i, c in enumerate(self.unique_data))

    def to_label_encoding(self):
        #label encoding
        label_encoded = [self.char_to_int[char] for char in self.series]
        return self.to_dataframe(label_encoded)

    def to_one_hot_encoding(self):
        #one-hot encoding
        integer_encoded = [self.char_to_int[char] for char in self.series]
        onehot_encoded = list()
        for value in integer_encoded:
            letter = [0 for _ in range(len(self.unique_data))]
            letter[value] = 1
            onehot_encoded.append(letter)
        return self.to_dataframe(onehot_encoded)

    def to_binary_encoding(self, pad_len = None):
        #binary encoding

        '''
        this function is deprecated
        pros: it reduce pandas cloumns than one-hot encoding.
              it makes and read csv file faster than one-hot encoding. 
        cons: it take more times to convergence losses than one-hot encoding.
              it can't completely convergence losses when one-hot encoding do. 
        '''
        warn_msg = "`to_binary_encoding` is deprecated. but will not be removed."
        warnings.warn(warn_msg, UserWarning)
        
        if pad_len == None:
            pad_len = len("{0:b}".format(self.unique_data.size))
        integer_encoded = [self.char_to_int[char] for char in self.series]
        binary_encoded = [self.to_binary_array(int_val, pad_len) for int_val in integer_encoded]
        return self.to_dataframe(binary_encoded)
    
    def to_binary_array(self, int_val, pad_len):
        binary_char_array = "{0:b}".format(int_val).zfill(pad_len)
        binary_int_array = [int(char) for char in binary_char_array]
        return binary_int_array

    def to_dataframe(self, encoded):
        dataframe = pd.DataFrame(encoded)
        dataframe = dataframe.rename(lambda x: self.series_name + '_enc_' + str(x), axis='columns')
        return dataframe

from sklearn.feature_extraction.text import CountVectorizer

class TextValueEncoder(object):
    
    __slots__ = ['series', 'series_name', 'vectorizer']

    def __init__(self,series):
        self.series = series
        self.series_name = series.name
        self.vectorizer = CountVectorizer()
        #self.vectorizer = CountVectorizer(min_df=0.10, max_df=0.90) # You can range Document Frequency

    def to_bow_encoding(self):
        #bow encoding
        corpus = self.series.values
        bow_encoded = self.vectorizer.fit_transform(corpus)
        #feature_name = self.vectorizer.get_feature_names()
        return self.to_dataframe(bow_encoded.toarray())

    def to_dataframe(self, encoded):
        dataframe = pd.DataFrame(encoded)
        dataframe = dataframe.rename(lambda x: self.series_name + '_enc_' + str(x), axis='columns')
        return dataframe

    #def getvectorizer(self):
    #    return self.vectorizer
    
class DateValueEncoder(object):
    
    __slots__ = ['series', 'series_name']

    def __init__(self,series):
        self.series = series
        self.series_name = series.name

    def to_year(self):
        encoded = pd.to_datetime(self.series).dt.year
        return self.rename(encoded, 'year')

    def to_month(self):
        encoded = pd.to_datetime(self.series).dt.month
        return self.rename(encoded, 'month')

    def to_day(self):
        encoded = pd.to_datetime(self.series).dt.day
        return self.rename(encoded, 'day')

    def to_dayofweek(self):
        encoded = pd.to_datetime(self.series).dt.dayofweek
        return self.rename(encoded, 'dayofweek')
    
    def rename(self, encoded, title):
        renamed = encoded.rename(self.series_name + title, axis='columns')
        return renamed
    
class NumericValueEncoder(object):

    __slots__ = ['series', 'series_name']

    def __init__(self,series):
        self.series = series
        self.series_name = series.name
        
    def normalize(self):
        mean = self.series.mean()
        std = self.series.std()
        encoded = self.series.map(lambda x: round((x - mean) / std / 10 + 0.5 , 2)).astype(float)
        return encoded
        
if __name__ == "__main__":
    
    data = pd.Series(["apple","amazon","google","facebook","microsoft","apple","apple","amazon","amazon","google", None], name = 'company_name')
    encoder = CategoryValueEncoder(data)
    #encoded = encoder.to_label_encoding()
    #encoded = encoder.to_one_hot_encoding()
    encoded = encoder.to_binary_encoding()
    print(encoded)
    
    fluits = ["apple & banana apple","maron","melon","grape & apple","lemon","melon & maron","orange","queie","orange"]
    data = pd.Series(fluits, name = 'fruits_name')
    encoder = TextValueEncoder(data)
    encoded = encoder.to_bow_encoding()
    print(encoded)
    
    counts = [0, 1, 2, 4, 8, 16, 32]
    data = pd.Series(counts, name = 'counts')
    encoder = NumericValueEncoder(data)
    encoded = encoder.normalize()
    print(encoded)