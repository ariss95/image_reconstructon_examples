from myDCTdict import DCTDictionary
#from sparselandtools.dictionaries import DCTDictionary
import matplotlib.pyplot as plt
from math import sqrt

# create dictionary
dct_dictionary = DCTDictionary(16, 32)
plt.imshow(dct_dictionary, cmap='gray')
print(dct_dictionary.shape)
print(type(dct_dictionary))
plt.title('matplotlib.pyplot.imshow() function Example', 
                                     fontweight ="bold")
plt.show()