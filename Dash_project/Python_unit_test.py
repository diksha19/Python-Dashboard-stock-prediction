import unittest
import calc
import pandas as pd
#from Errors import Errors as e
from DescriptiveAnalysis import stats

# initialize list of lists
data = [['20', 10], ['25', 15], ['45', 20]]

# Create the pandas DataFrame
df = pd.DataFrame(data, columns=['Open', 'Close'])

# print dataframe.
df



class TestMyTestCase(unittest.TestCase):
    def test_stats(self):
        #print('ok')
        #result = calc.add(10,5)
        #self.assertEqual(result,15)

        result = stats(df, 'Mean')
        self.assertEqual(result,15)
        result = stats(df, 'Median')
        self.assertEqual(result, 15)
        result = stats(df, 'Mean')
        self.assertEqual(result, 1)

        # = stats(df, 'Median')
        #self.assertEqual(result, 703.359985)
        #result = stats(df, 'Mean')
        #self.assertEqual(result, 12345)




if __name__ == '__main__':
    unittest.main()
