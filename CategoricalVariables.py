'''
    Resources for data wrangling. Mostly for dichotomic classification.
'''

import pandas as pd
import scipy.stats as stats
import numpy as np

def Cramers_V(data: pd.DataFrame, var1: str, var2: str) -> np.float:
    '''
        Cramers' V is a metric used to test the correlation
        between two categorical variables.

        input:
                data: pd.DataFrame
                var1: str Name of the first variable.
                var2: str Name of the second variable

        output:
                : float Cramers' V.

    '''
    cont_table = pd.crosstab(data[var1], data[var2])
    X2 = stats.chi2_contingency(cont_table, correction=False)[0]
    n = cont_table.sum().sum()
    minDim = min(cont_table.shape) - 1
    return np.sqrt((X2/n) / minDim)


def WoE_computation(data: pd.DataFrame, variable: str, woe_name: str, target: str) -> pd.DataFrame:
    '''
        This function computes the Weight Of Evidence. This is a way of encoding
        categorical variables in a single continous variable using the target.

        The WoE is widely used in finance modelisation and requires the target 
        variable to be dichotomic.

        input:
                data: pd.DataFrame
                variable: str name of the variable
                woe_name: str name of the woe variable
                target: str name of the target variable. The target variable
                        needs to be dichotomic and take values {0, 1}
        output:
                woe: pd.DataFrame
                
    '''

    woe = data.groupby(variable).agg({'default': ['count', 'sum']})
    woe.columns = ['_'.join(col) for col in woe.columns.values]

    #We compute the total WoE
    total_default = data[target].mean()
    tot_WoE = np.log((1-total_default)/total_default)

    # Each WoE
    woe[woe_name] = tot_WoE - np.log((woe[target+'_count'] - woe[target+'_sum'])/woe[target+'_sum'])
    return woe

def create_WoE_column(data: pd.DataFrame, woe_table: pd.DataFrame, woe_name: str) -> pd.DataFrame:
    '''
        This function creates a woe column. It's fed with the output of the `WoE_computation`
        function.
    '''

    variable = woe_table.index.name
    # print(variable)
    #data.set_index(variable, inplace=True)
    return pd.merge(data, woe_table[woe_name], how='left', on=variable, suffixes=('', ''))