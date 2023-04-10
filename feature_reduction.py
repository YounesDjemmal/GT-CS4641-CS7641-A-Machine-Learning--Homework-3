import pandas as pd
import statsmodels.api as sm

class FeatureReduction(object):
    def __init__(self):
        pass

    @staticmethod
    def forward_selection(data, target, significance_level=0.05): # 9 pts
        '''
        Args:
            data: (pandas data frame) contains the feature matrix
            target: (pandas series) represents target feature to search to generate significant features
            significance_level: (float) threshold to reject the null hypothesis
        Return:
            forward_list: (python list) contains significant features. Each feature
            name is a string
        '''
        initial_features = data.columns.tolist()
        forward_list = []

        while (len(initial_features)>0):
            rem_features = list(set(initial_features)-set(forward_list))
            new_pval = pd.Series(index =rem_features)

            for col in rem_features:
                model = sm.OLS(target, sm.add_constant(data[forward_list+[col]])).fit()
                new_pval[col] = model.pvalues[col]

            min_p_value = new_pval.min()

            if(min_p_value<significance_level):
                forward_list.append(new_pval.idxmin())
            else:
                break
        return forward_list

    @staticmethod
    def backward_elimination(data, target, significance_level = 0.05): # 9 pts
        '''
        Args:
            data: (pandas data frame) contains the feature matrix
            target: (pandas series) represents target feature to search to generate significant features
            significance_level: (float) threshold to reject the null hypothesis
        Return:
            backward_list: (python list) contains significant features. Each feature
            name is a string
        '''
        backward_list = data.columns.tolist()
        while(len(backward_list)>0):
            feature_plus_const = sm.add_constant(data[backward_list])
            p_values = sm.OLS(target, feature_plus_const).fit().pvalues[1: ]

            max_p_value = p_values.max()
            if(max_p_value >= significance_level):
                excluded_feature = p_values.idxmax()
                backward_list.remove(excluded_feature)
            else:
                break 
        return backward_list
