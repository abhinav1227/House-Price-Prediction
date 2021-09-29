#importing all the libraries

import pandas as pd
import numpy as np

'''import seaborn as sns
import matplotlib.pyplot as plt
'''
from scipy.stats import skew, norm
from scipy.stats import boxcox_normmax
from scipy.special import boxcox1p

from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import (VarianceThreshold, SelectPercentile,
                                       mutual_info_regression)
'''from sklearn.model_selection import (train_test_split, cross_val_score,
                                     GridSearchCV, RepeatedKFold)
from sklearn.linear_model import (Ridge, RidgeCV, Lasso, LassoCV, ElasticNet,
                                 LassoLarsCV, LinearRegression)
from sklearn.metrics import mean_absolute_error
import xgboost as xgb
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from mlxtend.regressor import StackingCVRegressor
'''
#functions

def missing_impute(df):
    '''
    We are operating under the assumptio. that the nan values are nothing more
    than the 'NA' value present in the data description, and imputing based on
    this assumption
    '''

    ###categorical columns

    #data with large chunks of missing value
    max_missing_feats = ['Alley', 'MiscFeature', 'Fence', 'PoolQC', 'FireplaceQu']

    #for grage categorical features
    garage_feats = ['GarageCond', 'GarageQual', 'GarageFinish', 'GarageType']

    #basement categorical columns
    bsmt_feats = ['BsmtCond','BsmtQual','BsmtExposure','BsmtFinType1',
                    'BsmtFinType2']
    #others
    other_feat = ['MasVnrType']

    #fill_by_mode
    mode_feat = ['Electrical']

    a=max_missing_feats+garage_feats+bsmt_feats+other_feat
    for feat in a:
        df[feat] = df[feat].fillna('None')

    for feat in mode_feat:
        df[feat] = df[feat].fillna(df[feat].mode()[0])

    ### numerical columns

    num_feats = ['GarageYrBlt', 'MasVnrArea']
    for feat in num_feats:
        df[feat] = df[feat].fillna(df[feat].mean())

    #lost frontage
    df.groupby('Neighborhood')['LotFrontage'].mean()

    '''To achieve the intended result(i.e use the values above)
    we use 'transform' method'''

    df['LotFrontage']=df.groupby('Neighborhood')['LotFrontage'].transform(
    lambda value: value.fillna(value.mean()))

    return df

def categorical_encoding(df):

    import pandas as pd
    #Since MSSubClass is a categorical feature, but recognised as int
    df.loc[:,'MSSubClass'] =  df.loc[:,'MSSubClass'].apply(str)
    cat_features = list(df.select_dtypes(include='object').columns)

    #features in diff cols
    '''for col in cat_features:
        print(col,'  ', df[col].unique())
'''
    #viewing features with large number of categories
    '''for col in cat_features:
        print(col,'  ',len(df[col].unique()), 'labels')
'''
    #imputing features with large no of categories
    #filtering to avoid the cure of dimensionality

    feat_top_10 = ['Neighborhood','Condition2', 'Condition1']
    feat_top_5 = ['HouseStyle','RoofMatl']

    #one hot encoding for nominal categorical features
    feat_one_hot = ['MSSubClass', 'Street','LotShape', 'LandContour', 'LotConfig','LandSlope', 'BldgType',
                    'RoofStyle', 'MasVnrType', 'Foundation', 'Heating', 'CentralAir','Electrical',
                    'GarageType', 'PavedDrive', 'SaleCondition','GarageFinish','Fence','Alley','MiscFeature',
                    'BsmtExposure','GarageYrBlt', 'BsmtFinType1','BsmtFinType2']

    # label encoding for ordinal categorical features
    ordinal_feat = ['ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond',
                    'HeatingQC', 'GarageQual', 'GarageCond',
                    'PoolQC','FireplaceQu']


    for feat in feat_top_10:
        top_10 = [x for x in df[feat].value_counts().sort_values(
            ascending=False).head(10).index]
        for label in top_10:
            df.loc[:,feat+'_'+label]=np.where(df.loc[:,feat]==label, 1, 0)

    for feat in feat_top_5:
        top_5 = [x for x in df[feat].value_counts().sort_values(
            ascending=False).head(10).index]
        for label in top_5:
            df.loc[:,feat+'_'+label]=np.where(df.loc[:,feat]==label, 1, 0)

    def dummies(x, df1):
        temp = pd.get_dummies(df1[x], prefix=x, drop_first=True).astype('int32')
        df1 = pd.concat([df1, temp], axis=1)
        df1.drop([x], axis=1, inplace=True)
        return df1

    for feat in feat_one_hot:
        df = dummies(feat, df)


    le=LabelEncoder()
    for feat in ordinal_feat:
        df.loc[:,feat]=le.fit_transform(df.loc[:,feat])

    df.drop(columns=feat_top_5, inplace=True)
    df.drop(columns=feat_top_10, inplace=True)
    return df

def feature_Selection(df, sale_price):

    num_features = list(df.select_dtypes(include=('int64','float64')).columns)

    ###dropping constant features using variance threshold




    var_thre = VarianceThreshold(threshold=0)
    var_thre.fit(df[num_features])

    const_columns = [col for col in num_features if col not in
                     df[num_features].columns[var_thre.get_support()]]
    print('No of constant columns:',len(const_columns))


    ###using correlation coeffiecient, to avoid duplicacy


    df_num = df[num_features]

    #finding how many features are related to each other

    def correlation(df1, threshold):
        corr_col = set() #to avoid duplicacy
        corr_matrix = df1.corr()
        for i in range(len(corr_matrix.columns)):
            for j in range(i):
                if corr_matrix.iloc[i,j]>threshold:
                    colname = corr_matrix.columns[i]
                    corr_col.add(colname)
        return corr_col

    correlated_features = correlation(df_num, 0.7)
    len(correlated_features)

    #we will drop these correlated features

    df.drop(correlated_features, axis=1, inplace=True)


    ###using information gain, mutual information is synonym to IG





    mutual_info = mutual_info_regression(df_num, sale_price)
    mutual_info = pd.Series(mutual_info)
    mutual_info.index = df_num.columns
    mutual_info.sort_values(ascending=False)



    selected_top_cols = SelectPercentile(mutual_info_regression, percentile=40)
    selected_top_cols.fit(df_num, sale_price)

    selected_top_cols.get_support()

    unimp_cols = [col for col in num_features if col not in df_num.columns
                 [selected_top_cols.get_support()]]
    df.drop(unimp_cols, axis=1, inplace=True)
    return df

def normalization(df, sale_price):
    #skew and kurtosis of sale price

    print('Skewness: %f' % sale_price.skew())
    print('Kurtosis: %f' % sale_price.kurt())

    #applying log transformation to remove skewness since its right skewed

    sale_price = np.log1p(sale_price)


    #fixing skewed numerical column

    #finding skewed numerical cols

    num_features = list(df.select_dtypes(include=('int64','float64')).columns)
    skews_column = df[num_features].apply(lambda x: skew(x)).sort_values(
        ascending=False)
    high_skew = skews_column[skews_column>0.5]
    skew_index = high_skew.index

    print(f'There are {high_skew.shape[0]} numerical features with Skew  > 0.5')
    skews_column

    #normalizing skewed features
    for i in skew_index:
        df[i] = boxcox1p(df[i], boxcox_normmax(df[i]+1))

    return df, sale_price

def predict_house_price(config, model):
    if type(config) == dict:
        df = pd.DataFrame(config)
    else:
        df = config
    y_pred = model.predict(df)
    return y_pred

