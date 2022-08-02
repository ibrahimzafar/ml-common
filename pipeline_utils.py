from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
import pandas as pd
import lightgbm as lgbm
from lifelines import CoxPHFitter
import shap


class ToPandas(BaseEstimator, TransformerMixin):
    # credits to Abhijeet Pokhriyal
    def __init__(self , colnames):
        self.colnames = colnames

    def fit(self,X,y=None):
        return self

    def transform(self, X,y=None):
        df = pd.DataFrame(X , columns=self.colnames)
        return df

def to_string(x):
  return pd.DataFrame(x).astype('str')

def to_float(x):
  return pd.DataFrame(x).astype('float')

str_transform = FunctionTransformer(to_string)
float_transform = FunctionTransformer(to_float)


def get_pipeline(numeric_features, categorical_features, passthrough_features=[]):
    numeric_transformer = Pipeline(
                                    steps=[
                                        ('to_float', float_transform),
                                        ("impute_num_col", SimpleImputer(strategy="median"))#,
                                        # ("scaler", StandardScaler())
                                        ]
                                )
    categorical_transformer = Pipeline(
                                    steps=[
                                        ('to_string', str_transform),
                                        ('imputer_cat_col', SimpleImputer(strategy="most_frequent")),
                                        ('ordinal_encode_categories', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))#,
                                    # 
                                    # ('toPandas', ToPandas(numeric_features))
                                    ]
                                )
    numpreprocessor = ColumnTransformer(
        transformers=[
            ("numerical", numeric_transformer, numeric_features),
        ], remainder='drop', sparse_threshold=0
    )
    catpreprocessor = ColumnTransformer(
        transformers=[
            ("categorical", categorical_transformer, categorical_features)#,
        ], remainder='drop', sparse_threshold=0
    )
    passthroughpreprocessor = ColumnTransformer(
    transformers=[
        ("passthrough_columns", 'passthrough', passthrough_features),
    ], remainder='drop'
    )

    numpipeline = Pipeline(steps=[("numpreprocessor", numpreprocessor)])
    catpipeline = Pipeline(steps=[("catpreprocessor", catpreprocessor)])
    passthroughpipeline = Pipeline(steps=[("passthroughpreprocessor", passthroughpreprocessor)])

    data_pipeline = Pipeline(steps = [
                                ('combine_num_and_cat_cols', FeatureUnion([('numpipeline', numpipeline), ('catpipeline', catpipeline), ('passthroughpipeline', passthroughpipeline)])), 
                                ('toPandas', ToPandas(numeric_features+categorical_features+passthrough_features))
                                    ])
    return data_pipeline




def create_encoder(encoder, df, columns):
    """
    Usage:
    encoder = create_encoder(OrdinalEncoder(), X_train, categorical_features)
    """
    return encoder.fit(df[columns])
    
def get_categories(encoder, row, columns):
    inv_tr = encoder.inverse_transform(row[columns])
    return pd.DataFrame({'feature':columns, 'value':inv_tr[0]})

def get_feature_mapping(encoder, categorical_features):
    """
    Usage:
    cat_feat_mapping = get_feature_mapping(encoder)
    """
    encoding = encoder.categories_
    encoding_mapping = {categorical_features[i]:dict(zip((encoding[i]), range(len(encoding[i])))) for i in range(len(categorical_features))}
    return encoding_mapping

# encoder = data_pipeline.named_steps['combine_num_and_cat_cols'].transformer_list[1][1]['catpreprocessor'].transformers[0][1]['ordinal_encoder']
def get_prediction_example(encoder, df, y, row_num, model, categorical_features, shap_tree_explainer):
    """
    Usage:
    get_prediction_example(encoder, X_test_features, y_test, 6, lgbm_model, categorical_features, tree_explain)
    """
    print(get_categories(encoder, df.iloc[row_num:row_num+1], categorical_features))
    print('Actual: ', y.iloc[row_num:row_num+1].values)
    print('Predicted: ', model.predict(df.iloc[row_num:row_num+1]))
    shap_valuess = shap_tree_explainer(df.iloc[row_num:row_num+1])
    # visualize the first prediction's explanation
    shap.plots.waterfall(shap_valuess[0])

def bin_minority_classes(df, col, fill_val='other', threshhold=0.05):
    """
    Usage:
    bin_minority_classes(df, 'household_composition', 'U')
    bin_minority_classes(df, 'channel', 'other')
    """
    tmp = df[col].value_counts(normalize=True).reset_index()
    minority = tmp[tmp[col]<threshhold]['index'].values
    majority = tmp[tmp[col]>=threshhold]['index'].values
    minmap_dict = {grp:fill_val for grp in df[col].unique() if grp in minority}
    majmap_dict = {grp:grp for grp in df[col].unique() if grp in majority}
    minmap_dict.update(majmap_dict)
    df[col] = df[col].map(minmap_dict)

def impute_missing_values(df, impute_dict):
    """
    Usage:
    impute_dict = {'person_1_occupation_group_v2':'0U', 
    'person_1_marital_status': '0U',
    'estimated_household_income_range_code_v6': 'U',
    'home_business': 'U',
    'behaviorbank_presence_of_credit_card':'U',
    'propertyrealty_home_heat_indicator':0,
    'propertyrealty_home_air_conditioning':0,
    'channel': 'UNKNOWN'}
    impute_missing_values(df, impute_dict)
    """
    for key in impute_dict.keys():
        df[key] = df[key].fillna(impute_dict[key])

def map_values(df, replace_dict):
    """
    Usage:
    replace_dict = {'person_1_marital_status': [('5U', '0U'), ('5M', 'M'), ('1M', 'M')], 
                     'dwelling_type': [('A', 'MF'), ('M', 'MF'), ('S', 'SF')]}
    map_values(df, replace_dict)
    """
    for col in replace_dict.keys():
        for a,b in replace_dict[col]:
            df[col] = df[col].replace(a,b)


class LGBMRegressorWrapper(BaseEstimator):
    def __init__(self, param_dict=None, cat_cols=None):
        self.cat_cols = cat_cols
        if param_dict!=None:
            self.estimator=lgbm.LGBMRegressor(**param_dict)
        else:
            self.estimator = lgbm.LGBMRegressor()

    def fit(self,X,y):
        if self.cat_cols==None:
            self.estimator.fit(X,y)
        else:
            self.estimator.fit(X,y,categorical_feature=self.cat_cols)
        return self

    def fit_transform(self,X,y):
        self.fit(X,y)
        return self.transform(X)

    def transform(self,X):
        return self.estimator.predict(X)

class LGBMClassifierWrapper(BaseEstimator):
    def __init__(self, param_dict=None, cat_cols=None):
        self.cat_cols = cat_cols
        if param_dict!=None:
            self.estimator=lgbm.LGBMClassifier(**param_dict)
        else:
            self.estimator = lgbm.LGBMClassifier()

    def fit(self,X,y):
        if self.cat_cols==None:
            self.estimator.fit(X,y)
        else:
            self.estimator.fit(X,y,categorical_feature=self.cat_cols)
        return self

    def fit_transform(self,X,y):
        self.fit(X,y)
        return self.transform(X)

    def transform(self,X):
        return self.estimator.predict_proba(X)


class CoxPHWrapper(BaseEstimator):
    def __init__(self, duration_col='tenure_length', event_col='churn', strata=None, penalizer=0.001):
        self.duration_col = duration_col
        self.event_col = event_col        
        self.stratify_cols=strata
        self.estimator = CoxPHFitter(penalizer=penalizer)

    def fit(self,X,y):
        if self.stratify_cols==None:
            self.estimator.fit(X, duration_col=self.duration_col, 
                            event_col=self.event_col,
                            show_progress=True)
        else:
            self.estimator.fit(X, duration_col=self.duration_col, 
                            event_col=self.event_col, strata=self.stratify_cols, 
                            show_progress=True)
        return self

    def fit_transform(self,X,y):
        self.fit(X,y)
        return self.transform(X)

    def transform(self,X):
        return self.estimator.predict_median(X)
        