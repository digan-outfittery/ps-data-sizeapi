import numpy as np
import pandas as pd

from sizemodel.job.size_consistency_checker.size_consistency_checker import \
    SizeChecker
from sklearn.linear_model import LinearRegression

shoesize_mapping = {
            '00380': 38,
            '00385': 38.5,
            '00390': 39,
            '00395': 39.5,
            '00400': 40,
            '00405': 40.5,
            '00406': 40.6,
            '00410': 41,
            '00413': 41.3,
            '00415': 41.5,
            '04142': 41.5,
            '00411': 41.5,
            '00420': 42,
            '00425': 42.5,
            '00426': 42.6,
            '00430': 43,
            '00433': 43.3,
            '00435': 43.5,
            '04344': 43.5,
            '00440': 44,
            '00445': 44.5,
            '00446': 44.6,
            '00450': 45,
            '00453': 45.3,
            '00455': 45.5,
            '04546': 45.5 ,
            '00460': 46,
            '00465': 46.5,
            '00466': 46.6,
            '00470': 47,
            '00473': 47.3,
            '00475': 47.5,
            '00480': 48,
            '00485': 48.5
            }
shirtsize_mapping = {
            'XS': 1,
            "S": 2,
            "M": 3,
            "L": 4,
            "XL": 5,
            "XXL": 6,
            "XXXL": 7,
            "3XL": 7,
            "4XL": 8,
            "5XL": 9,
            "6XL": 10
}


def add_numeric_id(df, col_name):
    '''
    Add a column to the df containing numerical ids starting from zero.
    Return df with "id__<col_name>" and a dict from index to the original identifier.
    '''
    assert col_name in df

    tmp = df[col_name].unique()
    tmp = pd.DataFrame(tmp, columns=['orig_id'])
    tmp['i__' + col_name] = tmp.index
    # link the indexes to their real columns
    df = pd.merge(df, tmp, left_on=col_name, right_on='orig_id', how='left')
    df = df.drop('orig_id', 1)
    idx2orig = tmp['orig_id'].to_dict()
    return df, idx2orig


def filter_min_obs(df, min_obs, obs_col='item_size_id'):
    """Filter the df for observations occurring at least min_obs times"""
    enough_obs_item_size_ids = df.groupby(obs_col).items_kept.sum() >= min_obs
    enough_obs_item_size_ids = enough_obs_item_size_ids[enough_obs_item_size_ids]
    enough_obs_item_size_ids = enough_obs_item_size_ids.index
    return df[df[obs_col].isin(enough_obs_item_size_ids)]


def make_size_priors(df, base_col, size_feedback_offset=None):
    """Make sizepriors (returns a series) from the average values in the base_col grouped by item_size_id.
    If size_feedback_offset is set, sizereturns are counted with the respective offset
    """
    df['size_est_base'] = df[base_col]
    df['size_est_base'][(df['items_kept'] + df['feedback_too_small'].fillna(0)
                         + df['feedback_too_large'].fillna(0)) != 1] = np.nan
    if size_feedback_offset:
        df['size_est_base'][df['feedback_too_small'] == 1] -= 1.
        df['size_est_base'][df['feedback_too_large'] == 1] += 1.
    mu_item_size_priors = df.groupby('i__item_size_id')['size_est_base'].mean()
    avg_default_sizes = df.groupby('nav_size_code')['size_est_base'].mean()
    mu_item_size_priors[mu_item_size_priors.isnull()] = df['nav_size_code'].map(avg_default_sizes)
    return mu_item_size_priors


def add_linreg(df, name, target_col, feat_cols):
    """Add two columns to the given df with a linear regressor prediction for the target col
     and the mean between the target and this prediction as 'linreg_<name>' and half_linreg_<name>.
    """
    lr = LinearRegression()  # linreg for width
    lr.fit(X=df[feat_cols], y=df[target_col])
    df['linreg_' + name] = lr.predict(df[feat_cols])
    df['half_linreg_' + name] = (df['linreg_' + name] + df[target_col]) / 2
    return df


def prepare_data_shirtsize(df,  min_obs=3, size_feedback_offset=0.7):
    data_dict = {}
    df = df.copy()
    # clean data
    df = df[df['trousers_size_width'].str.len() == 2]
    df = df[df['trousers_size_length'].str.len() == 2]
    sc = SizeChecker()
    size_consistency_df = sc.find_wrong_input(df)
    w_size_inconsistent = (~size_consistency_df).any(1)
    size_cols = ['weight_in_kg', 'height_in_cm', 'trousers_size_width', 'trousers_size_length',
                 'num_shirt_size_cust', 'date_of_birth']
    df = df[~w_size_inconsistent].dropna(subset=size_cols)
    # we only train on items where we have enough observations (this should reduce overfitting)
    df = filter_min_obs(df, min_obs=min_obs, obs_col='item_size_id')

    lr = LinearRegression()
    X_train = df[['weight_in_kg', 'height_in_cm', 'trousers_size_width', 'customer_age']].astype(float)
    y_train = df['num_shirt_size_cust']
    lr.fit(X=X_train, y=y_train)

    # make more exact size estimation
    df['linreg_size'] = lr.predict(X_train)
    df['half_linreg_size'] = (df['linreg_size'] + df['num_shirt_size_cust'])/2
    df['selective_linreg_size'] = df['linreg_size']
    df['selective_linreg_size'][abs(df['linreg_size']
                                    - df['num_shirt_size_cust']) > 0.6] = df['half_linreg_size']

    # the pymc3 model needs numeric index identifiers for all variables, so produce them
    df, idx2cust = add_numeric_id(df, 'customer_id')
    df, idx2item = add_numeric_id(df, 'item_size_id')
    mu_item_priors = make_size_priors(df, 'linreg_size', size_feedback_offset=size_feedback_offset)
    # df, idx2brandsize = add_numeric_id(df, 'brandsize')
    data_dict['obs_kept'] = df.items_kept.values
    data_dict['obs_toobig'] = df.feedback_too_large.fillna(0).values
    data_dict['obs_toosmall'] = df.feedback_too_small.fillna(0).values
    data_dict['customers'] = df['i__customer_id'].values
    data_dict['num_custs'] = df['i__customer_id'].nunique()
    data_dict['items'] = df['i__item_size_id'].values
    data_dict['num_items'] = df['i__item_size_id'].nunique()
    # data_dict['brands'] = df['i_brandsize'].values # TODO test if including brands yields significant improvements

    data_dict['mu_cust_priors'] = df.groupby('i__customer_id')['selective_linreg_size'].mean().values.astype('float32')
    data_dict['mu_item_priors'] = mu_item_priors.values.astype('float32')

    return data_dict, idx2cust, idx2item


def prepare_data_trousers_size(df, min_obs=3, size_feedback_offset=None):
    df = df.copy()

    # clean data - customers have to have both trousers sizes and they should be consistent
    df = df[df['trousers_size_width'].str.len() == 2]
    df = df[df['trousers_size_length'].str.len() == 2]
    sc = SizeChecker()
    size_consistency_df = sc.find_wrong_input(df)
    w_size_inconsistent = (~size_consistency_df).any(1)
    size_cols = ['weight_in_kg', 'height_in_cm', 'trousers_size_width', 'trousers_size_length',
                 'num_shirt_size_cust', 'date_of_birth']
    df = df[~w_size_inconsistent].dropna(subset=size_cols)
    # we only train on items where we have enough observations (this should reduce overfitting)
    df = filter_min_obs(df, min_obs=min_obs, obs_col='item_size_id')

    # make more exact size estimation
    df = add_linreg(df=df, name='cust_width', target_col='trousers_size_width',
                    feat_cols=['weight_in_kg', 'height_in_cm', 'num_shirt_size_cust', 'customer_age'])
    df = add_linreg(df=df, name='cust_length', target_col='trousers_size_length',
                    feat_cols=['height_in_cm', 'customer_age'])

    # the model needs numeric index identifiers for all variables, so add them
    df, idx2cust = add_numeric_id(df, 'customer_id')
    df, idx2item = add_numeric_id(df, 'item_size_id')
    # df, idx2brandsize = add_numeric_id(df, 'brandsize')

    data_dict = {}
    data_dict['mu_cust_trousers_length_priors'] = df.groupby('i__customer_id')['half_linreg_cust_length'].mean().values.astype('float32')
    data_dict['mu_cust_trousers_width_priors'] = df.groupby('i__customer_id')['half_linreg_cust_width'].mean().values.astype('float32')

    # make (hopefully better) itemsize priors that are based on customersize-estimation
    mu_item_length_priors = make_size_priors(df, 'half_linreg_cust_length', size_feedback_offset=size_feedback_offset * 2).values
    mu_item_width_priors = make_size_priors(df, 'half_linreg_cust_width', size_feedback_offset=size_feedback_offset * 2).values

    # finally store the data in a dict
    data_dict['obs_kept'] = df.items_kept.values
    data_dict['obs_toobig'] = df.feedback_too_large.fillna(0).values
    data_dict['obs_toosmall'] = df.feedback_too_small.fillna(0).values
    data_dict['customers'] = df['i__customer_id'].values
    data_dict['num_custs'] = df['i__customer_id'].nunique()
    data_dict['items'] = df['i__item_size_id'].values
    data_dict['num_items'] = df['i__item_size_id'].nunique()
    # data_dict['brands'] = df['i_brandsize'].values # TODO test if including brands yields significant improvements
    data_dict['mu_item_trousers_length_priors'] = mu_item_length_priors.astype('float32')
    data_dict['mu_item_trousers_width_priors'] = mu_item_width_priors.astype('float32')

    return data_dict, idx2cust, idx2item


def prepare_data_shoesize(df, min_obs=3, size_feedback_offset=None):
    data_dict = {}
    # convert shoesizes into floats and remove the asics which are in uk or us sizesystem
    df['num_shoe_size'] = df['nav_size_code'].map(shoesize_mapping)
    df = df[df['shoe_size'].str.len() == 2]
    df['shoe_size'] = df['shoe_size'].astype(int)
    df = df.dropna(subset=['shoe_size']).copy()
    df = filter_min_obs(df, min_obs=min_obs, obs_col='item_size_id')

    # the pymc3 model needs numeric index identifiers for all variables, so produce them
    df, idx2cust = add_numeric_id(df, 'customer_id')
    df, idx2item = add_numeric_id(df, 'item_size_id')
    # df, idx2brandsize = add_numeric_id(df, 'brandsize')

    data_dict['obs_kept'] = df.items_kept.values
    data_dict['obs_toobig'] = df.feedback_too_large.fillna(0).values
    data_dict['obs_toosmall'] = df.feedback_too_small.fillna(0).values
    data_dict['customers'] = df['i__customer_id'].values
    data_dict['num_custs'] = df['i__customer_id'].nunique()
    data_dict['items'] = df['i__item_size_id'].values
    data_dict['num_items'] = df['i__item_size_id'].nunique()
    # data_dict['brands'] = df['i_brandsize'].values # test if including brands yields significant improvements - leave out for now since it slows down everything with little improvement

    mu_cust_priors = df.groupby('i__customer_id')['num_shoe_size'].mean()
    # some customers might have had orders without any shoes, so we take the supplied size than
    mu_cust_priors_backup = df.drop_duplicates(subset=['customer_id']).set_index(
                'i__customer_id')['shoe_size'].to_dict()
    mu_cust_priors[mu_cust_priors.isnull()] = \
        mu_cust_priors.reset_index()['i__customer_id'].map(mu_cust_priors_backup)
    data_dict['mu_cust_priors'] = mu_cust_priors.values.astype('float32')

    # make (hopefully better) itemsize priors that are based on customersize-estimation
    df['size_est_base'] = df['num_shoe_size']
    df['size_est_base'][(df['items_kept'] + df['feedback_too_small'].fillna(0)
                         + df['feedback_too_large'].fillna(0)) != 1] = np.nan
    df['size_est_base'][df['feedback_too_small'] == 1] -= size_feedback_offset
    df['size_est_base'][df['feedback_too_large'] == 1] += size_feedback_offset
    mu_item_priors = df.groupby('i__item_size_id')['size_est_base'].mean()
    avg_default_sizes = df.groupby('nav_size_code')['size_est_base'].mean()

    mu_item_priors[mu_item_priors.isnull()] = df['nav_size_code'].map(avg_default_sizes)
    data_dict['mu_item_priors'] = mu_item_priors.values.astype('float32')
    return data_dict, idx2cust, idx2item
