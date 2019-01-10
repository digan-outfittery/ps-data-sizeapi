import logging
import time

import numpy as np
import pandas as pd
import tensorflow as tf

import edward
from sizemodel.size_consistency_checker.size_consistency_checker import \
    SizeChecker
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LinearRegression

log = logging.getLogger(__name__)


shirt_size_cat = ['Knitted tops', 'Suits', 'Shorts',
                  'Shirt short sleeves',
                  'Shirts business', 'Vest', 'Underwear', 'Blazers',
                  'Coats', 'Sweaters', 'T-Shirts', 'Belts',
                  'Shirts casual', 'Leather Jackets', 'Jackets']
shoes_size_cat = ['Shoes casual', 'Shoes business']
trousers_size_cat = ['Pants', 'Jeans']

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


def ed_gaussian(x, fmax=0.5, loc=0., scale=1., min_val=0.02):
    return fmax * tf.exp(-(tf.square(x - loc)/(2 * tf.square(scale)))) + min_val


def ed_sigmoid(x, right_shift=0.4, compr=2.5, min=0.02, max=0.98):
    ''' function for a sigmoid-like activation '''
    x_int = compr * (x - right_shift)
    f_x = x_int / (tf.sqrt(1 + tf.square(x_int)))
    # now transform into the area between min and max (normal range is (-1, 1) )
    return f_x * (0.5 * max - 0.5 * min) + 0.5 * (min + max)


def make_edward_model(mu_cust_priors,
                      mu_item_priors,
                      customers,
                      items,
                      num_custs,
                      num_items,
                      obs_kept,
                      obs_toosmall,
                      obs_toobig,
                      gauss_max=0.7,
                      sigmoid_max=0.8,
                      right_shift=0.4):
    #  data placeholders - here the index column is fed in
    cust_ph = tf.placeholder(tf.int32, [None])
    item_ph = tf.placeholder(tf.int32, [None])

    cust_sizes_std = 1.
    item_sizes_std = 0.8

    # these are the prior distributions of our algorithm
    cust_sizes = edward.models.StudentT(df=2., loc=tf.ones(num_custs) * mu_cust_priors, scale=tf.ones(num_custs))
    item_sizes = edward.models.Normal(loc=tf.ones(num_items) * mu_item_priors, scale=tf.ones(num_items) * item_sizes_std)

    # gather maps the index (in the placeholder tensor) to the variable of the right customer/item)
    size_diff = tf.gather(item_sizes, item_ph) - tf.gather(cust_sizes, cust_ph)

    matching = edward.models.Bernoulli(probs=ed_gaussian(size_diff, fmax=gauss_max, scale=0.4))
    too_big = edward.models.Bernoulli(probs=ed_sigmoid(size_diff, compr=2.5, max=sigmoid_max, right_shift=right_shift))
    too_small = edward.models.Bernoulli(probs=ed_sigmoid(-size_diff, max=sigmoid_max, right_shift=right_shift))

    data = {
        cust_ph: customers,
        item_ph: items,
        matching: obs_kept,
        too_big: obs_toobig,
        too_small: obs_toosmall
    }

    # initialize posterior and give it a good starting point
    # (the q_* variables are the distributions [to optimize while running] we look at in the end)

    # set startingpoint: "prior values" as mean and a softplus of a noisy 1 as std
    q_cust_sizes = edward.models.Normal(
    loc=tf.Variable(mu_cust_priors),
    scale=tf.nn.softplus(tf.Variable(tf.random_normal([num_custs], mean=1.))))
    q_item_sizes = edward.models.Normal(
    loc=tf.Variable(mu_item_priors),
    scale=tf.nn.softplus(tf.Variable(tf.random_normal([num_items], mean=1.))))
    latent_variables = {
        cust_sizes: q_cust_sizes,
        item_sizes: q_item_sizes
    }
    inference = edward.KLqp(latent_variables, data)
    return (inference, {'cust_sizes': q_cust_sizes, 'item_sizes': q_item_sizes})


def make_edward_model_trousers(
            mu_cust_trousers_length_priors,
            mu_cust_trousers_width_priors,
            mu_item_trousers_length_priors,
            mu_item_trousers_width_priors,
            customers,
            items,
            num_custs,
            num_items,
            obs_kept,
            obs_toosmall,
            obs_toobig,
            gauss_max=0.3,
            sigmoid_max=0.9,
            right_shift=0.5):
    #  data placeholders - here the index column is fed in
    cust_ph = tf.placeholder(tf.int32, [None])
    item_ph = tf.placeholder(tf.int32, [None])

    cust_sizes_std = 2.0
    item_sizes_std = 1.0

    # these are the prior distributions of our algorithm
    cust_trousers_width = edward.models.Normal(loc=tf.ones(num_custs) * mu_cust_trousers_width_priors, scale=tf.ones(num_custs) * item_sizes_std)
    cust_trousers_length = edward.models.Normal(loc=tf.ones(num_custs) * mu_cust_trousers_length_priors, scale=tf.ones(num_custs) * item_sizes_std)
    item_trousers_width = edward.models.Normal(loc=tf.ones(num_items) * mu_item_trousers_width_priors, scale=tf.ones(num_items) * item_sizes_std)
    item_trousers_length = edward.models.Normal(loc=tf.ones(num_items) * mu_item_trousers_length_priors, scale=tf.ones(num_items) * item_sizes_std)

    # gather maps the index (in the placeholder tensor) to the variable of the right customer/item)
    diff_trousers_width = tf.gather(item_trousers_width, item_ph) - tf.gather(cust_trousers_width, cust_ph)
    diff_trousers_length = tf.gather(item_trousers_length, item_ph) - tf.gather(cust_trousers_length, cust_ph)

    matching = edward.models.Bernoulli(probs=ed_gaussian(
                tf.maximum(diff_trousers_width, diff_trousers_length), fmax=gauss_max, scale=1.0))
    too_big = edward.models.Bernoulli(probs=ed_sigmoid(
                tf.maximum(diff_trousers_width, diff_trousers_length), max=sigmoid_max, right_shift=2*right_shift))
    too_small = edward.models.Bernoulli(probs=ed_sigmoid(
                -tf.minimum(diff_trousers_width, diff_trousers_length), max=sigmoid_max, right_shift=2*right_shift))

    data = {
        cust_ph: customers,
        item_ph: items,
        matching: obs_kept,
        too_big: obs_toobig,
        too_small: obs_toosmall
    }

    # initialize posterior and give it a good starting point
    # (the q_* variables are the distributions [to optimize while running] we look at in the end)

    # startingpoint is the "prior values" as mean and a softplus of a noisy 1 as std
    q_cust_trousers_width = edward.models.Normal(
        loc=tf.Variable(mu_cust_trousers_width_priors),
        scale=tf.nn.softplus(tf.Variable(tf.random_normal([num_custs], mean=1.))))
    q_cust_trousers_length = edward.models.Normal(
        loc=tf.Variable(mu_cust_trousers_length_priors),
        scale=tf.nn.softplus(tf.Variable(tf.random_normal([num_custs], mean=1.))))
    q_item_trousers_width = edward.models.Normal(
        loc=tf.Variable(mu_item_trousers_width_priors),
        scale=tf.nn.softplus(tf.Variable(tf.random_normal([num_items], mean=1.))))
    q_item_trousers_length = edward.models.Normal(
        loc=tf.Variable(mu_item_trousers_length_priors),
        scale=tf.nn.softplus(tf.Variable(tf.random_normal([num_items], mean=1.))))
    latent_variables = {
        cust_trousers_width: q_cust_trousers_width,
        cust_trousers_length: q_cust_trousers_length,
        item_trousers_width: q_item_trousers_width,
        item_trousers_length: q_item_trousers_length,
    }
    inference = edward.KLqp(latent_variables, data)
    return (inference, {'cust_trouserswidth': q_cust_trousers_width,
                        'cust_trouserslength': q_cust_trousers_length,
                        'item_trouserswidth': q_item_trousers_width,
                        'item_trouserslength': q_item_trousers_length
                        })


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


class SizeClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self,
                 cust_sdev_thresh=2.5,
                 item_sdev_thresh=2.5,
                 max_mean_diff=2.0,
                 min_mean_diff=0.5,
                 trousers_multiple=2.0,
                 size_feedback_offset=1.,
                 min_obs=4,
                 gauss_max=0.35,
                 sigmoid_max=0.9,
                 right_shift=0.5):
        self.cust_sdev_thresh = cust_sdev_thresh
        self.item_sdev_thresh = item_sdev_thresh
        self.max_mean_diff = max_mean_diff
        self.min_mean_diff = min_mean_diff
        self.trousers_multiple = trousers_multiple
        self.size_feedback_offset = size_feedback_offset
        self.min_obs = min_obs
        self.gauss_max = gauss_max
        self.sigmoid_max = sigmoid_max
        self.right_shift = right_shift

    def fit(self, *, df_train):
        assert 'item_no' in df_train and 'nav_size_code' in df_train
        # prepare the data
        df_train['item_size_id'] = df_train['item_no'] + "_" + df_train['nav_size_code']
        df_train['customer_age'] = df_train['date_observed'].dt.year - df_train['date_of_birth'].dt.year
        df_train['num_shirt_size_cust'] = df_train['shirt_size'].map(shirtsize_mapping)
        self.sizes = {}

        # build and estimate the edwardlib variational inference model for shirts
        data_dict, idx2cust, idx2item = prepare_data_shirtsize(
                    df_train.query('category in {}'.format(shirt_size_cat)), min_obs=self.min_obs,
                    size_feedback_offset=self.size_feedback_offset)
        model, posteriors = make_edward_model(**data_dict, gauss_max=self.gauss_max,
                                              sigmoid_max=self.sigmoid_max, right_shift=self.right_shift)
        model.run(n_iter=500)
        self._store_sizes(df_train, posteriors, idx2cust, idx2item,
                          size_cat_name='shirtsize', cust_size_col='shirt_size')

        # shoesizes ...
        data_dict, idx2cust, idx2item = prepare_data_shoesize(
                    df_train.query('category in {}'.format(shoes_size_cat)), min_obs=self.min_obs,
                    size_feedback_offset=self.size_feedback_offset)
        model, posteriors = make_edward_model(**data_dict, gauss_max=self.gauss_max,
                                              sigmoid_max=self.sigmoid_max, right_shift=self.right_shift)
        model.run(n_iter=500)
        self._store_sizes(df_train, posteriors, idx2cust, idx2item,
                          size_cat_name='shoesize', cust_size_col=None)

        # ... the same for trousers sizes
        data_dict, idx2cust, idx2item = prepare_data_trousers_size(
                    df_train.query('category in {}'.format(trousers_size_cat)), min_obs=self.min_obs,
                    size_feedback_offset=self.size_feedback_offset)
        model, posteriors = make_edward_model_trousers(**data_dict, gauss_max=self.gauss_max,
                                                       sigmoid_max=self.sigmoid_max, right_shift=self.right_shift)
        model.run(n_iter=500)
        self._store_sizes(df_train, posteriors, idx2cust, idx2item, size_cat_name='trouserslength',
                          cust_size_col=None, var_name='trouserslength')
        self._store_sizes(df_train, posteriors, idx2cust, idx2item, size_cat_name='trouserswidth',
                          cust_size_col=None, var_name='trouserswidth')

    def predict(self, df):
        '''prediction for the one-customer-case'''
        orig_idx = df.index

        df['item_size_id'] = df['item_no'] + "_" + df['nav_size_code']
        # assert(df['customer_id'].nunique() == 1)

        # split the df in to one for each size-category-group
        df_shoes = df.query('category in ({})'.format(shoes_size_cat)).copy()
        df_shirts = df.query('category in ({})'.format(shirt_size_cat)).copy()
        df_trousers = df.query('category in ({})'.format(trousers_size_cat)).copy()

        df_shirts = self._pred_cat(df_shirts, 'shirtsize',
                                   orig_custsize_name='shirt_size')
        df_shoes = self._pred_cat(df_shoes, 'shoesize',
                                  orig_custsize_name='shoe_size')
        df_trouserswidth = self._pred_cat(df_trousers, 'trouserswidth',
                                          orig_custsize_name='trousers_size_width')
        df_trouserslength = self._pred_cat(df_trousers, 'trouserslength',
                                           orig_custsize_name='trousers_size_length')

        # combine the different predictions for trousers into one
        df_trouserslength['pred_too_big'] |= df_trouserswidth['pred_too_big']
        df_trouserslength['pred_too_small'] |= df_trouserswidth['pred_too_small']

        df = pd.concat([df_shoes, df_trouserslength, df_shirts])
        df = df.reindex(orig_idx)
        return df['pred_too_small'].fillna(0), df['pred_too_big'].fillna(0)

    def _store_sizes(self, df_train, posteriors, idx2cust, idx2item, size_cat_name, cust_size_col, var_name='sizes'):
        """Store the sizes given by the posteriors into self.sizes[size_cat_name]"""
        means = posteriors['cust_' + var_name].mean().eval()
        sdevs = np.sqrt(posteriors['cust_' + var_name].variance().eval())

        df_cust = pd.DataFrame(data=[means, sdevs],
                               index=['mean_cust', 'std_cust']).T
        means = posteriors['item_' + var_name].mean().eval()
        # for items the variance is stored instead of the scale like in studentt
        sdevs = np.sqrt(posteriors['item_' + var_name].variance().eval())
        df_item = pd.DataFrame(data=[means, sdevs],
                               index=['mean_item', 'std_item']).T
        df_cust['customer_id'] = df_cust.reset_index()['index'].map(idx2cust)
        df_item['item_size_id'] = df_item.reset_index()['index'].map(idx2item)
        self.sizes[size_cat_name] = {'cust': df_cust.set_index('customer_id').to_dict()}
        self.sizes[size_cat_name]['item'] = df_item.set_index('item_size_id').to_dict()

        # make default item size
        df_item = df_item.merge(df_train[['item_size_id', 'nav_size_code']], on='item_size_id', how='left')
        self.sizes[size_cat_name]['default_item'] = df_item.groupby('nav_size_code')['mean_item'].mean().to_dict()
        if cust_size_col is not None:
            df_cust = df_cust.merge(df_train[['customer_id', cust_size_col]], on='customer_id', how='left')
            self.sizes[size_cat_name]['default_cust'] = df_cust.groupby(cust_size_col)['mean_cust'].mean().to_dict()
        else:
            self.sizes[size_cat_name]['default_cust'] = {str(i) : i for i in range(25, 55)}

    def _pred_cat(self, df, sizecat_name, category_multiplyer=1.0,
                  orig_itemsize_name='nav_size_code', orig_custsize_name='shirt_sizes'):
        """The actual prediction taking the estimated mean and stdev and predict
        a size mismatch whenever the area of n stdev around the means of items and customes don't overlap"""
        sizes = self.sizes[sizecat_name]
        cust_sdev_thresh = self.cust_sdev_thresh * category_multiplyer
        item_sdev_thresh = self.item_sdev_thresh * category_multiplyer
        max_mean_diff = self.max_mean_diff
        min_mean_diff = self.min_mean_diff

        if df['customer_id'].nunique() == 1:
            cust_id = df['customer_id'].unique()[0]
            df['cust_mean'] = sizes['cust']['mean_cust'].get(cust_id, np.nan)
            df['cust_std'] = sizes['cust']['std_cust'].get(cust_id, np.nan)
        else:
            df['cust_mean'] = df['customer_id'].map(sizes['cust']['mean_cust'])
            df['cust_std'] = df['customer_id'].map(sizes['cust']['std_cust'])
        df['item_mean'] = df['item_size_id'].map(sizes['item']['mean_item'])
        df['item_std'] = df['item_size_id'].map(sizes['item']['std_item'])

        # for the sizes where we do not have data from the training we fill with the default sizes
        df['cust_mean'] = df['cust_mean'].fillna(df[orig_custsize_name].map(sizes['default_cust']))
        df['item_mean'] = df['item_mean'].fillna(df[orig_itemsize_name].map(sizes['default_item']))
        default_std = 1.
        df['cust_std'] = df['cust_std'].fillna(default_std)
        df['item_std'] = df['item_std'].fillna(default_std)

        df['cust_upper'] = df['cust_mean'] + df['cust_std'] * cust_sdev_thresh
        df['item_lower'] = df['item_mean'] - df['item_std'] * item_sdev_thresh
        df['cust_lower'] = df['cust_mean'] - df['cust_std'] * cust_sdev_thresh
        df['item_upper'] = df['item_mean'] + df['item_std'] * item_sdev_thresh

        df['pred_too_big'] = df['cust_upper'] < df['item_lower']
        df['pred_too_small'] = df['item_upper'] < df['cust_lower']

        df.loc[df['item_mean'] - df['cust_mean'] > max_mean_diff, 'pred_too_big'] = True
        df.loc[df['cust_mean'] - df['item_mean'] > max_mean_diff, 'pred_too_small'] = True
        df.loc[(df['item_mean'] - df['cust_mean']).abs() < min_mean_diff, 'pred_too_big'] = False
        df.loc[(df['item_mean'] - df['cust_mean']).abs() < min_mean_diff, 'pred_too_small'] = False
        return df
