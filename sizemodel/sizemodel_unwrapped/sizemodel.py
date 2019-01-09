import logging
import time

import numpy as np
import pandas as pd

from sizemodel.size_consistency_checker.size_consistency_checker import \
    SizeChecker
from sklearn.base import BaseEstimator, ClassifierMixin

from .data_preparation import (prepare_data_shirtsize, prepare_data_shoesize,
                               prepare_data_trousers_size, shirtsize_mapping)
from .edward_models import make_edward_model, make_edward_model_trousers

log = logging.getLogger(__name__)

# define which item categories get which predictions
shirt_size_cat = ['Knitted tops', 'Suits', 'Shorts',
                  'Shirt short sleeves',
                  'Shirts business', 'Vest', 'Underwear', 'Blazers',
                  'Coats', 'Sweaters', 'T-Shirts', 'Belts',
                  'Shirts casual', 'Leather Jackets', 'Jackets']
shoes_size_cat = ['Shoes casual', 'Shoes business']
trousers_size_cat = ['Pants', 'Jeans']


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

        self.cust_sizes = pd.DataFrame(data=df_train['customer_id'].unique(), columns=['customer_id'])
        self.item_sizes = pd.DataFrame(data=df_train['item_size_id'].unique(), columns=['item_size_id'])

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

    # def predict(self, df):
    #     '''prediction for the one-customer-case'''
    #     orig_idx = df.index
    #
    #     df['item_size_id'] = df['item_no'] + "_" + df['nav_size_code']
    #     # assert(df['customer_id'].nunique() == 1)
    #
    #     # split the df in to one for each size-category-group
    #     df_shoes = df.query('category in ({})'.format(shoes_size_cat)).copy()
    #     df_shirts = df.query('category in ({})'.format(shirt_size_cat)).copy()
    #     df_trousers = df.query('category in ({})'.format(trousers_size_cat)).copy()
    #
    #     df_shirts = self._pred_cat(df_shirts, 'shirtsize',
    #                                orig_custsize_name='shirt_size')
    #     df_shoes = self._pred_cat(df_shoes, 'shoesize',
    #                               orig_custsize_name='shoe_size')
    #     df_trouserswidth = self._pred_cat(df_trousers, 'trouserswidth',
    #                                       orig_custsize_name='trousers_size_width')
    #     df_trouserslength = self._pred_cat(df_trousers, 'trouserslength',
    #                                        orig_custsize_name='trousers_size_length')
    #
    #     # combine the different predictions for trousers into one
    #     df_trouserslength['pred_too_big'] |= df_trouserswidth['pred_too_big']
    #     df_trouserslength['pred_too_small'] |= df_trouserswidth['pred_too_small']
    #
    #     df = pd.concat([df_shoes, df_trouserslength, df_shirts])
    #     df = df.reindex(orig_idx)
    #     return df['pred_too_small'].fillna(0), df['pred_too_big'].fillna(0)

    def _store_sizes(self, df_train, posteriors, idx2cust, idx2item, size_cat_name, cust_size_col, var_name='sizes'):
        """Store the sizes given by the posteriors into self.sizes[size_cat_name]"""
        means = posteriors['cust_' + var_name].mean().eval()
        sdevs = np.sqrt(posteriors['cust_' + var_name].variance().eval())

        df_cust = pd.DataFrame(data=[means, sdevs],
                               index=['mean_cust_' + size_cat_name, 'std_cust_' + size_cat_name]).T
        means = posteriors['item_' + var_name].mean().eval()

        # for items the variance is stored instead of the scale like in studentt
        sdevs = np.sqrt(posteriors['item_' + var_name].variance().eval())
        df_item = pd.DataFrame(data=[means, sdevs],
                               index=['mean_item_' + size_cat_name, 'std_item_' + size_cat_name]).T
        df_cust['customer_id'] = df_cust.reset_index()['index'].map(idx2cust)
        df_item['item_size_id'] = df_item.reset_index()['index'].map(idx2item)

        self.cust_sizes = self.cust_sizes.merge(df_cust, on='customer_id', how='left')
        self.item_sizes = self.item_sizes.merge(df_item, on='item_size_id', how='left')

        #size_cat_name = 'Shoe Size' .... etc
        #TODO: Make into 2 separate dataframes and remove size_cat_name from article sizes
        self.sizes[size_cat_name] = {'cust': df_cust.set_index('customer_id').to_dict()}
        self.sizes[size_cat_name]['item'] = df_item.set_index('item_size_id').to_dict()

        # make default item size
        df_item = df_item.merge(df_train[['item_size_id', 'nav_size_code']], on='item_size_id', how='left')
        self.sizes[size_cat_name]['default_item'] = df_item.groupby('nav_size_code')['mean_item_' + size_cat_name].mean().to_dict()
        if cust_size_col is not None:
            df_cust = df_cust.merge(df_train[['customer_id', cust_size_col]], on='customer_id', how='left')
            self.sizes[size_cat_name]['default_cust'] = df_cust.groupby(cust_size_col)['mean_cust_' + size_cat_name].mean().to_dict()
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
