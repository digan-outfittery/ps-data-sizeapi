import pandas as pd
import datetime as dt
from pkg_resources import resource_filename
import os
import logging

from sizemodel.dataloading import load_from_db, load_yaml
from sizemodel.sizemodel_atl_estimator import \
    SizeClassifier as SizeClassifierATL
from sizemodel.sizemodel_unwrapped.sizemodel import SizeClassifier
from sizemodel.db.redis_interaction import RedisStore
from sizemodel.db.pgsql_interaction import insert_into_table_batched, run_sql
from sizemodel.db.connections import mldb as pgsql_conn
from sklearn.metrics import roc_auc_score

log = logging.getLogger(__name__)

DROP_AND_CREATE_MLDB_TABLES_SQL = resource_filename(__name__, 'resources/create_tables.sql')

def main():

    redis_store = RedisStore()

    log.info('Training the ATL size model')
    # Run the ATL model and get the customer, item sizes
    df_item_size, df_customer_size = run_size_model()

    log.info('Transforming the output data')
    # #Transform them into the required format for DB upload
    df_customer_size = _transform_cust_size_df(df_customer_size)
    df_item_size = _transform_item_size_df(df_item_size)

    log.info('Storing the data in the redis database')
    #Store the sizes in the redis database
    redis_store.set_customer_df_sizes(df_customer_size)
    redis_store.set_customer_df_sizes(df_customer_size)

    # store_customer_sizes_in_redis(df_customer_size)
    # store_article_sizes_in_redis(df_item_size)

    log.info('Storing the data in MLDB')
    #Drop & Create the required tables in MLDB
    run_sql(
        sql_file=DROP_AND_CREATE_MLDB_TABLES_SQL,
        conn=pgsql_conn.connect()
    )
    #Upload the size tables into the empty tables
    insert_into_table_batched(
        df=df_item_size,
        table=os.environ['MLDB_ITEM_SIZE_TABLE'],
        cursor=pgsql_conn.connect().cursor()
    )



def store_customer_sizes_in_redis(customer_size_df):
    '''
    Loop through the customer_sizes DataFrame and store each row in the redis database

    Args:
        customer_size_df: The dataframe of customer sizes
    '''

    log.info('Storing customer sizes for %s customers', len(customer_size_df))

    redis_store = RedisStore()

    counter = 0
    for idx, row in customer_size_df.iterrows():

        counter += 1
        if counter % 10 == 0:
            print('{0} rows => {1}'.format(counter, dt.datetime.now()))
        customer_id = row['customer_id']
        size_object = row['size_object']
        redis_store.set_customer_sizes(customer_id, size_object)


def store_article_sizes_in_redis(article_size_df):
    '''
    Loop through the article_sizes DataFrame and store each row in the redis database

    Args:
        article_size_df: The dataframe of article sizes
    '''

    log.info('Storing article sizes for %s articles', len(article_size_df))

    redis_store = RedisStore()

    for idx, row in article_size_df.iterrows():
        article_id = row['item_size_id']
        size_object = row['size_object']
        redis_store.set_article_sizes(article_id, size_object)




def _transform_cust_size_df(size_df):
    '''
    Transform the raw output from the ATLSizeModel into the required format for storage

    Args:
        size_df: The raw ATLSizeModel output

    Returns:
        pd.DataFrame: dataframe with columns: id, upload_time, sizes_dict
    '''


    upload_date = dt.datetime.now()

    #Convert df_item_size to the required format for storage
    ret_df = pd.DataFrame(
        {
            'customer_id': row['customer_id'],
            'model_timestamp': upload_date,
            'size_object': {
                'customerId': row['customer_id'],
                'modelTimestamp': str(upload_date),
                'sizes': [
                            {
                                'name': 'shoeSize',
                                'mu': row['mean_cust_shoesize'],
                                'sigma': row['std_cust_shoesize']
                            },
                            {
                                'name': 'tShirtSize',
                                'mu': row['mean_cust_shirtsize'],
                                'sigma': row['std_cust_shirtsize']
                            },
                            {
                                'name': 'trouserSizeWidth',
                                'mu': row['mean_cust_trouserswidth'],
                                'sigma': row['std_cust_trouserswidth']
                            },
                            {
                                'name': 'trouserSizeLength',
                                'mu': row['mean_cust_trouserslength'],
                                'sigma': row['std_cust_trouserslength']
                            }
                        ]
            }

        }
        for idx, row in size_df.iterrows()
    )

    return ret_df


def _transform_item_size_df(size_df):
    '''
    Transform the raw output from the ATLSizeModel into the required format for storage

    Args:
        size_df: The raw ATLSizeModel output

    Returns:
        pd.DataFrame: dataframe with columns: id, upload_time, sizes_dict
    '''

    upload_date = dt.datetime.now()
    # Convert df_item_size to the required format for storage
    ret_df = pd.DataFrame(
        {
            'article_id': row['item_size_id'],
            'model_timestamp': upload_date,
            'size_object': {
                'articleId': row['item_size_id'],
                'modelTimestamp': str(upload_date),
                'sizes': [
                            {
                                'name': 'shoeSize',
                                'mu': row['mean_item_shoesize'],
                                'sigma': row['std_item_shoesize']
                            },
                            {
                                'name': 'tShirtSize',
                                'mu': row['mean_item_shirtsize'],
                                'sigma': row['std_item_shirtsize']
                            },
                            {
                                'name': 'trouserSizeWidth',
                                'mu': row['mean_item_trouserswidth'],
                                'sigma': row['std_item_trouserswidth']
                            },
                            {
                                'name': 'trouserSizeLength',
                                'mu': row['mean_item_trouserslength'],
                                'sigma': row['std_item_trouserslength']
                            }
                        ]
            }
        }
        for idx, row in size_df.iterrows()
    )

    return ret_df




# def test_sizemodel_atlversion(config_path='sizemodel/resources/baseconfig.yml'):
#     # load config file
#     config = load_yaml(config_path)
#     data = load_from_db(config['data'])
#     df_train = data['df_train']
#     df_test = data['df_pred']
#
#     # train the model
#     sm = SizeClassifierATL(**config['estimator']['init']['kwargs'])
#     sm.fit(df_train=df_train)
#
#     # predict
#     too_small_pred, too_big_pred = sm.predict(df_test)
#
#     # evaluation
#     print(roc_auc_score(df_test['feedback_too_small'], too_small_pred))
#     print(roc_auc_score(df_test['feedback_too_large'], too_big_pred))
#
#     df_test['too_small_pred'] = too_small_pred
#     print(df_test.groupby('too_small_pred')[['items_kept', 'feedback_too_small']].mean())
#     df_test['too_big_pred'] = too_big_pred
#     print(df_test.groupby('too_big_pred')[['items_kept', 'feedback_too_large']].mean())
#     print('everything done')


def run_size_model(config_path='sizemodel/resources/baseconfig.yml'):
    # load config file
    config = load_yaml(config_path)
    data = load_from_db(config['data']) #TODO: Change date_end in config file
    df_train = data['df_train']
    df_test = data['df_pred']

    # train the model
    sm = SizeClassifier(**config['estimator']['init']['kwargs'])
    sm.fit(df_train=df_train)

    #TODO: sm.sizes[default_category_name] .......

    # get the dataframes with the customer and item sizes
    df_customer_size = sm.cust_sizes.dropna()
    df_item_size = sm.item_sizes  # TODO we have to do a mapping if we want to provide sizes on article_id level

    # small evaluation to confirm everything worked
    df_test['item_size_id'] = df_test['item_no'] + '_' + df_test['nav_size_code']
    df_test = df_test.merge(df_item_size, on='item_size_id')
    df_test = df_test.merge(df_customer_size, on='customer_id')
    df_test['sizediff_shirt'] = (df_test['mean_item_shirtsize'] - df_test['mean_cust_shirtsize'])
    df_test['sizediff_shirt_q'] = pd.qcut(df_test['sizediff_shirt'], 10)
    print(df_test.dropna(subset=['sizediff_shirt']).groupby('sizediff_shirt_q')[['items_kept', 'feedback_too_small', 'feedback_too_large']].mean())

    return df_item_size, df_customer_size



if __name__ == '__main__':
    main()