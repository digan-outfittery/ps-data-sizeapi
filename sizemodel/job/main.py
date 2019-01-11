import pandas as pd
import datetime as dt
from pkg_resources import resource_filename
import os
import logging
import json

from sizemodel.job.dataloading import load_from_db, load_yaml
from sizemodel.job.sizemodel_atl_estimator import \
    SizeClassifier as SizeClassifierATL
from sizemodel.job.sizemodel_unwrapped.sizemodel import SizeClassifier
from sizemodel.utils.db.redis_interaction import RedisStore
from sizemodel.utils.db.pgsql_interaction import insert_into_table_batched, run_sql
from sizemodel.utils.db.connections import mldb as pgsql_conn
from sklearn.metrics import roc_auc_score

log = logging.getLogger(__name__)

DROP_AND_CREATE_MLDB_TABLES_SQL = resource_filename(__name__, 'resources/create_tables.sql')

def main():

    redis_store = RedisStore()

    log.info('Training the ATL size model')
    # Run the ATL model and get the customer, item sizes
    df_item_size, df_existing_customer_size, dct_new_customer_sizes = run_size_model()

    log.info('Transforming the output data')
    # #Transform them into the required format for DB upload
    df_existing_customer_size = _transform_existing_cust_size_df(df_existing_customer_size)
    df_new_customer_size = _transform_new_cust_size_dct(dct_new_customer_sizes)
    df_item_size = _transform_item_size_df(df_item_size)

    log.info('Storing the data in the redis database')
    #Store the sizes in the redis database
    redis_store.set_customer_df_sizes(df_existing_customer_size)
    redis_store.set_new_customer_df_sizes(df_new_customer_size)
    redis_store.set_article_df_sizes(df_item_size)

    log.info('Storing the data in MLDB')
    conn = pgsql_conn.connect()
    #Drop & Create the required tables in MLDB
    run_sql(
        sql_file=DROP_AND_CREATE_MLDB_TABLES_SQL,
        conn=conn
    )
    #Upload the size tables into the empty tables
    insert_into_table_batched(
        df=df_existing_customer_size,
        table=os.environ['MLDB_EXISTING_CUST_SIZE_TABLE'],
        cursor=conn.cursor()
    )
    insert_into_table_batched(
        df=df_new_customer_size,
        table=os.environ['MLDB_NEW_CUST_SIZE_TABLE'],
        cursor=conn.cursor()
    )
    insert_into_table_batched(
        df=df_item_size,
        table=os.environ['MLDB_ITEM_SIZE_TABLE'],
        cursor=conn.cursor()
    )
    conn.commit()


def _transform_existing_cust_size_df(size_df):
    '''
    Transform the raw output from the ATLSizeModel into the required format for storage

    Args:
        size_df: The raw ATLSizeModel output

    Returns:
        pd.DataFrame: dataframe with columns: id, upload_time, sizes_dict
    '''


    upload_date = dt.datetime.now().isoformat()

    #Convert df_item_size to the required format for storage
    ret_dct = pd.DataFrame(
        {
            'customer_id': int(row['customer_id']),
            'model_timestamp': upload_date,
            'size_object': json.dumps({
                'customerId': int(row['customer_id']),
                'modelTimestamp': '{0}{1}'.format(str(upload_date),'Z'),
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
            })

        }
        for idx, row in size_df.iterrows()
    )

    return ret_dct

def _transform_item_size_df(size_df):
    '''
    Transform the raw output from the ATLSizeModel into the required format for storage

    Args:
        size_df: The raw ATLSizeModel output

    Returns:
        pd.DataFrame: dataframe with columns: id, upload_time, sizes_dict
    '''

    #Convert all NaN values to None for json upload to MLDB
    ret_df = size_df.astype(object).where(pd.notnull(size_df), None)

    upload_date = dt.datetime.now().isoformat()
    # Convert df_item_size to the required format for storage
    ret_df = pd.DataFrame(
        {
            'article_id': row['item_size_id'],
            'model_timestamp': upload_date,
            'size_object': json.dumps({
                'articleId': row['item_size_id'],
                'modelTimestamp': '{0}{1}'.format(str(upload_date),'Z'),
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
            })
        }
        for idx, row in ret_df.iterrows()
    )

    return ret_df

def _transform_new_cust_size_dct(size_dct):
    '''

    Args:
        size_dct: The original new customer sizes dictionary

    Returns:
        pd.DataFrame: dataframe with columns: id, upload_time, sizes_dict
    '''

    size_key_mapper = {
        'shoesize': 'shoeSize',
        'shirtsize': 'shirtSize',
        'trouserslength': 'trouserSizeLength',
        'trouserswidth': 'trouserSizeWidth'
    }

    upload_date = dt.datetime.now().isoformat()

    all_sizes = []
    for key, value in size_dct.items():
        model_key = size_key_mapper[key]
        for cust_size, model_size in value.items():
            id = '{0}:{1}'.format(model_key, cust_size)
            size_object = json.dumps({
                'name': model_key,
                'mu': model_size,
                'sigma': None
            })
            row = [id, upload_date, size_object]
            all_sizes.append(row)

    colnames = ['id', 'model_timestamp', 'size_object']

    return pd.DataFrame(all_sizes, columns=colnames)




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


def run_size_model(config_path='sizemodel/job/resources/baseconfig.yml'):
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
    df_existing_customer_size = sm.cust_sizes.dropna()
    df_item_size = sm.item_sizes  # TODO we have to do a mapping if we want to provide sizes on article_id level

    # small evaluation to confirm everything worked
    df_test['item_size_id'] = df_test['item_no'] + '_' + df_test['nav_size_code']
    df_test = df_test.merge(df_item_size, on='item_size_id')
    df_test = df_test.merge(df_existing_customer_size, on='customer_id')
    df_test['sizediff_shirt'] = (df_test['mean_item_shirtsize'] - df_test['mean_cust_shirtsize'])
    df_test['sizediff_shirt_q'] = pd.qcut(df_test['sizediff_shirt'], 10)
    print(df_test.dropna(subset=['sizediff_shirt']).groupby('sizediff_shirt_q')[['items_kept', 'feedback_too_small', 'feedback_too_large']].mean())

    dct_new_customer_sizes = sm.new_cust_sizes

    return df_item_size, df_existing_customer_size, dct_new_customer_sizes



if __name__ == '__main__':
    main()