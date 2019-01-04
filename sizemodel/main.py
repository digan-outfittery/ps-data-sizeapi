import pandas as pd

from sizemodel.dataloading import load_from_db, load_yaml
from sizemodel.sizemodel_atl_estimator import \
    SizeClassifier as SizeClassifierATL
from sizemodel.sizemodel_unwrapped.sizemodel import SizeClassifier
from sklearn.metrics import roc_auc_score


# def test_sizemodel_atlversion(config_path='sizemodel/config/baseconfig.yml'):
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


def run_size_model(config_path='sizemodel/config/baseconfig.yml'):
    # load config file
    config = load_yaml(config_path)
    data = load_from_db(config['data'])
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