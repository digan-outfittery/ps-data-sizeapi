import datetime
import logging

import pandas as pd
import yaml

import dateparser
from sizemodel.utils.db.connections import mldb as db_conn
from sizemodel.job.picklecache import PickleCache

log = logging.getLogger(__name__)


def load_yaml(filename):
    with open(filename) as f:
        d = yaml.load(f)
    return d


def to_datetime(date):
    if isinstance(date, datetime.datetime):
        return date
    if isinstance(date, datetime.date):
        return date
    if isinstance(date, str):
        return dateparser.parse(date)
    else:
        raise ValueError


def load_from_db(config, invalidate=False):
    data = {}
    for data_id, data_desc in config.items():
        cache = PickleCache(data_desc)
        if invalidate:
            cache.invalidate()
        if cache.check():
            data[data_id] = cache.get()
        else:
            data[data_id] = fetch_data(data_desc)
            cache.put(data[data_id])
    return data


def fetch_data(instructions):
    # fetcher_name = instructions['fetcher']
    # fetcher = lookup_object(fetcher_name) # just take datasource as fetcher
    kwargs = instructions.get('kwargs', {})
    date_begin = to_datetime(instructions['date_begin'])
    date_end = to_datetime(instructions['date_end'])
    data = datasource(date_begin=date_begin, date_end=date_end, **kwargs)
    return data


def datasource(*, date_begin, date_end, columns):
    sql_columns = ', '.join(columns)
    q = '''
        select
            {columns}
        from ml.atl__datasource
        where date_observed >= %(date_begin)s
            and date_observed < %(date_end)s
    '''.format(columns=sql_columns)

    with db_conn.connect() as conn:
        dates = {'date_begin': date_begin, 'date_end': date_end}
        df = pd.read_sql_query(q, conn, params=dates)

    return df
