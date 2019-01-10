import logging
import pandas as pd

log = logging.getLogger(__name__)


def run_sql(sql_file, conn, args_dict={}):

    cursor = conn.cursor()

    sql = get_sql(sql_file)

    cursor.execute(sql, args_dict)
    conn.commit()
    log.info('Completed running SQL file: {}'.format(sql_file))



def fetch_from(query, conn, parameters={}):
    """fetch data as DataFrame

    Args:
       query (str): query string
       conn (object): connection object
       parameters (dict): query parameters

    Returns:
       :obj:`pandas.DataFrame`: fetched data
    """
    log.info("ENGINE: {}".format(conn.engine()))
    log.info("QUERY:\n{}".format(query))
    params_str = '\n'.join(['  {}: {}'.format(k, v) for k, v in parameters.items()])
    log.info('PARAMETERS:\n{}\n'.format(params_str))

    with conn.connect() as conn:
        with conn.cursor() as cursor:
            # import ipdb; ipdb.set_trace()
            cursor.execute(query, parameters)
            df = df_fetch(cursor)
    log.info("found shape: {}".format(df.shape))

    return df


def insert_into_table_batched(df, table, cursor, batch_size=5000):
    """insert data into table batchwise

    Args:
       df (:obj:`pandas.DataFrame`): data which will be inserted into database
       table (str): name of table
       cursor (cursor): database connection cursor
    """
    columns = df.columns
    columns_comma = '"' + '", "'.join(columns) + '"'

    query = 'INSERT INTO {table} ({columns_comma}) VALUES '
    query = query.format(table=table,
                         columns_comma=columns_comma)

    lsts = df.values.tolist()
    subs = list(chunker(lsts, batch_size))

    args = '(' + ', '.join(len(columns) * ['%s']) + ')'

    log.info("insert {} batches with {} rows each".format(len(subs), batch_size))

    n = len(subs)
    for i, lst in enumerate(subs):
        values = ', '.join(cursor.mogrify(args, x).decode('utf-8') for x in lst)
        equery = query + values

        cursor.execute(equery)

        log.info('  batch {0:>4}/{1:>4}'.format(i+1, n))



def get_sql(file):

    f = open(file, 'r')
    sql = f.read()
    f.close()
    return sql


def df_fetch(cursor):
    """fetch data and put in DataFrame

    Args:
       cursor (cursor): database connection cursor

    Returns:
       :obj:`pandas.DataFrame`
    """
    records = cursor.fetchall()

    columns = None
    if cursor.description:
        columns = [d[0] for d in cursor.description]
    if columns:
        df = pd.DataFrame(records, columns=columns, dtype=object)
    else:
        df = pd.DataFrame(records, dtype=object)

    return df

def chunker(seq, size):
    """return chunks of seq of size size

    Args:
       seq (list): list like sequence
       size (int): size of chunk

    Returns:
       tuple(list): chunked sequence
    """
    return (seq[pos:pos + size] for pos in range(0, len(seq), size))

