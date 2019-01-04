import os

import numpy as np
import psycopg2
from psycopg2.extensions import AsIs, register_adapter
from psycopg2.extras import RealDictCursor

import sqlalchemy


def addapt_numpy(dtype):
    return AsIs(dtype)


register_adapter(dict, psycopg2.extras.Json)
register_adapter(np.int64, addapt_numpy)
register_adapter(np.float64, addapt_numpy)


class mldb:
    @classmethod
    def get_info(cls):
        return {
            'host': os.environ['DB_MLDB_POSTGRES_PORT_5432_TCP_ADDR'],
            'port': os.environ['DB_MLDB_POSTGRES_PORT_5432_TCP_PORT'],
            'user': os.environ['DB_MLDB_POSTGRES_USER'],
            'password': os.environ['DB_MLDB_POSTGRES_PASSWORD'],
            'dbname': os.environ['DB_MLDB_POSTGRES_DATABASE'],
            'cursor_factory': RealDictCursor
        }

    @classmethod
    def connect(cls):
        info = cls.get_info()
        return psycopg2.connect(**info)

    @classmethod
    def engine(cls):
        info = cls.get_info()
        return sqlalchemy.create_engine(
            'postgresql://{user}:{password}@{host}:{port}/{dbname}'.format(
                user=info['user'], password=info['password'],
                host=info['host'], port=info['port'],
                dbname=info['dbname']
            )
        )


class paul:
    @classmethod
    def get_info(cls):
        return {
            'host': os.environ['DB_PAUL_POSTGRES_PORT_5432_TCP_ADDR'],
            'port': os.environ['DB_PAUL_POSTGRES_PORT_5432_TCP_PORT'],
            'dbname': os.environ['DB_PAUL_POSTGRES_DATABASE'],
            'user': os.environ['DB_PAUL_POSTGRES_USER'],
            'password': os.environ['DB_PAUL_POSTGRES_PASSWORD'],
            'connect_timeout': 5
        }

    @classmethod
    def connect(cls):
        info = cls.get_info()
        return psycopg2.connect(**info)

    @classmethod
    def engine(cls):
        info = cls.get_info()
        return sqlalchemy.create_engine(
            'postgresql://{user}:{password}@{host}:{port}/{dbname}'.format(
                user=info['user'], password=info['password'],
                host=info['host'], port=info['port'],
                dbname=info['dbname']
            )
        )
