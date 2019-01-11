import os
import sys
import logging
import socket
import datetime
import simplejson
import traceback
import uuid
import numpy as np
from dateutil import tz
from collections import ChainMap
from logstash.formatter import LogstashFormatterBase
from logstash import TCPLogstashHandler
from logging import StreamHandler


class CustomEncoder(simplejson.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, datetime.datetime):
            return obj.isoformat()
        elif isinstance(obj, datetime.date):
            return obj.isoformat().strftime('%Y-%m-%dT%H:%M:%S.%fZ')
        elif isinstance(obj, datetime.timedelta):
            return (datetime.datetime.min + obj).time().strftime('%Y-%m-%dT%H:%M:%S.%fZ')
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, ChainMap):
            return dict(obj)
        else:
            return super(CustomEncoder, self).default(obj)


def format_record(record, fields_extra):
    record.message = record.getMessage()
    data = {
        '@timestamp': datetime.datetime.utcnow().replace(tzinfo=tz.UTC),
        'message': record.msg,
        'levelname': record.levelname,
        'logger': record.name,
        'path': record.pathname
    }

    if record.exc_info is not None:
        data['stacktrace'] = ''.join(traceback.format_exception(*record.exc_info))
        data['exception_type'] = ''.join(record.exc_info[0].__name__)
        data['exception_value'] = ''.join(repr(record.exc_info[1]))

    if hasattr(record, 'custom_fields') and isinstance(record.custom_fields, dict):
        custom_fields = {str(k): str(v) for k, v in record.custom_fields.items()}
        data.update(custom_fields)
    data.update(fields_extra)

    return data


class CustLogAdapter(logging.LoggerAdapter):
    def __init__(self, log, **kwargs):
        extra = {'custom_fields': kwargs}
        super().__init__(log, extra)

    def add(self, **kwargs):
        self.extra['custom_fields'].update(kwargs)


class PsLogstashFormatter(LogstashFormatterBase):
    def __init__(self, **fields_extra):
        self.fields_extra = fields_extra
        super().__init__()

    def format(self, record):
        data = format_record(record, self.fields_extra)
        return bytes(simplejson.dumps(data, cls=CustomEncoder), 'utf-8')


class JsonExtraFieldFormatter(logging.Formatter):
    def __init__(self, **fields_extra):
        self.fields_extra = fields_extra
        super().__init__()

    def format(self, record):
        data = format_record(record, self.fields_extra)
        return simplejson.dumps(data, cls=CustomEncoder)


class LogstashHandler(TCPLogstashHandler):
    '''
    A TCP Logstash Handler that sends extra fields ( with static values ) in all messages.
    '''
    def __init__(self, host, port, **fields_extra):
        TCPLogstashHandler.__init__(self, host, port)
        formatter = PsLogstashFormatter(**fields_extra)
        self.setFormatter(formatter)


class StdoutHandler(StreamHandler):
    def __init__(self):
        StreamHandler.__init__(self, stream=sys.stdout)
        formatter = logging.Formatter(
            '%(asctime)s %(levelname)s [pid: %(process)d] %(name)s - %(message)s')
        self.setFormatter(formatter)


class JSONStdoutHandler(StreamHandler):
    def __init__(self, **fields_extra):
        StreamHandler.__init__(self, stream=sys.stdout)
        formatter = JsonExtraFieldFormatter(**fields_extra)
        self.setFormatter(formatter)


def update_fields_extra_of_formatter(logger, **fields_extra):
    """update fields_extra of formatter

    update and add fields_extra of formatter of each handler of parent logger which have fields_extra
    attribute.

    """
    for handler in logger.parent.handlers:
        if hasattr(handler.formatter, 'fields_extra'):
            handler.formatter.fields_extra.update(fields_extra)


def setup_logging(app_root, verbose=False, **fields_extra):
    app_env = os.environ.get('APP_ENVIRONMENT')
    # set debug=True if app environment is not staging or production
    debug = app_env not in ('staging', 'production')
    # overwrite debug level by passed argument or environment variable VERBOSE
    debug = True if verbose or os.environ.get('VERBOSE') else debug

    logger = logging.getLogger(app_root)
    logging_level = logging.DEBUG if debug else logging.INFO
    logger.setLevel(logging_level)

    fields_extra.update({
        'appEnvironment': app_env,
        'instance_uuid': str(uuid.uuid4()),
        'hostname': socket.gethostname(),
        'log_start_time': datetime.datetime.utcnow().replace(tzinfo=tz.UTC)
    })

    if app_env is not None:
        log_stsh_host = os.environ.get('LOGSTASH_PORT_9998_TCP_ADDR')
        if log_stsh_host:
            log_stsh_port = os.environ.get('LOGSTASH_PORT_9998_TCP_PORT')
            handler = LogstashHandler(log_stsh_host, log_stsh_port, **fields_extra)
        else:
            handler = JSONStdoutHandler(**fields_extra)
        handler.setLevel(logging_level)
        logger.addHandler(handler)
    stdhandler = StdoutHandler()
    stdhandler.setLevel(logging_level)
    logger.addHandler(stdhandler)

    logger.info("logger with log level '{}' has been set up".format(logging.getLevelName(logger.getEffectiveLevel())))
