import flask
import logging
from flask import request
from flask_healthcheck import Healthcheck

from prometheus_client import generate_latest, CollectorRegistry
from prometheus_client import Counter, Histogram
from prometheus_client import multiprocess

import sizemodel
from sizemodel.app.deciders.atl_sizemodel_decider import ATLSizeModelDecider
from sizemodel.app.utils import kblog

decider = ATLSizeModelDecider()

app = flask.Flask(__name__)
log = logging.getLogger(__name__)

healthcheck = Healthcheck(app)

FLASK_REQUEST_LATENCY = Histogram(__name__.replace('.', '_') + '_request_latency_seconds', 'Flask Request Latency')
FLASK_REQUEST_COUNT = Counter(__name__.replace('.', '_') + '_request_count', 'Flask Request Count',
                              ['method', 'endpoint', 'http_status'])


def after_request(response):
    FLASK_REQUEST_COUNT.labels(request.method, request.path, response.status_code).inc()
    return response



@healthcheck
def health():
    return True


@app.before_first_request
def setup_app():
    '''
    Don't forget to setup @app.errorhandler to log exceptions
    '''


    kblog.setup_logging(app_root='tinder',
                        appName='ps-data-tinder',
                        appVersion=sizemodel.__version__)

    app.after_request(after_request)
    log.info('Worker got first request')


@app.errorhandler(Exception)
def unhandled_exception(e):
    log.exception('Unhandled exception: {}'.format(repr(e)))
    response = flask.jsonify({'errors': [repr(e)]})
    response.status_code = 400
    return response



@app.route('/decision', methods=['POST'])
def decide():
    """return a certain decision based on request

    Returns:
        dict: decision
    """
    return do_decision_logic()



@FLASK_REQUEST_LATENCY.time()
def do_decision_logic():
    request_data = flask.request.get_json()

    try:
        decider.validate_request(request_data)

    except Exception as e:
        log.exception('Error validating request for request "{}"'
                      .format(request_data.get('data').get('meta').get('correlationId')))
        response = flask.jsonify({'errors': ['invalid request', str(e), repr(e)]})
        response.status_code = 400
        return response

    response_data = decider.decide(request_data)

    decider.validate_response(response_data)

    try:
        decider.validate_response(response_data)
    except Exception as e:
        raise ValueError('Response for request "{}" is not valid: \n\n {}, \n\n{}'
                         .format(response_data['data']['meta']['correlationId'], response_data, repr(e)))

    import ipdb; ipdb.set_trace()
    response = flask.jsonify(response_data)
    response.status_code = 201

    return response


@app.route('/', methods=['GET'])
def status():
    return 'OK', 200


@app.route('/stats', methods=['GET'])
def metrics():
    registry = CollectorRegistry()
    multiprocess.MultiProcessCollector(registry)
    return generate_latest(registry), 200


def debug():
    app.run('0.0.0.0',
            port=8081,
            debug=True,
            use_debugger=False,
            use_reloader=True)



if __name__ == '__main__':
    debug()
