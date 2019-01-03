#!/bin/bash

newrelic-admin run-program gunicorn -t 180 -c /appenv/bin/gunicorn_conf.py -w $GUNICORN_WORKERS -b 0.0.0.0:$GUNICORN_PORT sizeapi.app:app
