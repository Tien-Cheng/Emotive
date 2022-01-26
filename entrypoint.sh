#!/bin/sh
exec gunicorn --config /app/gunicorn_config.py app:app