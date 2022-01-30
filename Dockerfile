FROM python:3.8-slim
RUN apt-get update -y
RUN apt-get install -y postgresql-server-dev-all
RUN pip install psycopg2-binary
RUN mkdir /app
# Only copy requirements so as to avoid reinstalling them every time if the other code changes
COPY requirements.txt /app/requirements.txt 
RUN pip install -r /app/requirements.txt
COPY . /app
WORKDIR /app
EXPOSE 5000
CMD gunicorn --bind 0.0.0.0:$PORT app:app
