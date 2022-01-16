from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager
import os

# Create the Flask app
app = Flask(__name__)

if "TESTING" in os.environ:
    app.config.from_envvar('TESTING')
    print("Using config for TESTING")
else:
    app.config.from_pyfile('config.cfg')
    print(">>> Using default config")

login_manager = LoginManager(app)
 
# Instantiate SQLAlchemy to handle db process
db = SQLAlchemy(app)

#run the file routes.py
from application import routes