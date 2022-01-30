import os

from flask import Flask
from flask_cors import CORS
from flask_heroku import Heroku
from flask_login import LoginManager
from flask_sqlalchemy import SQLAlchemy

# Create the Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS

if "TESTING" in os.environ:
    app.config.from_envvar("TESTING")
    print("Using config for TESTING")
elif "DEVELOPMENT" in os.environ:
    app.config.from_envvar("DEVELOPMENT")
    print("Using config for DEVELOPMENT")
else:
    app.config.from_pyfile("config_deploy.cfg")
    print("Using Deployment configuration")

# Fix to get Heroku PostgreSQL to work
uri = os.getenv("DATABASE_URL")  # or other relevant config var
if uri is not None and uri.startswith("postgres://"):
    os.environ["DATABASE_URL"] = uri.replace("postgres://", "postgresql://", 1)
heroku = Heroku(app)
login_manager = LoginManager(app)

# Instantiate SQLAlchemy to handle db process
db = SQLAlchemy(app)

# run the file routes.py
from application import routes
