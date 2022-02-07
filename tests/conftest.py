from datetime import datetime as dt
from application.models import User

import pytest
from app import app as flask_app
from application import db
from selenium import webdriver
from selenium.webdriver.chrome.options import Options as chOpts
from selenium.webdriver.chrome.service import Service


@pytest.fixture
def app():
    yield flask_app


@pytest.fixture
def client(app):
    print(app.static_url_path)
    yield app.test_client()


@pytest.fixture
def chrome():
    options = chOpts()
    options.add_argument("--headless")
    options.add_argument("--no-sandbox")

    service = Service("/chromedriver/stable/chromedriver")
    yield webdriver.Chrome(service=service, options=options)


@pytest.fixture
def populate_users():
    users = [
        {
            "username": "jane",
            "password": "Password1234!",
        },
        {
            "username": "mark",
            "password": "Password567890!",
        },
    ]
    for user in users:
        try:
            db.session.add(
                User(
                    username=user["username"],
                    password=user["password"],
                    created_on=dt.utcnow(),
                )
            )
            db.session.commit()
        except Exception as error:
            print(error)
            db.session.rollback()
