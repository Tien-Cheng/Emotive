from datetime import datetime as dt

import pytest
from app import app as flask_app
from application import db
from application.models import User


@pytest.fixture
def app():
    yield flask_app


@pytest.fixture
def client(app):
    print(app.static_url_path)
    yield app.test_client()


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
