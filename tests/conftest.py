import pytest
from app import app as flask_app


@pytest.fixture
def app():
    yield flask_app

@pytest.fixture
def client(app):
    print(app.static_url_path)
    yield app.test_client()