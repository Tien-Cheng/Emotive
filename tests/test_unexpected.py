"""
Test how app handles unexpected errors via the error handler
"""
import json

import pytest


@pytest.mark.parametrize(
    "endpoint",
    [
        ("invalid_link", 404),
        ("history", 303),
        ("", 200),
        ("login", 200),
        ("register", 200),
        ("predict", 303),
    ],
)
def test_route_codes(client, endpoint, capsys):
    with capsys.disabled():
        endpoint, code = endpoint[0], endpoint[1]
        response = client.get(f"/{endpoint}")
        assert response.status_code == code
