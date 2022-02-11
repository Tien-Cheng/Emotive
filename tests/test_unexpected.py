"""
Test how app handles unexpected errors via the error handler
"""
from io import BytesIO

import pytest


@pytest.mark.parametrize(
    "endpoint",
    [
        ("invalid_link", 404),
        ("history", 302),
        ("", 200),
        ("login", 200),
        ("register", 200),
        ("predict", 302),
    ],
)
def test_route_codes(client, endpoint, capsys):
    with capsys.disabled():
        endpoint, code = endpoint[0], endpoint[1]
        response = client.get(f"/{endpoint}")
        assert response.status_code == code


@pytest.mark.parametrize("imgname", ["corrupted_img.png"])
def test_corrupted_img(client, imgname, capsys):
    with capsys.disabled():
        data = {
            "file": (
                BytesIO(open(f"./tests/test_files/{imgname}", "rb").read()),
                imgname,
            )
        }

        response = client.post(
            "/api/predict", data=data, content_type="multipart/form-data"
        )

        assert response.status_code == 400
