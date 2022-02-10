from multiprocessing.connection import Client
import pytest

from datetime import datetime as dt
from flask import json


@pytest.mark.parametrize(
    "userlist",
    [
        ["john", "password1"],
        ["terry", "password2"],
    ],
)
def test_user_add_api(client, userlist, capsys):
    with capsys.disabled():
        # Prepare JSON
        data = {
            "username": userlist[0],
            "password": userlist[1],
        }

        # Send post request to API
        response = client.post(
            "/api/user/add", data=json.dumps(data), content_type="application/json"
        )


        response_body = json.loads(response.get_data(as_text=True))

        # Check response status code
        assert response.status_code == 200

        # Check consistent content type
        assert response.headers["Content-Type"] == "application/json"

        # Check result is consistent with what was sent
        assert response_body["id"], "User ID was not returned"


@pytest.mark.usefixtures("populate_users")
@pytest.mark.parametrize(
    "userlist",
    [1, 2],
)
def test_user_get_api(client, userlist, capsys):
    with capsys.disabled():
        # Send post request to API
        response = client.get(
            f"/api/user/{userlist}", content_type="application/json"
        )

        response_body = json.loads(response.get_data(as_text=True))

        # Check response status code
        assert response.status_code == 200

        # Check consistent content type
        assert response.headers["Content-Type"] == "application/json"

        # Check result is consistent with what was sent
        assert response_body["id"] == userlist , "User ID was not correctly returned"
        assert response_body["username"], "Username was not returned"
        assert response_body["password"], "Password was not returned"


@pytest.mark.usefixtures("populate_users")
def test_user_get_all_api(client, capsys):
    with capsys.disabled():
        # Send post request to API
        response = client.get("/api/user/all", content_type="application/json")
        response_body = json.loads(response.get_data(as_text=True))

        # Check response status code
        assert response.status_code == 200

        # Check consistent content type
        assert response.headers["Content-Type"] == "application/json"

        # Check response
        for user in response_body:
            assert user["id"], "User ID was not correctly returned"
            assert user["username"], "Username was not returned"
            assert user["password"], "Password was not returned"
        assert len(response_body) == 4 # 2 from populate_users and 2 from add user unit test

@pytest.mark.usefixtures("populate_users")
@pytest.mark.parametrize(
    "userlist",
    [1, 2],
)
def test_user_delete_api(client, userlist, capsys):
    with capsys.disabled():
        # Send post request to API
        response = client.delete(
            f"/api/user/delete/{userlist}", content_type="application/json"
        )
        response_body = json.loads(response.get_data(as_text=True))

        # Check response status code
        assert response.status_code == 200

        # Check consistent content type
        assert response.headers["Content-Type"] == "application/json"

        # Check result is consistent with what was sent
        assert response_body["result"] == "ok"