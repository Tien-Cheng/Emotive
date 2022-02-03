import pytest

from datetime import datetime as dt
from flask import json


@pytest.mark.parametrize(
    "userlist",
    [
        ["user1", "password1"],
        ["user2", "password2"],
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
            "/api/user-add/", data=json.dumps(data), content_type="application/json"
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
    "id_list",
    [3, 4],
)
def test_user_get_api(client, userlist, capsys):
    with capsys.disabled():
        raise NotImplementedError



@pytest.mark.usefixtures("populate_users")
def test_user_get_all_api(client, userlist, capsys):
    with capsys.disabled():
        raise NotImplementedError


@pytest.mark.usefixtures("populate_users")
@pytest.mark.parametrize(
    "id_list",
    [3, 4],
)
def test_user_delete_api(client, userlist, capsys):
    with capsys.disabled():
        raise NotImplementedError