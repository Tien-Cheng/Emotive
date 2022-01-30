from datetime import datetime as dt

import pytest
from application.models import User
from werkzeug.security import generate_password_hash


@pytest.mark.parametrize(
    "userlist",
    [["user@yahoo.com", "Password1234!@#$"], ["test_user@ichat.com", "fadd$$@!45FF"]],
)
def test_UserClass(userlist, capsys):
    with capsys.disabled():
        created = dt.utcnow()
        new_user = User(
            username=userlist[0],
            password=userlist[1],
            created_on=created,
        )
        # Assert statements
        assert new_user.verify_password(userlist[1])
        assert new_user.username == userlist[0]
        assert new_user.created_on == created


@pytest.mark.xfail(reason="Missing inputs", strict=True)
@pytest.mark.parametrize(
    "userlist",
    [
        [],
        [None, None],
        [None],
        ["email@email.com", None],
        [None, "Password1234!@#$"],
    ],
)
def test_UserClassValidation_missing(userlist, capsys):
    test_UserClass(userlist, capsys)
