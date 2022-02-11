from datetime import datetime as dt

import pytest
from application.models import User


@pytest.mark.parametrize(
    "userlist",
    [["tan", "Password1234!@#$"], ["lee", "fadd$$@!45FF"]],
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
        ["tay", None],
        [None, "Password1234!@#$"],
    ],
)
def test_UserClassValidation_missing(userlist, capsys):
    test_UserClass(userlist, capsys)


@pytest.mark.xfail(reason="Invalid inputs", strict=True)  # Out of Range Inputs
@pytest.mark.parametrize(
    "userlist",
    [["yeet111", "password"], ["", "empty username"], ["empty password", ""], ["", ""]],
)
def test_UserClassValidation_invalid(userlist, capsys):
    test_UserClass(userlist, capsys)
