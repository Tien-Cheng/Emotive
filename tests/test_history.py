from datetime import datetime as dt
from datetime import timedelta

import pytest
from application.models import Prediction


# Convert a list of confidence to a dictionary
def conf_to_dict(conf):
    emotion_list = (
        "angry",
        "fearful",
        "surprised",
        "happy",
        "neutral",
        "sad",
        "disgusted",
    )
    return {k: v for k, v in zip(emotion_list, conf)}


# Test if the class correctly stores the values
@pytest.mark.parametrize(
    "predlist",
    [
        # id, fk_user_id, emotion, file_path, prediction, predicted_on
        (
            1,
            1,
            "happy",
            "happy.jpg",
            conf_to_dict([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]),
            dt.now(),
        ),
        (
            2,
            2,
            "sad",
            "sad.png",
            conf_to_dict([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]),
            dt.now(),
        ),
    ],
)
def test_PredClass(predlist, capsys):
    with capsys.disabled():

        new_prediction = Prediction(
            id=predlist[0],
            fk_user_id=predlist[1],
            emotion=predlist[2],
            file_path=predlist[3],
            prediction=predlist[4],
            predicted_on=predlist[5],
        )

        assert new_prediction.id == predlist[0]
        assert new_prediction.fk_user_id == predlist[1]
        assert new_prediction.emotion == predlist[2]
        assert new_prediction.file_path == predlist[3]
        assert new_prediction.prediction == predlist[4]
        assert new_prediction.predicted_on == predlist[5]


# Test if storing out of range values score raise an error
@pytest.mark.xfail(reason="out of range")
@pytest.mark.parametrize(
    "predlist",
    [
        # Confidence < 1
        [
            3,
            1,
            "happy",
            "happy.jpg",
            conf_to_dict([-0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]),
            dt.now(),
        ],
        [
            3,
            1,
            "happy",
            "happy.jpg",
            conf_to_dict([0.1, -0.2, 0.3, 0.4, 0.5, 0.6, 0.7]),
            dt.now(),
        ],
        [
            3,
            1,
            "happy",
            "happy.jpg",
            conf_to_dict([0.1, 0.2, -0.3, 0.4, 0.5, 0.6, 0.7]),
            dt.now(),
        ],
        [
            3,
            1,
            "happy",
            "happy.jpg",
            conf_to_dict([0.1, 0.2, 0.3, -0.4, 0.5, 0.6, 0.7]),
            dt.now(),
        ],
        [
            3,
            1,
            "happy",
            "happy.jpg",
            conf_to_dict([0.1, 0.2, 0.3, 0.4, -0.5, 0.6, 0.7]),
            dt.now(),
        ],
        [
            3,
            1,
            "happy",
            "happy.jpg",
            conf_to_dict([0.1, 0.2, 0.3, 0.4, 0.5, -0.6, 0.7]),
            dt.now(),
        ],
        [
            3,
            1,
            "happy",
            "happy.jpg",
            conf_to_dict([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, -0.7]),
            dt.now(),
        ],
        # Confidence > 1
        [
            3,
            1,
            "happy",
            "happy.jpg",
            conf_to_dict([1.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]),
            dt.now(),
        ],
        [
            3,
            1,
            "happy",
            "happy.jpg",
            conf_to_dict([0.1, 1.2, 0.3, 0.4, 0.5, 0.6, 0.7]),
            dt.now(),
        ],
        [
            3,
            1,
            "happy",
            "happy.jpg",
            conf_to_dict([0.1, 0.2, 1.3, 0.4, 0.5, 0.6, 0.7]),
            dt.now(),
        ],
        [
            3,
            1,
            "happy",
            "happy.jpg",
            conf_to_dict([0.1, 0.2, 0.3, 1.4, 0.5, 0.6, 0.7]),
            dt.now(),
        ],
        [
            3,
            1,
            "happy",
            "happy.jpg",
            conf_to_dict([0.1, 0.2, 0.3, 0.4, 1.5, 0.6, 0.7]),
            dt.now(),
        ],
        [
            3,
            1,
            "happy",
            "happy.jpg",
            conf_to_dict([0.1, 0.2, 0.3, 0.4, 0.5, 1.6, 0.7]),
            dt.now(),
        ],
        [
            3,
            1,
            "happy",
            "happy.jpg",
            conf_to_dict([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 1.7]),
            dt.now(),
        ],
        # Predicted_on < 01/01/2020
        [
            3,
            1,
            "happy",
            "happy.jpg",
            conf_to_dict([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]),
            dt(2020, 1, 1) - timedelta(days=1),
        ],
        # Predicted_on > now
        [
            3,
            1,
            "happy",
            "happy.jpg",
            conf_to_dict([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]),
            dt.now() + timedelta(days=1),
        ],
        # Unrecognised emotion
        [
            3,
            1,
            "cool",
            "cool.jpg",
            conf_to_dict([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]),
            dt.now(),
        ],
    ],
)
def test_PredOOR(predlist, capsys):
    test_PredClass(predlist, capsys)


# Test if inserting None raise an error
@pytest.mark.xfail(reason="None value")
@pytest.mark.parametrize(
    "predlist",
    [
        [
            None,
            1,
            "happy",
            "happy.jpg",
            conf_to_dict([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]),
            dt.now(),
        ],
        [
            3,
            None,
            "happy",
            "happy.jpg",
            conf_to_dict([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]),
            dt.now(),
        ],
        [
            3,
            1,
            None,
            "happy.jpg",
            conf_to_dict([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]),
            dt.now(),
        ],
        [
            3,
            1,
            "happy",
            None,
            conf_to_dict([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]),
            dt.now(),
        ],
        [3, 1, "happy", "happy.jpg", None, dt.now()],
        [
            3,
            1,
            "happy",
            "happy.jpg",
            conf_to_dict([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]),
            None,
        ],
    ],
)
def test_PredNone(predlist, capsys):
    test_PredClass(predlist, capsys)


# Test if inserting differrent data type raise an error
@pytest.mark.xfail(reason="different data type")
@pytest.mark.parametrize(
    "predlist",
    [
        # id (int)
        [
            True,
            1,
            "happy",
            "happy.jpg",
            conf_to_dict([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]),
            dt.now(),
        ],
        [
            {},
            1,
            "happy",
            "happy.jpg",
            conf_to_dict([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]),
            dt.now(),
        ],
        [
            set(),
            1,
            "happy",
            "happy.jpg",
            conf_to_dict([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]),
            dt.now(),
        ],
        [
            "str",
            1,
            "happy",
            "happy.jpg",
            conf_to_dict([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]),
            dt.now(),
        ],
        [
            (),
            1,
            "happy",
            "happy.jpg",
            conf_to_dict([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]),
            dt.now(),
        ],
        [
            1.1,
            1,
            "happy",
            "happy.jpg",
            conf_to_dict([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]),
            dt.now(),
        ],
        [
            [],
            1,
            "happy.jpg",
            conf_to_dict([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]),
            dt.now(),
        ],
        # fk_user_id (int)
        [
            3,
            True,
            "happy",
            "happy.jpg",
            conf_to_dict([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]),
            dt.now(),
        ],
        [
            3,
            {},
            "happy",
            "happy.jpg",
            conf_to_dict([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]),
            dt.now(),
        ],
        [
            3,
            set(),
            "happy",
            "happy.jpg",
            conf_to_dict([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]),
            dt.now(),
        ],
        [
            3,
            "str",
            "happy",
            "happy.jpg",
            conf_to_dict([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]),
            dt.now(),
        ],
        [
            3,
            (),
            "happy",
            "happy.jpg",
            conf_to_dict([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]),
            dt.now(),
        ],
        [
            3,
            1.1,
            "happy",
            "happy.jpg",
            conf_to_dict([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]),
            dt.now(),
        ],
        [
            3,
            [],
            "happy",
            "happy.jpg",
            conf_to_dict([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]),
            dt.now(),
        ],
        # emotion (str)
        [
            3,
            1,
            True,
            "happy.jpg",
            conf_to_dict([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]),
            dt.now(),
        ],
        [
            3,
            1,
            {},
            "happy.jpg",
            conf_to_dict([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]),
            dt.now(),
        ],
        [
            3,
            1,
            set(),
            "happy.jpg",
            conf_to_dict([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]),
            dt.now(),
        ],
        [
            3,
            1,
            1,
            "happy.jpg",
            conf_to_dict([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]),
            dt.now(),
        ],
        [
            3,
            1,
            (),
            "happy.jpg",
            conf_to_dict([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]),
            dt.now(),
        ],
        [
            3,
            1,
            1.1,
            "happy.jpg",
            conf_to_dict([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]),
            dt.now(),
        ],
        [
            3,
            1,
            [],
            "happy.jpg",
            conf_to_dict([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]),
            dt.now(),
        ],
        # file_path (str)
        [
            3,
            1,
            "happy",
            True,
            conf_to_dict([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]),
            dt.now(),
        ],
        [
            3,
            1,
            "happy",
            {},
            conf_to_dict([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]),
            dt.now(),
        ],
        [
            3,
            1,
            "happy",
            set(),
            conf_to_dict([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]),
            dt.now(),
        ],
        [3, 1, "happy", 1, conf_to_dict([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]), dt.now()],
        [
            3,
            1,
            "happy",
            (),
            conf_to_dict([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]),
            dt.now(),
        ],
        [
            3,
            1,
            "happy",
            1.1,
            conf_to_dict([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]),
            dt.now(),
        ],
        [
            3,
            1,
            "happy",
            [],
            conf_to_dict([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]),
            dt.now(),
        ],
        # prediction (dict/list)
        [3, 1, "happy", "happy.jpg", True, dt.now()],
        [3, 1, "happy", "happy.jpg", set(), dt.now()],
        [3, 1, "happy", "happy.jpg", "str", dt.now()],
        [3, 1, "happy", "happy.jpg", 1, dt.now()],
        [3, 1, "happy", "happy.jpg", (), dt.now()],
        [3, 1, "happy", "happy.jpg", 1.1, dt.now()],
        # confidence (int/float)
        [
            3,
            1,
            "happy",
            "happy.jpg",
            conf_to_dict([True, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]),
            dt.now(),
        ],
        [
            3,
            1,
            "happy",
            "happy.jpg",
            conf_to_dict([{}, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]),
            dt.now(),
        ],
        [
            3,
            1,
            "happy",
            "happy.jpg",
            conf_to_dict([set(), 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]),
            dt.now(),
        ],
        [
            3,
            1,
            "happy",
            "happy.jpg",
            conf_to_dict(["str", 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]),
            dt.now(),
        ],
        [
            3,
            1,
            "happy",
            "happy.jpg",
            conf_to_dict([(), 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]),
            dt.now(),
        ],
        [
            3,
            1,
            "happy",
            "happy.jpg",
            conf_to_dict([[], 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]),
            dt.now(),
        ],
        # different confidence key data types (unhashable dtypes are not included)
        [
            3,
            1,
            "happy",
            "happy.jpg",
            {
                k: v
                for k, v in zip(
                    (
                        True,
                        "fearful",
                        "surprised",
                        "happy",
                        "neutral",
                        "sad",
                        "disgusted",
                    ),
                    [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
                )
            },
            dt.now(),
        ],
        [
            3,
            1,
            "happy",
            "happy.jpg",
            {
                k: v
                for k, v in zip(
                    (1, "fearful", "surprised", "happy", "neutral", "sad", "disgusted"),
                    [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
                )
            },
            dt.now(),
        ],
        [
            3,
            1,
            "happy",
            "happy.jpg",
            {
                k: v
                for k, v in zip(
                    (
                        1.1,
                        "fearful",
                        "surprised",
                        "happy",
                        "neutral",
                        "sad",
                        "disgusted",
                    ),
                    [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
                )
            },
            dt.now(),
        ],
        [
            3,
            1,
            "happy",
            "happy.jpg",
            {
                k: v
                for k, v in zip(
                    (
                        (),
                        "fearful",
                        "surprised",
                        "happy",
                        "neutral",
                        "sad",
                        "disgusted",
                    ),
                    [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
                )
            },
            dt.now(),
        ],
        # predicted_on (datetime)
        [
            3,
            1,
            "happy",
            "happy.jpg",
            conf_to_dict([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]),
            True,
        ],
        [
            3,
            1,
            "happy",
            "happy.jpg",
            conf_to_dict([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]),
            {},
        ],
        [
            3,
            1,
            "happy",
            "happy.jpg",
            conf_to_dict([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]),
            set(),
        ],
        [
            3,
            1,
            "happy",
            "happy.jpg",
            conf_to_dict([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]),
            "str",
        ],
        [
            3,
            1,
            "happy",
            "happy.jpg",
            conf_to_dict([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]),
            1,
        ],
        [
            3,
            1,
            "happy",
            "happy.jpg",
            conf_to_dict([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]),
            (),
        ],
        [
            3,
            1,
            "happy",
            "happy.jpg",
            conf_to_dict([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]),
            1.1,
        ],
        [
            3,
            1,
            "happy",
            "happy.jpg",
            conf_to_dict([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]),
            [],
        ],
    ],
)
def test_PredDtype(predlist, capsys):
    test_PredClass(predlist, capsys)


# Testing other invalid values
@pytest.mark.xfail(reason="Invalid inputs")
@pytest.mark.parametrize(
    "predlist",
    [
        # empty image path
        [
            1,
            1,
            "happy",
            "",
            conf_to_dict([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]),
            dt.now(),
        ],
        # unrecognised confidence key
        [
            1,
            1,
            "happy",
            "happy.jpg",
            {
                k: v
                for k, v in zip(
                    (
                        "cool",
                        "fearful",
                        "surprised",
                        "happy",
                        "neutral",
                        "sad",
                        "disgusted",
                    ),
                    [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
                )
            },
            dt.now(),
        ],
    ],
)
def test_PredOther(predlist, capsys):
    test_PredClass(predlist, capsys)
