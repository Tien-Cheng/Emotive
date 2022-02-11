import itertools
import json

import pytest

"""
# [History] API Testing
* Add API
* Get API
* Get All API
* Get All API (with filters)
* Delete API
"""

emotion_list = ("angry", "fearful", "surprised", "happy", "neutral", "sad", "disgusted")

# Convert a list of confidence to a dictionary
def conf_to_dict(conf):
    return {k: v for k, v in zip(emotion_list, conf)}


# Test add API
@pytest.mark.parametrize(
    "histList",
    [
        # fk_user_id, emotion, file_path, prediction, prediction_id
        (1, "angry", "angry.jpg", conf_to_dict([0.7, 0.2, 0.3, 0.4, 0.5, 0.6, 0.1])),
        (
            1,
            "fearful",
            "fearful.jpg",
            conf_to_dict([0.1, 0.7, 0.3, 0.4, 0.5, 0.6, 0.2]),
        ),
        (
            1,
            "surprised",
            "surprised.jpg",
            conf_to_dict([0.1, 0.2, 0.7, 0.4, 0.5, 0.6, 0.3]),
        ),
        (2, "happy", "happy.png", conf_to_dict([0.1, 0.2, 0.3, 0.7, 0.5, 0.6, 0.4])),
        (
            2,
            "neutral",
            "neutral.png",
            conf_to_dict([0.1, 0.2, 0.3, 0.4, 0.7, 0.6, 0.5]),
        ),
        (2, "sad", "sad.png", conf_to_dict([0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 0.6])),
        (
            2,
            "disgusted",
            "disgusted.png",
            conf_to_dict([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]),
        ),
    ],
)

# Testing adding predictions
def test_addAPI(client, histList, capsys):
    with capsys.disabled():

        data = {
            "fk_user_id": histList[0],
            "emotion": histList[1],
            "file_path": histList[2],
            "prediction": histList[3],
        }

        response = client.post(
            "/api/history/add", data=json.dumps(data), content_type="application/json"
        )

        assert response.status_code == 200
        assert response.headers["Content-Type"] == "application/json"

        response_body = json.loads(response.get_data(as_text=True))
        assert response_body["id"]


# Test get API
@pytest.mark.parametrize(
    "histList",
    [
        # fk_user_id, emotion, file_path, prediction, prediction_id
        (1, "angry", "angry.jpg", conf_to_dict([0.7, 0.2, 0.3, 0.4, 0.5, 0.6, 0.1]), 1),
        (
            1,
            "fearful",
            "fearful.jpg",
            conf_to_dict([0.1, 0.7, 0.3, 0.4, 0.5, 0.6, 0.2]),
            2,
        ),
        (
            1,
            "surprised",
            "surprised.jpg",
            conf_to_dict([0.1, 0.2, 0.7, 0.4, 0.5, 0.6, 0.3]),
            3,
        ),
        (2, "happy", "happy.png", conf_to_dict([0.1, 0.2, 0.3, 0.7, 0.5, 0.6, 0.4]), 4),
        (
            2,
            "neutral",
            "neutral.png",
            conf_to_dict([0.1, 0.2, 0.3, 0.4, 0.7, 0.6, 0.5]),
            5,
        ),
        (2, "sad", "sad.png", conf_to_dict([0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 0.6]), 6),
        (
            2,
            "disgusted",
            "disgusted.png",
            conf_to_dict([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]),
            7,
        ),
    ],
)

# Testing getting individual predictions
def test_getAPI(client, histList, capsys):
    with capsys.disabled():
        response = client.get(
            f"/api/history/get/{histList[4]}", content_type="application/json"
        )

        assert response.status_code == 200
        assert response.headers["Content-Type"] == "application/json"

        response_body = json.loads(response.get_data(as_text=True))

        assert response_body["id"] == histList[4]
        assert response_body["fk_user_id"] == histList[0]
        assert response_body["emotion"] == histList[1]
        assert response_body["file_path"] == histList[2]
        assert response_body["prediction"] == histList[3]


# Test get all API
@pytest.mark.parametrize(
    "histList",
    [
        [
            (
                1,
                "angry",
                "angry.jpg",
                conf_to_dict([0.7, 0.2, 0.3, 0.4, 0.5, 0.6, 0.1]),
                1,
            ),
            (
                1,
                "fearful",
                "fearful.jpg",
                conf_to_dict([0.1, 0.7, 0.3, 0.4, 0.5, 0.6, 0.2]),
                2,
            ),
            (
                1,
                "surprised",
                "surprised.jpg",
                conf_to_dict([0.1, 0.2, 0.7, 0.4, 0.5, 0.6, 0.3]),
                3,
            ),
        ],
        [
            (
                2,
                "happy",
                "happy.png",
                conf_to_dict([0.1, 0.2, 0.3, 0.7, 0.5, 0.6, 0.4]),
                4,
            ),
            (
                2,
                "neutral",
                "neutral.png",
                conf_to_dict([0.1, 0.2, 0.3, 0.4, 0.7, 0.6, 0.5]),
                5,
            ),
            (2, "sad", "sad.png", conf_to_dict([0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 0.6]), 6),
            (
                2,
                "disgusted",
                "disgusted.png",
                conf_to_dict([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]),
                7,
            ),
        ],
    ],
)

# Test get all API
def test_getAllAPI(client, histList, capsys):
    with capsys.disabled():
        response = client.get(
            f"/api/history/{histList[0][0]}?col_sort=id&dir=asc",
            content_type="application/json",
        )

        assert response.status_code == 200
        assert response.headers["Content-Type"] == "application/json"

        response_body = json.loads(response.get_data(as_text=True))

        if histList[0][0] == 1:
            assert len(response_body) == 3
        else:
            assert len(response_body) == 4

        for pl, res in zip(histList, response_body):
            assert pl[4] == res["id"]
            assert pl[0] == res["fk_user_id"]
            assert pl[1] == res["emotion"]
            assert pl[2] == res["file_path"]
            assert pl[3] == res["prediction"]


# Test get all API with filters
def allPossibleFilters():
    fill_all = []
    fil_base = []
    fil_emotions = []

    for pp in [3, 9]:
        link_ext = ""
        link_ext += f"per_page={pp}&"

        for cs in ["id", "emotion", "file_path"]:
            link_ext += f"col_sort={cs}&"
            for so in ["asc", "desc"]:
                link_ext += f"dir={so}&"

                fil_base.append(link_ext)

    for i in range(1, 8):
        for j in itertools.combinations(emotion_list, i):
            tmp = ""
            for e in j:
                tmp += f"c{e.capitalize()}=1&"
            fil_emotions.append(tmp)

    for l in fil_base:
        for f in fil_emotions:
            fill_all.append([(l + f)[:-1]])

    return fill_all


@pytest.mark.parametrize("filters", allPossibleFilters())
def test_getAllAPI_filters(client, filters, capsys):
    with capsys.disabled():
        response = client.get(
            f"/api/history/{2}?" + filters[0], content_type="application/json"
        )

        assert response.status_code == 200
        assert response.headers["Content-Type"] == "application/json"

        response_body = json.loads(response.get_data(as_text=True))

        assert len(response_body) in list(range(5))


# Test delete API
@pytest.mark.parametrize("histIds", range(1, 8))
def test_deleteAPI(client, histIds, capsys):
    with capsys.disabled():
        response = client.delete(
            f"/api/history/delete/{histIds}", content_type="application/json"
        )

        assert response.status_code == 200
        assert response.headers["Content-Type"] == "application/json"

        response_body = json.loads(response.get_data(as_text=True))

        assert response_body["success"] == True
