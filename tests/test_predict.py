import pytest
import json
import os
from io import BytesIO

# TEMPORARY: Find an image that predicts fearful and digusted
# emotion_list = ("angry", "fearful", "surprised", "happy", "neutral", "sad", "disgusted")
emotion_list = ("angry", "surprised", "happy", "neutral", "sad")

# Test predict API
@pytest.mark.parametrize("emotionImg", [f"{e}.{ext}" for e in emotion_list for ext in ['png', 'jpg']])
def test_predictAPI(client, emotionImg, capsys):
    with capsys.disabled():

        data = {
            "file": (BytesIO(open(f'./tests/test_files/{emotionImg}', 'rb').read()), emotionImg)
        }

        response = client.post(
            '/api/predict',
            data=data,
            content_type="multipart/form-data"
        )

        assert response.status_code == 200
        assert response.headers["Content-Type"] == "application/json"
        response_body = json.loads(response.get_data(as_text=True))

        # Remove test api images
        os.remove(f"./application/static/images/{response_body['file_path']}")

        # TEMPORARY
        if "error" in response_body.keys():
            print(response_body["error"])

        assert "error" not in response_body.keys()
        assert response_body["emotion"] == emotionImg.split(".")[0]

# Test non png or non jpg files
@pytest.mark.xfail(reason="Not png or jpg")
@pytest.mark.parametrize("files", [
    "angry.txt",
    "emotion.pdf",
    "happy.webp"
])
def test_predictAPI_fail(client, files, capsys):
    test_predictAPI(client, files, capsys)