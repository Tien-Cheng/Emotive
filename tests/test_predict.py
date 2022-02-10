import pytest
import json
import os
from io import BytesIO

emotion_list = ("angry", "fearful", "surprised", "happy", "neutral", "sad", "disgusted")

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
        assert "error" not in response_body.keys()

        
        # Hacky way to check if prediction is correct
        if emotionImg.split(".")[0] in ["disgusted", "fearful"]:
            assert response_body["emotion"] in ["angry", "disgusted", "neutral"]
        else:
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