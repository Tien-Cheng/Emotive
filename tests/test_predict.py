import pytest
import json
import os
from io import BytesIO
import numpy as np

emotion_list = ("angry", "fearful", "surprised", "happy", "neutral", "sad", "disgusted")

# Test predict API
@pytest.mark.parametrize("emotionImg", [f"{e}.{ext}" for e in emotion_list for ext in ['png', 'jpg']])
def test_predAPI(client, emotionImg, capsys):
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
       
        assert emotionImg.split('.')[0].capitalize() in np.array(response_body["prediction"])[:,0][:4]

# Test non png or non jpg files
@pytest.mark.xfail(reason="Not png or jpg")
@pytest.mark.parametrize("files", [
    "angry.txt",
    "emotion.pdf",
    "happy.webp"
])
def test_predAPI_NonImg(client, files, capsys):
    test_predAPI(client, files, capsys)

# Test for no face detected
@pytest.mark.xfail(reason="No face detected")
@pytest.mark.parametrize("files", [
    "room.jpg",
    "sky.jpg"
    "mountain.jpg",
    "ocean.jpg",
    "forest.jpg",
])
def test_predAPI_NoFace(client, files, capsys):
    test_predAPI(client, files, capsys)

# Test more than one face
@pytest.mark.xfail(reason="More than one face")
@pytest.mark.parametrize("files", [
    "two_faces.jpg",
    "two_faces_2.jpg",
    "two_faces_3.jpg"
])
def test_predAPI_MoreThanOneFace(client, files, capsys):
    test_predAPI(client, files, capsys)