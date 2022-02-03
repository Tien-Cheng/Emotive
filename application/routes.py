import datetime
import json
import os
import urllib
from datetime import datetime as dt
from os import getcwd

import cv2
import numpy as np
import plotly
import plotly.graph_objects as go
import requests
from flask import (flash, abort, json, jsonify, make_response, redirect,
                   render_template, request, url_for)
from flask_login import current_user, login_user, logout_user
from PIL import Image
from plotly.subplots import make_subplots
from sqlalchemy.sql.expression import column
from werkzeug.exceptions import InternalServerError

from application import app, db
from application.forms import LoginForm, RegisterForm
from application.models import Prediction, User

# ===== Functions and Global variables ===== >>>

emotion_list = ("angry", "fearful", "surprised", "happy", "neutral", "sad", "disgusted")

# Create the database if not exist
db.create_all()

# Function to add new prediction or user
def add_to_db(new_pred):
    try:
        db.session.add(new_pred)
        db.session.commit()
        return new_pred.id
    except Exception as error:
        db.session.rollback()
        flash(error, "danger")
        return None


# Getting the histories in pages
def get_history(
    user_id,
    page=None,
    per_page=9,
    order_by="predicted_on",
    desc=True,
    emotion_filter=None,
):

    try:
        order_by = column(order_by)

        if desc:
            order_by = order_by.desc()
        else:
            order_by = order_by.asc()

        if emotion_filter == None:
            emotion_filter = emotion_list

        results = (
            db.session.query(Prediction)
            .filter(
                Prediction.fk_user_id == user_id, Prediction.emotion.in_(emotion_filter)
            )
            .order_by(order_by)
        )

        if page is None:
            return results.all()
        else:
            return results.paginate(page=page, per_page=per_page)

    except Exception as error:
        flash(str(error), "danger")


# Sorting the prediction dictionary by probabilities
def sort_prediction(pred_dict):
    return [
        (k.capitalize(), v)
        for k, v in sorted(pred_dict.items(), key=lambda item: item[1], reverse=True)
    ]


# Quotes
with open("./application/static/quotes.json") as f:
    global_quotes = json.load(f)

# Plot User Activity
def plot_history(history):

    fig = go.Figure()

    emotion_color_map = {
        "angry": "#FF5858",
        "happy": "#55F855",
        "neutral": "#57D9F1",
        "surprised": "#FFE858",
        "sad": "#629AF3",
        "fearful": "#FFB858",
        "disgusted": "#7F68F4",
    }

    for emotion in emotion_list:

        fig.add_trace(
            go.Histogram(
                name=emotion.capitalize(),
                x=[
                    i.predicted_on
                    for i in history
                    if i.prediction[0][0].lower() == emotion
                ],
                nbinsx=20,
                marker_line_width=1.5,
                marker_line_color="white",
                marker_color=emotion_color_map[emotion],
            )
        )

    fig.update_layout(paper_bgcolor="white", plot_bgcolor="white", barmode="stack")
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor="#E5E5E5")
    fig.update_layout(margin=dict(l=10, b=10, r=130, t=20))

    html_file_path = f"{getcwd()}/application/static/file.html"

    plotly.offline.plot(
        fig, include_plotlyjs=False, filename=html_file_path, auto_open=False
    )

    plotly_txt = '<script src="https://cdn.plot.ly/plotly-latest.min.js"></script>'

    with open(html_file_path, "r+") as f:
        content = f.read()
        f.seek(0, 0)
        f.write(plotly_txt + "\n" + content)

    return html_file_path


# ===== Error Handler ===== >>>

@app.errorhandler(Exception)
def error_handler(error):
    if not hasattr(error, "name") or not hasattr(error, "code"):
        error = InternalServerError
        error.name = "Internal Server Error"
        error.code = 500
    return render_template("error.html", error=error, page="error", userInfo=current_user), error.code


# ===== Routes ===== >>>


@app.route("/set-cookie", methods=["GET"])
def change_cookie():
    auto_capture = request.args.get("autoCapture")
    redir_page = request.args.get("page", "dashboard")
    resp = make_response(redirect(url_for(redir_page)))
    resp.set_cookie("autoCapture", auto_capture)
    return resp


@app.route("/", methods=["GET"])
def homepage():

    resp = make_response(
        render_template(
            "index.html",
            page="home",
            userInfo=current_user,
        )
    )

    if 'autoCapture' in request.cookies:
        resp.set_cookie('autoCapture', request.cookies.get('autoCapture'))
    else:
        resp.set_cookie("autoCapture", "ON")
        
    return resp


# For new users who want to register
@app.route("/register", methods=["GET", "POST"])
def register():
    form = RegisterForm()

    if request.method == "POST":
        if form.validate_on_submit():
            username = form.username.data
            password = form.password.data

            # Check if user exist
            user = User.query.filter_by(username=username).first()

            # If user does not exist, add to database
            if not user:

                try:
                    user_db = User(
                        username=str(username),
                        password=str(password),
                        created_on=dt.now(),
                    )
                except Exception as e:
                    flash(e, "danger")
                    return render_template(
                        "register.html",
                        page="register",
                        userInfo=current_user,
                        form=form,
                    )

                add_to_db(user_db)

                flash("You're registered! Try to login!", "green")
            else:
                flash("Account already exist. Try to login!", "dark")
            return redirect(url_for("login"))
        else:
            flash("Register failed!", "red")

    return render_template(
        "register.html", page="register", userInfo=current_user, form=form
    )


# Login is the default page the user will see
# If the user is authenticated, they will be redirected to the prediction form
# If the user is not signed in, they'll be redirected to login
@app.route("/login", methods=["GET", "POST"])
def login():
    form = LoginForm()

    if request.method == "POST":
        if form.validate_on_submit():
            username = form.username.data
            password = form.password.data

            # Check if user exist
            user = User.query.filter_by(username=username).first()

            # Verify if user exist and the passwords are the same
            if user and user.verify_password(password):
                login_user(user)
                return redirect(url_for("predict"))
            else:
                flash("Login invalid!", "red")
        else:
            flash("Login failed!", "red")

    # Authenticated users will be redirected to /predict
    elif request.method == "GET":
        if current_user.is_authenticated:
            return redirect(url_for("predict"))

    # If user is not authenticated
    resp = make_response(
        render_template("login.html", page="login", userInfo=current_user, form=form)
    )

    resp.set_cookie("autoCapture", "ON")

    return resp


# For users to log out
@app.route("/logout", methods=["GET"])
def logout():
    logout_user()
    flash("You're logged out.", "green")
    return redirect(url_for("login"))


# Prediction form and prediction
@app.route("/predict", methods=["GET", "POST"])
def predict():

    # Unauthenticated user will be redirected to login
    if not current_user.is_authenticated:
        flash("Unauthorized: You're not logged in!", "red")
        return redirect(url_for("login"))

    if request.method == "POST":
        upload_time = dt.now().strftime("%Y%m%d%H%M%S%f")
        imgName = f"{current_user.username.strip().replace(' ', '_')}_{upload_time}.png"
        imgPath = f"./application/static/images/{imgName}"

        # Using file upload
        if "file" in request.files.keys():

            f = request.files["file"]
            ext = f.filename.split(".")[-1]

            if f.filename == "":
                flash("You did not upload any image!", "red")
                return redirect(url_for("predict"))

            # Handle non-standard images
            if ext not in ["png"]:
                flash("Upload only png!", "red")
                return redirect(url_for("predict"))

            f.save(imgPath)

        # Using WebCam
        elif request.data:

            image_b64 = request.data.decode("utf-8")
            response = urllib.request.urlopen(image_b64)

            with open(imgPath, "wb") as f:
                f.write(response.file.read())

        else:
            flash("You did not use WebCam or File Upload!", "red")
            return redirect(url_for("predict"))

        # === Crop the faces in the image ===>

        image = cv2.imread(imgPath)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        faceCascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        faces = faceCascade.detectMultiScale(
            gray, scaleFactor=1.3, minNeighbors=3, minSize=(30, 30)
        )

        if len(faces) < 1:

            os.remove(imgPath)  # Remove image from directory
            flash("No face detected!", "red")
            return redirect(url_for("predict"))
        
        elif len(faces) > 1:

            os.remove(imgPath)
            flash("Multiple faces detected!", "red")
            return redirect(url_for("predict"))

        for idx, (x, y, w, h) in enumerate(
            faces
        ):
            cv2.rectangle(
                image, (x - 5, y - 5), (x + w + 5, y + h + 5), (255, 59, 86), 2
            )
            roi_gray = gray[y : y + h, x : x + w]

            # Cropped black and white face
            cv2.imwrite(
                f"./application/static/images/faces/{current_user.username.strip().replace(' ', '_')}_{upload_time}_{idx}_face.png",
                roi_gray,
            )

        # === Send image to TF model server ===>

        # Waiting for AI model to output an array of 7 probability scores
        data_instance = np.asarray(Image.fromarray(roi_gray).resize((48, 48)))
        # From shape of (48,48) to (1,48,48,1)
        data_instance = np.expand_dims(np.expand_dims(data_instance, axis=2), axis=0)

        json_response = requests.post(
            "https://doaa-ca2-emotive-model.herokuapp.com/v1/models/img_classifier:predict",
            data=json.dumps(
                {
                    "signature_name": "serving_default",
                    "instances": data_instance.tolist(),
                }
            ),
            headers={"content-type": "application/json"},
        )

        predictions = json.loads(json_response.text)["predictions"]
        print("\n\n", predictions, "\n\n", np.array(predictions).shape, "\n\n")

        # === Save image metadata to database ===>

        prediction_to_db = {
            expression: probability
            for expression, probability in zip(
                [
                    "angry",
                    "disgusted",
                    "fearful",
                    "happy",
                    "neutral",
                    "sad",
                    "surprised",
                ],
                predictions[0],
            )
        }

        prediction = Prediction(
            fk_user_id=int(current_user.id),
            emotion=sort_prediction(prediction_to_db)[0][0].lower(),
            file_path=str(imgName),
            prediction=prediction_to_db,
            predicted_on=dt.now(),
        )

        pred_id = add_to_db(prediction)
        history = Prediction.query.filter_by(id=pred_id).first()

        history.prediction = sort_prediction(history.prediction)

        return render_template(
            "result.html", page="results", userInfo=current_user, history=history
        )

    return render_template(
        "predict.html",
        page="predict",
        userInfo=current_user,
        autoCapture=request.cookies.get("autoCapture"),
    )


# Contains the history of the predictions in table form
@app.route("/history", methods=["GET"])
def history():

    # Unauthenticated user will be redirected to login
    if not current_user.is_authenticated:
        flash("Unauthorized: You're not logged in!", "red")
        return redirect(url_for("login"))

    page = int(request.args.get("page", 1))
    per_page = int(request.args.get("per_page", 9))
    col_sort = request.args.get("col_sort", "predicted_on")
    desc = request.args.get("dir", "desc") == "desc"

    emotion_filter = {k: 1 for k in emotion_list}

    req_dict_keys = request.args.keys()

    # check if req_dict_keys does not contain all elements in emotion_list
    if not all(f"c{el.capitalize()}" not in req_dict_keys for el in emotion_list):
        for e in emotion_list:
            emotion_filter[e] = int(request.args.get(f"c{e.capitalize()}", 0))

    history = get_history(
        current_user.id,
        page,
        per_page,
        col_sort,
        desc,
        [k for k, v in emotion_filter.items() if v == 1],
    )

    for i, p in enumerate(history.items):
        history.items[i].prediction = sort_prediction(p.prediction)

    return render_template(
        "history.html",
        page="history",
        userInfo=current_user,
        history=history,
        col_sort=col_sort,
        desc=desc,
        per_page=per_page,
        emotion_filter=emotion_filter,
    )


# Delete history with id argument
@app.route("/history/delete", methods=["POST"])
def delete_history():

    # Unauthenticated user will be redirected to login
    if not current_user.is_authenticated:
        flash("Unauthorized: You're not logged in!", "red")
        return redirect(url_for("login"))

    history_id = request.args.get("id")
    history = Prediction.query.filter_by(id=history_id).first()

    # Check if history exist
    if history:
        # Check if the predictor is the current user
        if history.fk_user_id != current_user.id:
            flash("Unauthorized: You're not the predictor!", "red")
            return redirect(url_for("history"))

        else:
            # Remove image from the folder
            os.remove(f"./application/static/images/{history.file_path}")

            Prediction.query.filter_by(id=history_id).delete()
            db.session.commit()

            flash(f"Deleted history with id = {history_id}!", "green")
    else:
        flash(f"No history with id = {history_id}!", "red")

    history = get_history(current_user.id)
    return redirect(url_for("history"))


# Contains the individual prediction history
@app.route("/history/<int:history_id>", methods=["GET"])
def results(history_id):

    # Unauthenticated user will be redirected to login
    if not current_user.is_authenticated:
        flash("Unauthorized: You're not logged in!", "red")
        return redirect(url_for("login"))

    history = Prediction.query.filter_by(id=history_id).first()

    # If history exist, render results
    if history:
        # Check if the predictor is the same as current user
        if history.fk_user_id != current_user.id:
            flash("Unauthorized: You're not the predictor!", "red")
            return redirect(url_for("history"))
        else:
            history.prediction = sort_prediction(history.prediction)

            return render_template(
                "result.html", page="results", userInfo=current_user, history=history
            )

    # If not, render histories table
    else:
        flash(f"No history with id = {history_id}!", "red")

        history = get_history(current_user.id)
        return redirect(url_for("history"))


# Shows the user's dashboard
@app.route("/dashboard", methods=["GET"])
def dashboard():

    # Unauthenticated user will be redirected to login
    if not current_user.is_authenticated:
        flash("Unauthorized: You're not logged in!", "red")
        return redirect(url_for("login"))

    date_filter = request.args.get("date_filter", "All Time")

    date_filter_map = {
        "1 Day": "days=1",
        "3 Days": "days=3",
        "1 Week": "days=7",
        "1 Month": "days=30",
        "3 Months": "days=90",
        "6 Months": "days=180",
        "1 Year": "days=365",
        "All Time": "weeks=5300",
    }

    date_lower_bound = dt.now() - eval(
        f"datetime.timedelta({date_filter_map[date_filter]})"
    )

    history = (
        db.session.query(Prediction)
        .filter(
            Prediction.fk_user_id == current_user.id,
            Prediction.predicted_on > date_lower_bound,
        )
        .all()
    )

    total_photos = len(history)

    emotion_counter = {k: 0 for k in emotion_list}
    est_face = {k: None for k in emotion_list}

    data_usage_mb = 0

    # Getting the right data
    for i, p in enumerate(history):
        history[i].prediction = sort_prediction(p.prediction)

        emotion = history[i].prediction[0][0].lower()
        emotion_counter[emotion] += 1
        
        try:
            data_usage_mb += (os.path.getsize(f"{getcwd()}/application/static/images/{history[i].file_path}") / 1e6)
        except FileNotFoundError:
            print(f"{getcwd()}/application/static/images/{history[i].file_path} not found")
        
        if est_face[emotion] == None:
            est_face[emotion] = history[i]
        elif est_face[emotion].prediction[0][1] < history[i].prediction[0][1]:
            est_face[emotion] = history[i]

    emotion_counter = [
        (k.capitalize(), v)
        for k, v in sorted(
            emotion_counter.items(), key=lambda item: item[1], reverse=True
        )
    ]

    print(est_face)

    return render_template(
        "dashboard.html",
        page="dashboard",
        userInfo=current_user,
        est_face=est_face,
        emotion_counter=emotion_counter,
        total_photos=total_photos,
        data_usage_mb=data_usage_mb,
        html_file_path=plot_history(history),
        quote=global_quotes[emotion_counter[0][0].lower()][np.random.randint(0, 3)],
        autoCapture=request.cookies.get("autoCapture"),
        date_filter=date_filter,
    )


# =========== APIs ===========

# API for prediction
@app.route("/api/predict", methods=["POST"])
def api_predict():

    upload_time = dt.now().strftime("%Y%m%d%H%M%S%f")
    imgName = f"api_{upload_time}.png"
    imgPath = f"./application/static/images/{imgName}"

    # Using file upload
    if "file" in request.files.keys():

        f = request.files["file"]
        ext = f.filename.split(".")[-1]

        if f.filename == "":
            return jsonify({"error": "No image uploaded"})

        # Handle non-standard images
        if ext not in ["png"]:
            return jsonify({"error": "Only PNG images are allowed"})

        f.save(imgPath)

    # Using WebCam
    elif request.data:

        image_b64 = request.data.decode("utf-8")
        response = urllib.request.urlopen(image_b64)

        with open(imgPath, "wb") as f:
            f.write(response.file.read())

    else:
        return jsonify({"error": "No image uploaded"})

    # === Crop the faces in the image ===>

    image = cv2.imread(imgPath)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    faceCascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    
    faces = faceCascade.detectMultiScale(
        gray, scaleFactor=1.3, minNeighbors=3, minSize=(30, 30)
    )

    if len(faces) < 1:

        os.remove(imgPath)  # Remove image from directory
        return jsonify({"error": "No face detected"})
    
    elif len(faces) > 1:

        os.remove(imgPath)
        return jsonify({"error": "Multiple faces detected"})

    for idx, (x, y, w, h) in enumerate(
        faces
    ):
        cv2.rectangle(
            image, (x - 5, y - 5), (x + w + 5, y + h + 5), (255, 59, 86), 2
        )
        roi_gray = gray[y : y + h, x : x + w]

        # Cropped black and white face
        cv2.imwrite(
            f"./application/static/images/faces/api_{upload_time}_{idx}_face.png",
            roi_gray,
        )

    # === Send image to TF model server ===>

    # Waiting for AI model to output an array of 7 probability scores
    data_instance = np.asarray(Image.fromarray(roi_gray).resize((48, 48)))
    # From shape of (48,48) to (1,48,48,1)
    data_instance = np.expand_dims(np.expand_dims(data_instance, axis=2), axis=0)

    json_response = requests.post(
        "https://doaa-ca2-emotive-model.herokuapp.com/v1/models/img_classifier:predict",
        data=json.dumps(
            {
                "signature_name": "serving_default",
                "instances": data_instance.tolist(),
            }
        ),
        headers={"content-type": "application/json"},
    )

    predictions = json.loads(json_response.text)["predictions"]

    # === Save image metadata to database ===>

    prediction_to_db = {
        expression: probability
        for expression, probability in zip(
            [
                "angry",
                "disgusted",
                "fearful",
                "happy",
                "neutral",
                "sad",
                "surprised",
            ],
            predictions[0],
        )
    }

    prediction = Prediction(
        fk_user_id=int(1),
        emotion=sort_prediction(prediction_to_db)[0][0].lower(),
        file_path=str(imgName),
        prediction=prediction_to_db,
        predicted_on=dt.now(),
    )

    pred_id = add_to_db(prediction)
    history = Prediction.query.filter_by(id=pred_id).first()

    history.prediction = sort_prediction(history.prediction)

    return jsonify(history)

# API for adding history
@app.route("/api/history/add", methods=["GET"])
def api_add_history():
    # Retrieve the json file posted from client
    data = request.get_json()

    # Add the history to the database
    history = Prediction(
        fk_user_id=int(1),
        emotion=data["emotion"],
        file_path=data["file_path"],
        prediction=data["prediction"],
        predicted_on=dt.now(),
    )

    hist_id = add_to_db(history)

    return jsonify({"id": hist_id})


# API for getting all history
@app.route("/api/history", methods=["GET"])
def api_get_all_history():

    page = int(request.args.get("page", 1))
    per_page = int(request.args.get("per_page", 9))
    col_sort = request.args.get("col_sort", "predicted_on")
    desc = request.args.get("dir", "desc") == "desc"

    emotion_filter = {k: 1 for k in emotion_list}

    req_dict_keys = request.args.keys()

    # check if req_dict_keys does not contain all elements in emotion_list
    if not all(f"c{el.capitalize()}" not in req_dict_keys for el in emotion_list):
        for e in emotion_list:
            emotion_filter[e] = int(request.args.get(f"c{e.capitalize()}", 0))

    history = get_history(
        current_user.id,
        page,
        per_page,
        col_sort,
        desc,
        [k for k, v in emotion_filter.items() if v == 1],
    )

    for i, p in enumerate(history.items):
        history.items[i].prediction = sort_prediction(p.prediction)
    
    return jsonify(history)

# API for getting history by id
@app.route("/api/history/get/<int:history_id>", methods=["GET"])
def api_get_history(history_id):

    history_id = request.arg

    # Get the history from the database
    history = Prediction.query.filter_by(id=history_id).first()

    return jsonify(history)


# API for deleting history
@app.route("/api/history/delete/<int:history_id>", methods=["GET"])
def api_delete_history(history_id):
    
    history = Prediction.query.filter_by(id=history_id).first()

    # Check if history exist
    if history:
        # Remove image from the folder
        os.remove(f"./application/static/images/{history.file_path}")

        Prediction.query.filter_by(id=history_id).delete()
        db.session.commit()

        return jsonify({"success": True})

    else:
        return jsonify({"error": "History not found"})

# ========= APIs Users =========

# API: add users
@app.route("/api/user-add", methods=["POST"])
def api_user_add():

    # Retrieve the json file posted from client
    data = request.get_json()

    # Create a user object store all data for db action
    user_id = add_to_db(
        User(
            username=data["username"],
            password=data["password"],
            created_on=dt.now(),
        )
    )

    # Return the result of the db action
    return jsonify({"id": user_id})

# API get users
@app.route("/api/user-get/<id>", methods=["GET"])
def api_user_get(id):

    # Retrieve the user using id from client
    user = User.query.filter_by(id=id).first()

    # Prepare a dictionary for json conversion
    data = {
        "id": user.id,
        "username": user.username,
        "password": user.password,
    }

    # Convert the data to json
    result = jsonify(data)
    return result

# API get all users
@app.route("/api/get-users", methods=["GET"])
def get_all_users():

    all_users = []

    for e in User.query.all():
        all_users.append(
            {
                "id": e.id,
                "username": e.username,
                "password": e.password,
            }
        )

    return jsonify(all_users)


# API delete users
@app.route("/api/user-delete/<id>", methods=["GET"])
def api_user_delete(id):
    User.query.filter_by(id=id).delete()
    return jsonify({"result": "ok"})


# ========== Others ==========
