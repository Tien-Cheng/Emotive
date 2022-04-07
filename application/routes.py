import datetime
import json
import os
import shutil
import urllib
from datetime import datetime as dt
from datetime import timedelta
from os import getcwd

import cv2
import numpy as np
import pandas as pd
import plotly
import plotly.graph_objects as go
import requests
from flask import (abort, flash, json, jsonify, make_response, redirect,
                   render_template, request, url_for)
from flask_login import current_user, login_user, logout_user
from PIL import Image
from plotly.subplots import make_subplots
from sqlalchemy.sql.expression import column
from werkzeug.exceptions import InternalServerError

from application import app, db
from application.forms import LoginForm, RegisterForm
from application.models import Prediction, User

# ===== Functions and Global variables ===== #

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

    try:
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

        emotion_score_map = {
            "angry": -1,
            "happy": 1,
            "neutral": 0.5,
            "surprised": 0.2,
            "sad": -0.2,
            "fearful": -0.3,
            "disgusted": -0.5,
        }

        # List of Emotions for Items in History
        sorted_history = sorted(history, key=lambda x: x.predicted_on)

        dates = [h.predicted_on for h in sorted_history]
        net = [emotion_score_map[h.prediction[0][0].lower()] for h in sorted_history]

        df = pd.DataFrame({"dates": dates, "net": net})

        try:

            binned_dates = pd.to_datetime(
                np.linspace(
                    pd.Timestamp(dates[0]).value, pd.Timestamp(dates[-1]).value, 17
                )
            )
            net_emotion = []

            for idx, bd in enumerate(binned_dates[1:]):
                net_emotion.append(
                    df[(binned_dates[idx] < df.dates) & (df.dates <= bd)].net.sum()
                )

            colours = ["#55F855" if i > 0 else "#FF5858" for i in net_emotion]
            binned_dates = binned_dates[1:]

        except:

            net_emotion, binned_dates, colours = [], [], []

        # Plot Histogram
        for emotion in emotion_list:
            fig.add_trace(
                go.Histogram(
                    name=emotion.capitalize(),
                    x=[
                        i.predicted_on.date()
                        for i in history
                        if i.prediction[0][0].lower() == emotion
                    ],
                    nbinsx=20,
                    marker_line_width=1.5,
                    marker_line_color="white",
                    marker_color=emotion_color_map[emotion],
                )
            )

        fig.add_trace(
            go.Bar(
                name="Net Emotion",
                y=net_emotion,
                x=binned_dates,
                marker_color=colours,
                visible=False,
            )
        )

        # Add an update menu to allow the selection of different plots
        fig.update_layout(
            updatemenus=[
                dict(
                    type="buttons",
                    buttons=[
                        dict(
                            label="1",
                            method="update",
                            args=[
                                # Make histograms visible and hide line plot
                                dict(
                                    visible=[
                                        True,
                                        True,
                                        True,
                                        True,
                                        True,
                                        True,
                                        True,
                                        False,
                                    ]
                                ),
                                dict(
                                    barmode="stack",
                                    margin=dict(l=10, b=10, r=130, t=40),
                                ),
                            ],
                        ),
                        dict(
                            label="2",
                            method="update",
                            args=[
                                dict(
                                    visible=[
                                        False,
                                        False,
                                        False,
                                        False,
                                        False,
                                        False,
                                        False,
                                        True,
                                    ]
                                ),
                                dict(
                                    hovermode="x", margin=dict(l=10, b=20, r=30, t=40)
                                ),
                            ],
                        ),
                    ],
                )
            ]
        )

        fig.update_layout(
            paper_bgcolor="white",
            plot_bgcolor="white",
            barmode="stack",
        )

        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor="#E5E5E5")
        fig.update_layout(margin=dict(l=10, b=10, r=130, t=40))

        # Set location of plot selection menu
        fig["layout"]["updatemenus"][0]["pad"] = dict(r=10, t=5)

        html_file_path = f"{getcwd()}/application/static/file.html"

        # Generate HTML for plot embedding
        plotly.offline.plot(
            fig, include_plotlyjs=False, filename=html_file_path, auto_open=False
        )

        plotly_txt = '<script src="https://cdn.plot.ly/plotly-latest.min.js"></script>'

        # Add plotly.js
        with open(html_file_path, "r+") as f:
            content = f.read()
            f.seek(0, 0)
            f.write(plotly_txt + "\n" + content)

    except Exception as e:
        print(e)

    return html_file_path


# ===== Error Handler ===== #
class API_Error(Exception):
    def __init__(self, message, status_code=400):
        super().__init__()
        self.message = message
        self.status_code = status_code


@app.errorhandler(Exception)
def error_handler(error):

    if not hasattr(error, "name") or not hasattr(error, "code"):
        error = InternalServerError
        error.name = "Internal Server Error"
        error.code = 500

    return (
        render_template("error.html", error=error, page="error", userInfo=current_user),
        error.code,
    )


@app.errorhandler(API_Error)
def api_error_handler(error):
    return jsonify({"message": error.message}), error.status_code


# ===== Routes ===== #

# Populate the database with images for demonstration purposes
@app.route("/demo/populate", methods=["GET"])
def demo():

    # Unauthenticated user will be redirected to login
    if not current_user.is_authenticated:
        flash("Unauthorized: You're not logged in!", "red")
        return redirect(url_for("login"))

    # If there are enough histories, don't populate the database
    if db.session.query(Prediction).count() > 50:
        flash("Demo database already populated!", "info")
        return redirect(url_for("dashboard"))

    # Get a random date since a month ago
    def random_date():
        delta = timedelta(days=30)
        start = dt.now() - delta
        int_delta = (delta.days * 24 * 60 * 60) + delta.seconds
        random_second = np.random.randint(int_delta)
        return start + timedelta(seconds=random_second)

    # Get a random image from the demo folder
    for idx in np.random.randint(7, size=100):

        emotion = emotion_list[idx]
        upload_time = dt.now().strftime("%Y%m%d%H%M%S%f")

        img_path = f"./application/static/demo/{emotion}.jpg"
        img_name = (
            f"demo_{current_user.username.strip().replace(' ', '_')}_{upload_time}.jpg"
        )
        dest_path = f"./application/static/images/{img_name}"

        prediction_to_db = dict()

        for e in emotion_list:
            if e == emotion:
                prediction_to_db[e] = np.random.uniform(0.7, 0.95)
            else:
                prediction_to_db[e] = np.random.uniform(0.0, 0.4)

        try:
            shutil.copy(img_path, dest_path)

            prediction = Prediction(
                fk_user_id=int(current_user.id),
                emotion=emotion,
                file_path=str(img_name),
                prediction=prediction_to_db,
                predicted_on=random_date(),
            )

            add_to_db(prediction)

        except Exception as e:
            print(e)
            flash("Error while copying files!", "red")
            return redirect(url_for("dashboard"))

    flash("Demo images added successfully!", "green")
    return redirect(url_for("dashboard"))


# Remove histories no images in the directory
# This is because newly added images are remove in heroku after inactivity
@app.route("/demo/remove-images", methods=["GET"])
def demo_remove_images():

    # Unauthenticated user will be redirected to login
    if not current_user.is_authenticated:
        flash("Unauthorized: You're not logged in!", "red")
        return redirect(url_for("login"))

    histories = db.session.query(Prediction).all()

    try:

        for history in histories:
            if not os.path.isfile(f"./application/static/images/{history.file_path}"):
                db.session.query(Prediction).filter_by(id=history.id).delete()
                db.session.commit()
                print(f">>> Removed history with {history.file_path}")

        flash("Histories with missing images removed!", "green")
        return redirect(url_for("dashboard"))

    except:
        flash("Error while removing images!", "red")
        return redirect(url_for("dashboard"))


# Set cookie for auto capture
@app.route("/set-cookie", methods=["GET"])
def change_cookie():
    auto_capture = request.args.get("autoCapture")
    redir_page = request.args.get("page", "dashboard")
    resp = make_response(redirect(url_for(redir_page)))
    resp.set_cookie("autoCapture", auto_capture)
    return resp


# Route for homepage
@app.route("/", methods=["GET"])
def homepage():

    resp = make_response(
        render_template(
            "index.html",
            page="home",
            userInfo=current_user,
        )
    )

    if "autoCapture" in request.cookies:
        resp.set_cookie("autoCapture", request.cookies.get("autoCapture"))

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

                flash("Registered! Try to login!", "green")

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
@app.route("/logout", methods=["POST"])
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
        imgName = f"{current_user.username.strip().replace(' ', '_')}_{upload_time}"
        imgPath = f"./application/static/images/{imgName}"

        # Using file upload
        if "file" in request.files.keys():

            f = request.files["file"]
            ext = f.filename.split(".")[-1]

            if f.filename == "":
                flash("You did not upload any image!", "red")
                return redirect(url_for("predict"))

            # Handle non-standard images
            if ext not in ["png", "jpg", "jpeg"]:
                flash("Upload only png or jpg!", "red")
                return redirect(url_for("predict"))

            imgNameExt = f"{imgName}.{ext}"
            imgPathExt = f"{imgPath}.{ext}"
            f.save(imgPathExt)

        # Using WebCam
        elif request.data:

            image_b64 = request.data.decode("utf-8")
            response = urllib.request.urlopen(image_b64)
            imgNameExt = f"{imgName}.png"
            imgPathExt = f"{imgPath}.png"
            with open(imgPathExt, "wb") as f:
                f.write(response.file.read())

        else:
            flash("You did not use WebCam or File Upload!", "red")
            #return redirect(url_for("predict"))
            return render_template(
                "predict.html",
                page="predict",
                userInfo=current_user,
                autoCapture=request.cookies.get("autoCapture"),
            )
        # === Crop the faces in the image ===>
        try:
            image = cv2.imread(imgPathExt)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            faceCascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            )

            faces = faceCascade.detectMultiScale(
                gray, scaleFactor=1.3, minNeighbors=3, minSize=(30, 30)
            )
        except:
            flash("Unable to process the image. The image may be corrupted!", "red")
            return render_template(
                "predict.html",
                page="predict",
                userInfo=current_user,
                autoCapture=request.cookies.get("autoCapture"),
            )
        if len(faces) < 1:

            # Remove image from directory
            os.remove(imgPathExt)
            flash("No face detected!", "red")
            #return redirect(url_for("predict"))
            return render_template(
                "predict.html",
                page="predict",
                userInfo=current_user,
                autoCapture=request.cookies.get("autoCapture"),
            )
        elif len(faces) > 1:

            os.remove(imgPathExt)
            flash("Multiple faces detected!", "red")
            #return redirect(url_for("predict"))
            return render_template(
            "predict.html",
            page="predict",
            userInfo=current_user,
            autoCapture=request.cookies.get("autoCapture"),
            )
        for (x, y, w, h) in faces:

            cv2.rectangle(
                image, (x - 5, y - 5), (x + w + 5, y + h + 5), (255, 59, 86), 2
            )
            roi_gray = gray[y : y + h, x : x + w]

        # === Send image to TF model server ===>

        # Waiting for AI model to output an array of 7 probability scores
        ori_face = np.asarray(Image.fromarray(roi_gray).resize((48, 48)))

        # From shape (48,48) to (48,48,3)
        data_instance = np.zeros((48, 48, 3))
        data_instance[:, :, 0] = ori_face
        data_instance[:, :, 1] = ori_face
        data_instance[:, :, 2] = ori_face

        # From shape of (48,48,3) to (1,48,48,3)
        data_instance = np.expand_dims(data_instance, axis=0)
        try:
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
        except:
            flash(
                "Model Failed To Predict. A likely reason is that the model is facing high demand at the moment.",
                "red",
            )
            # return redirect(url_for("predict"))
            return render_template(
                "predict.html",
                page="predict",
                userInfo=current_user,
                autoCapture=request.cookies.get("autoCapture"),
            )
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
            file_path=str(imgNameExt),
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

    # Get request arguments
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

        # Check if image file exists
        if not os.path.isfile(f"./application/static/images/{p.file_path}"):
            history.items[i].file_path = "default.png"

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
            try:
                os.remove(f"./application/static/images/{history.file_path}")
            except:
                print("Image not found")

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

            # Check if image file exists
            if not os.path.isfile(f"./application/static/images/{history.file_path}"):
                history.file_path = "default.png"

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
            data_usage_mb += (
                os.path.getsize(f"./application/static/images/{history[i].file_path}")
                / 1e6
            )
        except FileNotFoundError:
            print(f"{history[i].file_path} not found")

        # Check if image file exists
        if not os.path.isfile(f"./application/static/images/{p.file_path}"):
            history[i].file_path = "default.png"

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


# ========== APIs ========== #

# API for prediction
@app.route("/api/predict", methods=["POST"])
def api_predict():

    if "LOGIN_DISABLED" in app.config and app.config["LOGIN_DISABLED"]:
        user_id = 99
    elif not current_user.is_authenticated:
        raise API_Error("Not Logged In", 401)
    else:
        user_id = current_user.id
    upload_time = dt.now().strftime("%Y%m%d%H%M%S%f")
    imgName = f"api_{upload_time}"
    imgPath = f"./application/static/images/{imgName}"

    # Using file upload
    if "file" in request.files.keys():

        f = request.files["file"]
        ext = f.filename.split(".")[-1]

        if f.filename == "":
            raise API_Error("No file provided")

        # Handle non-standard images
        if ext not in ["png", "jpg"]:
            raise API_Error("Only PNG and JPG files are allowed!")

        imgNameExt = f"{imgName}.{ext}"
        imgPathExt = f"{imgPath}.{ext}"
        f.save(imgPathExt)

    else:
        raise API_Error("No file uploaded!")

    # === Crop the faces in the image ===>
    try:
        image = cv2.imread(imgPathExt)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        faceCascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )

        faces = faceCascade.detectMultiScale(
            gray, scaleFactor=1.3, minNeighbors=3, minSize=(30, 30)
        )
    except:
        raise API_Error("Unable to process image, image may be corrupted!")

    if len(faces) < 1:

        # Remove image from directory
        os.remove(imgPathExt)
        raise API_Error("No face detected!")

    elif len(faces) > 1:

        os.remove(imgPathExt)
        raise API_Error("Multiple faces detected!")

    for (x, y, w, h) in faces:

        cv2.rectangle(image, (x - 5, y - 5), (x + w + 5, y + h + 5), (255, 59, 86), 2)

        roi_gray = gray[y : y + h, x : x + w]

    # === Send image to TF model server ===>

    # Waiting for AI model to output an array of 7 probability scores
    ori_face = np.asarray(Image.fromarray(roi_gray).resize((48, 48)))

    # From shape (48,48) to (48,48,3)
    data_instance = np.zeros((48, 48, 3))
    data_instance[:, :, 0] = ori_face
    data_instance[:, :, 1] = ori_face
    data_instance[:, :, 2] = ori_face

    # From shape of (48,48,3) to (1,48,48,3)
    data_instance = np.expand_dims(data_instance, axis=0)
    try:
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
    except Exception as e:
        print(e)
        raise API_Error(
            "Model unable to predict image. It is likely that the model is facing high demand at the moment and thus cannot process your request.",
            500,
        )

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
        fk_user_id=user_id,
        emotion=sort_prediction(prediction_to_db)[0][0].lower(),
        file_path=str(imgNameExt),
        prediction=prediction_to_db,
        predicted_on=dt.now(),
    )

    pred_id = add_to_db(prediction)
    history = Prediction.query.filter_by(id=pred_id).first()

    history.prediction = sort_prediction(history.prediction)

    data = {
        "id": history.id,
        "fk_user_id": history.fk_user_id,
        "emotion": history.emotion,
        "file_path": history.file_path,
        "prediction": history.prediction,
    }

    return jsonify(data)


# ===== APIs Predictions =====#

# API for adding history
@app.route("/api/history/add", methods=["POST"])
def api_add_history():
    # Retrieve the json file posted from client
    data = request.get_json()

    # Add the history to the database
    history = Prediction(
        fk_user_id=data["fk_user_id"],
        emotion=data["emotion"],
        file_path=data["file_path"],
        prediction=data["prediction"],
        predicted_on=dt.now(),
    )

    hist_id = add_to_db(history)

    return jsonify({"id": hist_id})


# API for getting history by id
@app.route("/api/history/get/<int:history_id>", methods=["GET"])
def api_get_history(history_id):
    history = Prediction.query.filter_by(id=history_id).first()
    if history is None:
        raise API_Error("Entry not found", 404)
    if "LOGIN_DISABLED" in app.config and app.config["LOGIN_DISABLED"]:
        user_id = history.fk_user_id
    elif not current_user.is_authenticated:
        raise API_Error("Not Logged In", 401)
    else:
        user_id = current_user.id
    # Get the history from the database
    if user_id != history.fk_user_id:
        raise API_Error("Not logged in as correct user", 403)

    data = {
        "id": history.id,
        "fk_user_id": history.fk_user_id,
        "emotion": history.emotion,
        "file_path": history.file_path,
        "prediction": history.prediction,
    }

    return jsonify(data)


# API for getting all history
@app.route("/api/history/<int:user_id>", methods=["GET"])
def api_get_all_history(user_id):
    if "LOGIN_DISABLED" in app.config and app.config["LOGIN_DISABLED"]:
        pass
    elif not current_user.is_authenticated:
        raise API_Error("Not Logged In", 401)
    else:
        if user_id != current_user.id:
            raise API_Error("Not logged in as correct user", 403)
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
        user_id,
        page,
        per_page,
        col_sort,
        desc,
        [k for k, v in emotion_filter.items() if v == 1],
    )

    histories = []

    for h in history.items:
        histories.append(
            {
                "id": h.id,
                "fk_user_id": h.fk_user_id,
                "emotion": h.emotion,
                "file_path": h.file_path,
                "prediction": h.prediction,
            }
        )

    return jsonify(histories)


# API for deleting history
@app.route("/api/history/delete/<int:history_id>", methods=["DELETE"])
def api_delete_history(history_id):

    history = Prediction.query.filter_by(id=history_id).first()
    if "LOGIN_DISABLED" in app.config and app.config["LOGIN_DISABLED"]:
        user_id = history.fk_user_id
    elif not current_user.is_authenticated:
        raise API_Error("Not Logged In", 401)
    else:
        user_id = current_user.id
    # Get the history from the database
    if user_id != history.fk_user_id:
        raise API_Error("Not logged in as correct user", 403)

    # Check if history exist
    if history:

        Prediction.query.filter_by(id=history_id).delete()
        db.session.commit()

        return jsonify({"success": True})

    else:
        return jsonify({"error": "History not found"})


# ===== APIs Users ===== #

# API: Login
@app.route("/api/user/login", methods=["POST"])
def api_user_login():
    # Retrieve login data
    data = request.get_json()
    if data is None:
        raise TypeError(
            "Invalid request type. Ensure data is in the form of a json file."
        )

    username = data["username"]
    password = data["password"]

    # Check if user exists
    user = User.query.filter_by(username=username).first()

    if not user:
        raise API_Error("User not found", 404)
    elif not user.verify_password(password):
        raise API_Error("Invalid password", 403)
    else:
        login_user(user)
        return jsonify({"Result": "Logged In!", "id": user.id})


# API: Logout
@app.route("/api/user/logout", methods=["POST"])
def api_user_logout():
    logout_user()
    return jsonify({"Result": "Logged Out!"})


# API: add users
@app.route("/api/user/add", methods=["POST"])
def api_user_add():

    # Retrieve the json file posted from client
    data = request.get_json()
    if "LOGIN_DISABLED" not in app.config and not app.config["LOGIN_DISABLED"]:
        raise API_Error("Add User is disabled for now", 403)
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
@app.route("/api/user/<id>", methods=["GET"])
def api_user_get(id):
    if "LOGIN_DISABLED" in app.config and app.config["LOGIN_DISABLED"]:
        pass
    elif not current_user.is_authenticated:
        raise API_Error("Not Logged In", 401)
    else:
        if id != current_user.id:
            raise API_Error("Not logged in as correct user", 403)
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
@app.route("/api/user/all", methods=["GET"])
def get_all_users():
    if "LOGIN_DISABLED" not in app.config and not app.config["LOGIN_DISABLED"]:
        raise API_Error(
            "Get all users is only available for testing at the moment", 403
        )
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
@app.route("/api/user/delete/<id>", methods=["DELETE"])
def api_user_delete(id):
    if "LOGIN_DISABLED" in app.config and app.config["LOGIN_DISABLED"]:
        pass
    elif not current_user.is_authenticated:
        raise API_Error("Not Logged In", 401)
    else:
        if id != current_user.id:
            raise API_Error("Not logged in as correct user", 403)
    User.query.filter_by(id=id).delete()
    return jsonify({"result": "ok"})
