from datetime import datetime as dt

from flask_login import UserMixin
from sqlalchemy.orm import validates
from werkzeug.security import check_password_hash, generate_password_hash

from application import db, login_manager

emotion_list = ("angry", "fearful", "surprised", "happy", "neutral", "sad", "disgusted")


@login_manager.user_loader
def get_user(id):
    return User.query.get(id)


class Prediction(db.Model):
    __tablename__ = "prediction"

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    fk_user_id = db.Column(db.Integer, db.ForeignKey("users.id"))
    emotion = db.Column(db.String)
    file_path = db.Column(db.String, nullable=False)
    prediction = db.Column(db.PickleType, nullable=False)
    predicted_on = db.Column(db.DateTime, nullable=False)

    # === Validation ===>
    # dtype
    # valid values (ie postive/negative)
    # range (if any)
    # length of string (if any)

    @validates("id")
    def validate_id(self, _, id):
        if type(id) is not int:
            raise AssertionError("Prediction Id must be an Integer")
        if id <= 0:
            raise AssertionError("Prediction Id must be positive")
        return id

    @validates("fk_user_id")
    def validate_fk_user_id(self, _, fk_user_id):
        if type(fk_user_id) is not int:
            raise AssertionError("Foreign Key User Id must be an Integer")
        if fk_user_id <= 0:
            raise AssertionError("Foreign Key User Id must be positive")
        return fk_user_id

    @validates("emotion")
    def validate_emotion(self, _, emotion):
        if type(emotion) is not str:
            raise AssertionError("Emotion must be a String")
        if emotion not in emotion_list:
            raise AssertionError("Emotion is not recognised")
        return emotion

    @validates("file_path")
    def validate_file_path(self, _, file_path):
        if type(file_path) is not str:
            raise AssertionError("File path must be a String")
        if len(file_path) <= 0:
            raise AssertionError("File path must not be empty")
        return file_path

    @validates("prediction")
    def validate_prediction(self, _, prediction):
        if type(prediction) not in [dict, list]:
            raise AssertionError("Prediction must be a Dictionary or List")

        if type(prediction) is dict:
            if len(prediction.keys()) != 7:
                raise AssertionError("Prediction must contain 7 emotions")
            for e in emotion_list:
                if type(e) is not str:
                    raise AssertionError("Prediction confidence key must a string")
                if e not in prediction:
                    raise AssertionError("Prediction emotion is not recognised")
                if type(prediction[e]) not in [float, int]:
                    raise AssertionError("Prediction confidence must be a float or int")
                if prediction[e] < 0 or prediction[e] > 1:
                    raise AssertionError("Prediction must be between 0 and 1")

        if type(prediction) is list:
            if len(prediction) != 7:
                raise AssertionError("Prediction must contain 7 emotions")
            for e in prediction:
                if len(e) != 2:
                    raise AssertionError(
                        "Prediction must be a list of tuples of length 2"
                    )
                if type(e[0]) is not str:
                    raise AssertionError("Prediction emotion must be a string")
                if e[0].lower() not in emotion_list:
                    raise AssertionError("Prediction emotion is not recognised")
                if type(e[1]) not in [int, float]:
                    raise AssertionError("Prediction confidence must be a float")
                if e[1] < 0 or e[1] > 1:
                    raise AssertionError("Prediction must be between 0 and 1")

        return prediction

    @validates("predicted_on")
    def validate_predicted_on(self, key, predicted_on):
        if type(predicted_on) is not dt:
            raise AssertionError("Date of Prediction must be a datetime")
        if predicted_on > dt.now():
            raise AssertionError("Date of Prediction must be in the past")
        if predicted_on < dt(2020, 1, 1):
            raise AssertionError("Date of Prediction must be after 01/01/2020")
        return predicted_on


class User(db.Model, UserMixin):
    __tablename__ = "users"

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    username = db.Column(db.String, nullable=False, unique=True)
    password = db.Column(db.String, nullable=False)
    created_on = db.Column(db.DateTime, nullable=False)

    def __init__(self, username, password, created_on):
        self.username = username
        self.password = generate_password_hash(password)
        self.created_on = created_on

    def verify_password(self, pwd):
        return check_password_hash(self.password, pwd)

    # === Validation ===

    @validates("id")
    def validate_id(self, key, id):
        if type(id) is not int:
            raise AssertionError("User Id must be an Integer")
        if id <= 0:
            raise AssertionError("User Id must be positive")
        return id

    @validates("username")
    def validate_username(self, key, username):
        if type(username) is not str:
            raise AssertionError("Username must be a String")
        if len(username) <= 0:
            raise AssertionError("Username must not be empty")
        if not username.isalpha():
            raise AssertionError("Username must contain only letters")
        return username

    @validates("password")
    def validate_password(self, key, password):
        if type(password) is not str:
            raise AssertionError("Password must be a String")
        if len(password) <= 0:
            raise AssertionError("Password must not be empty")
        return password

    @validates("created_on")
    def validate_created_on(self, key, created_on):
        if type(created_on) is not dt:
            raise AssertionError(
                "Date of User creation must be a datetime"
            )
        return created_on
