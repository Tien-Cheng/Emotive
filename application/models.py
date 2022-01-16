from application import db, login_manager
from sqlalchemy.orm import validates
from datetime import datetime as dt
from flask_login import UserMixin
from werkzeug.security import generate_password_hash, check_password_hash

@login_manager.user_loader
def get_user(id):
    return User.query.get(id)

class Prediction(db.Model):
    __tablename__ = 'prediction'

    id           = db.Column(db.Integer, primary_key=True, autoincrement=True)
    fk_user_id   = db.Column(db.Integer, db.ForeignKey('users.id'))
    file_path    = db.Column(db.String, nullable=False)
    prediction   = db.Column(db.PickleType, nullable=False)
    predicted_on = db.Column(db.DateTime, nullable=False)

    # === Validation ===>
    # dtype
    # valid values (ie postive/negative)
    # range (if any)
    # length of string (if any)

    @validates('id') 
    def validate_id(self, key, id):
        if type(id) is not int: raise AssertionError('Prediction Id must be an Integer')
        if id <= 0: raise AssertionError('Prediction Id must be positive')
        return id
    
    @validates('fk_user_id') 
    def validate_fk_user_id(self, key, fk_user_id):
        if type(fk_user_id) is not int: raise AssertionError('Foreign Key User Id must be an Integer')
        if fk_user_id <= 0: raise AssertionError('Foreign Key User Id must be positive')
        return fk_user_id

    @validates('file_path')
    def validate_file_path(self, key, file_path):
        if type(file_path) is not str: raise AssertionError('Closest Car must be a String')
        if len(file_path) <= 0: raise AssertionError('Closest Car must not be empty')
        return file_path

    # Validate PickleType here
    # ...

    @validates('predicted_on') 
    def validate_predicted_on(self, key, predicted_on):
        if type(predicted_on) is not dt: raise AssertionError('Date of Prediction must be a datetime')
        return predicted_on

class User(db.Model, UserMixin):
    __tablename__ = 'users'
    
    id         = db.Column(db.Integer, primary_key=True, autoincrement=True)
    username   = db.Column(db.String, nullable=False)
    password   = db.Column(db.String, nullable=False)
    created_on = db.Column(db.DateTime, nullable=False)

    def __init__(self, username, password, created_on):
        self.username = username
        self.password = generate_password_hash(password)
        self.created_on = created_on

    def verify_password(self, pwd):
        return check_password_hash(self.password, pwd)
    
    # === Validation ===

    @validates('id') 
    def validate_id(self, key, id):
        if type(id) is not int: raise AssertionError('User Id must be an Integer')
        if id <= 0: raise AssertionError('User Id must be positive')
        return id
    
    @validates('username')
    def validate_username(self, key, username):
        if type(username) is not str: raise AssertionError('Username must be a String')
        if len(username) <= 0: raise AssertionError('Username must not be empty')
        return username
    
    @validates('password') 
    def validate_password(self, key, password):
        if type(password) is not str: raise AssertionError('Password must be a String')
        if len(password) <= 0: raise AssertionError('Password must not be empty')
        return password
    
    @validates('created_on') 
    def validate_created_on(self, key, created_on):
        if type(created_on) is not dt: raise AssertionError('Date of User creation must be a datetime')
        return created_on