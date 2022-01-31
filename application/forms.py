from flask_wtf import FlaskForm
from wtforms import FloatField, SubmitField, IntegerField, SelectField, StringField, PasswordField
from wtforms.validators import Length, InputRequired, ValidationError, NumberRange
from datetime import datetime as dt
 
class LoginForm(FlaskForm):
    username = StringField(
        "Username",
        validators=[InputRequired()]
    )

    password = PasswordField(
        "Password",
        validators=[InputRequired()]
    )

    submit = SubmitField("Login")

class RegisterForm(FlaskForm):
    username = StringField(
        "Username",
        validators=[InputRequired()]
    )

    password = PasswordField(
        "Password",
        validators=[InputRequired()]
    )

    submit = SubmitField("Register")

    def validate_username(self, username):
        if not username.data.isalpha():
            raise ValidationError('Username must contain only letters')
        elif ' ' in username.data:
            raise ValidationError('Username must not contain spaces')