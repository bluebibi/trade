from flask_wtf import FlaskForm
import wtforms as f
from wtforms.validators import DataRequired


class LoginForm(FlaskForm):
    email = f.StringField('이메일', validators=[DataRequired()])
    password = f.PasswordField('비밀번호', validators=[DataRequired()])
    display = ['email', 'password']


class UserForm(FlaskForm):
    email = f.StringField('이메일', validators=[DataRequired()])
    password = f.PasswordField('비밀번호', validators=[DataRequired()])
    name = f.StringField('이름', validators=[DataRequired()])

    display = ['email', 'password', 'name']