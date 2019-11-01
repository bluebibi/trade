from flask import Blueprint, redirect, session, render_template, request, flash
from db.database import db, User
from forms import LoginForm
from flask_login import login_required, login_user, logout_user, current_user


auth_blueprint = Blueprint('auth', __name__)


@auth_blueprint.route('/mypage')
@login_required
def _user():
    q = db.session.query(User).filter(User.email == session['user']['email'])
    user = q.first()
    return render_template("subpage/mypage.html", user=user)


@auth_blueprint.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect('/auth/check_registration')

    form = LoginForm()

    if request.method == 'POST':
        if form.validate_on_submit():
            email = form.data['email']
            password = form.data['password']

            q = db.session.query(User).filter(User.email == email)
            user = q.first()

            if user is not None:
                if user.authenticate(password):
                    login_user(user)
                    session['user'] = user.to_json()
                    return redirect('/auth/check_registration')
                else:
                    flash('Invalid username/password combination')
                    return redirect('/auth/login')

    return render_template('login.html', form=form)


@auth_blueprint.route("/logout")
@login_required
def logout():
    logout_user()
    return redirect('/')