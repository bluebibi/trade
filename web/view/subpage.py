from flask import Blueprint, render_template

subpage_blueprint = Blueprint('subpage', __name__)


@subpage_blueprint.route('/models')
def _models():
    return render_template("subpage/models.html", menu="models")


@subpage_blueprint.route('/data_collects')
def news_main():
    return render_template("subpage/data_collects.html", menu="data_collects")
