from flask import Flask
from resource_check import resource

application = Flask(__name__)
application.register_blueprint(resource)


@application.route("/")
def say_hello(username="World"):
    return '<p>Hello {0}!</p>'.format(username)


if __name__ == "__main__":
    application.debug = True
    application.run()