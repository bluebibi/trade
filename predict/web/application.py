import requests
from flask import Flask, url_for, render_template, request, make_response
import logging
logging.basicConfig(filename='test.log', level=logging.DEBUG)


# print a nice greeting.
def say_hello(username = "World"):
    return '<p>Hello %s!</p>\n' % username


# some bits of text for the page.
header_text = '''
    <html>\n<head> <title>EB Flask Test</title> </head>\n<body>
    '''

instructions = '''
    <p><em>Hint</em>: This is a RESTful web service! Append a username
    to the URL (for example: <code>/Thelonious</code>) to say hello to
    someone specific.</p>\n
    '''

home_link = '<p><a href="/">Back</a></p>\n'
footer_text = '</body>\n</html>'

# EB looks for an 'application' callable by default.
application = Flask(__name__)

# add a rule for the index page.
# application.add_url_rule(
#     '/',
#     'index',
#     (lambda: header_text + say_hello() + instructions + footer_text)
# )

# add a rule when the page is accessed with a name appended to the site
# URL.
# application.add_url_rule(
#     '/<username>',
#     'hello',
#     (lambda username: header_text + say_hello(username) + home_link + footer_text)
# )


@application.route("/message/<int:message_id>")
def get_message(message_id):
    print(type(message_id))
    return("message id: {0}".format(message_id))


@application.route("/hello/")
def hello():
    response = requests.get("http://www.google.co.kr")
    return response.text


@application.route('/hello/<user>')
def hello_name(user):
   return render_template('index.html', name=user, data=20)


@application.route("/profile/<username>")
def get_profile(username):
    return "profile: " + username

@application.route('/hello_loop')
def hello_loop():
    value_list = ['list1', 'list2', 'list3']
    value = 50
    return render_template('index.html', values=value_list, data=value)


@application.errorhandler(404)
def page_not_found(error):
    application.logger.error(error)
    return "<h1>404 Error</h1>", 404


@application.before_request
def before_request():
    print("!!!!!")


@application.after_request
def after_request(response):
    print("?????")
    return response


@application.teardown_request
def teardown_request(response):
    print("-----------")
    return response


@application.teardown_appcontext
def teardown_appcontext(exception):
    print("#################")


@application.route('/get')
def get_cookie():
    username = request.cookies.get('username')
    return render_template('variable.html', name=username)


@application.route('/add/<user>')
def add_cookie(user):
    resp = make_response(render_template('simple.html'))
    resp.set_cookie('username', user)
    return resp


# run the app.
if __name__ == "__main__":
    # Setting debug to True enables debug output. This line should be
    # removed before deploying a production app.
    # print(application.jinja_env.globals)

    application.debug = True
    application.run()



    # with application.test_request_context():
    #     print(url_for('hello'))
    #     print(url_for('get_profile', username='flash'))