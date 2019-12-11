import json
import requests


class PushSlack:
    def __init__(self, SLACK_WEBHOOK_URL_1, SLACK_WEBHOOK_URL_2):
        if SLACK_WEBHOOK_URL_1 and SLACK_WEBHOOK_URL_1.startswith("http"):
            self.webhook_url_1 = SLACK_WEBHOOK_URL_1
        else:
            self.webhook_url_1 = None

        if SLACK_WEBHOOK_URL_2 and SLACK_WEBHOOK_URL_2.startswith("http"):
            self.webhook_url_2 = SLACK_WEBHOOK_URL_2
        else:
            self.webhook_url_2 = None

    def send_message(self, username=None, message=None):
        slack_data = {'text': message}

        if self.webhook_url_1:
            requests.post(
                self.webhook_url_1,
                data=json.dumps(slack_data),
                headers={'Content-Type': 'application/json'}
            )

        if self.webhook_url_2:
            requests.post(
                self.webhook_url_2,
                data=json.dumps(slack_data),
                headers={'Content-Type': 'application/json'}
            )

    def send_message_to_manager(self, username=None, message=None):
        slack_data = {'text': message}

        if self.webhook_url_1:
            requests.post(
                self.webhook_url_1,
                data=json.dumps(slack_data),
                headers={'Content-Type': 'application/json'}
            )
