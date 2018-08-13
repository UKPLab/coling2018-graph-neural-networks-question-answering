from flask import Flask
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

from questionanswering.qaserver.server import qaserver

app.register_blueprint(qaserver, url_prefix="/question-answering")
