from os import name
from flask import Flask, jsonify, request, render_template
from bot2 import predict_class, getResponse, intents

app = Flask(__name__, template_folder='templates', static_folder='static')


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/chat')
def index():
    return render_template('index.html')


@app.route('/send', methods=['POST'])
def post():
    user_input = request.form['user_input']
    print(user_input)
    prediction = predict_class(user_input)
    res = getResponse(prediction, intents)
    print(res)
    return jsonify({
        'status': True,
        'bot_response': '{}'.format(res)
    })


app.run()
