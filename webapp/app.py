"""Helllo World!

Python program using Flask for Forest Fire Predictions
GUI using the flask module

"""

# import Flask Library

import traceback
from flask import Flask, render_template, request
import math


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


app = Flask(__name__)

# you might want to just straightaway remove this function+decorator
@app.route('/')
def homepage():
    return render_template('homepage.html')
    # return render_template('index.html')
    # return '<h1>Hi, Welcome to this session</h1>'


@app.route('/forestfirepreds')
def index():
    return render_template('index.html')
    # return '<h1>Hi, Welcome to this session</h1>'


def monthToNum(shortMonth):

    return {
        'jan': 1,
        'feb': 2,
        'mar': 3,
        'apr': 4,
        'may': 5,
        'jun': 6,
        'jul': 7,
        'aug': 8,
        'sep': 9,
        'oct': 10,
        'nov': 11,
        'dec': 12
    }[shortMonth]


@app.route('/forestfirepredssend', methods=['POST'])
def send():
    if request.method == 'POST':
        x = request.form['x']
        y = request.form['y']
        month = request.form['month']
        day = request.form['day']
        FFMC = request.form['FFMC']
        DMC = request.form['DMC']
        DC = request.form['DC']
        ISI = request.form['ISI']
        temp = request.form['temp']
        RH = request.form['RH']
        wind = request.form['wind']
        rain = request.form['rain']

        # perform appropriate error handling yourself
        try:
            # here you can load the model using numpy loading and saving, or pickle or h5 file handling
            res = float(x)*float(y)*monthToNum(month)*float(day) * float(FFMC) * \
                float(DMC)*float(DMC)*float(ISI) * float(temp) * \
                float(RH)*float(wind)*float(rain)
            return render_template('index.html', results="The percentage of the Forest Fire occuring is : "+str(round(100*sigmoid(res), 3))+"%")
        except Exception as e:
            traceback.print_exc()
            err_message = "An internal error occurred\nPlease try again\nError Message : " + \
                "Full Traceback:" + traceback.format_exc()
            return render_template('index.html', results=err_message)


if __name__ == "__main__":
    # import sys
    # print(sys.version)
    # 3.7.6 (default, Jan  8 2020, 20:23:39) [MSC v.1916 64 bit (AMD64)]
    app.run(debug=True)
    # maybe it runs now!
