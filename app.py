from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)
model = pickle.load(open("data1.pkl", "rb"))


@app.route("/")
def home():
    return render_template('index.html')


@app.route("/index", methods=["POST", "GET"])
def pre():

    if request.method == "GET":
        sl = request.args.get('se_len')
        sw = request.args.get('se_wid')
        pl = request.args.get('pe_len')
        pw = request.args.get('pe_wid')

        # load saved model
        x = np.array([[sl, sw, pl, pw]])
        pred = model.predict(x)
        str = ''.join(pred)
        return render_template("index.html", data=str)
    else:
        return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True)
