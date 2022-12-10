from flask import Flask, render_template, request, jsonify
from collections import Counter
import pandas as pd

app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html")

@app.route("/state/<string:name>")
def state(name):
    df = pd.read_csv('incd.csv')
    return jsonify(State=name, Age_Adjusted_Incidence_Rate = float(df[df['State'] == name]['Age-Adjusted Incidence Rate']))

@app.route("/info", methods=["GET"])
def info():
    name = request.args.get("state", None)
    df = pd.read_csv('incd.csv')
    if name in df['State'].value_counts().keys():
        return render_template("analyze.html", analysis=state(name).get_data(as_text=True), name=name)
    else: 
        return render_template("error.html")

if __name__ == "__main__":
    app.run(debug=True)
