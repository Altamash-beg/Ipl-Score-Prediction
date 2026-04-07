from flask import Flask, request, jsonify
import pickle
import pandas as pd
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # allow React frontend

# Load trained pipeline
pipe = pickle.load(open("beg.pkl", "rb"))

# Load dataset for teams
df = pd.read_csv("deliveries.csv")
teams = sorted(df["batting_team"].unique())

@app.route("/teams", methods=["GET"])
def get_teams():
    """
    Returns a list of batting teams for dropdown
    """
    return jsonify({"teams": teams})

@app.route("/predict", methods=["POST"])
def predict():
    """
    Receives JSON with features and returns predicted inning score
    """
    try:
        data = request.json

        # React sends all derived features
        input_df = pd.DataFrame({
            "batting_team": [data["batting_team"]],
            "bowling_team": [data["bowling_team"]],
            "cum_runs": [float(data["cum_runs"])],
            "ball_left": [float(data["ball_left"])],
            "wickets_left": [float(data["wickets_left"])],
            "crr": [float(data["crr"])],
            "last5_runs": [float(data["last5_runs"])]
        })

        prediction = int(pipe.predict(input_df)[0])
        return jsonify({"prediction": prediction})

    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(debug=True)
