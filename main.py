import streamlit as st
import pickle
import pandas as pd

# Load trained pipeline
pipe = pickle.load(open('beg.pkl', 'rb'))

# Load dataset to fetch teams
df = pd.read_csv("deliveries.csv")
teams = sorted(df["batting_team"].unique())

# Streamlit UI
st.set_page_config(page_title="Cricket Innings Predictor", page_icon="🏏")
st.title("🏏 Cricket Innings Score Predictor")

# Team selection
col1, col2 = st.columns(2)
with col1:
    batting_team = st.selectbox("Select Batting Team", teams)
with col2:
    bowling_team = st.selectbox("Select Bowling Team", [t for t in teams if t != batting_team])

# Inning selection
innings = st.selectbox("Select Innings", [1, 2])

# Match situation inputs
st.subheader("📊 Current Match Situation")

col3, col4, col5 = st.columns(3)
with col3:
    cum_runs = st.number_input("Cumulative Runs", min_value=0, step=1)
with col4:
    overs = st.number_input("Overs Completed", min_value=0.0, max_value=20.0, step=0.1)
with col5:
    wickets = st.number_input("Wickets Fallen", min_value=0, max_value=10, step=1)

last5_runs = st.number_input("Runs in Last 5 Overs", min_value=0, step=1)

# Prediction
if st.button("🔮 Predict Innings Score"):
    try:
        # Derived features
        balls_bowled = int(overs * 6)
        ball_left = 120 - balls_bowled
        wickets_left = 10 - wickets
        crr = cum_runs / overs if overs > 0 else 0

        # Build input dataframe
        input_df = pd.DataFrame({
            "batting_team": [batting_team],
            "bowling_team": [bowling_team],
            "innings": [innings],
            "cum_runs": [cum_runs],
            "ball_left": [ball_left],
            "wickets_left": [wickets_left],
            "crr": [crr],
            "last5_runs": [last5_runs]
        })

        # Predict
        predicted_innings_score = pipe.predict(input_df)[0]
        st.success(f"🏆 Predicted Innings Score: **{int(predicted_innings_score)} runs**")

    except Exception as e:
        st.error(f"Error: {e}")
