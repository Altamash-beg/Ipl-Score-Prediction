import streamlit as st
import pickle
import pandas as pd
import numpy as np

# --- PAGE CONFIG ---
st.set_page_config(page_title="IPL Pro-Predictor", page_icon="🏏", layout="wide")

# --- CUSTOM CSS ---
st.markdown("""
    <style>
    .main {
        background-color: #0e1117;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        background-color: #ff4b4b;
        color: white;
        font-weight: bold;
        border: none;
    }
    .stButton>button:hover {
        background-color: #ff3333;
        border: none;
        color: white;
    }
    .prediction-card {
        padding: 25px;
        border-radius: 12px;
        background-color: #1e2130;
        border-left: 5px solid #ff4b4b;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.3);
    }
    </style>
    """, unsafe_allow_html=True)

# Load model
try:
    pipe = pickle.load(open("beg.pkl", "rb"))
except FileNotFoundError:
    st.error("Model file 'beg.pkl' not found. Please ensure the model is trained and saved in the same directory.")

# Teams List
teams = ['Rajasthan Royals', 'Royal Challengers Bangalore', 'Sunrisers Hyderabad', 
         'Delhi Capitals', 'Chennai Super Kings', 'Gujarat Titans', 
         'Lucknow Super Giants', 'Kolkata Knight Riders', 'Punjab Kings', 'Mumbai Indians']

# --- SIDEBAR: MATCH SETTINGS ---
st.sidebar.image("https://upload.wikimedia.org/wikipedia/en/thumb/8/84/Indian_Premier_League_Official_Logo.svg/1200px-Indian_Premier_League_Official_Logo.svg.png", width=150)
st.sidebar.title("Match Settings")

batting_team = st.sidebar.selectbox("Batting Team", sorted(teams))
bowling_team = st.sidebar.selectbox("Bowling Team", sorted(teams), index=1)

# Ensure teams aren't the same
if batting_team == bowling_team:
    st.sidebar.warning("Batting and Bowling teams must be different!")

# --- MAIN UI ---
st.title("🏏 IPL AI Score Predictor")
st.markdown("---")

col1, col2, col3 = st.columns([1, 1, 1], gap="large")

with col1:
    st.subheader("Match Progress")
    overs = st.number_input("Overs Completed", min_value=5.0, max_value=19.5, value=10.0, step=0.1, help="Input as 10.3 for 10 overs and 3 balls")
    wickets_left = st.slider("Wickets Remaining", 0, 10, 8)

with col2:
    st.subheader("Current Statistics")
    current_score = st.number_input("Current Runs", min_value=0, value=80, step=1)
    last_five = st.number_input("Runs in Last 5 Overs", min_value=0, value=40, step=1)

with col3:
    st.subheader("Live Analytics")
    # Quick calculations for the display
    overs_int = int(overs)
    balls_fraction = round((overs - overs_int) * 10)
    balls_bowled = (overs_int * 6) + balls_fraction
    
    if balls_bowled > 0:
        crr = round(current_score / (balls_bowled / 6), 2)
        st.metric("Current Run Rate (CRR)", crr)
        st.progress(balls_bowled / 120)
        st.caption(f"{balls_bowled} balls bowled / 120 total")
    else:
        st.info("Input overs to see live stats.")

st.markdown("---")

# --- PREDICTION LOGIC ---
if st.button("🚀 PREDICT FINAL SCORE"):
    
    # Calculate features
    balls_left = 120 - balls_bowled
    crr = current_score / (balls_bowled / 6) if balls_bowled > 0 else 0
    
    # Phase logic
    if balls_bowled <= 36: phase = "Powerplay"
    elif balls_bowled <= 90: phase = "Middle"
    else: phase = "Death"

    # Input DataFrame construction
    input_df = pd.DataFrame({
        "batting_team": [batting_team],
        "bowling_team": [bowling_team],
        "phase": [phase],
        "cum_runs": [current_score],
        "balls_left": [balls_left],
        "wickets_left": [wickets_left],
        "crr": [crr],
        "last5_runs": [last_five]
    })

    # Model Prediction
    result = pipe.predict(input_df)
    predicted_score = int(result[0])
    
    # --- OUTPUT DISPLAY ---
    res_col1, res_col2 = st.columns(2)
    
    with res_col1:
        st.markdown(f"""
            <div class="prediction-card">
                <h3 style='color: #ff4b4b; margin-bottom: 10px;'>Predicted Final Total</h3>
                <h1 style='font-size: 85px; margin: 0; font-weight: 800;'>{predicted_score}</h1>
                <p style='color: #a3a8b4;'>Estimated range: {predicted_score - 5} to {predicted_score + 5}</p>
            </div>
        """, unsafe_allow_html=True)
        
    with res_col2:
        # Qualitative Insights
        momentum_text = "Standard"
        if last_five > 55: momentum_text = "🔥 Explosive"
        elif last_five < 30: momentum_text = "🐌 Struggling"
        
        st.metric("Momentum Status", momentum_text)
        
        projected_remainder = predicted_score - current_score
        st.write(f"**Analysis:**")
        st.write(f"- {batting_team} is projected to score **{projected_remainder}** more runs.")
        st.write(f"- Required Run Rate to reach this: **{round((projected_remainder*6)/balls_left, 2) if balls_left > 0 else 0}**")