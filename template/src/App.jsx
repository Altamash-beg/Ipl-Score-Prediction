import React, { useState, useEffect } from "react";
import "./App.css";

function App() {
  const [teams, setTeams] = useState([]);
  const [battingTeam, setBattingTeam] = useState("");
  const [bowlingTeam, setBowlingTeam] = useState("");
  const [cumRuns, setCumRuns] = useState("");
  const [ballLeft, setBallLeft] = useState("");
  const [wicketsLeft, setWicketsLeft] = useState("");
  const [crr, setCrr] = useState("");
  const [last5Runs, setLast5Runs] = useState("");
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    fetch("http://localhost:5000/teams")
      .then((res) => res.json())
      .then((data) => setTeams(data.teams))
      .catch((err) => console.error(err));
  }, []);

  const handlePredict = async () => {
    if (!battingTeam || !bowlingTeam) {
      alert("Please select both teams!");
      return;
    }

    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const res = await fetch("http://localhost:5000/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          batting_team: battingTeam,
          bowling_team: bowlingTeam,
          cum_runs: parseFloat(cumRuns),
          ball_left: parseFloat(ballLeft),
          wickets_left: parseFloat(wicketsLeft),
          crr: parseFloat(crr),
          last5_runs: parseFloat(last5Runs),
        }),
      });

      if (!res.ok) throw new Error("Prediction failed");
      const data = await res.json();
      setResult(data.prediction);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="container">
      <h1 className="title">🏏 Cricket Innings Score Predictor</h1>

      <div className="row">
        <div className="col">
          <label>Batting Team</label>
          <select
            value={battingTeam}
            onChange={(e) => setBattingTeam(e.target.value)}
          >
            <option value="">Select Batting Team</option>
            {teams.map((team, idx) => (
              <option key={idx} value={team}>
                Batting Team: {team}
              </option>
            ))}
          </select>
        </div>

        <div className="col">
          <label>Bowling Team</label>
          <select
            value={bowlingTeam}
            onChange={(e) => setBowlingTeam(e.target.value)}
          >
            <option value="">Select Bowling Team</option>
            {teams
              .filter((team) => team !== battingTeam)
              .map((team, idx) => (
                <option key={idx} value={team}>
                  Bowling Team: {team}
                </option>
              ))}
          </select>
        </div>
      </div>

      <div className="row">
        <div className="col">
          <label>Cumulative Runs</label>
          <input
            type="number"
            value={cumRuns}
            onChange={(e) => setCumRuns(e.target.value)}
          />
        </div>

        <div className="col">
          <label>Balls Left</label>
          <input
            type="number"
            value={ballLeft}
            onChange={(e) => setBallLeft(e.target.value)}
          />
        </div>

        <div className="col">
          <label>Wickets Left</label>
          <input
            type="number"
            value={wicketsLeft}
            onChange={(e) => setWicketsLeft(e.target.value)}
          />
        </div>
      </div>

      <div className="row">
        <div className="col">
          <label>Current Run Rate (CRR)</label>
          <input
            type="number"
            step="0.1"
            value={crr}
            onChange={(e) => setCrr(e.target.value)}
          />
        </div>

        <div className="col">
          <label>Runs in Last 5 Overs</label>
          <input
            type="number"
            value={last5Runs}
            onChange={(e) => setLast5Runs(e.target.value)}
          />
        </div>
      </div>

      <button onClick={handlePredict} disabled={loading}>
        {loading ? "Predicting..." : "🔮 Predict Innings Score"}
      </button>

      {result !== null && (
        <div className="result">
          🏆 Predicted Innings Score: <strong>{result.toFixed(0)}</strong>
        </div>
      )}

      {error && <div className="error">Error: {error}</div>}

      <footer>
        Ensure the Flask backend is running on{" "}
        <code>http://localhost:5000</code>
      </footer>
    </div>
  );
}

export default App;
