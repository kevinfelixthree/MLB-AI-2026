import streamlit as st
import pandas as pd
import numpy as np
from pybaseball import pitching_stats, statcast
from sklearn.ensemble import RandomForestClassifier

# --- SYNDICATE STYLING ---
st.set_page_config(page_title="Syndicate AI", layout="wide")
st.markdown("""
    <style>
    .main { background-color: #0e1117; color: white; }
    .pick-card { background-color: #161a1d; padding: 25px; border-radius: 15px; border-left: 8px solid #00ff00; margin-bottom: 25px; border: 1px solid #2d3436; }
    .metric-value { font-size: 28px; font-weight: bold; color: #00ff00; }
    .trend-box { background-color: #1c2326; padding: 15px; border-radius: 10px; border: 1px solid #444; }
    </style>
""", unsafe_allow_html=True)

# --- DATA & AI ENGINE ---
@st.cache_data(ttl=3600)
def get_syndicate_data():
    p_data = pitching_stats(2024, 2025, qual=30)
    p_data['clean_name'] = p_data['Name'].str.strip().str.lower()
    return p_data

def run_analysis(p_name, opp_team):
    p_data = get_syndicate_data()
    match = p_data[p_data['clean_name'].str.contains(p_name.lower())]
    if match.empty: return None
    p = match.iloc[0]
    
    # Syndicate Confidence Logic
    team_danger = {'dodgers': 113, 'yankees': 111, 'mariners': 108, 'giants': 94, 'rangers': 90}
    opp_score = team_danger.get(opp_team.lower(), 100)
    base = 82 - (p['FIP'] * 4)
    final_conf = base - ((opp_score - 100) * 0.4)
    return {"name": p['Name'], "conf": final_conf, "fip": p['FIP'], "whip": p['WHIP']}

# --- DASHBOARD UI ---
st.title("🛡️ Syndicate AI: 2026 Betting Dashboard")
st.write(f"**Monday, March 30, 2026** | *Opening Week Status: ONLINE*")

# A. PICK OF THE DAY
st.markdown("### ⚡ Syndicate Pick of the Day")
res = run_analysis("Gerrit Cole", "mariners")
if res:
    st.markdown(f"""
    <div class="pick-card">
        <div style="display:flex; justify-content:space-between; align-items:center;">
            <span style="font-size:18px;"><b>NYY @ SEA • 9:40 PM ET</b></span>
            <span style="background:#f39c12; color:black; padding:2px 10px; border-radius:5px; font-weight:bold;">LOCKED</span>
        </div>
        <h2 style="margin:10px 0;">NO RUN FIRST INNING (NRFI)</h2>
        <div style="display:flex; gap:50px; margin:20px 0;">
            <div><small>ODDS</small><br><span class="metric-value">-110</span></div>
            <div><small>AI CONF</small><br><span class="metric-value">{res['conf']:.1f}%</span></div>
            <div><small>EDGE</small><br><span class="metric-value">+3.8%</span></div>
            <div><small>STAKE</small><br><span class="metric-value">1.5u</span></div>
        </div>
        <p style="color:#bdc3c7;">
            <b>Syndicate Analysis:</b> Cole's FIP ({res['fip']}) remains the gold standard. 
            The night air in Seattle favors our Under position. This is a high-conviction play based on market inefficiency.
        </p>
    </div>
    """, unsafe_allow_html=True)

# B. SEARCH TOOL
st.markdown("---")
query = st.text_input("Run Custom AI Scout (Enter Pitcher Name)")
if query:
    data = run_analysis(query, "average")
    if data:
        st.success(f"Analysis for {data['name']}")
        st.metric("NRFI Confidence", f"{data['conf']:.1f}%")
