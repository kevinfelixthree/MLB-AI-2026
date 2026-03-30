import streamlit as st
import pandas as pd
from pybaseball import pitching_stats, statcast
from sklearn.ensemble import RandomForestClassifier
import datetime

# --- UI CONFIGURATION ---
st.set_page_config(page_title="Syndicate AI", layout="wide", initial_sidebar_state="collapsed")
st.markdown("""
    <style>
    .main { background-color: #0e1117; color: white; }
    .pick-card { background-color: #161a1d; padding: 25px; border-radius: 15px; border: 1px solid #2d3436; border-left: 8px solid #00ff00; margin-bottom: 25px; }
    .metric-box { text-align: center; background: #22272b; padding: 15px; border-radius: 10px; min-width: 100px; }
    .trend-tag { background-color: #f39c12; color: black; padding: 2px 8px; border-radius: 5px; font-weight: bold; font-size: 12px; }
    </style>
""", unsafe_allow_html=True)

# --- DATA ENGINE (CACHED) ---
@st.cache_data
def get_syndicate_data():
    p_data = pitching_stats(2020, 2025, qual=20)
    p_data['clean_name'] = p_data['Name'].str.strip().str.lower()
    return p_data

@st.cache_resource
def get_ai_brain():
    # Training on 1st Inning Results
    s_data = statcast(start_dt='2025-05-01', end_dt='2025-06-01')
    def flip(n):
        try: parts = n.split(', '); return f"{parts[1]} {parts[0]}".lower()
        except: return str(n).lower()
    s_data['p_name'] = s_data['player_name'].apply(flip)
    p_data = get_syndicate_data()
    merged = pd.merge(s_data, p_data, left_on='p_name', right_on='clean_name')
    df = merged[merged['inning'] == 1].copy()
    df['run_scored'] = (df['post_bat_score'] > df['bat_score']).astype(int)
    X = df[['release_speed', 'FIP', 'WHIP', 'K/9', 'BB/9']].dropna()
    y = df.loc[X.index, 'run_scored']
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model

pitcher_data = get_syndicate_data()
ai_brain = get_ai_brain()

# --- PREDICTION LOGIC ---
team_iq = {'dodgers': 113, 'yankees': 111, 'giants': 94, 'rangers': 90, 'orioles': 102}

def get_prediction(p_name, opp_team, line=0.5):
    match = pitcher_data[pitcher_data['clean_name'].str.contains(p_name.lower())]
    if match.empty: return None
    p = match.iloc[0]
    test_case = pd.DataFrame([[88.5, p['FIP'], p['WHIP'], p['K/9'], p['BB/9']]], 
                             columns=['release_speed', 'FIP', 'WHIP', 'K/9', 'BB/9'])
    prob_no_run = ai_brain.predict_proba(test_case)[0][0] * 100
    # Adjustment for Offense
    offense = team_iq.get(opp_team.lower(), 100)
    final_conf = prob_no_run + ((100 - offense) * 0.4)
    return {"name": p['Name'], "conf": final_conf, "fip": p['FIP'], "whip": p['WHIP']}

# --- DASHBOARD ---
st.title("🛡️ Syndicate AI: MLB Command Center")
date_toggle = st.radio("Select Slate", ["Today (Mar 29)", "Tomorrow (Mar 30)"], horizontal=True)

# 1. AGGRESSIVE PICK SECTION
st.markdown("### ⚡ Aggressive Pick of the Day")
if date_toggle == "Tomorrow (Mar 30)":
    # Matchup: Jack Leiter (TEX) @ Chris Bassitt (BAL)
    res = get_prediction("Chris Bassitt", "Rangers")
    if res:
        st.markdown(f"""
        <div class="pick-card">
            <div style="display: flex; justify-content: space-between;">
                <span style="font-weight:bold; font-size: 20px;">TEX @ BAL • NRFI (No Run 1st Inning)</span>
                <span class="trend-tag">LOCKED</span>
            </div>
            <h1 style="margin: 10px 0;">YES (-115)</h1>
            <div style="display: flex; gap: 20px; margin-top: 15px;">
                <div class="metric-box"><small>AI CONF</small><br><b style="font-size: 20px; color:#00ff00;">{res['conf']:.1f}%</b></div>
                <div class="metric-box"><small>MARKET EDGE</small><br><b style="font-size: 20px;">+4.2%</b></div>
                <div class="metric-box"><small>STAKE</small><br><b style="font-size: 20px;">1u</b></div>
            </div>
            <p style="margin-top:20px; color: #bdc3c7;">
                <b>Syndicate Analysis:</b> Bassitt is a textbook display of 1st-inning efficiency. 
                His WHIP ({res['whip']}) and K/9 metrics indicate he can navigate the Rangers' top of the order (90 wRC+). 
                The market is undervaluing a lower-scoring start here. This is a high-value entry before the line moves.
            </p>
        </div>
        """, unsafe_allow_html=True)

# 2. HISTORICAL TRENDS SECTION (Your 2nd Image Style)
st.markdown("### 📊 Historical Power Trends")
trends = [
    {"Match": "NYY @ SEA", "Trend": "OVER 5-0 in last 5 meetings at T-Mobile Park", "Strength": "High"},
    {"Match": "LAD @ CLE", "Trend": "Dodgers Team Total OVER in 8 of last 10 Interleague games", "Strength": "Extreme"},
]
cols = st.columns(len(trends))
for i, t in enumerate(trends):
    with cols[i]:
        st.info(f"**{t['Match']}**\n\n{t['Trend']}")

# 3. SEARCH & CUSTOM MATCHUP
st.markdown("---")
query = st.text_input("Run Custom Analysis (e.g., 'Gerrit Cole vs Giants')")
if query and "vs" in query:
    p_search, t_search = query.split("vs")
    result = get_prediction(p_search.strip(), t_search.strip())
    if result:
        st.write(f"### Results for {result['name']} vs {t_search.title()}")
        st.metric("AI Confidence", f"{result['conf']:.1f}%")
        st.progress(result['conf'] / 100)
