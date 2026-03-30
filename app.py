import streamlit as st
import pandas as pd
from pybaseball import pitching_stats, statcast
from sklearn.ensemble import RandomForestClassifier
import warnings

warnings.filterwarnings('ignore')

# --- CONFIGURATION & UI ---
st.set_page_config(page_title="MLB AI Command Center", layout="wide")
st.title("⚾ MLB AI Prediction Command Center (2026)")

# --- STEP 1: DATA ENGINE (CACHED) ---
@st.cache_data(ttl=3600) # Refreshes every hour
def load_mlb_data():
    # Pulling 2020-2025 data as our historical foundation
    p_data = pitching_stats(2020, 2025, qual=30)
    p_data['clean_name'] = p_data['Name'].str.strip().str.lower()
    
    # Pulling a sample of Statcast for the AI "Answer Key"
    s_data = statcast(start_dt='2025-05-01', end_dt='2025-05-15')
    return p_data, s_data

# --- STEP 2: BRAIN ENGINE ---
@st.cache_resource
def train_ai(p_data, s_data):
    def flip_name(name):
        try:
            parts = name.split(', ')
            return f"{parts[1]} {parts[0]}".strip().lower()
        except: return str(name).strip().lower()

    s_data['pitcher_name'] = s_data['player_name'].apply(flip_name)
    merged = pd.merge(s_data, p_data, left_on='pitcher_name', right_on='clean_name')
    nrfi_df = merged[merged['inning'] == 1].copy()
    nrfi_df['run_scored'] = (nrfi_df['post_bat_score'] > nrfi_df['bat_score']).astype(int)
    
    features = ['release_speed', 'FIP', 'WHIP', 'K/9', 'BB/9']
    X = nrfi_df[features].dropna()
    y = nrfi_df.loc[X.index, 'run_scored']
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model

# --- LOAD DATA & BRAIN ---
with st.spinner("Waking up the AI..."):
    pitcher_data, statcast_data = load_mlb_data()
    nrfi_model = train_ai(pitcher_data, statcast_data)

# --- STEP 3: MATCHUP LOGIC ---
team_iq = {'dodgers': 113, 'yankees': 111, 'mariners': 108, 'mets': 105, 'braves': 102, 'rangers': 90, 'nationals': 87} # (Add more here!)

def get_analysis(p_name, opp_team):
    match = pitcher_data[pitcher_data['clean_name'].str.contains(p_name.lower())]
    if match.empty: return None
    p = match.iloc[0]
    
    # AI Prediction
    test_case = pd.DataFrame([[88.0, p['FIP'], p['WHIP'], p['K/9'], p['BB/9']]], columns=['release_speed', 'FIP', 'WHIP', 'K/9', 'BB/9'])
    base_prob = nrfi_model.predict_proba(test_case)[0][0] * 100
    
    # Adjust for Team Strength
    offense = team_iq.get(opp_team.lower(), 100)
    final_nrfi = base_prob + ((100 - offense) * 0.5)
    
    # Team Total Prediction (League avg ~4.45)
    projected_runs = 4.45 * (p['FIP'] / 4.13) * (offense / 100)
    
    return {"name": p['Name'], "nrfi": final_nrfi, "runs": projected_runs, "fip": p['FIP']}

# --- STEP 4: THE INTERFACE ---
query = st.text_input("Search (e.g. 'Gerrit Cole vs Nationals' or 'Logan Webb')")

if query:
    if "vs" in query:
        parts = query.split("vs")
        p_name = parts[0].strip()
        t_name = parts[1].strip()
        
        result = get_analysis(p_name, t_name)
        if result:
            col1, col2 = st.columns(2)
            col1.metric("NRFI Confidence", f"{result['nrfi']:.1f}%")
            col2.metric(f"{t_name.title()} Projected Runs", f"{result['runs']:.2f}")
            
            if result['nrfi'] > 70: st.success("🔥 BEST BET: NRFI")
            elif result['nrfi'] < 55: st.error("❌ AVOID: High Run Risk")
    else:
        # Single Player Scout
        result = get_analysis(query, "average team")
        if result:
            st.subheader(f"Pitcher Scout: {result['name']}")
            st.write(f"Standard NRFI Probability: **{result['nrfi']:.1f}%**")
            st.write(f"Season FIP: {result['fip']}")