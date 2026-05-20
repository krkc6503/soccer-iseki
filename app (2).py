import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import kagglehub
import os
from sklearn.linear_model import LinearRegression
import matplotlib.font_manager as fm

st.title("⚽ サッカー選手 実市場価値予測AI")

# 日本語フォント設定
font_path = "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc"
jp_font = fm.FontProperties(fname=font_path)

# データ取得
path = kagglehub.dataset_download("davidcariboo/player-scores")
df = pd.read_csv(os.path.join(path, "players.csv"))

df = df[[
    "name",
    "current_club_name",
    "position",
    "date_of_birth",
    "market_value_in_eur",
    "highest_market_value_in_eur"
]].dropna()

# 年齢
df["date_of_birth"] = pd.to_datetime(df["date_of_birth"])
df["Age"] = 2026 - df["date_of_birth"].dt.year

position_map = {
    "Goalkeeper": 1,
    "Defender": 2,
    "Midfield": 3,
    "Attack": 4
}
df["PositionNum"] = df["position"].map(position_map).fillna(3)

X = df[["Age", "PositionNum"]]
y = df["market_value_in_eur"]

model = LinearRegression()
model.fit(X, y)

club = st.selectbox("クラブ", sorted(df["current_club_name"].unique()))
club_df = df[df["current_club_name"] == club]

player = st.selectbox("選手", club_df["name"])
selected = club_df[club_df["name"] == player].iloc[0]

pred = model.predict([[selected["Age"], selected["PositionNum"]]])
st.write(f"予測市場価値: {pred[0]:,.0f} €")

# レーダーチャート
labels = ["年齢", "現在価値", "最高価値", "ポジション"]

values = [
    selected["Age"] / df["Age"].max() * 100,
    selected["market_value_in_eur"] / df["market_value_in_eur"].max() * 100,
    selected["highest_market_value_in_eur"] / df["highest_market_value_in_eur"].max() * 100,
    selected["PositionNum"] / 4 * 100
]

values += values[:1]
angles = np.linspace(0, 2*np.pi, len(labels), endpoint=False).tolist()
angles += angles[:1]

fig, ax = plt.subplots(figsize=(6,6), subplot_kw=dict(polar=True))
ax.plot(angles, values, linewidth=2)
ax.fill(angles, values, alpha=0.25)

ax.set_xticks(angles[:-1])
ax.set_xticklabels(labels)
for label in ax.get_xticklabels():
    label.set_fontproperties(jp_font)

ax.set_ylim(0, 100)

st.pyplot(fig)
