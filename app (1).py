import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import kagglehub
import os
from sklearn.linear_model import LinearRegression

st.title("⚽ サッカー選手 実市場価値予測AI")

# データ取得
path = kagglehub.dataset_download("davidcariboo/player-scores")
df = pd.read_csv(os.path.join(path, "players.csv"))

# 必要列
df = df[[
    "name",
    "current_club_name",
    "position",
    "date_of_birth",
    "market_value_in_eur",
    "highest_market_value_in_eur"
]].dropna()

# 日本語辞書
club_jp = {
    "Real Madrid": "レアル・マドリード",
    "FC Barcelona": "バルセロナ",
    "Manchester City": "マンチェスター・シティ",
    "Arsenal FC": "アーセナル",
    "Liverpool FC": "リヴァプール",
    "Paris Saint-Germain": "パリ・サンジェルマン",
    "Bayern Munich": "バイエルン"
}

player_jp = {
    "Kylian Mbappe": "キリアン・エムバペ",
    "Jude Bellingham": "ジュード・ベリンガム",
    "Erling Haaland": "アーリング・ハーランド",
    "Bukayo Saka": "ブカヨ・サカ",
    "Mohamed Salah": "モハメド・サラー",
    "Vinicius Junior": "ヴィニシウス・ジュニオール",
    "Kevin De Bruyne": "ケヴィン・デ・ブライネ",
    "Harry Kane": "ハリー・ケイン"
}

df["ClubJP"] = df["current_club_name"].replace(club_jp)
df["ClubJP"] = df["ClubJP"].fillna(df["current_club_name"])

df["NameJP"] = df["name"].replace(player_jp)
df["NameJP"] = df["NameJP"].fillna(df["name"])

# 年齢
df["date_of_birth"] = pd.to_datetime(df["date_of_birth"])
df["Age"] = 2026 - df["date_of_birth"].dt.year

# ポジション数値化
position_map = {
    "Goalkeeper": 1,
    "Defender": 2,
    "Midfield": 3,
    "Attack": 4
}
df["PositionNum"] = df["position"].map(position_map).fillna(3)

# 学習
X = df[["Age", "PositionNum"]]
y = df["market_value_in_eur"]

model = LinearRegression()
model.fit(X, y)

# モード
mode = st.radio("選択", ["実在選手", "自分で入力"])

if mode == "実在選手":
    club = st.selectbox("クラブ", sorted(df["ClubJP"].unique()))
    club_df = df[df["ClubJP"] == club]

    player = st.selectbox("選手", club_df["NameJP"])
    selected = club_df[club_df["NameJP"] == player].iloc[0]

    st.subheader("選手データ")
    st.dataframe(selected.to_frame().T)

    pred = model.predict([[selected["Age"], selected["PositionNum"]]])
    st.subheader("予測市場価値")
    st.write(f"{pred[0]:,.0f} €")

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
    ax.set_ylim(0, 100)

    st.subheader("選手能力レーダーチャート")
    st.pyplot(fig)

else:
    age = st.slider("年齢", 16, 40, 22)
    position = st.selectbox("ポジション", ["Goalkeeper", "Defender", "Midfield", "Attack"])
    pos_num = position_map[position]

    if st.button("予測"):
        pred = model.predict([[age, pos_num]])
        st.subheader("予測市場価値")
        st.write(f"{pred[0]:,.0f} €")
