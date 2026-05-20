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

# 日本語表示用辞書
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
    "Harry Kane": "ハリー・ケイン",
    "Lionel Messi": "リオネル・メッシ",
    "Cristiano Ronaldo": "クリスティアーノ・ロナウド"
}

df["ClubJP"] = df["current_club_name"].replace(club_jp)
df["ClubJP"] = df["ClubJP"].fillna(df["current_club_name"])

df["NameJP"] = df["name"].replace(player_jp)
df["NameJP"] = df["NameJP"].fillna(df["name"])

# 架空選手追加
new_player = pd.DataFrame([{
    "name": "Awaji Taku",
    "current_club_name": "Real Madrid",
    "position": "Attack",
    "date_of_birth": "2005-01-01",
    "market_value_in_eur": 999999999,
    "highest_market_value_in_eur": 999999999,
    "ClubJP": "レアル・マドリード",
    "NameJP": "淡路卓"
}])

df = pd.concat([df, new_player], ignore_index=True)

# 年齢計算
df["date_of_birth"] = pd.to_datetime(df["date_of_birth"], errors="coerce")
df = df.dropna(subset=["date_of_birth"])
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

# モード選択
mode = st.radio("選択", ["実在選手", "自分で入力"])

if mode == "実在選手":
    search_name = st.text_input("選手名を入力（日本語・英語どちらでもOK）")

    filtered_players = df[
        df["NameJP"].str.contains(search_name, case=False, na=False) |
        df["name"].str.contains(search_name, case=False, na=False)
    ]

    if len(filtered_players) == 0:
        st.warning("該当する選手がいません")
    else:
        display_names = filtered_players["NameJP"] + "（" + filtered_players["ClubJP"] + "）"
        player_display = st.selectbox("選手を選択", display_names)

        selected = filtered_players.iloc[display_names.tolist().index(player_display)]

        st.subheader("選手データ")
        st.dataframe(selected.to_frame().T)

        pred = model.predict([[selected["Age"], selected["PositionNum"]]])
        st.subheader("予測市場価値")
        st.write(f"{pred[0]:,.0f} €")

        # レーダーチャート
        labels = ["Age", "Current", "Highest", "Position"]

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
        ax.set_xticklabels(labels, fontsize=12)
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
