import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import kagglehub
import os
from sklearn.linear_model import LinearRegression

st.title("⚽ サッカー選手 実市場価値予測AI")

path = kagglehub.dataset_download("davidcariboo/player-scores")
df = pd.read_csv(os.path.join(path, "players.csv"))

df = df[["name", "current_club_name", "position", "date_of_birth", "market_value_in_eur", "highest_market_value_in_eur"]].dropna()

df["date_of_birth"] = pd.to_datetime(df["date_of_birth"], errors="coerce")
df = df.dropna(subset=["date_of_birth"])
df["Age"] = 2026 - df["date_of_birth"].dt.year

player_jp = {
    "Kylian Mbappe": "エムバペ", "Jude Bellingham": "ベリンガム", "Erling Haaland": "ハーランド",
    "Bukayo Saka": "サカ", "Mohamed Salah": "サラー", "Vinicius Junior": "ヴィニシウス",
    "Kevin De Bruyne": "デブライネ", "Harry Kane": "ケイン", "Lionel Messi": "メッシ",
    "Cristiano Ronaldo": "ロナウド", "Neymar": "ネイマール", "Son Heung-min": "ソン"
}
club_jp = {
    "Real Madrid": "レアル・マドリード", "FC Barcelona": "バルセロナ", "Manchester City": "マンチェスター・シティ",
    "Arsenal FC": "アーセナル", "Liverpool FC": "リヴァプール", "Paris Saint-Germain": "パリ・サンジェルマン"
}

df["NameJP"] = df["name"].replace(player_jp).fillna(df["name"])
df["ClubJP"] = df["current_club_name"].replace(club_jp).fillna(df["current_club_name"])

new_player = pd.DataFrame([{
    "name": "Awaji Taku",
    "current_club_name": "Real Madrid",
    "position": "Attack",
    "date_of_birth": pd.Timestamp("2008-01-01"),
    "market_value_in_eur": 999999999,
    "highest_market_value_in_eur": 999999999,
    "NameJP": "淡路卓",
    "ClubJP": "レアル・マドリード",
    "Age": 18
}])

position_map = {"Goalkeeper": 1, "Defender": 2, "Midfield": 3, "Attack": 4}
df["PositionNum"] = df["position"].map(position_map).fillna(3)

X = df[["Age", "PositionNum"]]
y = df["market_value_in_eur"]
model = LinearRegression()
model.fit(X, y)

mode = st.radio("選択", ["実在選手", "自分で入力"])

if mode == "実在選手":
    show_awaji = st.toggle("フェンシング選手をオンにする")
    if show_awaji:
        work_df = pd.concat([df, new_player], ignore_index=True)
    else:
        work_df = df.copy()

    search_name = st.text_input("選手名を入力（日本語・英語OK）")
    filtered_players = work_df[
        work_df["NameJP"].str.contains(search_name, case=False, na=False) |
        work_df["name"].str.contains(search_name, case=False, na=False)
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

        labels = ["Age", "Current Value", "Highest Value", "Position"]
        values = [
            selected["Age"] / work_df["Age"].max() * 100,
            selected["market_value_in_eur"] / work_df["market_value_in_eur"].max() * 100,
            selected["highest_market_value_in_eur"] / work_df["highest_market_value_in_eur"].max() * 100,
            selected["PositionNum"] / 4 * 100,
        ]
        values += values[:1]
        angles = np.linspace(0, 2*np.pi, len(labels), endpoint=False).tolist()
        angles += angles[:1]

        fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
        ax.plot(angles, values, linewidth=2)
        ax.fill(angles, values, alpha=0.25)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(labels, fontsize=11)
        ax.set_ylim(0, 100)
        st.subheader("Player Radar Chart")
        st.pyplot(fig)

        st.subheader("Top 10 Market Value")
        top10 = work_df.sort_values("market_value_in_eur", ascending=False).head(10)
        fig2, ax2 = plt.subplots(figsize=(10, 5))
        ax2.barh(top10["name"], top10["market_value_in_eur"] / 1000000)
        ax2.invert_yaxis()
        ax2.set_xlabel("Market Value (Million €)")
        ax2.set_ylabel("Player")
        st.pyplot(fig2)

else:
    age = st.slider("年齢", 16, 40, 22)
    position = st.selectbox("ポジション", ["Goalkeeper", "Defender", "Midfield", "Attack"])
    pos_num = position_map[position]
    if st.button("予測"):
        pred = model.predict([[age, pos_num]])
        st.subheader("予測市場価値")
        st.write(f"{pred[0]:,.0f} €")
