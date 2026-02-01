# API used: https://steamspy.com/api.php
# This script extracts data for all Steam games, ranks them by average playtime, and saves the result in a txt file

import requests
import pathlib


url = "https://steamspy.com/api.php?request=all"
response = requests.get(url)
data = response.json()  


game_data = []
for appid, game in data.items():
    try:
        name = game.get("name", "Unknown")
        owners = game.get("owners", "Unknown")
        avg_playtime = int(game.get("average_forever", 0))
        if avg_playtime > 0:
            game_data.append((name, owners, avg_playtime))
    except Exception: 
        continue

# Rank the games by average playtime 
game_data_sorted = sorted(game_data, key=lambda x: x[2], reverse=True)

# Save only the top 100 games to CSV (Rank, Name, Average_Playtime)
output_file = pathlib.Path.cwd() / "steam_top100_summary.csv"
with output_file.open("w", encoding="utf-8") as f:
    f.write("Rank,Name,Average_Playtime(min)\n")
    for idx, game in enumerate(game_data_sorted[:100], start=1):
        f.write(f"{idx},{game[0]},{game[2]}\n")

# Save the top 100 games into a txt file
txt_output = pathlib.Path.cwd() / "steam_top100_summary.txt"
with txt_output.open("w", encoding="utf-8") as f:
    f.write("Top 100 Steam Games by Average Playtime\n")
    f.write("----------------------------------------\n")
    for idx, game in enumerate(game_data_sorted[:100], start=1):
        f.write(f"{idx}. {game[0]}: {game[2]} minutes\n")

print("Top 100 games by average playtime written to steam_top100_summary.csv and steam_top100_summary.txt")


