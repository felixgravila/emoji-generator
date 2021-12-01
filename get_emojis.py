#%%

import requests
from bs4 import BeautifulSoup
import base64
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

emojis_url = "https://unicode.org/emoji/charts/full-emoji-list.html"
to_use_emoji_sources = ["Appl", "Goog", "FB", "Wind", "Twtr", "Sams"]
base_data_dir = "emojis"

# %%

emojilist_req = requests.get(emojis_url)
soup = BeautifulSoup(emojilist_req.text, "html.parser")
emoji_rows = soup.find("body").find("div", {"class": "main"}).find("table").find_all("tr")

# %%

if not os.path.isdir(base_data_dir):
    os.makedirs(base_data_dir)

for s in to_use_emoji_sources:
    subpath = os.path.join(base_data_dir, s)
    if not os.path.isdir(subpath):
        os.makedirs(subpath)

idx_source_map = {}

for i,c in enumerate(emoji_rows[2].find_all("th")):
    if c.text in to_use_emoji_sources:
        idx_source_map[i] = c.text

#%%

def get_image_for_cell(cell):
    icon = cell.find("img")
    if icon is not None:
        b64_data = icon.attrs['src']
        base64_str = b64_data.split(",")[1]
        base64_bytes = base64_str.encode("UTF-8")
        return base64.decodebytes(base64_bytes)
    else:
        return None


for row in tqdm(emoji_rows[3:]):
    cols = row.find_all("td")
    if len(cols) == 15:

        col_nr = int(cols[0].text)
        desc = cols[-1].text.replace(" ", "_")

        for i, source in idx_source_map.items():
            img = get_image_for_cell(cols[i])
            if img is not None:
                filename = f"{col_nr:04d}_{desc}.png"
                with open(os.path.join(base_data_dir, source, filename), "wb") as f:
                    f.write(img)
