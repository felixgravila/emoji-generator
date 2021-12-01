#%%

import matplotlib.pyplot as plt
import os
import numpy as np
from dataclasses import dataclass

sources = ['Goog', 'FB', 'Wind', 'Appl', 'Twtr', 'Sams']

@dataclass
class Emoji():
    desc: str
    source: str
    data: np.ndarray

def get_emojis_from_source(source: str, root_dir="emojis"):
    return [
        Emoji(
            desc=img.replace(".png", ""),
            source=source,
            data=plt.imread(os.path.join(root_dir, source, img))
        )
        for img in os.listdir(os.path.join(root_dir, source))
    ]

def get_all_emojis(sources=sources, root_dir="emojis"):
    return {
        k: get_emojis_from_source(k, root_dir) 
        for k in sources
    }



# %%

if __name__ == "__main__":

    all_emojis = get_all_emojis()


# %%
