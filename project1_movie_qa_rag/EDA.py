import pandas as pd

df = pd.read_csv("wiki_movie_plots_deduped.csv")
print(df.info())


length_list = [len(text) for text in df["Plot"]]
length_series = pd.Series(length_list)
stats = length_series.describe()
print(stats)  