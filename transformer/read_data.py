import pyarrow.parquet as pq

table = pq.read_table(
    "data/iwslt2017_zh_en/train.parquet"
)

samples = table.to_pylist()

english = samples[0]["translation"]["en"]
chinese = samples[0]["translation"]["zh"]

print(english)
print(chinese)