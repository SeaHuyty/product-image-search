## Data notes

- `engine/data/styles.csv`: ~44.4k rows. Columns: `id, gender, masterCategory, subCategory, articleType, baseColour, season, year, usage, productDisplayName`.
- `engine/data/images/<id>.jpg`: image per row. **Some ids have no image** (e.g. 10036, 10038), so rows must be filtered to those with an existing file.
- **No description column.** Build one from attributes, e.g.
  `"Men's Casual Navy Blue Shirt - Topwear, Fall 2011"`
  Template: `{gender} {usage} {baseColour} {articleType} - {subCategory}, {season} {year}`, skipping null fields.
- Known dataset quirks to handle when loading:
  - A few malformed lines with extra commas -> `pd.read_csv(..., on_bad_lines="skip")`.
  - Null `productDisplayName` / `baseColour` etc. -> drop rows with no name; skip null fields in description.
  - Possibly corrupt images -> `try/except` per image in indexing, log and skip.
