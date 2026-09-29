# Maharashtra SWCD Watershed Priority Portal

## 1. Put these files in one folder

The application expects the complete shapefile set:

- Final_Priority_Total.shp
- Final_Priority_Total.shx
- Final_Priority_Total.dbf
- Final_Priority_Total.prj
- Final_Priority_Total.cpg
- app.py
- requirements.txt

## 2. Create an environment

Recommended:

```bash
conda create -n swcd_app python=3.11 -y
conda activate swcd_app
pip install -r requirements.txt
```

## 3. Run

```bash
streamlit run app.py
```

The browser will open the SWCD portal.

## 4. Main features

- Maharashtra watershed cluster map
- District filter
- Taluka filter
- Cluster search
- Project/cluster selection
- Project profile
- Ten thematic indicators
- Overall / district / taluka ranks
- Parameter visualization
- Full attribute table
- PDF report generation

## 5. PDF

Selecting a project and pressing **Generate Project PDF** creates:

1. Project summary
2. Project location map
3. Ten thematic parameter statistics
4. Ranking and planning profile

The PDF is generated from the values already present in `Final_Priority_Total`.

## 6. Important

The app does not recalculate or alter the supplied priority methodology.

`FIN_PRI` is preserved as the original final rank.

The five map classes are only a visualization grouping:

- Very High = top 20% of ranks
- High = 20–40%
- Moderate = 40–60%
- Low = 60–80%
- Very Low = bottom 20%

These display classes are not new analytical priority scores.
