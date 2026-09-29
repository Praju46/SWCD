
import io
import os
from datetime import datetime

import geopandas as gpd
import pandas as pd
import numpy as np
import streamlit as st
import folium
from streamlit_folium import st_folium
import matplotlib.pyplot as plt

from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.utils import ImageReader
from reportlab.lib import colors
from reportlab.platypus import Table, TableStyle
from reportlab.lib.units import mm


# ============================================================
# APP CONFIG
# ============================================================

st.set_page_config(
    page_title="Maharashtra SWCD Watershed Priority Portal",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="expanded",
)

DATA_FILE = "Final_Priority_Total.shp"

# Friendly name -> actual field in Final_Priority_Total
FIELD_MAP = {
    "Cluster": "CLUSTER",
    "District": "DISTRICT",
    "Taluka": "TALUKA",
    "Area (ha)": "AREA_HA",

    "Groundwater fluctuation": "VAL_GWL",
    "Untapped drains (%)": "VAL_UNTAP",
    "Rainfed agriculture (%)": "VAL_RAIN",
    "Soil erosion": "VAL_EROS",
    "Agricultural area (ha)": "VAL_AGRI",
    "Wasteland (ha)": "VAL_WASTE",
    "SC population": "VAL_SC",
    "ST population": "VAL_ST",
    "Active job cards": "VAL_JC",
    "Landholding": "VAL_LH",

    "Overall score": "SCR_O",
    "District score": "SCR_D",
    "Taluka score": "SCR_T",
    "Overall rank": "RNK_O",
    "District rank": "RNK_D",
    "Taluka rank": "RNK_T",
    "Final priority rank": "FIN_PRI",
}

PARAMETERS = [
    ("Groundwater fluctuation", "VAL_GWL"),
    ("Untapped drains (%)", "VAL_UNTAP"),
    ("Rainfed agriculture (%)", "VAL_RAIN"),
    ("Soil erosion", "VAL_EROS"),
    ("Agricultural area (ha)", "VAL_AGRI"),
    ("Wasteland (ha)", "VAL_WASTE"),
    ("SC population", "VAL_SC"),
    ("ST population", "VAL_ST"),
    ("Active job cards", "VAL_JC"),
    ("Landholding", "VAL_LH"),
]


# ============================================================
# CSS
# ============================================================

st.markdown(
    """
    <style>
    .main-title {
        font-size: 2rem;
        font-weight: 700;
        margin-bottom: 0.1rem;
    }
    .subtitle {
        font-size: 1rem;
        color: #555;
        margin-bottom: 1rem;
    }
    .metric-card {
        padding: 0.85rem;
        border-radius: 0.65rem;
        border: 1px solid #dddddd;
        background: #fafafa;
        min-height: 105px;
    }
    .small-note {
        font-size: 0.78rem;
        color: #666;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# DATA
# ============================================================

@st.cache_data(show_spinner="Loading Final_Priority_Total...")
def load_data(path):
    gdf = gpd.read_file(path)

    # Standardize field names to upper case.
    gdf.columns = [str(c).upper() for c in gdf.columns]

    # Convert numeric fields.
    for field in FIELD_MAP.values():
        if field in gdf.columns and field not in ["CLUSTER", "DISTRICT", "TALUKA"]:
            gdf[field] = pd.to_numeric(gdf[field], errors="coerce")

    if gdf.crs is None:
        raise ValueError(
            "The input layer has no CRS. Define the CRS before running the app."
        )

    source_crs = gdf.crs
    if gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs(4326)

    # Display class is ONLY for map visualization.
    # Original FIN_PRI is never changed.
    if "FIN_PRI" in gdf.columns:
        rank = gdf["FIN_PRI"].rank(method="min", ascending=True)
        pct = rank / max(len(gdf), 1)

        gdf["DISPLAY_CLASS"] = pd.cut(
            pct,
            bins=[0, 0.20, 0.40, 0.60, 0.80, 1.0],
            labels=[
                "Very High (top 20%)",
                "High (20–40%)",
                "Moderate (40–60%)",
                "Low (60–80%)",
                "Very Low (bottom 20%)",
            ],
            include_lowest=True,
        ).astype(str)
    else:
        gdf["DISPLAY_CLASS"] = "Not available"

    return gdf, str(source_crs)


def find_data_file():
    candidates = [
        DATA_FILE,
        os.path.join(os.path.dirname(os.path.abspath(__file__)), DATA_FILE),
        os.path.join(os.getcwd(), DATA_FILE),
        os.path.join("data", DATA_FILE),
    ]

    for p in candidates:
        if os.path.exists(p):
            return p

    return None


DATA_PATH = find_data_file()

if DATA_PATH is None:
    st.error(
        "Final_Priority_Total.shp was not found. Put the complete shapefile "
        "set (.shp, .shx, .dbf, .prj and .cpg) in the same folder as app.py."
    )
    st.stop()

try:
    gdf, SOURCE_CRS = load_data(DATA_PATH)
except Exception as e:
    st.error(f"Could not load the watershed layer: {e}")
    st.stop()


# ============================================================
# HELPERS
# ============================================================

def safe_num(value, digits=2):
    if pd.isna(value):
        return "—"
    try:
        return f"{float(value):,.{digits}f}"
    except Exception:
        return str(value)


def safe_int(value):
    if pd.isna(value):
        return "—"
    try:
        return f"{int(round(float(value))):,}"
    except Exception:
        return str(value)


def get_row(cluster_id):
    result = gdf[gdf["CLUSTER"].astype(str) == str(cluster_id)]
    if result.empty:
        return None
    return result.iloc[0]


def filter_data(data, district, taluka, search):
    out = data

    if district != "All":
        out = out[out["DISTRICT"].astype(str) == district]

    if taluka != "All":
        out = out[out["TALUKA"].astype(str) == taluka]

    if search.strip():
        query = search.strip().lower()
        out = out[
            out["CLUSTER"]
            .astype(str)
            .str.lower()
            .str.contains(query, na=False)
        ]

    return out


def rank_class(rank):
    try:
        rank = float(rank)
        n = len(gdf)
        ratio = rank / max(n, 1)

        if ratio <= 0.20:
            return "Very High"
        if ratio <= 0.40:
            return "High"
        if ratio <= 0.60:
            return "Moderate"
        if ratio <= 0.80:
            return "Low"
        return "Very Low"
    except Exception:
        return "Not available"


def map_style(feature):
    rank = feature["properties"].get("FIN_PRI")

    try:
        rank = float(rank)
        ratio = rank / max(len(gdf), 1)

        if ratio <= 0.20:
            fill = "#b2182b"
        elif ratio <= 0.40:
            fill = "#ef8a62"
        elif ratio <= 0.60:
            fill = "#fddbc7"
        elif ratio <= 0.80:
            fill = "#d1e5f0"
        else:
            fill = "#2166ac"
    except Exception:
        fill = "#cccccc"

    return {
        "fillColor": fill,
        "color": "#666666",
        "weight": 0.5,
        "fillOpacity": 0.65,
    }


def make_map(data, selected_cluster=None, height=600):
    if data.empty:
        center = [20.75, 78.95]
    else:
        try:
            union = data.geometry.union_all()
            centroid = union.centroid
            center = [centroid.y, centroid.x]
        except Exception:
            center = [20.75, 78.95]

    m = folium.Map(
        location=center,
        zoom_start=7,
        control_scale=True,
        tiles="CartoDB positron",
    )

    show = data.copy()

    # Simplify only the browser copy to improve performance.
    if len(show) > 900:
        show["geometry"] = show.geometry.simplify(
            0.0005,
            preserve_topology=True,
        )

    popup_fields = [
        "CLUSTER",
        "DISTRICT",
        "TALUKA",
        "AREA_HA",
        "FIN_PRI",
        "RNK_O",
        "RNK_D",
        "RNK_T",
    ]

    popup_fields = [x for x in popup_fields if x in show.columns]

    aliases_all = {
        "CLUSTER": "Cluster",
        "DISTRICT": "District",
        "TALUKA": "Taluka",
        "AREA_HA": "Area (ha)",
        "FIN_PRI": "Final priority rank",
        "RNK_O": "Overall rank",
        "RNK_D": "District rank",
        "RNK_T": "Taluka rank",
    }

    aliases = [aliases_all[x] for x in popup_fields]

    tooltip = folium.GeoJsonTooltip(
        fields=popup_fields,
        aliases=aliases,
        localize=True,
        sticky=False,
        labels=True,
    )

    folium.GeoJson(
        show,
        name="Watershed clusters",
        tooltip=tooltip,
        style_function=map_style,
        highlight_function=lambda feature: {
            "weight": 2,
            "color": "#111111",
            "fillOpacity": 0.8,
        },
    ).add_to(m)

    if selected_cluster is not None:
        selected = gdf[
            gdf["CLUSTER"].astype(str) == str(selected_cluster)
        ].copy()

        if not selected.empty:
            folium.GeoJson(
                selected,
                name="Selected project",
                style_function=lambda feature: {
                    "fillColor": "#ffff00",
                    "color": "#000000",
                    "weight": 4,
                    "fillOpacity": 0.35,
                },
                tooltip=folium.GeoJsonTooltip(
                    fields=[
                        "CLUSTER",
                        "DISTRICT",
                        "TALUKA",
                        "AREA_HA",
                        "FIN_PRI",
                    ],
                    aliases=[
                        "Cluster",
                        "District",
                        "Taluka",
                        "Area (ha)",
                        "Final rank",
                    ],
                ),
            ).add_to(m)

            bounds = selected.total_bounds
            m.fit_bounds(
                [
                    [bounds[1], bounds[0]],
                    [bounds[3], bounds[2]],
                ]
            )

    folium.LayerControl(collapsed=False).add_to(m)

    return m


def static_map_png(row):
    fig, ax = plt.subplots(figsize=(10, 6))

    selected = gpd.GeoDataFrame(
        [row],
        geometry=[row.geometry],
        crs=gdf.crs,
    )

    # Context around selected cluster.
    try:
        minx, miny, maxx, maxy = selected.total_bounds
        dx = max((maxx - minx) * 0.6, 0.01)
        dy = max((maxy - miny) * 0.6, 0.01)

        context = gdf.cx[
            minx - dx:maxx + dx,
            miny - dy:maxy + dy
        ]

        if context.empty:
            context = gdf
    except Exception:
        context = gdf

    context.plot(
        ax=ax,
        facecolor="none",
        edgecolor="0.65",
        linewidth=0.5,
    )

    selected.plot(
        ax=ax,
        facecolor="0.35",
        edgecolor="black",
        linewidth=2,
        alpha=0.45,
    )

    ax.set_title(
        f"{row['CLUSTER']} | {row['DISTRICT']} | {row['TALUKA']}",
        fontsize=13,
        fontweight="bold",
    )

    ax.set_axis_off()
    fig.tight_layout()

    buffer = io.BytesIO()
    fig.savefig(
        buffer,
        format="png",
        dpi=220,
        bbox_inches="tight",
    )
    plt.close(fig)

    buffer.seek(0)
    return buffer.getvalue()


def parameter_chart_png(row):
    labels = [x[0] for x in PARAMETERS]

    raw = []
    for _, field in PARAMETERS:
        raw.append(pd.to_numeric(row[field], errors="coerce"))

    values = np.asarray(raw, dtype=float)

    valid = np.isfinite(values)
    normalized = np.zeros(len(values), dtype=float)

    if valid.any():
        mn = np.nanmin(values)
        mx = np.nanmax(values)

        if mx > mn:
            normalized[valid] = (
                (values[valid] - mn) /
                (mx - mn)
            )
        else:
            normalized[valid] = 1.0

    fig, ax = plt.subplots(figsize=(10, 5.5))

    y = np.arange(len(labels))
    ax.barh(y, normalized)

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlim(0, 1)
    ax.set_xlabel(
        "Normalized value for visual comparison only"
    )
    ax.set_title(
        "Project Parameter Profile"
    )
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.2)

    fig.tight_layout()

    buffer = io.BytesIO()
    fig.savefig(
        buffer,
        format="png",
        dpi=220,
        bbox_inches="tight",
    )
    plt.close(fig)

    buffer.seek(0)
    return buffer.getvalue()


def build_pdf(row):
    """
    Generate a 4-page PDF:
      1. Project summary
      2. Project location map
      3. Ten thematic parameters
      4. Ranking and planning information
    """

    output = io.BytesIO()

    page_width, page_height = A4

    pdf = canvas.Canvas(
        output,
        pagesize=A4,
    )

    cluster = str(row["CLUSTER"])

    # --------------------------------------------------------
    # PAGE 1
    # --------------------------------------------------------

    pdf.setFont("Helvetica-Bold", 20)
    pdf.drawString(
        25 * mm,
        page_height - 28 * mm,
        "MAHARASHTRA SWCD",
    )

    pdf.setFont("Helvetica-Bold", 16)
    pdf.drawString(
        25 * mm,
        page_height - 40 * mm,
        "Watershed Project Profile",
    )

    pdf.setFont("Helvetica", 10)
    pdf.drawString(
        25 * mm,
        page_height - 49 * mm,
        "GIS-based watershed prioritization project",
    )

    summary = [
        ["Cluster", cluster],
        ["District", str(row["DISTRICT"])],
        ["Taluka", str(row["TALUKA"])],
        ["Area (ha)", safe_num(row["AREA_HA"])],
        ["Final priority rank", safe_int(row["FIN_PRI"])],
        ["Priority display class", rank_class(row["FIN_PRI"])],
        ["Overall rank", safe_int(row["RNK_O"])],
        ["District rank", safe_int(row["RNK_D"])],
        ["Taluka rank", safe_int(row["RNK_T"])],
        ["Overall score", safe_num(row["SCR_O"])],
        ["District score", safe_num(row["SCR_D"])],
        ["Taluka score", safe_num(row["SCR_T"])],
    ]

    table = Table(
        summary,
        colWidths=[
            65 * mm,
            75 * mm,
        ],
    )

    table.setStyle(
        TableStyle(
            [
                ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
                ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 10),
                ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
                ("BACKGROUND", (0, 0), (0, -1), colors.whitesmoke),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )

    table.wrapOn(
        pdf,
        page_width,
        page_height,
    )

    table.drawOn(
        pdf,
        25 * mm,
        page_height - 155 * mm,
    )

    pdf.setFont("Helvetica", 8)
    pdf.drawString(
        25 * mm,
        15 * mm,
        f"Generated: {datetime.now().strftime('%d-%m-%Y %H:%M')}",
    )

    pdf.showPage()

    # --------------------------------------------------------
    # PAGE 2 - MAP
    # --------------------------------------------------------

    pdf.setFont("Helvetica-Bold", 16)
    pdf.drawString(
        20 * mm,
        page_height - 22 * mm,
        "Project Location Map",
    )

    map_image = static_map_png(row)

    pdf.drawImage(
        ImageReader(io.BytesIO(map_image)),
        15 * mm,
        42 * mm,
        width=180 * mm,
        height=220 * mm,
        preserveAspectRatio=True,
        anchor="c",
    )

    pdf.setFont("Helvetica", 8)
    pdf.drawString(
        20 * mm,
        25 * mm,
        "Selected watershed cluster shown against surrounding watershed clusters.",
    )

    pdf.showPage()

    # --------------------------------------------------------
    # PAGE 3 - PARAMETERS
    # --------------------------------------------------------

    pdf.setFont("Helvetica-Bold", 16)
    pdf.drawString(
        20 * mm,
        page_height - 22 * mm,
        "Ten Thematic Parameter Statistics",
    )

    rows = [
        ["Parameter", "Field", "Value"]
    ]

    for label, field in PARAMETERS:
        value = row[field]

        if field in [
            "VAL_SC",
            "VAL_ST",
            "VAL_JC",
            "VAL_LH",
        ]:
            display = safe_int(value)
        else:
            display = safe_num(value)

        rows.append(
            [label, field, display]
        )

    table = Table(
        rows,
        colWidths=[
            85 * mm,
            35 * mm,
            45 * mm,
        ],
    )

    table.setStyle(
        TableStyle(
            [
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )

    table.wrapOn(
        pdf,
        page_width,
        page_height,
    )

    table.drawOn(
        pdf,
        20 * mm,
        page_height - 155 * mm,
    )

    chart = parameter_chart_png(row)

    pdf.drawImage(
        ImageReader(io.BytesIO(chart)),
        20 * mm,
        25 * mm,
        width=170 * mm,
        height=80 * mm,
        preserveAspectRatio=True,
        anchor="c",
    )

    pdf.showPage()

    # --------------------------------------------------------
    # PAGE 4 - RANKING / PLANNING
    # --------------------------------------------------------

    pdf.setFont("Helvetica-Bold", 16)
    pdf.drawString(
        20 * mm,
        page_height - 22 * mm,
        "Priority and Planning Profile",
    )

    rank_rows = [
        ["Indicator", "Value"],
        ["Final priority rank", safe_int(row["FIN_PRI"])],
        ["Overall rank", safe_int(row["RNK_O"])],
        ["District rank", safe_int(row["RNK_D"])],
        ["Taluka rank", safe_int(row["RNK_T"])],
        ["Overall score", safe_num(row["SCR_O"])],
        ["District score", safe_num(row["SCR_D"])],
        ["Taluka score", safe_num(row["SCR_T"])],
    ]

    table = Table(
        rank_rows,
        colWidths=[
            80 * mm,
            60 * mm,
        ],
    )

    table.setStyle(
        TableStyle(
            [
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
                ("FONTSIZE", (0, 0), (-1, -1), 10),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )

    table.wrapOn(
        pdf,
        page_width,
        page_height,
    )

    table.drawOn(
        pdf,
        20 * mm,
        page_height - 105 * mm,
    )

    pdf.setFont("Helvetica-Bold", 12)
    pdf.drawString(
        20 * mm,
        page_height - 125 * mm,
        "Planning note",
    )

    notes = [
        "The values shown are the indicators stored in the supplied Final_Priority_Total layer.",
        "The application does not recalculate or modify the original priority rank.",
        "The five-category map classification is only a visualization grouping of the final rank.",
        "Priority results should be followed by field verification and technical feasibility assessment before implementation.",
    ]

    y = page_height - 138 * mm

    pdf.setFont("Helvetica", 9)

    for note in notes:
        # Basic line wrapping.
        words = note.split()
        lines = []
        current = ""

        for word in words:
            candidate = (current + " " + word).strip()

            if len(candidate) > 95:
                lines.append(current)
                current = word
            else:
                current = candidate

        if current:
            lines.append(current)

        for line in lines:
            pdf.drawString(
                25 * mm,
                y,
                "• " + line,
            )
            y -= 5 * mm

        y -= 2 * mm

    pdf.setFont(
        "Helvetica-Oblique",
        8,
    )

    pdf.drawString(
        20 * mm,
        15 * mm,
        "GIS decision-support report based on the supplied watershed priority dataset.",
    )

    pdf.showPage()

    pdf.save()

    output.seek(0)

    return output.getvalue()


# ============================================================
# SIDEBAR FILTERS
# ============================================================

st.sidebar.title("SWCD Watershed Portal")
st.sidebar.caption("Maharashtra watershed project prioritization")

page = st.sidebar.radio(
    "Navigation",
    [
        "Dashboard",
        "Project Explorer",
        "Methodology",
        "Data Dictionary",
    ],
)

st.sidebar.divider()

districts = [
    "All"
] + sorted(
    gdf["DISTRICT"]
    .dropna()
    .astype(str)
    .unique()
    .tolist()
)

selected_district = st.sidebar.selectbox(
    "District",
    districts,
)

if selected_district == "All":
    taluka_source = gdf
else:
    taluka_source = gdf[
        gdf["DISTRICT"].astype(str)
        == selected_district
    ]

talukas = [
    "All"
] + sorted(
    taluka_source["TALUKA"]
    .dropna()
    .astype(str)
    .unique()
    .tolist()
)

selected_taluka = st.sidebar.selectbox(
    "Taluka",
    talukas,
)

search_cluster = st.sidebar.text_input(
    "Search cluster",
    placeholder="Example: Risod_C4",
)

filtered = filter_data(
    gdf,
    selected_district,
    selected_taluka,
    search_cluster,
)

st.sidebar.metric(
    "Clusters in current view",
    f"{len(filtered):,}",
)


# ============================================================
# DASHBOARD
# ============================================================

if page == "Dashboard":

    st.markdown(
        '<div class="main-title">'
        'Maharashtra SWCD Watershed Priority Portal'
        '</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="subtitle">'
        'Interactive exploration of watershed cluster priorities, '
        'statistics and project reports.'
        '</div>',
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Total clusters",
        f"{len(gdf):,}",
    )

    c2.metric(
        "Districts",
        f"{gdf['DISTRICT'].nunique():,}",
    )

    c3.metric(
        "Talukas",
        f"{gdf['TALUKA'].nunique():,}",
    )

    c4.metric(
        "Total area (ha)",
        f"{gdf['AREA_HA'].sum():,.0f}",
    )

    st.subheader(
        "Watershed Priority Map"
    )

    st.caption(
        "The exact FIN_PRI rank is retained. "
        "The map colours are a five-group visualization only."
    )

    map_obj = make_map(
        filtered,
        height=620,
    )

    st_folium(
        map_obj,
        width=None,
        height=620,
        returned_objects=[],
        key="dashboard_map",
    )

    st.subheader(
        "Priority distribution in current view"
    )

    class_order = [
        "Very High (top 20%)",
        "High (20–40%)",
        "Moderate (40–60%)",
        "Low (60–80%)",
        "Very Low (bottom 20%)",
    ]

    distribution = (
        filtered["DISPLAY_CLASS"]
        .value_counts()
        .reindex(class_order)
        .fillna(0)
        .astype(int)
    )

    st.bar_chart(
        distribution
    )


# ============================================================
# PROJECT EXPLORER
# ============================================================

elif page == "Project Explorer":

    st.markdown(
        '<div class="main-title">'
        'Watershed Project Explorer'
        '</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        "Select a watershed cluster/project to view its complete statistics "
        "and generate a PDF report."
    )

    if filtered.empty:
        st.warning(
            "No watershed clusters match the selected filters."
        )
        st.stop()

    cluster_list = (
        filtered["CLUSTER"]
        .astype(str)
        .sort_values()
        .tolist()
    )

    selected_cluster = st.selectbox(
        "Select watershed project / cluster",
        cluster_list,
    )

    row = get_row(
        selected_cluster
    )

    if row is None:
        st.error(
            "The selected cluster was not found."
        )
        st.stop()

    st.divider()

    st.subheader(
        f"Project Profile — {selected_cluster}"
    )

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "District",
        str(row["DISTRICT"]),
    )

    c2.metric(
        "Taluka",
        str(row["TALUKA"]),
    )

    c3.metric(
        "Area (ha)",
        safe_num(row["AREA_HA"]),
    )

    c4.metric(
        "Final priority rank",
        safe_int(row["FIN_PRI"]),
    )

    c5, c6, c7, c8 = st.columns(4)

    c5.metric(
        "Overall rank",
        safe_int(row["RNK_O"]),
    )

    c6.metric(
        "District rank",
        safe_int(row["RNK_D"]),
    )

    c7.metric(
        "Taluka rank",
        safe_int(row["RNK_T"]),
    )

    c8.metric(
        "Overall score",
        safe_num(row["SCR_O"]),
    )

    st.info(
        f"Display class for mapping only: "
        f"**{rank_class(row['FIN_PRI'])}**"
    )

    # --------------------------------------------------------
    # Selected project map
    # --------------------------------------------------------

    st.subheader(
        "Selected Project Map"
    )

    project_map = make_map(
        filtered,
        selected_cluster=selected_cluster,
    )

    st_folium(
        project_map,
        width=None,
        height=600,
        returned_objects=[],
        key=f"project_map_{selected_cluster}",
    )

    # --------------------------------------------------------
    # Ten indicators
    # --------------------------------------------------------

    st.subheader(
        "Ten Thematic Indicators"
    )

    metric_columns = st.columns(5)

    for i, (label, field) in enumerate(PARAMETERS):

        value = row[field]

        if field in [
            "VAL_SC",
            "VAL_ST",
            "VAL_JC",
            "VAL_LH",
        ]:
            display = safe_int(value)
        else:
            display = safe_num(value)

        with metric_columns[i % 5]:
            st.markdown(
                f"""
                <div class="metric-card">
                    <b>{label}</b><br>
                    <span style="font-size:1.3rem;">{display}</span><br>
                    <span class="small-note">{field}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # --------------------------------------------------------
    # Charts
    # --------------------------------------------------------

    st.write("")

    left, right = st.columns(2)

    with left:

        st.subheader(
            "Parameter profile"
        )

        chart_values = []

        for label, field in PARAMETERS:
            value = pd.to_numeric(
                row[field],
                errors="coerce",
            )

            chart_values.append(
                [label, value]
            )

        chart_df = pd.DataFrame(
            chart_values,
            columns=["Parameter", "Value"],
        )

        # Normalize only for visual comparison because
        # parameters have different units.
        values = chart_df["Value"].to_numpy(
            dtype=float
        )

        if np.isfinite(values).any():
            mn = np.nanmin(values)
            mx = np.nanmax(values)

            if mx > mn:
                chart_df["Normalized"] = (
                    (values - mn) / (mx - mn)
                )
            else:
                chart_df["Normalized"] = 1.0
        else:
            chart_df["Normalized"] = 0.0

        st.bar_chart(
            chart_df.set_index("Parameter")[
                "Normalized"
            ]
        )

        st.caption(
            "Normalized values are for visual comparison only; "
            "the raw values above are the actual project statistics."
        )

    with right:

        st.subheader(
            "Ranking summary"
        )

        ranking = pd.DataFrame(
            {
                "Rank": [
                    row["RNK_O"],
                    row["RNK_D"],
                    row["RNK_T"],
                ]
            },
            index=[
                "Overall",
                "District",
                "Taluka",
            ],
        )

        st.bar_chart(
            ranking
        )

    # --------------------------------------------------------
    # Full attributes
    # --------------------------------------------------------

    with st.expander(
        "View complete project attributes"
    ):
        attributes = (
            row.drop(
                labels=["geometry"],
                errors="ignore",
            )
            .to_frame("Value")
        )

        st.dataframe(
            attributes,
            use_container_width=True,
        )

    # --------------------------------------------------------
    # PDF
    # --------------------------------------------------------

    st.divider()

    st.subheader(
        "Generate Project PDF"
    )

    st.write(
        "The PDF contains the project profile, selected-project map, "
        "ten thematic indicators, rankings and planning notes."
    )

    if st.button(
        "📄 Generate Project PDF",
        type="primary",
        use_container_width=True,
    ):

        with st.spinner(
            "Generating PDF report..."
        ):
            pdf_bytes = build_pdf(
                row
            )

        filename = (
            f"{selected_cluster}"
            "_Watershed_Project_Report.pdf"
        )

        st.success(
            "PDF generated successfully."
        )

        st.download_button(
            label="⬇️ Download Project PDF",
            data=pdf_bytes,
            file_name=filename,
            mime="application/pdf",
            use_container_width=True,
        )


# ============================================================
# METHODOLOGY
# ============================================================

elif page == "Methodology":

    st.markdown(
        '<div class="main-title">'
        'Watershed Prioritization Methodology'
        '</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        The application uses the supplied Final_Priority_Total watershed
        cluster layer as the operational project database. It is designed
        to expose the cluster statistics already calculated in the project
        and to provide an interactive project-level reporting workflow.
        """
    )

    st.subheader(
        "Overall workflow"
    )

    st.markdown(
        """
        **Untreated microwatersheds → Watershed clusters → Thematic
        indicators → Cluster-level statistics → Priority scoring/ranking
        → Project profile → PDF report**
        """
    )

    st.subheader(
        "Ten thematic parameters"
    )

    methodology_table = pd.DataFrame(
        [
            ["1", "Groundwater Level", "VAL_GWL"],
            ["2", "Agricultural Area", "VAL_AGRI"],
            ["3", "Wasteland", "VAL_WASTE"],
            ["4", "Rainfed Agriculture", "VAL_RAIN"],
            ["5", "Active Job Cards", "VAL_JC"],
            ["6", "SC Population", "VAL_SC"],
            ["7", "ST Population", "VAL_ST"],
            ["8", "Landholding", "VAL_LH"],
            ["9", "Untapped Drains", "VAL_UNTAP"],
            ["10", "Soil Erosion", "VAL_EROS"],
        ],
        columns=[
            "No.",
            "Parameter",
            "Layer field",
        ],
    )

    st.dataframe(
        methodology_table,
        hide_index=True,
        use_container_width=True,
    )

    st.subheader(
        "Ranking structure in the supplied layer"
    )

    ranking_table = pd.DataFrame(
        [
            ["Overall", "SCR_O", "RNK_O"],
            ["District", "SCR_D", "RNK_D"],
            ["Taluka", "SCR_T", "RNK_T"],
            ["Final priority", "—", "FIN_PRI"],
        ],
        columns=[
            "Level",
            "Score field",
            "Rank field",
        ],
    )

    st.dataframe(
        ranking_table,
        hide_index=True,
        use_container_width=True,
    )

    st.info(
        "The application does not recalculate the supplied priority "
        "methodology. It reads the scores and ranks already present in "
        "Final_Priority_Total."
    )


# ============================================================
# DATA DICTIONARY
# ============================================================

elif page == "Data Dictionary":

    st.markdown(
        '<div class="main-title">'
        'Data Dictionary'
        '</div>',
        unsafe_allow_html=True,
    )

    dictionary = []

    for description, field in FIELD_MAP.items():

        dtype = (
            str(gdf[field].dtype)
            if field in gdf.columns
            else "MISSING"
        )

        dictionary.append(
            [
                description,
                field,
                dtype,
            ]
        )

    st.dataframe(
        pd.DataFrame(
            dictionary,
            columns=[
                "Description",
                "Field",
                "Data type",
            ],
        ),
        hide_index=True,
        use_container_width=True,
    )

    st.subheader(
        "Layer information"
    )

    info = pd.DataFrame(
        [
            ["Features", len(gdf)],
            ["Fields", len(gdf.columns)],
            ["Source CRS", SOURCE_CRS],
            ["Application map CRS", "EPSG:4326"],
            ["Districts", gdf["DISTRICT"].nunique()],
            ["Talukas", gdf["TALUKA"].nunique()],
        ],
        columns=[
            "Item",
            "Value",
        ],
    )

    st.table(info)
