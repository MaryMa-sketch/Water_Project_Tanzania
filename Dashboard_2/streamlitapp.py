import streamlit as st
from PIL import Image
import pandas as pd
import plotly.express as px
import joblib, json
import numpy as np
from pathlib import Path
import category_encoders as ce

# preprocessing pipeline
# --- Column groups ---

numeric_cols = ['age','lat_5dp','lon_5dp', 'amount_tsh_nonzero_log','population_clean_log']
ohe_cols = ['basin','source_type','quality_group','extraction_type','management','quantity','waterpoint_type']
targetenc_cols = ['ward','installer_clean']

def _coerce_object_and_nan(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # ensure object dtype (avoids pandas NA semantics) and replace pd.NA with np.nan
    for c in df.columns:
        df[c] = df[c].astype("object")
    return df.replace({pd.NA: np.nan})


# Base paths
BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "best_xgb_pipe.pkl"
LABEL_PATH = BASE_DIR / "inv_label_map.json"
PRED_PATH  = BASE_DIR / "final_predictions.csv"

@st.cache_data
def load_data():
    df = pd.read_csv(PRED_PATH)
    return df

@st.cache_resource
def load_model_and_labels():
    import joblib, json
    model = joblib.load(MODEL_PATH)
    with open(LABEL_PATH, "r") as f:
        inv_label_map = json.load(f)
    return model, inv_label_map

# Load model + labels
best_xgb_pipe, inv_label_map = load_model_and_labels()


def predict_single(input_dict):
    """
    input_dict: raw feature values from the form.
    Returns: predicted_label (str), probs_df (pd.DataFrame)
    """
    df_input = pd.DataFrame([input_dict])

    # Predict encoded label
    y_pred_enc = best_xgb_pipe.predict(df_input)
    y_pred_enc = int(y_pred_enc[0])
    pred_label = inv_label_map[str(y_pred_enc)]

    # Probabilities
    proba = best_xgb_pipe.predict_proba(df_input)[0]
    classes_encoded = best_xgb_pipe.classes_
    class_labels = [inv_label_map[str(i)] for i in classes_encoded]

    probs_df = pd.DataFrame(
        {"status": class_labels, "probability": proba}
    ).sort_values("probability", ascending=False)

    return pred_label, probs_df

@st.cache_resource
def compute_shap_stats(df):
    """Compute SHAP values and importance stats for the XGBoost model."""

    # 1. Extract preprocessor + model from the pipeline
    pre = best_xgb_pipe.named_steps["preprocess"]
    xgb_model = best_xgb_pipe.named_steps["model"]

    # 2. Build X matrix with just model features
    feature_cols = numeric_cols + ohe_cols + targetenc_cols
    X = df[feature_cols].copy()

    # (Optional) use a subset to speed things up
    #if len(X) > 3000:
        #X = X.sample(3000, random_state=42)

    # 3. Transform data
    X_trans = pre.transform(X)

    # 4. Build feature names (same logic you used in the notebook)
    num_feature_names = list(numeric_cols)

    ohe_pipe = pre.named_transformers_["ohe"]
    ohe_enc = ohe_pipe.named_steps["ohe"]

    ohe_feature_names = []
    for col, cats in zip(ohe_cols, ohe_enc.categories_):
        for cat in cats:
            ohe_feature_names.append(f"{col}__{cat}")

    target_feature_names = list(targetenc_cols)
    feature_names = num_feature_names + ohe_feature_names + target_feature_names

    # sanity check
    assert X_trans.shape[1] == len(feature_names), (
        f"Transformed shape {X_trans.shape[1]} vs feature names {len(feature_names)}"
    )

    # 5. SHAP for XGBoost (multi-class)
    explainer = shap.TreeExplainer(xgb_model)
    shap_values = explainer.shap_values(X_trans)   # list of 3 arrays [n_samples, n_features]

    # 6. Mean |SHAP| across classes (and samples)
    sv_stack = np.stack(shap_values, axis=0)       # (n_classes, n_samples, n_features)
    mean_abs_all = np.mean(np.abs(sv_stack), axis=(0, 1))  # (n_features,)

    # 7. Per-class mean |SHAP|
    mean_abs_per_class = [np.mean(np.abs(sv), axis=0) for sv in shap_values]  # list of (n_features,)

    return feature_names, mean_abs_all, mean_abs_per_class



st.set_page_config(
    page_title="Water Pump Functionality Prediction",
    layout="wide"
)

def load_logo_base64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()

logo_base64 = load_logo_base64("redi_logo.png")   

st.markdown(
   f"""
    <style>
        .top-banner {{background-color: #E9F4F9;  padding: 18px 40px; border-radius: 0 0 8px 8px;
        margin-bottom: 20px;}}
        .banner-content {{display: flex; align-items: center; gap: 24px;}}
        .banner-text h1 {{margin: 0; font-size: 40px;}}
        .banner-text h3 {{margin: 6px 0 4px 0; font-weight: 500;}}
        .banner-text p {{margin: 0; font-size: 15px;}}
        .block-container {{max-width: 1100px;}}
    </style>

    <div class="top-banner">
      <div class="banner-content">
        <img src="data:image/png;base64,{logo_base64}" width="90">
         <div class="banner-text">
          <h1>Water Pump Functionality Prediction</h1>
           <h3>Mary Makhutu & Geetha Malaichamy</h3>
           <p>
             Data Source:
            <a href="https://www.drivendata.org/competitions/7/pump-it-up-data-mining-the-water-table/"
               target="_blank">
               https://www.drivendata.org/competitions/7/pump-it-up-data-mining-the-water-table/
            </a>
          </p>
        </div>
     </div>
   </div>
   """,
   unsafe_allow_html=True,
)

st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to",
    ["Introduction","Data Overview", "Model Results", "Feature Importance", "Maps", "Prediction Interface", "Conclusion", "Feature Exploration"]
)

# =========================================================
#   GLOBAL STYLE OVERRIDES (Deep Blue Headings) #2E86C1
# =========================================================
st.markdown("""
<style>

/* ---- Page Title (H1) ---- */
h1 {
    color: #1F3A93 !important;      /* Deep Blue */
    font-weight: 800 !important;
    letter-spacing: -0.5px !important;
}

/* ---- Section Title (H2) ---- */
h2 {
    color: #253449 !important;      /* Slightly darker blue-gray */
    font-weight: 700 !important;
    margin-top: 1.2em !important;
}

/* ---- Sub-heading (H3) ---- */
h3 {
    color: #3C546A !important;      /* Medium blue-gray */
    font-weight: 600 !important;
}

</style>
""", unsafe_allow_html=True)

st.markdown("""
<style>

/* === Deep Blue Divider Styles === */

/* Thin blue line */
hr.blue-line {border: none; border-top: 2px solid #1F3A93;   /* Deep Blue */ margin: 1.5em 0;
}

/* Medium thickness */
hr.blue-line-medium {border: none; border-top: 4px solid #2E86C1;   /* Medium Blue */ margin: 2em 0;
}

/* Gradient divider (premium look) */
hr.blue-gradient {border: 0; height: 3px; background: linear-gradient(to right, #1F3A93, #2E86C1, #5DADE2);
margin: 2em 0;
}

</style>
""", unsafe_allow_html=True)

st.markdown("""
<style>

/* Increase body text size across the app */
p, li, ul, ol {
    font-size: 20px !important;
    line-height: 1.6 !important;
    color: #253449 !important;  /* matches your theme */
}

/* Optional: increase bullet spacing for readability */
ul, ol {
    margin-top: 0.4em !important;
    margin-bottom: 0.4em !important;
}

</style>
""", unsafe_allow_html=True)

STATUS_ORDER = ["functional", "functional needs repair", "non functional"]
STATUS_COLORS = {
    "functional": "#7E22CE",              # purple
    "functional needs repair": "#EAB308", # yellow
    "non functional": "#DC2626",          # red
}

def style_sample_table(df: pd.DataFrame, target_col: str):
    sample = df.copy()

    # Style only if the status column exists
    if target_col in sample.columns:

        def color_status(val):
            color = STATUS_COLORS.get(val, None)
            if color:
                return f"color: {color}; font-weight:600;"
            return ""

        styler = sample.style.applymap(color_status, subset=[target_col])
    else:
        styler = sample.style

    # Header + zebra stripes
    styler = styler.set_table_styles(
        [
            {
                "selector": "th",
                "props": "background-color:#F4F6F7; color:#253449; font-weight:bold;",
            },
            {
                "selector": "tbody tr:nth-child(even)",
                "props": "background-color:#FBFBFB;",
            },
        ]
    )

    return styler

# ------------------ MAIN PAGE CONTENT ------------------

if page == "Introduction":

    st.markdown(
    "<h1 style='font-size: 42px; font-weight: 700; margin-bottom: 0.2em;'>Introduction</h1>",
    unsafe_allow_html=True
)
    
    # Top section: Why it matters
    col1, col2 = st.columns([2, 2])

    with col1:
        st.markdown("### 💡 What Affects Pump Functionality ?")
        st.write(
            "Rural water pumps are essential for providing clean water in Tanzania, "
            "but many fail unexpectedly without timely maintenance."
        )

    with col2:
        st.markdown("###  🔮 Aim ")
        st.write("This project aims to predict whether a water pump is:")
        st.markdown(
            """
  🟣 **functional**  
  🟡 **functional needs repair**  
  🔴 **non-functional**
            """
        )

    st.markdown("<hr class='blue-line'>", unsafe_allow_html=True)

    #st.markdown("---")

    # Feature groups
    st.markdown("### How ?")

    fcol1, fcol2 = st.columns(2)

    with fcol1:
        st.markdown(
            """
**🌍 Location & Time**

- Geographic location  
- Administrative features  
- Temporal features (age)
            """
        )

    with fcol2:
        st.markdown(
            """
**🚰 Usage & Pump Characteristics**

- Population  
- Amount of water
- Water source & extraction type
            """
        )

    st.markdown("<hr class='blue-line'>", unsafe_allow_html=True)
    #st.markdown("---")

    # Project goal
    st.markdown("### 🎯 Project Goal")

    st.markdown("""
<div style="
    background-color: #F4F6F7; padding: 20px 25px; border-radius: 10px; border-left: 6px solid #1F3A93;
">
<p style="font-size: 17px; color: #2C3E50;">
To support NGOs and government agencies in making <strong>data-driven maintenance decisions</strong>:
</p>

<ul style="color: #2C3E50; font-size: 16px;"> <li>Prioritize repairs</li> <li>Reduce pump downtime</li>
<li>Improve long-term water access for communities</li>
</ul>

</div>
""", unsafe_allow_html=True)
    

elif page == "Data Overview":

    st.markdown("<h1>Data Overview</h1>", unsafe_allow_html=True)

    st.markdown("<hr class='blue-line'>", unsafe_allow_html=True)

    # --------------------- ROW 1: Dataset Overview + Processing Pipeline ---------------------
    col1, col2 = st.columns(2, gap="small")

    with col1:
        st.markdown("### 📊 Data Sets")
        st.markdown(
            """
Train set : <span style="color:#1E8449; font-weight:700;">59,400</span> rows, 40 columns  
Test set : <span style="color:#B03A2E; font-weight:700;">14,850</span> rows, 40 columns  
Target set : 
<span style="color:#DC2626; font-weight:600;">non functional</span>, 
<span style="color:#EAB308; font-weight:600;">functional needs repair</span> 
and <span style="color:#7E22CE; font-weight:600;">functional</span>
    """,
    unsafe_allow_html=True

        )

    with col2:
        st.markdown("### ⚙️ Data Processing")
        st.markdown(
            """
<b>Raw Data → Cleaning → Feature Engineering → Encoding → Model Training</b>
            """,unsafe_allow_html=True
        )
        
    st.markdown("<hr class='blue-line'>", unsafe_allow_html=True)

    col1, col2 = st.columns(2, gap="large")

# --- LEFT COLUMN ---
    with col1:
        with st.expander("### 🧹 Data Cleaning", expanded=False):
            st.markdown("""
- Handled zero values - treated them as NAN(population,age and amount_tsh) 
- Replaced out-of-range values with NAN (latitude, longitude, gps height)  
- Standardized categorical columns (trimmed whitespace: installer)  
- Explored class imbalance (target feature)
        """)
            
    
        with st.expander("### 🧰 Feature Engineering", expanded=False):
            st.markdown("""
- Computed **age** column  
- Handled outliers (population)  
- Log-transformation (amount_tsh)  
- Rounded latitude/longitude to 5 decimals
- Grouped installer into categories (**installer_cate**)  
        """)

    # --- RIGHT COLUMN ---
    with col2:
        with st.expander("### 🔄 Preprocessing Pipeline", expanded=False):
            st.markdown("""
- Numeric features: imputation + standard scaling
- Low-cardinality categoricals: one-hot encoding 
- High-cardinality categoricals: target encoding 
- Wrapped in a single ColumnTransformer + Pipeline  
        """)

        with st.expander("### 🧠 Model Training & Tuning", expanded=False):
            st.markdown("""
- **Random Forest** & **XGBoost Classifier**  
- Class imbalance - class_weight= **balanced**
- Tuning **RandomizedSearchCV**  
- Model valuation-**Accuracy**
        """)
    
    # --------------------- MODEL FEATURES SUMMARY ---------------------

    with st.expander("### 🧩 Model Features", expanded=False):
        f1, f2, f3 = st.columns(3)

        with f1:
            st.markdown("**Numeric features**")
            st.markdown(
                "\n".join([f"- `{c}`" for c in numeric_cols])
            )

        with f2:
            st.markdown("**One-hot encoded features**")
            st.markdown(
                "\n".join([f"- `{c}`" for c in ohe_cols])
            )

        with f3:
            st.markdown("**Target-encoded features**")
            st.markdown(
                "\n".join([f"- `{c}`" for c in targetenc_cols])
            )
    
elif page == "Model Results":
    st.header("Model Results")
    
    st.markdown("""
<style>
.table-style {width: 100%; border-collapse: collapse; font-size: 18px;
}
.table-style th {background-color: #0a4f70; color: white; padding: 10px; text-align: left;
}
.table-style td {padding: 10px; border-bottom: 1px solid #ddd;
}
.table-style tr:nth-child(even) { background-color: #f2f2f2;
}
.red-text {color: red; font-weight: 700;
}
</style>

<table class="table-style">
<tr> <th>Model</th> <th>Accuracy</th> <th>Precision_macro</th> <th>Recall_macro</th> <th>F1_macro</th> </tr>

<!--<tr>
    #<td>Random Forest</td> <td>0.8072</td> <td>0.6921</td> <td>0.6958</td> <td class="red-text">0.6935</#td>
</tr>-->

<tr> <td>XGBoost</td> <td>0.8087</td> <td>0.75</td> <td>0.66</td> <td class="red-text">0.69</td> </tr>
</table>
""", unsafe_allow_html=True)
     
    st.markdown("<br><br>", unsafe_allow_html=True)

    # ---- RANDOM FOREST EXPANDER ----

    #st.markdown("""
#<style>
#/* Table Layout */
#.conf-table {
#border-collapse: collapse; margin: 20px 0; width: 100%; font-size: 18px; text-align: center;}

#/* Top Bar (Header) */
#.conf-table thead th {background-color: #1f4e79; /* Deep blue */ color: white; padding: 12px; 
#font-weight: 700;
#}

#/* Actual class labels (left column) */
#.conf-table tbody th {background-color: #e8f0fb; padding: 10px; font-weight: 700;
#border-right: 1px solid #ccc;
#}

#/* Table Cells */
#.conf-table td {padding: 10px; border: 1px solid #ddd;
#}

#/* Alternate row shading */
#.conf-table tbody tr:nth-child(even) {background-color: #fafbff;}

#/* Red highlighting for correct predictions */
#.correct {color: #d60000; font-weight: 700;}
#</style>
#""", unsafe_allow_html=True)


    with st.expander("⚡ XGBoost – Confusion Matrix"):
        st.markdown("""
    
    <table class="conf-table">
    <thead>
        <tr> <th>Actual \\ Predicted</th> <th>Functional</th> <th>Needs Repair</th> <th>Non-Functional</th>
        </tr>
    </thead>

    <tbody>
        <tr> <th>Functional</th> <td class="correct">5794</td> <td>125</td> <td>533</td> </tr>
        <tr> <th>Needs Repair</th> <td>476</td> <td class="correct">262</td> <td>125</td> </tr>
        <tr> <th>Non-Functional</th> <td>958</td> <td>56</td> <td class="correct">3551</td> </tr>
    </tbody>
    </table>
    """, unsafe_allow_html=True)


    #with st.expander("🌲Random Forest – Confusion Matrix"):
       # st.markdown("""
    
    #<table class="conf-table">
    #<thead>
        #<tr>
           # <th>Actual \\ Predicted</th> <th>Functional</th> <th>Needs Repair</th> <th>Non-Functional</th>
        #</tr>
    #</thead>

    #<tbody>
        #<tr> <th>Functional</th> <td class="correct">5678</td> <td>226</td> <td>548</td> </tr>
        #<tr> <th>Needs Repair</th> <td>421</td> <td class="correct">304</td> <td>138</td> </tr>
        #<tr> <th>Non-Functional</th> <td>885</td> <td>73</td> <td class="correct">3607</td> </tr>
    #</tbody>
    #</table>
    #""", unsafe_allow_html=True)
        


elif page == "Feature Importance":

    st.markdown("<h1>Feature Importance (SHAP)</h1>", unsafe_allow_html=True)
    st.markdown("<hr class='blue-line'>", unsafe_allow_html=True)

    st.markdown("### 📊 Overall importance (all classes combined)")
    st.image("overall_classes.png", use_container_width=True)

    st.markdown("<hr class='blue-line'>", unsafe_allow_html=True)
    st.markdown("### 📈 Importance by class")

    CLASS_SHAP_IMAGES = {
    "functional": "shap_functional.png",
    "functional needs repair": "shap_functional_needs_repair.png",
    "non functional": "shap_non_functional.png",
}


# Dropdown with a hidden / neutral option
    class_options = ["(choose class …)"] + list(CLASS_SHAP_IMAGES.keys())

    selected_class = st.selectbox(
    "Show SHAP importance for:", 
    options=class_options,
    index=0,
)


    # Only show plot when a real class is chosen
    if selected_class != "(choose class …)":
        img_path = CLASS_SHAP_IMAGES[selected_class]
        st.image(img_path, use_container_width=True)
   # else:
        #st.info("Select a class from the dropdown to see its SHAP feature importance.")


elif page == "Maps":
    st.markdown("<h1>Geographic Exploration</h1>", unsafe_allow_html=True)
    st.markdown("<hr class='blue-line'>", unsafe_allow_html=True)

    # Check that required columns exist
    required_cols = {"lat_5dp", "lon_5dp", "status_pred"}
    if not required_cols.issubset(df.columns):
        missing = required_cols - set(df.columns)
        st.error(f"Missing required columns for map: {', '.join(missing)}")
    else:
        # ------------------ FILTERS ------------------
        st.markdown("### Filters")

        f1, f2, f3 = st.columns(3)

        # 1) Status filter (main requirement)
        with f1:
            status_filter = st.multiselect(
                "Pump status",
                options=STATUS_ORDER,
                default=[],
                placeholder="Choose status",
            )

        # 2) Ward filter (since you use 'ward')
        with f2:
            if "ward" in df.columns:
                ward_opts = ["All"] + sorted(df["ward"].dropna().unique().tolist())
                selected_ward = st.selectbox("Ward", ward_opts, index=0)
            else:
                selected_ward = "All"

        # 3) Pump type filter (Extraction_type)
        
        with f3:
            extract_col = "extraction_type"
            if extract_col in df.columns:
                extract_types = sorted(df[extract_col].dropna().unique())
                selected_extract = st.multiselect(
                    "Extraction type",
                    options=extract_types,
                    default=[],
                    placeholder="Choose types",
                )
            else:
                selected_extract = None

        # Extra filters row (age + population) – optional but nice
        st.markdown("<br>", unsafe_allow_html=True)
        f4, f5 = st.columns(2)

        with f4:
            if "age" in df.columns:
                min_age, max_age = int(df["age"].min()), int(df["age"].max())
                age_range = st.slider(
                    "Age (years)",
                    min_value=min_age,
                    max_value=max_age,
                    value=(min_age, max_age),
                )
            else:
                age_range = None

        with f5:
            if "lat_5dp" in df.columns:
                lat_min = float(df["lat_5dp"].min())
                lat_max = float(df["lat_5dp"].max())
                lat_range = st.slider(
                "Latitude range",
                min_value=round(lat_min, 2),
                max_value=round(lat_max, 2),
                value=(round(lat_min, 2), round(lat_max, 2)),
        )
            else:
                lat_range = None

            
        # ------------------ APPLY FILTERS ------------------
        map_df = df.copy()

        # status
        if status_filter:
            map_df = map_df[map_df["status_pred"].isin(status_filter)]

        # ward
        if selected_ward != "All" and "ward" in map_df.columns:
            map_df = map_df[map_df["ward"] == selected_ward]

        # pump type
        if selected_extract:
            map_df = map_df[map_df["extraction_type"].isin(selected_extract)]


        # age
        if age_range and "age" in map_df.columns:
            map_df = map_df[(map_df["age"] >= age_range[0]) & (map_df["age"] <= age_range[1])]

        # population log
        # latitude filter
        if lat_range and "lat_5dp" in map_df.columns:
            map_df = map_df[
            (map_df["lat_5dp"] >= lat_range[0])
            & (map_df["lat_5dp"] <= lat_range[1])
            ]
            

        # drop rows with missing/zero coords
        map_df = map_df.dropna(subset=["lat_5dp", "lon_5dp"])
        map_df = map_df[(map_df["lat_5dp"] != 0) & (map_df["lon_5dp"] != 0)]

        st.markdown("<hr class='blue-line'>", unsafe_allow_html=True)
        st.markdown("### Pump Locations by Status")

        if map_df.empty:
            st.warning("No pumps match the selected filters.")
        else:
            # ------------------ HOVER DATA ------------------
            hover_data = {}

            for c in [
                "status_actual", "ward", "age", "lat_5dp", "lon_5dp", "population_clean_log",
                "amount_tsh_nonzero_log", "installer_clean", "source_type",
                "quality_group", "extraction_type", "management","basin"
                "quantity", "waterpoint_type"
            ]:
                if c in map_df.columns:
                    hover_data[c] = True

            # Hide raw coords in hover (still used for map)
            hover_data["lat_5dp"] = True
            hover_data["lon_5dp"] = True 

            fig_map = px.scatter_mapbox(
                map_df,
                lat="lat_5dp",
                lon="lon_5dp",
                color="status_pred",
                color_discrete_map=STATUS_COLORS,
                hover_data=hover_data,
                zoom=5,
                height=600,
            )

            fig_map.update_layout(
                mapbox_style="open-street-map",
                margin=dict(l=0, r=0, t=0, b=0),
                legend_title_text="Predicted status",
            )

            st.plotly_chart(fig_map, width='stretch')

        # ------------------ SUMMARY ------------------
        st.markdown("<hr class='blue-line'>", unsafe_allow_html=True)
        st.markdown("### Status Breakdown for Filtered Pumps")

        if not map_df.empty:
            summary = (
                map_df["status_pred"]
                .value_counts()
                .reindex(STATUS_ORDER)
                .rename_axis("status")
                .reset_index(name="count")
            )
            fig_bar = px.bar(
                summary,
                x="status",
                y="count",
                color="status",
                color_discrete_map=STATUS_COLORS,
                text="count",
            )
            fig_bar.update_layout(xaxis_title="", yaxis_title="Number of pumps")
            st.plotly_chart(fig_bar, width='stretch')


           
elif page == "Prediction Interface":

    st.markdown("<h1>Predict Pump Status</h1>", unsafe_allow_html=True)
    st.markdown("<hr class='blue-line'>", unsafe_allow_html=True)

    st.markdown(
        "The model will predict whether the pump is **functional**, "
        "**functional needs repair**, or **non functional**."
    )

    st.markdown("<br>", unsafe_allow_html=True)

    # We'll use your main feature set:
    # lat_5dp, lon_5dp, ward, age, population_clean_log, amount_tsh_nonzero_log,
    # installer_clean, source_type, quality_group, extraction_type,
    # management, quantity, waterpoint_type

    col_left, col_right = st.columns(2, gap="large")

    with col_left:
        # Location & context
        lat = st.number_input("Latitude (lat_5dp)", value=float(df["lat_5dp"].median()))
        lon = st.number_input("Longitude (lon_5dp)", value=float(df["lon_5dp"].median()))

        if "ward" in df.columns:
            ward_options = sorted(df["ward"].dropna().unique())
            ward = st.selectbox("Ward", ward_options)
        else:
            ward = ""

        age = st.slider(
            "Pump age (years)",
            min_value=int(df["age"].min()),
            max_value=int(df["age"].max()),
            value=int(df["age"].median()),
        )

        # Raw population value -> we'll log-transform below
        pop_raw = st.number_input(
            "Population served (people)",
            min_value=0,
            value=int(np.expm1(df["population_clean_log"].median())) if "population_clean_log" in df.columns else 100,
        )

        amt_raw = st.number_input(
            "Amount_tsh (water amount)",
            min_value=0.0,
            value=float(np.expm1(df["amount_tsh_nonzero_log"].median())) if "amount_tsh_nonzero_log" in df.columns else 0.0,
        )

    with col_right:
        # Pump + source characteristics
        if "installer_clean" in df.columns:
            inst_options = sorted(df["installer_clean"].dropna().unique())
            installer = st.selectbox("Installer", inst_options)
        else:
            installer = ""

        if "source_type" in df.columns:
            src_options = sorted(df["source_type"].dropna().unique())
            source_type = st.selectbox("Source type", src_options)
        else:
            source_type = ""

        if "basin" in df.columns:
            basin_options = sorted(df["basin"].dropna().unique())
            basin = st.selectbox("Basin", basin_options)
        else:
            basin = ""


        if "quality_group" in df.columns:
            q_options = sorted(df["quality_group"].dropna().unique())
            quality_group = st.selectbox("Water quality group", q_options)
        else:
            quality_group = ""

        if "extraction_type" in df.columns:
            e_options = sorted(df["extraction_type"].dropna().unique())
            extraction_type = st.selectbox("Extraction type", e_options)
        else:
            extraction_type = ""

        if "management" in df.columns:
            m_options = sorted(df["management"].dropna().unique())
            management = st.selectbox("Management", m_options)
        else:
            management = ""

        if "quantity" in df.columns:
            qnty_options = sorted(df["quantity"].dropna().unique())
            quantity = st.selectbox("Quantity", qnty_options)
        else:
            quantity = ""

        if "waterpoint_type" in df.columns:
            wpt_options = sorted(df["waterpoint_type"].dropna().unique())
            waterpoint_type = st.selectbox("Waterpoint type", wpt_options)
        else:
            waterpoint_type = ""

    st.markdown("<hr class='blue-line'>", unsafe_allow_html=True)

    # Transform raw values into log features the model expects
    population_clean_log = np.log1p(pop_raw)  # log(1 + pop)
    # avoid log(0) issues
    amt_safe = max(amt_raw, 0.001)
    amount_tsh_nonzero_log = np.log1p(amt_safe)

    # Build input dict matching your training feature names
    input_features = {
        "lat_5dp": lat,
        "lon_5dp": lon,
        "ward": ward,
        "age": age,
        "population_clean_log": population_clean_log,
        "amount_tsh_nonzero_log": amount_tsh_nonzero_log,
        "basin":basin,
        "installer_clean": installer,
        "source_type": source_type,
        "quality_group": quality_group,
        "extraction_type": extraction_type,
        "management": management,
        "quantity": quantity,
        "waterpoint_type": waterpoint_type,
    }

    # Prediction button
    if st.button("🔮 Predict pump status"):
        pred_label, probs_df = predict_single(input_features)

        # Main result card
        st.markdown("<hr class='blue-line'>", unsafe_allow_html=True)
        st.markdown("### Prediction Result")

        color = STATUS_COLORS.get(pred_label, "#253449")
        st.markdown(
            f"""
<div style="
    padding: 18px 22px;
    border-radius: 10px;
    border-left: 6px solid {color};
    background-color: #F4F6F7;
">
<p style="font-size:18px; margin:0;">
Predicted status: <span style="font-weight:700; color:{color};">{pred_label}</span>
</p>
</div>
            """,
            unsafe_allow_html=True,
        )

        # Probability bar chart
        st.markdown("### Class probabilities")
        probs_df["color"] = probs_df["status"].map(STATUS_COLORS)
        fig_p = px.bar(
            probs_df,
            x="status",
            y="probability",
            color="status",
            color_discrete_map=STATUS_COLORS,
            text=probs_df["probability"].map(lambda x: f"{x:.2f}"),
        )
        fig_p.update_layout(
            xaxis_title="Status",
            yaxis_title="Probability",
            yaxis_range=[0, 1],
        )
        st.plotly_chart(fig_p, use_container_width=True)


elif page == "Conclusion":

    st.markdown("<h1>🏁 Conclusion</h1>", unsafe_allow_html=True)
    st.markdown("<hr class='blue-line'>", unsafe_allow_html=True)

    # Intro sentence
    #st.write(
        #"Pump functionality is driven primarily by "
        #"**geography and environmental conditions**, followed by "
        #"**pump type, installation quality, and usage demand**."
    #)

    # --- Key model findings card ---
    st.markdown(
        """
        <div style="
            background-color:#F4F6F7;
            padding:18px 22px;
            border-radius:12px;
            margin-top:1.2em;
        ">
          <p style="font-size:18px; font-weight:700; margin-bottom:0.5em;">
            🔍 What the model learned
          </p>
          <ul style="font-size:16px; line-height:1.6;">
            <li><b>Geographical-related features</b> (longitude, latitude, ward) together with
                <b>Technical aspects</b> (extraction type) are among the top drivers.</li>
            <li><b>Installation, usage, and temporal features</b> (age, population and installer)
                are also important signals.</li>
          </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # --- Why this matters card ---
    st.markdown(
        """
        <div style="
            background-color:#E8F4FF;
            padding:18px 22px;
            border-radius:12px;
            margin-top:1.5em;
        ">
          <p style="font-size:18px; font-weight:700; margin-bottom:0.5em;">
            🎯 Why these insights matter
          </p>
          <ul style="font-size:16px; line-height:1.6;">
            <li><b>Prioritizing maintenance</b> in high-risk regions.</li>
            <li><b>Guiding infrastructure investment</b> towards locations and pump types
                that are more resilient under local conditions.</li>
            <li><b>Flag pumps that combine high population, risky extraction types, and vulnerable locations.</li>
          </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<hr class='blue-line'>", unsafe_allow_html=True)

    # --- Limitations & next steps card ---
    st.markdown(
        """
        <div style="
            background-color:#F4F6F7;
            padding:18px 22px;
            border-radius:12px;
            margin-top:1.5em;
        ">
          <p style="font-size:18px; font-weight:700; margin-bottom:0.5em;">
            🐘 Limitations and ➡️ Next step
          </p>
          <ul style="font-size:16px; line-height:1.6;">
            <li><b>Class imbalance, false predictions, and data-quality issues</b> remain challenges.</li>
            <li><b>Future work could explore <b>more granular geographic data</b>, richer temporal usage patterns, and <b>additional socio-economic factors</b> around each pump.</li>
          </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )

elif page == "Feature Exploration":

    st.markdown("<h1>Feature Exploration</h1>", unsafe_allow_html=True)
    st.markdown("<hr class='blue-line'>", unsafe_allow_html=True)

    st.markdown(
        "Explore how different features relate to pump status. "
        )

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### Drill-down: filter by status and category")
    dcol1, dcol2 = st.columns(2)


    # ------------------ CONFIG ------------------
    target_choice = st.radio(
        "Use which status for coloring?",
        options=["Predicted status", "Actual status"],
        horizontal=True,
    )
    target_col = "status_pred" if target_choice == "Predicted status" else "status_actual"

    # define which features to offer
    numeric_features = [c for c in ["age", "population_clean_log", "amount_tsh_nonzero_log"] if c in df.columns]
    categorical_features = [
        c for c in [
            "basin", "ward", "installer_clean", "source_type", "quality_group",
            "extraction_type", "management", "quantity", "waterpoint_type"
        ] if c in df.columns
    ]

    analysis_type = st.radio(
        "What do you want to explore?",
        options=["Numeric feature vs status", "Categorical feature vs status"],
        horizontal=True,
    )

    st.markdown("<hr class='blue-line'>", unsafe_allow_html=True)

    # ------------------ NUMERIC FEATURE EXPLORATION ------------------
    if analysis_type == "Numeric feature vs status":

        if not numeric_features:
            st.warning("No numeric features available for exploration.")
        else:
            num_col = st.selectbox("Choose a numeric feature", numeric_features)

            st.markdown(f"### Distribution of `{num_col}` by {target_choice.lower()}")

            # Box plot
            fig_box = px.box(
                df,
                x=target_col,
                y=num_col,
                color=target_col,
                color_discrete_map=STATUS_COLORS,
                category_orders={target_col: STATUS_ORDER},
            )
            fig_box.update_layout(xaxis_title="", yaxis_title=num_col)
            st.plotly_chart(fig_box, use_container_width=True)

            st.markdown("<hr class='blue-line'>", unsafe_allow_html=True)
            st.markdown("### Drill-down: filter by status and value range")

            # Drill-down filters
            dcol1, dcol2 = st.columns(2)

            with dcol1:
                status_sel = st.multiselect(
                    "Filter by status",
                    options=STATUS_ORDER,
                    default=[],
                )

            with dcol2:
                min_v = float(df[num_col].min())
                max_v = float(df[num_col].max())
                v1, v2 = st.slider(
                    f"Range for {num_col}",
                    min_value=round(min_v, 2),
                    max_value=round(max_v, 2),
                    value=(round(min_v, 2), round(max_v, 2)),
                )

            drill_df = df.copy()
            if status_sel:
                drill_df = drill_df[drill_df[target_col].isin(status_sel)]
            drill_df = drill_df[(drill_df[num_col] >= v1) & (drill_df[num_col] <= v2)]

            st.markdown(
                f"Showing **{len(drill_df)}** pumps where `{num_col}` is between **{v1}** and **{v2}**."
            )

            # Summary
            if not drill_df.empty:
                st.markdown("#### Summary statistics")
                st.write(drill_df[[num_col]].describe().T)

                st.markdown("#### Sample of matching pumps")
                st.dataframe(drill_df.head(50))
                #sample = drill_df.head(50)
                #styled = style_sample_table(sample, target_col=target_col)
            else:
                st.info("No pumps match the selected filters.")

    # ------------------ CATEGORICAL FEATURE EXPLORATION ------------------
    else:  # analysis_type == "Categorical feature vs status"

        if not categorical_features:
            st.warning("No categorical features available for exploration.")
        else:
            cat_col = st.selectbox("Choose a categorical feature", categorical_features)

            st.markdown(f"### Counts of `{cat_col}` by {target_choice.lower()}")

            # Limit to top 20 categories for readability
            top_categories = df[cat_col].value_counts().head(20).index
            cat_df = df[df[cat_col].isin(top_categories)].copy()

            fig_cat = px.histogram(
                cat_df,
                x=cat_col,
                color=target_col,
                barmode="group",
                color_discrete_map=STATUS_COLORS,
                category_orders={target_col: STATUS_ORDER},
            )
            fig_cat.update_layout(
                xaxis_title=cat_col,
                yaxis_title="Number of pumps",
            )
            st.plotly_chart(fig_cat, use_container_width=True)

            st.markdown("<hr class='blue-line'>", unsafe_allow_html=True)
            st.markdown("### Drill-down: filter by status and category")

            dcol1, dcol2 = st.columns(2)

            with dcol1:
                status_sel = st.multiselect(
                    "Filter by status",
                    options=STATUS_ORDER,
                    default=[],
                    placeholder="Choose types",
                )

            with dcol2:
                cat_values = sorted(cat_df[cat_col].dropna().unique().tolist())
                cat_sel = st.multiselect(
                f"Filter {cat_col}",
                options=cat_values,
                default= [],
                #default=cat_values[:5] if len(cat_values) > 5 else cat_values,
    )

              
              

               
            drill_df = cat_df.copy()
            if status_sel:
                drill_df = drill_df[drill_df[target_col].isin(status_sel)]
            if cat_sel:
                drill_df = drill_df[drill_df[cat_col].isin(cat_sel)]

            st.markdown(
                f"Showing **{len(drill_df)}** pumps for selected {cat_col} values and status."
            )

            if not drill_df.empty:
                # show simple summary: counts by status
                st.markdown("#### Status counts in filtered subset")
                subset_counts = (
                    drill_df[target_col]
                    .value_counts()
                    .reindex(STATUS_ORDER)
                    .rename_axis("status")
                    .reset_index(name="count")
                )
                fig_sub = px.bar(
                    subset_counts,
                    x="status",
                    y="count",
                    color="status",
                    color_discrete_map=STATUS_COLORS,
                    text="count",
                )
                fig_sub.update_layout(xaxis_title="", yaxis_title="Number of pumps")
                st.plotly_chart(fig_sub, use_container_width=True)

                st.markdown("#### Sample of matching pumps")
                st.dataframe(drill_df.head(50))
                #sample = drill_df.head(50)
                #styled = style_sample_table(sample, target_col=target_col)

                #st.dataframe(styled, use_container_width=True)
            
            else:
                st.info("No pumps match the selected filters.")




                

    

    


   








    




    
