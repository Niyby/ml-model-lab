
import hashlib

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler


# --------------------------------------------------
# Configuration and default cleaning template
# --------------------------------------------------

st.set_page_config(
    page_title="ML Model Lab",
    layout="wide",
)

DEFAULT_CLEANING_CODE = '''# Default cleaning template
# The original uploaded dataframe is available as "df".
# Store your result in "cleaned_df".

cleaned_df = df.copy()

# Remove duplicate rows
cleaned_df = cleaned_df.drop_duplicates()

# Fill missing numerical values with the median
for col in cleaned_df.select_dtypes(include="number").columns:
    median_value = cleaned_df[col].median()
    if pd.notna(median_value):
        cleaned_df[col] = cleaned_df[col].fillna(median_value)

# Fill missing categorical values with the mode
for col in cleaned_df.select_dtypes(
    include=["object", "category", "string"]
).columns:
    mode = cleaned_df[col].mode(dropna=True)
    if not mode.empty:
        cleaned_df[col] = cleaned_df[col].fillna(mode.iloc[0])
'''


# --------------------------------------------------
# Session state
# --------------------------------------------------

defaults = {
    "original_df": None,
    "cleaned_df": None,
    "source_hash": None,
    "cleaning_editor": DEFAULT_CLEANING_CODE,
    "cleaning_applied": False,
    "model_results": None,
    "model_details": None,
    "experiment_history": [],
}

for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value


# --------------------------------------------------
# Helpers
# --------------------------------------------------

def get_active_df(use_cleaned=False):
    """Return the selected version of the dataset."""
    if use_cleaned and st.session_state.cleaned_df is not None:
        return st.session_state.cleaned_df

    return st.session_state.original_df


def show_dataset_metrics(df):
    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Rows", f"{len(df):,}")
    c2.metric("Columns", len(df.columns))
    c3.metric("Missing cells", f"{int(df.isna().sum().sum()):,}")
    c4.metric("Duplicate rows", f"{int(df.duplicated().sum()):,}")


def show_missing_summary(df):
    missing = pd.DataFrame({
        "Column": df.columns,
        "Missing": df.isna().sum().values,
        "Missing (%)": (df.isna().mean().values * 100).round(2),
    })

    missing = missing[missing["Missing"] > 0]

    if missing.empty:
        st.success("No missing values found.")
    else:
        st.warning(
            f"{int(df.isna().sum().sum()):,} missing cells "
            f"across {len(missing)} columns."
        )
        st.dataframe(
            missing,
            hide_index=True,
            use_container_width=True,
        )


def apply_filters(df, key_prefix):
    """Apply interactive filters without modifying the source dataset."""
    filtered = df.copy()

    st.subheader("Interactive Filters")

    numeric_cols = filtered.select_dtypes(
        include="number"
    ).columns.tolist()

    category_cols = [
        col for col in filtered.columns
        if filtered[col].nunique(dropna=True) <= 50
        and col not in numeric_cols
    ]

    if numeric_cols:
        selected_numeric = st.selectbox(
            "Filter numerical column",
            ["None"] + numeric_cols,
            key=f"{key_prefix}_numeric",
        )

        if selected_numeric != "None":
            series = filtered[selected_numeric].dropna()

            if not series.empty:
                low = float(series.min())
                high = float(series.max())

                if low < high:
                    selected_range = st.slider(
                        f"Range for {selected_numeric}",
                        min_value=low,
                        max_value=high,
                        value=(low, high),
                        key=f"{key_prefix}_range",
                    )

                    filtered = filtered[
                        filtered[selected_numeric].between(
                            selected_range[0],
                            selected_range[1],
                        )
                    ]

    if category_cols:
        selected_category = st.selectbox(
            "Filter categorical column",
            ["None"] + category_cols,
            key=f"{key_prefix}_category",
        )

        if selected_category != "None":
            choices = filtered[selected_category].dropna().unique().tolist()

            selected_values = st.multiselect(
                f"Keep values in {selected_category}",
                options=choices,
                default=choices,
                key=f"{key_prefix}_values",
            )

            filtered = filtered[
                filtered[selected_category].isin(selected_values)
            ]

    st.caption(
        f"Showing {len(filtered):,} of {len(df):,} rows after filtering."
    )

    return filtered


# --------------------------------------------------
# Sidebar navigation
# --------------------------------------------------

with st.sidebar:
    st.header("Navigation")

    workspace = st.selectbox(
        "Workspace",
        ["ML Model Lab", "LLM Experiments"],
        key="workspace",
    )

    if workspace == "ML Model Lab":
        page = st.radio(
            "ML Model Lab",
            [
                "Overview",
                "Data Cleaning",
                "Visualizations",
                "Train Models",
                "Model Evaluation",
                "Experiment History",
            ],
            key="ml_page",
        )
    else:
        page = "LLM Lab"

    st.divider()


# --------------------------------------------------
# Upload panel and missing-value summary
# --------------------------------------------------

upload_col, missing_col = st.columns([1.1, 1])

with upload_col:
    st.subheader("Dataset")
    uploaded_file = st.file_uploader(
        "Upload a CSV file",
        type=["csv"],
        help="Your uploaded data is processed by this local Streamlit app.",
    )

    if uploaded_file is not None:
        file_bytes = uploaded_file.getvalue()
        current_hash = hashlib.sha256(file_bytes).hexdigest()

        if current_hash != st.session_state.source_hash:
            try:
                new_df = pd.read_csv(
                    uploaded_file,
                    low_memory=False,
                )

                if new_df.empty:
                    st.error("This CSV contains no data rows.")
                elif len(new_df.columns) < 1:
                    st.error("No columns were found.")
                else:
                    st.session_state.original_df = new_df.copy(deep=True)
                    st.session_state.cleaned_df = None
                    st.session_state.source_hash = current_hash
                    st.session_state.cleaning_applied = False
                    st.session_state.cleaning_editor = (
                        DEFAULT_CLEANING_CODE
                    )
                    st.session_state.model_results = None
                    st.session_state.model_details = None
                    st.success("Dataset loaded successfully!")

            except Exception as error:
                st.error(f"Could not read the CSV: {error}")

with missing_col:
    st.subheader("Missing-Value Check")

    if st.session_state.original_df is not None:
        show_missing_summary(st.session_state.original_df)
    else:
        st.info("Upload a CSV to inspect missing values.")

if st.session_state.original_df is None:
    st.divider()
    st.subheader("Welcome to ML Model Lab")

    st.write(
        "Start by uploading a CSV dataset. You can then inspect its "
        "columns, clean a working copy, explore interactive charts, "
        "and train classification models."
    )

    st.info(
        "The original dataset is kept separately from the cleaned "
        "dataset. Uploading a new file resets the current cleaning "
        "result and model evaluation."
    )

    st.stop()


# --------------------------------------------------
# Dataset selected for downstream work
# --------------------------------------------------

st.divider()

use_cleaned = False

if st.session_state.cleaned_df is not None:
    use_cleaned = st.toggle(
        "Use cleaned dataset for this page",
        value=True,
        key=f"use_cleaned_{page}",
    )

active_df = get_active_df(use_cleaned)

st.caption(
    "Current data: "
    + ("Cleaned dataset" if use_cleaned else "Original dataset")
)


# --------------------------------------------------
# Overview
# --------------------------------------------------

if page == "Overview":
    st.header("Dataset Overview")
    show_dataset_metrics(st.session_state.original_df)

    st.subheader("Preview")
    st.dataframe(
        active_df.head(20),
        use_container_width=True,
        hide_index=True,
    )

    st.subheader("Column Information")

    info_df = pd.DataFrame({
        "Column": active_df.columns,
        "Data Type": active_df.dtypes.astype(str).values,
        "Missing Values": active_df.isna().sum().values,
        "Unique Values": [
            active_df[col].nunique(dropna=True)
            for col in active_df.columns
        ],
    })

    st.dataframe(
        info_df,
        use_container_width=True,
        hide_index=True,
    )

    st.subheader("Numerical Statistics")

    numeric_stats = active_df.describe().T

    if numeric_stats.empty:
        st.info("No numerical columns are available.")
    else:
        st.dataframe(numeric_stats, use_container_width=True)


# --------------------------------------------------
# Data Cleaning
# --------------------------------------------------

elif page == "Data Cleaning":
    st.header("Data Cleaning Workspace")

    original = st.session_state.original_df

    st.write(
        "Edit the cleaning script below. The variable `df` contains "
        "a fresh copy of your original uploaded dataset. Your script "
        "must assign the final result to `cleaned_df`."
    )

    before_col, after_col = st.columns(2)

    with before_col:
        st.subheader("Before Cleaning")
        show_dataset_metrics(original)

    with after_col:
        st.subheader("Current Cleaning Result")

        if st.session_state.cleaned_df is None:
            st.info("No cleaned dataset has been applied yet.")
        else:
            show_dataset_metrics(st.session_state.cleaned_df)

    st.subheader("Editable Python Cleaning Script")

    st.caption(
        "The default template remains preserved. Your edits are "
        "temporary and can be reset."
    )

    st.text_area(
        "Cleaning code",
        key="cleaning_editor",
        height=360,
        help=(
            "Use pandas and NumPy. Input: df. "
            "Required output: cleaned_df."
        ),
        label_visibility="collapsed",
    )

    apply_col, reset_col = st.columns(2)

    with apply_col:
        apply_clicked = st.button(
            "Apply Cleaning Code",
            type="primary",
            use_container_width=True,
        )

    with reset_col:
        reset_clicked = st.button(
            "Restore Default Code",
            use_container_width=True,
        )

    if reset_clicked:
        st.session_state.cleaning_editor = DEFAULT_CLEANING_CODE
        st.rerun()

    if apply_clicked:
        try:
            # Custom Python execution is intended only for trusted,
            # locally authored code. It is NOT a security sandbox.
            working_scope = {
                "df": original.copy(deep=True),
                "pd": pd,
                "np": np,
            }

            exec(
                st.session_state.cleaning_editor,
                {"pd": pd, "np": np},
                working_scope,
            )

            result = working_scope.get("cleaned_df")

            if not isinstance(result, pd.DataFrame):
                raise ValueError(
                    "Your script must create a pandas DataFrame "
                    "named cleaned_df."
                )

            if result.empty:
                raise ValueError(
                    "The cleaning script returned an empty dataset."
                )

            st.session_state.cleaned_df = result.copy(deep=True)
            st.session_state.cleaning_applied = True
            st.success("Cleaning applied to a separate dataset.")

        except Exception as error:
            st.error(f"Cleaning failed: {error}")

    if st.session_state.cleaned_df is not None:
        st.divider()
        st.subheader("Before vs After")

        before = original
        after = st.session_state.cleaned_df

        comparison = pd.DataFrame({
            "Measure": [
                "Rows",
                "Columns",
                "Missing cells",
                "Duplicate rows",
            ],
            "Before": [
                len(before),
                len(before.columns),
                int(before.isna().sum().sum()),
                int(before.duplicated().sum()),
            ],
            "After": [
                len(after),
                len(after.columns),
                int(after.isna().sum().sum()),
                int(after.duplicated().sum()),
            ],
        })

        st.dataframe(
            comparison,
            use_container_width=True,
            hide_index=True,
        )

        st.subheader("Cleaned Dataset Preview")
        st.dataframe(
            after.head(30),
            use_container_width=True,
            hide_index=True,
        )

        st.download_button(
            "Download Cleaned CSV",
            data=after.to_csv(index=False).encode("utf-8"),
            file_name="cleaned_dataset.csv",
            mime="text/csv",
        )


# --------------------------------------------------
# Interactive Visualizations
# --------------------------------------------------


elif page == "Visualizations":
    st.header("Visualization Studio")
    st.caption(
        "Explore distributions, compare categories, inspect relationships, "
        "and identify data quality issues."
    )

    chart_df = apply_filters(active_df, "viz").copy()

    if chart_df.empty:
        st.warning("No rows remain after filtering.")

    else:
        import pandas.api.types as ptypes

        # Identify column types
        numeric_cols = [
            col for col in chart_df.columns
            if ptypes.is_numeric_dtype(chart_df[col])
            and not ptypes.is_bool_dtype(chart_df[col])
        ]

        datetime_cols = [
            col for col in chart_df.columns
            if ptypes.is_datetime64_any_dtype(chart_df[col])
        ]

        # Include text, category, Boolean, and low-cardinality
        # numeric columns as possible categorical dimensions.
        categorical_cols = [
            col for col in chart_df.columns
            if col not in datetime_cols
            and (
                not ptypes.is_numeric_dtype(chart_df[col])
                or ptypes.is_bool_dtype(chart_df[col])
                or chart_df[col].nunique(dropna=True) <= 20
            )
        ]

        chart_type = st.selectbox(
            "Chart type",
            [
                "Categorical Distribution",
                "Grouped Comparison",
                "Box Plot",
                "Violin Plot",
                "Histogram",
                "Scatter Plot",
                "Line Chart",
                "Density Heatmap",
                "Correlation Heatmap",
                "Pie Chart",
                "Missing Values",
            ],
            key="viz_chart_type",
        )

        fig = None

        # Shared helper: prepare category labels and group the
        # less frequent categories into Other when necessary.
        def prepare_categories(series, top_n, include_missing):
            labels = series.astype("string")

            if include_missing:
                labels = labels.fillna("(Missing)")
            else:
                labels = labels[labels.notna()]

            labels = labels.astype(str)
            counts = labels.value_counts()

            if len(counts) > top_n:
                keep = counts.head(top_n).index
                labels = labels.where(labels.isin(keep), "Other")

            return labels

        # --------------------------------------------------
        # 1. CATEGORICAL DISTRIBUTION
        # --------------------------------------------------
        if chart_type == "Categorical Distribution":
            if categorical_cols:
                category_col = st.selectbox(
                    "Categorical column",
                    categorical_cols,
                    key="viz_dist_category",
                )

                col1, col2 = st.columns(2)

                with col1:
                    display_mode = st.selectbox(
                        "Measure",
                        ["Count", "Percentage"],
                        key="viz_dist_measure",
                    )

                with col2:
                    top_n = st.selectbox(
                        "Maximum categories",
                        [5, 10, 15, 20, 30],
                        index=2,
                        key="viz_dist_top_n",
                    )

                sort_order = st.selectbox(
                    "Sort categories",
                    [
                        "Frequency (descending)",
                        "Frequency (ascending)",
                        "Alphabetical",
                    ],
                    key="viz_dist_sort",
                )

                include_missing = st.checkbox(
                    "Include missing values as a category",
                    value=True,
                    key="viz_dist_missing",
                )

                labels = prepare_categories(
                    chart_df[category_col],
                    top_n,
                    include_missing,
                )

                summary = labels.value_counts().rename_axis(
                    "Category"
                ).reset_index(name="Count")

                if display_mode == "Percentage":
                    summary["Value"] = (
                        summary["Count"] / summary["Count"].sum() * 100
                    )
                    value_col = "Value"
                    axis_title = "Percentage (%)"
                else:
                    value_col = "Count"
                    axis_title = "Number of rows"

                if sort_order == "Frequency (descending)":
                    summary = summary.sort_values(
                        value_col, ascending=False
                    )
                elif sort_order == "Frequency (ascending)":
                    summary = summary.sort_values(
                        value_col, ascending=True
                    )
                else:
                    summary = summary.sort_values(
                        "Category", key=lambda s: s.str.lower()
                    )

                fig = px.bar(
                    summary,
                    x=value_col,
                    y="Category",
                    orientation="h",
                    text=value_col,
                    title=f"{display_mode} by {category_col}",
                    labels={
                        value_col: axis_title,
                        "Category": category_col,
                    },
                )

                fig.update_layout(
                    yaxis={"categoryorder": "array",
                           "categoryarray": summary["Category"].tolist()},
                    height=max(400, min(800, 30 * len(summary) + 150)),
                )

                if display_mode == "Percentage":
                    fig.update_traces(
                        texttemplate="%{x:.1f}%",
                        textposition="outside",
                    )
                else:
                    fig.update_traces(textposition="outside")

            else:
                st.info("No suitable categorical columns were found.")

        # --------------------------------------------------
        # 2. GROUPED COMPARISON
        # --------------------------------------------------
        elif chart_type == "Grouped Comparison":
            if categorical_cols and numeric_cols:
                category_col = st.selectbox(
                    "Group by",
                    categorical_cols,
                    key="viz_group_category",
                )

                numeric_col = st.selectbox(
                    "Numerical measure",
                    numeric_cols,
                    key="viz_group_numeric",
                )

                aggregation = st.selectbox(
                    "Aggregation",
                    ["Mean", "Median", "Sum", "Count"],
                    key="viz_group_aggregation",
                )

                top_n = st.selectbox(
                    "Maximum categories",
                    [5, 10, 15, 20, 30],
                    index=2,
                    key="viz_group_top_n",
                )

                labels = prepare_categories(
                    chart_df[category_col], top_n, True
                )

                temp = pd.DataFrame({
                    "Category": labels,
                    "Measure": chart_df[numeric_col].to_numpy(),
                })

                if aggregation == "Mean":
                    summary = temp.groupby(
                        "Category", as_index=False
                    )["Measure"].mean()
                elif aggregation == "Median":
                    summary = temp.groupby(
                        "Category", as_index=False
                    )["Measure"].median()
                elif aggregation == "Sum":
                    summary = temp.groupby(
                        "Category", as_index=False
                    )["Measure"].sum(min_count=1)
                else:
                    summary = temp.groupby(
                        "Category", as_index=False
                    )["Measure"].count()

                summary = summary.sort_values(
                    "Measure", ascending=False
                )

                fig = px.bar(
                    summary,
                    x="Category",
                    y="Measure",
                    title=f"{aggregation} of {numeric_col} by {category_col}",
                    labels={"Measure": aggregation, "Category": category_col},
                )

                fig.update_layout(
                    xaxis_tickangle=-35,
                    height=500,
                )
            else:
                st.info(
                    "This chart requires at least one categorical "
                    "column and one numerical column."
                )

        # --------------------------------------------------
        # 3. BOXPLOT / VIOLIN
        # --------------------------------------------------
        elif chart_type in ["Box Plot", "Violin Plot"]:
            if numeric_cols:
                y = st.selectbox(
                    "Numerical column",
                    numeric_cols,
                    key="viz_distribution_numeric",
                )

                group_options = ["None"] + categorical_cols
                group_by = st.selectbox(
                    "Group by (optional)",
                    group_options,
                    key="viz_distribution_group",
                )

                if chart_type == "Box Plot":
                    fig = px.box(
                        chart_df,
                        x=None if group_by == "None" else group_by,
                        y=y,
                        points="outliers",
                        title=f"Distribution of {y}",
                    )
                else:
                    fig = px.violin(
                        chart_df,
                        x=None if group_by == "None" else group_by,
                        y=y,
                        box=True,
                        points="outliers",
                        title=f"Distribution of {y}",
                    )

                if group_by != "None":
                    fig.update_layout(xaxis_tickangle=-35)
            else:
                st.info("This chart requires a numerical column.")

        # --------------------------------------------------
        # 4. HISTOGRAM
        # --------------------------------------------------
        elif chart_type == "Histogram":
            if numeric_cols:
                x = st.selectbox(
                    "Numerical column",
                    numeric_cols,
                    key="viz_hist_numeric",
                )

                bins = st.slider(
                    "Number of bins",
                    min_value=5,
                    max_value=100,
                    value=30,
                    key="viz_hist_bins",
                )

                color_options = [
                    col for col in categorical_cols
                    if chart_df[col].nunique(dropna=True) <= 20
                ]

                color = st.selectbox(
                    "Color by (optional)",
                    ["None"] + color_options,
                    key="viz_hist_color",
                )

                fig = px.histogram(
                    chart_df,
                    x=x,
                    nbins=bins,
                    color=None if color == "None" else color,
                    marginal="box",
                    title=f"Distribution of {x}",
                )
            else:
                st.info("This chart requires a numerical column.")

        # --------------------------------------------------
        # 5. SCATTER PLOT
        # --------------------------------------------------
        elif chart_type == "Scatter Plot":
            if len(numeric_cols) >= 2:
                col1, col2 = st.columns(2)

                with col1:
                    x = st.selectbox(
                        "X-axis",
                        numeric_cols,
                        key="viz_scatter_x",
                    )

                with col2:
                    y_options = [
                        col for col in numeric_cols if col != x
                    ]
                    y = st.selectbox(
                        "Y-axis",
                        y_options,
                        key="viz_scatter_y",
                    )

                color_options = [
                    col for col in categorical_cols
                    if chart_df[col].nunique(dropna=True) <= 20
                ]

                color = st.selectbox(
                    "Color by (optional)",
                    ["None"] + color_options,
                    key="viz_scatter_color",
                )

                fig = px.scatter(
                    chart_df,
                    x=x,
                    y=y,
                    color=None if color == "None" else color,
                    title=f"{y} vs {x}",
                )
            else:
                st.info("Scatter plots require two numerical columns.")

        # --------------------------------------------------
        # 6. LINE CHART
        # --------------------------------------------------
        elif chart_type == "Line Chart":
            x_options = numeric_cols + [
                col for col in datetime_cols if col not in numeric_cols
            ]

            if x_options and numeric_cols:
                x = st.selectbox(
                    "X-axis (numeric or date)",
                    x_options,
                    key="viz_line_x",
                )

                y = st.selectbox(
                    "Y-axis (numerical)",
                    numeric_cols,
                    key="viz_line_y",
                )

                if x == y:
                    st.warning("Choose different columns for the X and Y axes.")
                else:
                    line_df = chart_df[[x, y]].dropna().sort_values(x)

                    fig = px.line(
                        line_df,
                        x=x,
                        y=y,
                        title=f"{y} over {x}",
                    )
            else:
                st.info(
                    "A line chart requires a numerical Y-axis and "
                    "a numerical or date X-axis."
                )

        # --------------------------------------------------
        # 7. DENSITY HEATMAP
        # --------------------------------------------------
        elif chart_type == "Density Heatmap":
            if len(numeric_cols) >= 2:
                col1, col2 = st.columns(2)

                with col1:
                    x = st.selectbox(
                        "X-axis",
                        numeric_cols,
                        key="viz_density_x",
                    )

                with col2:
                    y = st.selectbox(
                        "Y-axis",
                        [col for col in numeric_cols if col != x],
                        key="viz_density_y",
                    )

                fig = px.density_heatmap(
                    chart_df,
                    x=x,
                    y=y,
                    title=f"Density of {y} vs {x}",
                )
            else:
                st.info("This chart requires two numerical columns.")

        # --------------------------------------------------
        # 8. CORRELATION HEATMAP
        # --------------------------------------------------
        elif chart_type == "Correlation Heatmap":
            if len(numeric_cols) >= 2:
                corr = chart_df[numeric_cols].corr()

                fig = px.imshow(
                    corr,
                    text_auto=".2f",
                    aspect="auto",
                    zmin=-1,
                    zmax=1,
                    color_continuous_scale="RdBu",
                    title="Feature Correlation Heatmap",
                )
            else:
                st.info(
                    "Correlation heatmaps require at least two numerical columns."
                )

        # --------------------------------------------------
        # 9. PIE CHART
        # --------------------------------------------------
        elif chart_type == "Pie Chart":
            if categorical_cols:
                category_col = st.selectbox(
                    "Category",
                    categorical_cols,
                    key="viz_pie_category",
                )

                top_n = st.selectbox(
                    "Maximum slices (including Other)",
                    [3, 5, 8, 10],
                    index=2,
                    key="viz_pie_top_n",
                )

                include_missing = st.checkbox(
                    "Include missing values",
                    value=True,
                    key="viz_pie_missing",
                )

                labels = prepare_categories(
                    chart_df[category_col],
                    top_n,
                    include_missing,
                )

                summary = labels.value_counts().rename_axis(
                    "Category"
                ).reset_index(name="Count")

                fig = px.pie(
                    summary,
                    names="Category",
                    values="Count",
                    title=f"Distribution of {category_col}",
                    hole=0.3,
                )

                fig.update_traces(textposition="inside",
                                  textinfo="percent+label")
            else:
                st.info("No suitable categorical columns were found.")

        # --------------------------------------------------
        # 10. MISSING VALUES
        # --------------------------------------------------
        elif chart_type == "Missing Values":
            missing = (
                chart_df.isna().sum()
                .rename_axis("Column")
                .reset_index(name="Missing Cells")
            )

            missing = missing[missing["Missing Cells"] > 0]

            if not missing.empty:
                missing = missing.sort_values(
                    "Missing Cells", ascending=False
                )

                fig = px.bar(
                    missing,
                    x="Missing Cells",
                    y="Column",
                    orientation="h",
                    text="Missing Cells",
                    title="Missing Values by Column",
                )

                fig.update_layout(
                    height=max(350, min(800, 30 * len(missing) + 120))
                )
                fig.update_traces(textposition="outside")
            else:
                st.success("There are no missing values in the filtered dataset.")

        # --------------------------------------------------
        # RENDER CHART
        # --------------------------------------------------
        if fig is not None:
            fig.update_layout(
                template="plotly_white",
                margin=dict(l=30, r=30, t=70, b=50),
            )

            st.plotly_chart(
                fig,
                use_container_width=True,
                key=f"viz_plot_{chart_type}",
            )

            with st.expander("View chart data"):
                st.dataframe(
                    chart_df,
                    use_container_width=True,
                    hide_index=True,
                )

# --------------------------------------------------
# Model Training
# --------------------------------------------------

elif page == "Train Models":
    st.header("Train and Compare Classification Models")

    df = active_df

    if len(df.columns) < 2:
        st.warning("At least two columns are required.")
        st.stop()

    target = st.selectbox("Prediction target", df.columns)
    test_pct = st.slider("Test set size (%)", 10, 40, 20, 5)
    seed = st.number_input(
        "Random seed",
        min_value=0,
        max_value=9999,
        value=42,
    )

    selected_models = st.multiselect(
        "Models to train",
        ["Logistic Regression", "Random Forest"],
        default=["Logistic Regression", "Random Forest"],
    )

    if st.button("Train Models", type="primary"):
        try:
            data = df.dropna(subset=[target]).copy()

            if len(data) < 10:
                raise ValueError("At least 10 usable rows are required.")

            X = data.drop(columns=[target])
            y_raw = data[target].astype(str)

            if X.empty or y_raw.nunique() < 2:
                raise ValueError(
                    "Select a target with at least two classes "
                    "and one feature column."
                )

            encoder = LabelEncoder()
            y = encoder.fit_transform(y_raw)

            counts = pd.Series(y).value_counts()

            if counts.min() < 2:
                raise ValueError(
                    "Every target class needs at least two rows."
                )

            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                test_size=test_pct / 100,
                random_state=int(seed),
                stratify=y,
            )

            numeric = X.select_dtypes(include="number").columns.tolist()
            categorical = [c for c in X.columns if c not in numeric]

            transformers = []

            if numeric:
                transformers.append((
                    "numeric",
                    Pipeline([
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]),
                    numeric,
                ))

            if categorical:
                transformers.append((
                    "categorical",
                    Pipeline([
                        (
                            "imputer",
                            SimpleImputer(strategy="most_frequent"),
                        ),
                        (
                            "encoder",
                            OneHotEncoder(handle_unknown="ignore"),
                        ),
                    ]),
                    categorical,
                ))

            if not transformers:
                raise ValueError("No usable features were found.")

            models = {
                "Logistic Regression": LogisticRegression(max_iter=1000),
                "Random Forest": RandomForestClassifier(
                    n_estimators=100,
                    random_state=int(seed),
                    class_weight="balanced",
                    n_jobs=-1,
                ),
            }

            if not selected_models:
                raise ValueError("Select at least one model.")

            rows = []
            details = {}

            with st.spinner("Training selected models..."):
                for name in selected_models:
                    pipeline = Pipeline([
                        (
                            "preprocessor",
                            ColumnTransformer(transformers=transformers),
                        ),
                        ("model", models[name]),
                    ])

                    pipeline.fit(X_train, y_train)
                    pred = pipeline.predict(X_test)

                    scores = {
                        "Model": name,
                        "Accuracy": accuracy_score(y_test, pred),
                        "Precision": precision_score(
                            y_test, pred,
                            average="weighted",
                            zero_division=0,
                        ),
                        "Recall": recall_score(
                            y_test, pred,
                            average="weighted",
                            zero_division=0,
                        ),
                        "F1-score": f1_score(
                            y_test, pred,
                            average="weighted",
                            zero_division=0,
                        ),
                    }

                    rows.append(scores)

                    details[name] = {
                        "pipeline": pipeline,
                        "predictions": pred,
                        "y_test": y_test,
                        "classes": encoder.classes_,
                        "feature_names": X.columns.tolist(),
                    }

            results = pd.DataFrame(rows)

            st.session_state.model_results = results
            st.session_state.model_details = details

            history_entry = {
                "Dataset rows": len(data),
                "Target": target,
                "Models": ", ".join(selected_models),
                "Test size (%)": test_pct,
                "Random seed": int(seed),
                "Best F1-score": float(results["F1-score"].max()),
            }

            st.session_state.experiment_history.append(history_entry)

            st.success("Training completed.")

        except Exception as error:
            st.error(f"Training failed: {error}")

    if st.session_state.model_results is not None:
        st.subheader("Model Comparison")

        results = st.session_state.model_results
        st.dataframe(
            results.style.format({
                "Accuracy": "{:.3f}",
                "Precision": "{:.3f}",
                "Recall": "{:.3f}",
                "F1-score": "{:.3f}",
            }),
            use_container_width=True,
            hide_index=True,
        )

        fig = px.bar(
            results.melt(
                id_vars="Model",
                var_name="Metric",
                value_name="Score",
            ),
            x="Model",
            y="Score",
            color="Metric",
            barmode="group",
            title="Model Performance Comparison",
            range_y=[0, 1],
        )

        st.plotly_chart(fig, use_container_width=True)


# --------------------------------------------------
# Model Evaluation
# --------------------------------------------------

elif page == "Model Evaluation":
    st.header("Model Evaluation")

    details = st.session_state.model_details

    if not details:
        st.info("Train at least one model to see evaluation results.")
    else:
        name = st.selectbox("Choose trained model", list(details.keys()))
        result = details[name]

        y_test = result["y_test"]
        pred = result["predictions"]
        classes = result["classes"]

        cm = confusion_matrix(
            y_test,
            pred,
            labels=np.arange(len(classes)),
        )

        fig = px.imshow(
            cm,
            x=classes,
            y=classes,
            text_auto=True,
            aspect="auto",
            labels={
                "x": "Predicted label",
                "y": "Actual label",
                "color": "Count",
            },
            title=f"{name}: Confusion Matrix",
        )

        st.plotly_chart(fig, use_container_width=True)

        report = classification_report(
            y_test,
            pred,
            labels=np.arange(len(classes)),
            target_names=classes,
            zero_division=0,
            output_dict=True,
        )

        st.subheader("Classification Report")
        st.dataframe(
            pd.DataFrame(report).T,
            use_container_width=True,
        )


# --------------------------------------------------
# Experiment History
# --------------------------------------------------

elif page == "Experiment History":
    st.header("Experiment History")

    history = st.session_state.experiment_history

    if history:
        history_df = pd.DataFrame(history)

        st.dataframe(
            history_df,
            use_container_width=True,
            hide_index=True,
        )

        st.download_button(
            "Download Experiment History",
            data=history_df.to_csv(index=False).encode("utf-8"),
            file_name="experiment_history.csv",
            mime="text/csv",
        )

        if st.button("Clear Experiment History"):
            st.session_state.experiment_history = []
            st.rerun()
    else:
        st.info(
            "No experiments yet. Train a model to record the first run."
        )


# --------------------------------------------------
# LLM Lab placeholder
# --------------------------------------------------

elif page == "LLM Lab":
    st.header("LLM Experiment Lab")

    st.write(
        "Next milestone: compare local Ollama models, prompts, "
        "response times, memory usage, and output quality."
    )

    st.info(
        "This section is a placeholder. No LLM experiment runs here yet."
    )
