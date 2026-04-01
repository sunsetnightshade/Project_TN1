from __future__ import annotations

from dataclasses import asdict
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import streamlit as st

from pca_tools import pca_beta_alpha, pca_fit_summary
from pipeline import build_and_serialize
from standardizer import render_aligned_matrix_heatmap


ROOT_DIR = Path(__file__).resolve().parent
STORAGE_DIR = ROOT_DIR / "storage"
OUTPUTS_DIR = ROOT_DIR / "outputs"
LATEST_DIR = OUTPUTS_DIR / "latest"
ARCHIVE_DIR = OUTPUTS_DIR / "archive"


@st.cache_data(show_spinner=False)
def _load_pickle(path: str) -> pd.DataFrame:
    return pd.read_pickle(path)


@st.cache_data(show_spinner=False)
def _load_scaler_params(path: str) -> dict:
    return pd.read_pickle(path)


def _run_build(
    *,
    start_date: date,
    end_date: date,
    missing_threshold: float,
) -> dict[str, object]:
    artifacts = build_and_serialize(
        start_date=start_date,
        end_date=end_date,
        missing_threshold=missing_threshold,
        root_dir=ROOT_DIR,
    )

    paths = {k: str(v) for k, v in artifacts.paths.items()}
    return {
        "prices": artifacts.prices,
        "cleaning": artifacts.cleaning,
        "returns": artifacts.aligned_log_returns,
        "standardized": artifacts.standardization.standardized,
        "scaler_params": artifacts.standardization.scaler_params,
        "paths": paths,
    }


def _build_page() -> None:
    st.subheader("Build")

    today = date.today()
    default_end = today
    default_start = today - timedelta(days=730)

    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        start = st.date_input("Start date", value=default_start)
    with col2:
        end = st.date_input("End date", value=default_end)
    with col3:
        missing_threshold = st.number_input(
            "Zombie missing threshold (fraction)",
            min_value=0.0,
            max_value=0.5,
            value=0.05,
            step=0.01,
        )

    if start >= end:
        st.error("Start date must be before end date.")
        return

    run = st.button("Run build", type="primary")
    if not run:
        st.caption("Click “Run build” to fetch, clean, align, standardize, and save.")
        return

    with st.spinner("Fetching data and building matrix…"):
        result = _run_build(start_date=start, end_date=end, missing_threshold=missing_threshold)
        st.session_state["latest"] = result

    cleaning = result["cleaning"]
    dropped = list(cleaning.dropped_primaries)
    repl = list(cleaning.replacements)

    if dropped:
        st.warning(f"Zombie tickers dropped: {', '.join(dropped)}")
        st.info("Replacements: " + ", ".join([f"{a} → {b}" for a, b in repl]))
    else:
        st.success("No zombie tickers detected.")

    st.write("Artifacts saved:")
    st.json(result["paths"])


def _matrix_page() -> None:
    st.subheader("Matrix")

    latest = st.session_state.get("latest")
    if latest is None:
        # Try to load from disk (if user built earlier via CLI)
        current = STORAGE_DIR / "current_matrix.pkl"
        if current.exists():
            standardized = _load_pickle(str(current))
            st.info("Loaded `storage/current_matrix.pkl` from disk.")
        else:
            st.warning("No matrix loaded yet. Go to Build and run the pipeline.")
            return
    else:
        standardized = latest["standardized"]

    st.caption("Heatmap is 30×T (tickers × days), rendered as a single continuous image.")
    fig = render_aligned_matrix_heatmap(standardized, heatmap_path=None)
    st.pyplot(fig, clear_figure=True, use_container_width=True)

    with st.expander("Alignment sanity check (US shift after log-returns)", expanded=False):
        st.markdown(
            "- **Rule**: compute log returns for *all* tickers first, then shift **US tickers** by 1 row, then `dropna()`.\n"
            "- **What to look for**: US series should be moved one day later relative to India, so the US move from last night lines up with India’s next morning date."
        )

        # Heuristic picks: one US + one India (if present)
        us_candidates = [c for c in standardized.columns if not c.endswith(".NS")]
        in_candidates = [c for c in standardized.columns if c.endswith(".NS")]
        us_pick = st.selectbox("US ticker (shifted)", options=us_candidates, index=0)
        in_pick = st.selectbox("India ticker (unshifted)", options=in_candidates, index=0)

        # Use the non-standardized aligned returns if available to avoid z-score confusion
        if latest is not None:
            aligned_returns = latest["returns"]
        else:
            # fallback: show standardized (still time-aligned, but scaled)
            aligned_returns = standardized

        preview = aligned_returns[[in_pick, us_pick]].tail(10)
        st.dataframe(preview)

    st.markdown("**Preview (standardized, T×30)**")
    st.dataframe(standardized.tail(15))

    # Downloads (30×T orientation)
    csv_30xt = standardized.T.copy()
    if isinstance(csv_30xt.columns, pd.DatetimeIndex):
        csv_30xt.columns = csv_30xt.columns.strftime("%Y-%m-%d")
    st.download_button(
        "Download standardized matrix (30×T CSV)",
        data=csv_30xt.to_csv(index=True).encode("utf-8"),
        file_name="standardized_matrix_30xT.csv",
        mime="text/csv",
    )


def _pca_page() -> None:
    st.subheader("PCA (Beta vs Alpha)")

    latest = st.session_state.get("latest")
    if latest is None:
        current = STORAGE_DIR / "current_matrix.pkl"
        if current.exists():
            standardized = _load_pickle(str(current))
            st.info("Loaded `storage/current_matrix.pkl` from disk.")
        else:
            st.warning("No matrix loaded yet. Go to Build and run the pipeline.")
            return
    else:
        standardized = latest["standardized"]

    k = st.slider("Number of components to treat as Beta (k)", min_value=1, max_value=10, value=1)
    summary = pca_fit_summary(standardized, n_components=min(k, standardized.shape[1]))

    col1, col2 = st.columns([1, 1])
    with col1:
        st.markdown("**Explained variance ratio**")
        st.dataframe(summary["explained_variance_ratio"])
    with col2:
        st.markdown("**Cumulative explained variance**")
        st.dataframe(summary["cumulative_explained_variance"])

    beta, alpha, pca = pca_beta_alpha(standardized, k=k)

    st.markdown("**Beta component (30×T heatmap)**")
    fig_b = render_aligned_matrix_heatmap(beta, heatmap_path=None, title=f"Beta (first {k} PC(s))")
    st.pyplot(fig_b, clear_figure=True, use_container_width=True)

    st.markdown("**Alpha residual (30×T heatmap)**")
    fig_a = render_aligned_matrix_heatmap(alpha, heatmap_path=None, title="Alpha residual (standardized − beta)")
    st.pyplot(fig_a, clear_figure=True, use_container_width=True)

    st.download_button(
        "Download alpha residual (30×T CSV)",
        data=alpha.T.to_csv(index=True).encode("utf-8"),
        file_name="alpha_residual_30xT.csv",
        mime="text/csv",
    )
    st.download_button(
        "Download beta component (30×T CSV)",
        data=beta.T.to_csv(index=True).encode("utf-8"),
        file_name="beta_component_30xT.csv",
        mime="text/csv",
    )

    st.caption("PCA model details (debug):")
    st.json(
        {
            "n_components": int(pca.n_components_),
            "k_beta": int(k),
            "feature_names": list(standardized.columns),
        }
    )


def main() -> None:
    st.set_page_config(page_title="Quant Matrix (NIFTY + NASDAQ)", layout="wide")
    st.title("Quant Matrix: NIFTY IT + NASDAQ (Aligned, Standardized) + PCA")

    with st.sidebar:
        st.header("Navigation")
        page = st.radio("Go to", options=["Build", "Matrix", "PCA"], index=0)

        st.divider()
        st.caption("Outputs")
        st.code(str(ROOT_DIR), language="text")
        st.caption("Saved heatmap (latest):")
        st.code(str(LATEST_DIR / "matrix_heatmap.png"), language="text")

    if page == "Build":
        _build_page()
    elif page == "Matrix":
        _matrix_page()
    else:
        _pca_page()


if __name__ == "__main__":
    main()

