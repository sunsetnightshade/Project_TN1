"""
app.py — Quant Matrix Streamlit Dashboard
NASDAQ-100 Tech | 30 Stocks | Real-time + Batch

Pages:
  Build          — Run pipeline, archive old outputs, show results
  Matrix (30×T)  — Standardised Z-score heatmap per ticker/day
  Correlation    — 30×30 pairwise Pearson correlation heatmap
  PCA            — Beta (systematic) / Alpha (residual) decomposition
  Archive        — Browse and compare previous pipeline runs
"""
from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import streamlit as st

from pca_tools import pca_beta_alpha, pca_fit_summary
from pipeline import build_and_serialize
from standardizer import render_aligned_matrix_heatmap, render_correlation_heatmap


ROOT_DIR = Path(__file__).resolve().parent
STORAGE_DIR = ROOT_DIR / "storage"
OUTPUTS_DIR = ROOT_DIR / "outputs"
LATEST_DIR = OUTPUTS_DIR / "latest"
ARCHIVE_DIR = OUTPUTS_DIR / "archive"


# ---------------------------------------------------------------------------
# Helpers — cached loaders
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def _load_pickle(path: str) -> pd.DataFrame:
    return pd.read_pickle(path)


def _try_load_matrix() -> pd.DataFrame | None:
    """Load current matrix from disk; return None if it doesn't exist."""
    p = STORAGE_DIR / "current_matrix.pkl"
    if not p.exists():
        return None
    return _load_pickle(str(p))


def _session_or_disk(session_key: str = "latest") -> pd.DataFrame | None:
    """Return standardized matrix from session state or fallback to disk."""
    latest = st.session_state.get(session_key)
    if latest is not None:
        return latest["standardized"]
    return _try_load_matrix()


# ---------------------------------------------------------------------------
# Sidebar — persistent file status panel
# ---------------------------------------------------------------------------

def _render_sidebar() -> str:
    with st.sidebar:
        st.image(
            "https://img.shields.io/badge/NASDAQ--100-30%20Stocks-0078D4?style=flat-square&logo=data:image/svg+xml;base64,",
            use_container_width=False,
        ) if False else None  # badge placeholder — skip broken URL

        st.markdown("## 📊 Quant Matrix")
        st.caption("NASDAQ-100 Tech — 30 Stocks")
        st.divider()

        page = st.radio(
            "**Navigate**",
            options=["🔨 Build", "📈 Matrix (30×T)", "🔗 Correlation (30×30)", "🧮 PCA", "🗂 Archive"],
            index=0,
            label_visibility="visible",
        )

        st.divider()
        st.markdown("**Latest Outputs**")
        _sidebar_file_status()

        st.divider()
        st.caption(f"Storage: `storage/`")
        st.caption(f"Outputs: `outputs/latest/`")

    return page


def _sidebar_file_status() -> None:
    """Show a compact status list of latest output files with sizes."""
    files_to_check = [
        ("matrix_heatmap.png", "Matrix heatmap"),
        ("correlation_heatmap.png", "Corr. heatmap"),
        ("standardized_matrix_30xT.csv", "Standardized CSV"),
        ("aligned_log_returns_30xT.csv", "Log returns CSV"),
        ("build_metadata.json", "Build metadata"),
        ("GUIDE.md", "Usage guide"),
    ]
    any_exists = False
    for fname, label in files_to_check:
        p = LATEST_DIR / fname
        if p.exists():
            size = p.stat().st_size
            size_str = f"{size/1024:.0f} KB" if size >= 1024 else f"{size} B"
            st.markdown(f"✅ `{label}` *{size_str}*")
            any_exists = True
        else:
            st.markdown(f"⬜ `{label}`")

    if not any_exists:
        st.info("No outputs yet. Go to **Build** → **Run build**.")


# ---------------------------------------------------------------------------
# Page: Build
# ---------------------------------------------------------------------------

def _build_page() -> None:
    st.subheader("🔨 Build Pipeline")
    st.markdown(
        "Fetches 2-year price history for 30 US tech stocks, detects zombie tickers, "
        "computes log returns, Z-score standardizes, and saves all outputs to `outputs/latest/`. "
        "**The previous run is automatically archived.**"
    )

    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        start = st.date_input("Start date", value=date.today() - timedelta(days=730))
    with col2:
        end = st.date_input("End date", value=date.today())
    with col3:
        missing_threshold = st.number_input(
            "Zombie threshold",
            min_value=0.0, max_value=0.5, value=0.05, step=0.01,
            help=">5% NaN in a ticker's history → it's a zombie and gets replaced by a reserve.",
        )

    if start >= end:
        st.error("Start date must be before end date.")
        return

    st.divider()
    run = st.button("▶  Run build", type="primary", use_container_width=True)

    if not run:
        # Show last build metadata if available
        meta_path = LATEST_DIR / "build_metadata.json"
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                st.info(
                    f"**Last build:** {meta.get('build_timestamp_utc', 'unknown')[:19].replace('T', ' ')} UTC  |  "
                    f"Shape: {meta.get('matrix_shape', {}).get('rows_T', '?')} days × "
                    f"{meta.get('matrix_shape', {}).get('cols_tickers', '?')} tickers"
                )
            except Exception:
                pass
        return

    with st.spinner("Building pipeline — this takes 10–30 seconds…"):
        try:
            artifacts = build_and_serialize(
                start_date=start, end_date=end,
                missing_threshold=missing_threshold, root_dir=ROOT_DIR,
            )
        except RuntimeError as exc:
            st.error(f"Build failed: {exc}")
            return

    st.session_state["latest"] = {
        "prices": artifacts.prices,
        "cleaning": artifacts.cleaning,
        "returns": artifacts.aligned_log_returns,
        "standardized": artifacts.standardization.standardized,
        "scaler_params": artifacts.standardization.scaler_params,
        "paths": {k: str(v) for k, v in artifacts.paths.items()},
    }

    # Results summary
    cleaning = artifacts.cleaning
    if cleaning.dropped_primaries:
        st.warning(f"⚠ Zombie tickers replaced: {', '.join(cleaning.dropped_primaries)}")
        for old, new in cleaning.replacements:
            st.caption(f"  {old} → {new} (reserve)")
    else:
        st.success("✅ All 30 primary tickers healthy — no zombies detected.")

    outliers = artifacts.standardization.scaler_params.get("correlation_outliers", [])
    if outliers:
        worst = outliers[0]
        st.error(
            f"⚠ {len(outliers)} low-correlation pair(s) — "
            f"worst: **{worst['ticker_a']} ↔ {worst['ticker_b']}** = {worst['correlation']:.3f}. "
            "Check the **Correlation** page."
        )
    else:
        st.success("✅ Correlation structure healthy — all pairs ≥ 0.3.")

    shape = artifacts.standardization.standardized.shape
    archived = artifacts.paths.get("archived_to")

    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Trading Days", shape[0])
    col_b.metric("Tickers", shape[1])
    col_c.metric("Corr. Outliers", len(outliers))

    if archived and str(archived) != "(none)":
        st.caption(f"Previous run archived → `{Path(str(archived)).name}`")

    st.divider()
    st.markdown("**Saved to `outputs/latest/`**")
    paths = artifacts.paths
    for key in ["matrix_heatmap", "correlation_heatmap", "standardized_csv",
                 "returns_csv", "guide", "build_metadata"]:
        p = paths.get(key)
        if p and Path(str(p)).exists():
            st.markdown(f"✅ `{Path(str(p)).name}`")

    st.info("💡 Open `outputs/latest/GUIDE.md` for a plain-English explanation of every file.")
    st.rerun()


# ---------------------------------------------------------------------------
# Page: Matrix (30×T)
# ---------------------------------------------------------------------------

def _matrix_page() -> None:
    st.subheader("📈 Matrix — 30×T Standardised Z-scores")

    standardized = _session_or_disk()
    if standardized is None:
        st.warning("No matrix found. Go to **Build** and run the pipeline first.")
        return

    st.caption(
        f"**{standardized.shape[0]} trading days × {standardized.shape[1]} tickers.** "
        "Colour = Z-score (🔴 high / 🔵 low relative to each ticker's own history)."
    )

    fig = render_aligned_matrix_heatmap(standardized, heatmap_path=None)
    st.pyplot(fig, clear_figure=True, use_container_width=True)

    with st.expander("Preview — last 15 rows (T×30 orientation)", expanded=False):
        st.dataframe(standardized.tail(15), use_container_width=True)

    st.divider()
    csv_30xt = standardized.T.copy()
    if isinstance(csv_30xt.columns, pd.DatetimeIndex):
        csv_30xt.columns = csv_30xt.columns.strftime("%Y-%m-%d")
    st.download_button(
        "⬇  Download standardized_matrix_30xT.csv",
        data=csv_30xt.to_csv(index=True).encode("utf-8"),
        file_name="standardized_matrix_30xT.csv",
        mime="text/csv",
        use_container_width=True,
    )


# ---------------------------------------------------------------------------
# Page: Correlation (30×30)
# ---------------------------------------------------------------------------

def _correlation_page() -> None:
    st.subheader("🔗 Correlation Heatmap — 30×30")

    standardized = _session_or_disk()
    if standardized is None:
        st.warning("No matrix found. Go to **Build** and run the pipeline first.")
        return

    low_threshold = st.slider(
        "Flag pairs below this correlation",
        min_value=0.0, max_value=1.0, value=0.30, step=0.05,
        help="Healthy US tech pairs are typically 0.5–0.9. Below 0.3 = suspicious.",
    )

    fig, outliers = render_correlation_heatmap(
        standardized, heatmap_path=None, low_threshold=low_threshold,
    )
    st.pyplot(fig, clear_figure=True, use_container_width=True)

    st.caption(
        "**Expected range for US Nasdaq-100 tech stocks: 0.5 – 0.9.** "
        "A near-zero or negative cell means two tickers are behaving independently — "
        "likely a data issue or a misspecified reserve ticker."
    )

    if outliers:
        st.error(f"⚠ {len(outliers)} suspicious pair(s) below {low_threshold:.2f}:")
        st.dataframe(pd.DataFrame(outliers), use_container_width=True, hide_index=True)
    else:
        st.success(f"✅ All pairs ≥ {low_threshold:.2f} — correlation structure looks healthy.")

    with st.expander("Full correlation matrix (numeric, colour-coded)", expanded=False):
        corr = standardized.corr()
        st.dataframe(
            corr.style.background_gradient(cmap="vlag", vmin=-1, vmax=1)
                      .format("{:.3f}"),
            use_container_width=True,
        )


# ---------------------------------------------------------------------------
# Page: PCA
# ---------------------------------------------------------------------------

def _pca_page() -> None:
    st.subheader("🧮 PCA — Beta / Alpha Decomposition")

    standardized = _session_or_disk()
    if standardized is None:
        st.warning("No matrix found. Go to **Build** and run the pipeline first.")
        return

    st.markdown(
        "PCA decomposes the return matrix into:\n"
        "- **Beta** (first *k* principal components) — the systematic/market factor\n"
        "- **Alpha** (residual) — idiosyncratic returns after subtracting Beta"
    )

    k = st.slider("Number of components to treat as Beta (k)", min_value=1, max_value=10, value=1)
    n = min(k, standardized.shape[1])
    summary = pca_fit_summary(standardized, n_components=n)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Explained variance ratio (per PC)**")
        st.dataframe(summary["explained_variance_ratio"], use_container_width=True)
    with col2:
        st.markdown("**Cumulative explained variance**")
        st.dataframe(summary["cumulative_explained_variance"], use_container_width=True)

    beta, alpha, pca_model = pca_beta_alpha(standardized, k=k)

    st.markdown(f"**Beta component — first {k} PC(s)**")
    fig_b = render_aligned_matrix_heatmap(beta, heatmap_path=None, title=f"Beta (first {k} PC)")
    st.pyplot(fig_b, clear_figure=True, use_container_width=True)

    st.markdown("**Alpha residual — idiosyncratic returns**")
    fig_a = render_aligned_matrix_heatmap(alpha, heatmap_path=None, title="Alpha residual")
    st.pyplot(fig_a, clear_figure=True, use_container_width=True)

    col_dl1, col_dl2 = st.columns(2)
    with col_dl1:
        st.download_button(
            "⬇  beta_component_30xT.csv",
            data=beta.T.to_csv(index=True).encode("utf-8"),
            file_name="beta_component_30xT.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with col_dl2:
        st.download_button(
            "⬇  alpha_residual_30xT.csv",
            data=alpha.T.to_csv(index=True).encode("utf-8"),
            file_name="alpha_residual_30xT.csv",
            mime="text/csv",
            use_container_width=True,
        )


# ---------------------------------------------------------------------------
# Page: Archive browser
# ---------------------------------------------------------------------------

def _archive_page() -> None:
    st.subheader("🗂 Archive — Browse Previous Runs")

    if not ARCHIVE_DIR.exists() or not any(ARCHIVE_DIR.iterdir()):
        st.info("No archived runs yet. Previous builds are automatically saved here.")
        return

    runs = sorted(
        [d for d in ARCHIVE_DIR.iterdir() if d.is_dir()],
        reverse=True,  # newest first
    )

    st.caption(f"{len(runs)} archived run(s) in `outputs/archive/`")

    selected = st.selectbox(
        "Select a run to inspect",
        options=runs,
        format_func=lambda p: _format_archive_name(p.name),
    )

    if selected is None:
        return

    st.markdown(f"**Run: `{selected.name}`**")
    files = sorted(selected.iterdir())

    if not files:
        st.warning("Archive folder is empty.")
        return

    # Summary from metadata if available
    meta_file = selected / "build_metadata.json"
    if meta_file.exists():
        try:
            meta = json.loads(meta_file.read_text(encoding="utf-8"))
            dr = meta.get("date_range", {})
            shape = meta.get("matrix_shape", {})
            zombies = meta.get("zombies_dropped", [])
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Start", dr.get("start", "?"))
            col2.metric("End", dr.get("end", "?"))
            col3.metric("Days × Tickers", f"{shape.get('rows_T','?')}×{shape.get('cols_tickers','?')}")
            col4.metric("Zombies", len(zombies))
            if zombies:
                st.warning(f"Zombies dropped: {', '.join(zombies)}")
        except Exception:
            pass

    st.divider()
    # File listing
    st.markdown("**Files in this archive:**")
    for f in files:
        size_kb = f.stat().st_size / 1024
        st.markdown(f"- `{f.name}` — {size_kb:.1f} KB")

    # Preview heatmaps if they exist
    matrix_png = selected / "matrix_heatmap.png"
    corr_png = selected / "correlation_heatmap.png"

    if matrix_png.exists() or corr_png.exists():
        st.divider()
        st.markdown("**Heatmap preview**")
        if matrix_png.exists() and corr_png.exists():
            c1, c2 = st.columns(2)
            c1.image(str(matrix_png), caption="Matrix heatmap (30×T)", use_container_width=True)
            c2.image(str(corr_png), caption="Correlation heatmap (30×30)", use_container_width=True)
        elif matrix_png.exists():
            st.image(str(matrix_png), caption="Matrix heatmap (30×T)", use_container_width=True)
        elif corr_png.exists():
            st.image(str(corr_png), caption="Correlation heatmap (30×30)", use_container_width=True)


def _format_archive_name(name: str) -> str:
    """Convert '20260409_103000' → '2026-04-09  10:30:00 UTC'"""
    try:
        date_part, time_part = name.split("_", 1)
        y, m, d = date_part[:4], date_part[4:6], date_part[6:8]
        hh, mm, ss = time_part[:2], time_part[2:4], time_part[4:6]
        return f"{y}-{m}-{d}  {hh}:{mm}:{ss} UTC"
    except Exception:
        return name


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    st.set_page_config(
        page_title="Quant Matrix — NASDAQ-100 Tech",
        layout="wide",
        page_icon="📊",
        initial_sidebar_state="expanded",
    )

    st.title("Quant Matrix: NASDAQ-100 Tech — 30 Stocks")

    page = _render_sidebar()

    if page.startswith("🔨"):
        _build_page()
    elif page.startswith("📈"):
        _matrix_page()
    elif page.startswith("🔗"):
        _correlation_page()
    elif page.startswith("🧮"):
        _pca_page()
    elif page.startswith("🗂"):
        _archive_page()


if __name__ == "__main__":
    main()
