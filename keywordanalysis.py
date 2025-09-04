import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# ------------------- PAGE CONFIG -------------------
st.set_page_config(page_title="Keyword Tools", layout="wide")
st.title("🔍 Keyword Tools Dashboard")

# ------------------- CREATE TABS -------------------
tab1, tab2 = st.tabs(["📈 Keyword Ranking Comparison", "📊 Keyword Analyzer"])

# ========================================================================
# 📌 TAB 1: KEYWORD RANKING COMPARISON TOOL
# ========================================================================
with tab1:
    st.header("📈 Keyword Ranking Comparison Tool")

    uploaded_file = st.file_uploader("Upload Excel File", type=["xlsx"], key="compare_upload")

    if uploaded_file:
        # Read sheet names first
        xls = pd.ExcelFile(uploaded_file)
        sheet_name = st.selectbox("Select Sheet", xls.sheet_names, key="compare_sheet")

        if sheet_name:
            df = pd.read_excel(xls, sheet_name=sheet_name)

            # Select keyword, URL, and volume columns
            keyword_col = st.selectbox("Select the Keyword Column Name", df.columns, key="compare_kw_col")
            url_col = st.selectbox("Select the URL Column Name", df.columns, key="compare_url_col")
            volume_col = st.selectbox("Select the Volume Column Name", df.columns, key="compare_vol_col")

            # Filter by keyword & URL values
            keyword_filter = st.multiselect("✅ Select Keywords to Show", options=df[keyword_col].dropna().unique(), key="compare_kw_filter")
            url_filter = st.multiselect("🌐 Select URLs to Show", options=df[url_col].dropna().unique(), key="compare_url_filter")

            filtered_df = df.copy()
            if keyword_filter:
                filtered_df = filtered_df[filtered_df[keyword_col].isin(keyword_filter)]
            if url_filter:
                filtered_df = filtered_df[filtered_df[url_col].isin(url_filter)]

            # Ranking columns (all except keyword, URL, and volume)
            ranking_cols = [col for col in df.columns if col not in [keyword_col, url_col, volume_col]]

            # Comparison mode selection
            compare_mode = st.radio("🔄 Select Comparison Mode", ["Compare by Date Range", "Compare by Month"], key="compare_mode")

            # ==================== DATE RANGE COMPARISON ====================
            if compare_mode == "Compare by Date Range":
                st.subheader("📅 Select Date Range to Compare Rankings")
                col1, col2 = st.columns(2)
                with col1:
                    start_date = st.selectbox("Start Date", ranking_cols, key="start_date")
                with col2:
                    end_date = st.selectbox("End Date", ranking_cols, index=len(ranking_cols) - 1, key="end_date")

            # ==================== MONTH-WISE COMPARISON ====================
            else:
                st.subheader("🗓️ Select Month Range to Compare Rankings")
                date_map = {}
                for col in ranking_cols:
                    try:
                        parsed_date = pd.to_datetime(col, errors='raise', dayfirst=True)
                        date_map[col] = parsed_date
                    except:
                        continue

                if not date_map:
                    st.error("❌ No valid date columns found for month-wise comparison.")
                    st.stop()

                month_map = {col: dt.strftime("%B %Y") for col, dt in date_map.items()}
                unique_months = sorted(set(month_map.values()), key=lambda x: pd.to_datetime(x, format="%B %Y"))

                col1, col2 = st.columns(2)
                with col1:
                    start_month = st.selectbox("Start Month", unique_months, key="start_month")
                with col2:
                    end_month = st.selectbox("End Month", unique_months, index=len(unique_months) - 1, key="end_month")

                start_cols = [col for col in month_map if month_map[col] == start_month]
                end_cols = [col for col in month_map if month_map[col] == end_month]

                if not start_cols or not end_cols:
                    st.error("❌ No columns found for selected months.")
                    st.stop()

                start_date = start_cols[0]
                end_date = end_cols[-1]

            # ==================== COMPARISON LOGIC ====================
            if start_date == end_date:
                st.warning("⚠️ Please select two different dates or months to compare.")
            else:
                df_compare = filtered_df[[keyword_col, url_col, volume_col, start_date, end_date]].copy()
                df_compare.columns = ['Keyword', 'URL', 'Volume', 'Start Rank', 'End Rank']
                df_compare['Start Rank'] = pd.to_numeric(df_compare['Start Rank'], errors='coerce')
                df_compare['End Rank'] = pd.to_numeric(df_compare['End Rank'], errors='coerce')

                # Calculate change
                df_compare['Change'] = df_compare['End Rank'] - df_compare['Start Rank']

                # Status calculation
                def rank_status(row):
                    if pd.isna(row['Start Rank']) and pd.notna(row['End Rank']):
                        return "Dropped"
                    if pd.notna(row['Start Rank']) and pd.isna(row['End Rank']):
                        return "Improved"
                    if pd.isna(row['Start Rank']) or pd.isna(row['End Rank']):
                        return "No Data"
                    if row['End Rank'] > row['Start Rank']:
                        return "Up"
                    elif row['End Rank'] < row['Start Rank']:
                        return "Down"
                    else:
                        return "No Change"

                df_compare['Status'] = df_compare.apply(rank_status, axis=1)

                # Show comparison table
                st.subheader("📊 Keyword Ranking Comparison Report")
                st.dataframe(df_compare.sort_values(by="Change", ascending=False), use_container_width=True)

                # Summary counts
                st.subheader("📋 Summary")
                total = len(df_compare)
                up = df_compare[df_compare['Status'] == 'Up']
                down = df_compare[df_compare['Status'] == 'Down']
                dropped = df_compare[df_compare['Status'] == 'Dropped']
                improved = df_compare[df_compare['Status'] == 'Improved']
                no_change = df_compare[df_compare['Status'] == 'No Change']

                st.markdown(f"""
                - ✅ **Total Keywords Shown:** {total}  
                - 📈 **Improved (ranked at start but missing at end):** {len(improved)}  
                - 📉 **Dropped (newly ranked at end):** {len(dropped)}  
                - 📊 **Rank Up:** {len(up)}  
                - 📉 **Rank Down:** {len(down)}  
                - ➖ **No Change:** {len(no_change)}  
                """)

                # Detailed tables
                st.subheader(f"📈 Keywords Improved ({len(improved)})")
                st.dataframe(improved[['Keyword', 'URL', 'Volume', 'Start Rank', 'End Rank', 'Change']], use_container_width=True)

                st.subheader(f"📉 Keywords Dropped ({len(dropped)})")
                st.dataframe(dropped[['Keyword', 'URL', 'Volume', 'Start Rank', 'End Rank', 'Change']], use_container_width=True)

                st.subheader(f"📊 Keywords Rank Up ({len(up)})")
                st.dataframe(up[['Keyword', 'URL', 'Volume', 'Start Rank', 'End Rank', 'Change']], use_container_width=True)

                st.subheader(f"📉 Keywords Rank Down ({len(down)})")
                st.dataframe(down[['Keyword', 'URL', 'Volume', 'Start Rank', 'End Rank', 'Change']], use_container_width=True)

                # Download CSV
                csv = df_compare.to_csv(index=False)
                st.download_button("📥 Download Report as CSV", csv, file_name="keyword_ranking_comparison.csv", mime="text/csv")

                # Visualizations
                st.subheader("📊 Visualizations")
                status_counts = df_compare['Status'].value_counts().reset_index()
                status_counts.columns = ['Status', 'Count']
                fig_bar = px.bar(status_counts, x='Status', y='Count', color='Status',
                                 title='Keyword Ranking Status Counts')
                st.plotly_chart(fig_bar, use_container_width=True)

                scatter_df = df_compare.dropna(subset=['Start Rank', 'End Rank'])
                if not scatter_df.empty:
                    fig_scatter = px.scatter(
                        scatter_df,
                        x='Start Rank', y='End Rank',
                        color='Status',
                        hover_data=['Keyword', 'URL', 'Volume', 'Change'],
                        title='Start Rank vs End Rank Scatter Plot'
                    )
                    st.plotly_chart(fig_scatter, use_container_width=True)

                change_df = df_compare.dropna(subset=['Change'])
                if not change_df.empty:
                    fig_hist = px.histogram(change_df, x='Change', nbins=30,
                                            title='Distribution of Rank Changes')
                    st.plotly_chart(fig_hist, use_container_width=True)

# ========================================================================
# 📌 TAB 2: KEYWORD ANALYZER TOOL
# ========================================================================
with tab2:
    st.header("📊 Keyword Analyzer Tool")

    uploaded_kw_file = st.file_uploader("Upload Keyword File (CSV or Excel)", type=["csv", "xlsx"], key="kw_upload")

    if uploaded_kw_file:
        # Read file based on extension
        if uploaded_kw_file.name.endswith(".csv"):
            kw_df = pd.read_csv(uploaded_kw_file)
        else:
            kw_df = pd.read_excel(uploaded_kw_file)

        required_cols = ["Keyword", "Difficulty", "Volume"]
        if not all(col in kw_df.columns for col in required_cols):
            st.error(f"❌ File must contain columns: {required_cols}")
            st.stop()

        # Convert numeric columns safely
        kw_df["Difficulty"] = pd.to_numeric(kw_df["Difficulty"], errors="coerce")
        kw_df["Volume"] = pd.to_numeric(kw_df["Volume"], errors="coerce")

        # Input: Keywords
        keywords_input = st.text_input("🔍 Enter Keywords (comma-separated)", key="kw_input")
        keywords = [kw.strip().lower() for kw in keywords_input.split(",") if kw.strip()]

        # Select match type
        match_type = st.radio("🔑 Select Keyword Match Type", ["Phrase Match", "Exact Match"], key="match_type")

        # Difficulty range is fixed (0 to 100)
        min_diff, max_diff = 0, 100
        min_vol, max_vol = int(kw_df["Volume"].min(skipna=True)), int(kw_df["Volume"].max(skipna=True))

        diff_range = st.slider("🎯 Select Difficulty Range", min_diff, max_diff, (min_diff, max_diff))
        vol_range = st.slider("📊 Select Volume Range", min_vol, max_vol, (min_vol, max_vol))

        filtered_kw_df = kw_df.copy()

        # Apply range filtering first
        filtered_kw_df = filtered_kw_df[
            (filtered_kw_df["Difficulty"].between(diff_range[0], diff_range[1])) &
            (filtered_kw_df["Volume"].between(vol_range[0], vol_range[1]))
        ]

        # Apply keyword filtering if any keyword entered
        not_found_keywords = []
        if keywords:
            if match_type == "Exact Match":
                filtered_kw_df = filtered_kw_df[
                    filtered_kw_df["Keyword"].str.lower().isin(keywords)
                ]
            else:  # Phrase Match → match all words, anywhere in the keyword
                filtered_kw_df = filtered_kw_df[
                    filtered_kw_df["Keyword"].str.lower().apply(
                        lambda x: any(all(word in x for word in kw.split()) for kw in keywords)
                    )
                ]

            # Find missing keywords properly against full sheet
            existing_keywords = kw_df["Keyword"].str.lower().tolist()
            not_found_keywords = [
                kw for kw in keywords if not any(all(w in ek for w in kw.split()) for ek in existing_keywords)
            ]

        # Show missing keywords message
        if keywords and not_found_keywords:
            st.warning(f"⚠️ These keywords were not found: {', '.join(not_found_keywords)}")

        # Show filtered results
        st.subheader(f"🔎 Filtered Keywords ({len(filtered_kw_df)})")
        st.dataframe(filtered_kw_df, use_container_width=True)

        # Download button
        csv_kw = filtered_kw_df.to_csv(index=False)
        st.download_button("📥 Download Filtered Results", csv_kw, file_name="filtered_keywords.csv", mime="text/csv")
