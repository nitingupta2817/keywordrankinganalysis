import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

st.set_page_config(page_title="Keyword Rank Tracker", layout="wide")
st.title("📈 Keyword Ranking Comparison Tool")

# ---------------------- EXISTING TOOL: RANKING COMPARISON ----------------------
uploaded_file = st.file_uploader("Upload Excel File for Ranking Comparison", type=["xlsx"])

if uploaded_file:
    # Read sheet names first
    xls = pd.ExcelFile(uploaded_file)
    sheet_name = st.selectbox("Select Sheet", xls.sheet_names)

    if sheet_name:
        df = pd.read_excel(xls, sheet_name=sheet_name)

        # Select keyword, URL, and volume columns
        keyword_col = st.selectbox("Select the Keyword Column Name", df.columns)
        url_col = st.selectbox("Select the URL Column Name", df.columns)
        volume_col = st.selectbox("Select the Volume Column Name", df.columns)

        # Filter by actual keyword and URL values
        keyword_filter = st.multiselect("✅ Select Keywords to Show", options=df[keyword_col].dropna().unique())
        url_filter = st.multiselect("🌐 Select URLs to Show", options=df[url_col].dropna().unique())

        filtered_df = df.copy()
        if keyword_filter:
            filtered_df = filtered_df[filtered_df[keyword_col].isin(keyword_filter)]
        if url_filter:
            filtered_df = filtered_df[filtered_df[url_col].isin(url_filter)]

        # Ranking columns (all except keyword, URL, and volume)
        ranking_cols = [col for col in df.columns if col not in [keyword_col, url_col, volume_col]]

        # ---------------- New Option for Comparison Mode ----------------
        compare_mode = st.radio("🔄 Select Comparison Mode", ["Compare by Date Range", "Compare by Month"])

        # ==================== DATE RANGE COMPARISON ====================
        if compare_mode == "Compare by Date Range":
            st.subheader("📅 Select Date Range to Compare Rankings")
            col1, col2 = st.columns(2)
            with col1:
                start_date = st.selectbox("Start Date", ranking_cols)
            with col2:
                end_date = st.selectbox("End Date", ranking_cols, index=len(ranking_cols) - 1)

        # ==================== MONTH-WISE COMPARISON ====================
        else:
            st.subheader("🗓️ Select Month Range to Compare Rankings")

            # Try parsing columns to datetime where possible
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

            # Create mapping: column -> Month Year
            month_map = {col: dt.strftime("%B %Y") for col, dt in date_map.items()}

            # Get unique sorted months
            unique_months = sorted(set(month_map.values()),
                                   key=lambda x: pd.to_datetime(x, format="%B %Y"))

            col1, col2 = st.columns(2)
            with col1:
                start_month = st.selectbox("Start Month", unique_months)
            with col2:
                end_month = st.selectbox("End Month", unique_months, index=len(unique_months) - 1)

            # Get all columns belonging to selected months
            start_cols = [col for col in month_map if month_map[col] == start_month]
            end_cols = [col for col in month_map if month_map[col] == end_month]

            if not start_cols or not end_cols:
                st.error("❌ No columns found for selected months.")
                st.stop()

            # Select first column of start month & last column of end month
            start_date = start_cols[0]
            end_date = end_cols[-1]

        # ==================== COMMON COMPARISON LOGIC ====================
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

            # ----- Detailed Tables (Always Open) -----
            st.subheader(f"📈 Keywords Improved (missing at end) ({len(improved)})")
            st.dataframe(improved[['Keyword', 'URL', 'Volume', 'Start Rank', 'End Rank', 'Change']], use_container_width=True)

            st.subheader(f"📉 Keywords Dropped (newly ranked) ({len(dropped)})")
            st.dataframe(dropped[['Keyword', 'URL', 'Volume', 'Start Rank', 'End Rank', 'Change']], use_container_width=True)

            st.subheader(f"📊 Keywords Rank Up ({len(up)})")
            st.dataframe(up[['Keyword', 'URL', 'Volume', 'Start Rank', 'End Rank', 'Change']], use_container_width=True)

            st.subheader(f"📉 Keywords Rank Down ({len(down)})")
            st.dataframe(down[['Keyword', 'URL', 'Volume', 'Start Rank', 'End Rank', 'Change']], use_container_width=True)

            # Download option
            csv = df_compare.to_csv(index=False)
            st.download_button("📥 Download Report as CSV", csv, file_name="keyword_ranking_comparison.csv", mime="text/csv")

            # ----------- GRAPHS --------------
            st.subheader("📊 Visualizations")

            # Bar chart of Status counts
            status_counts = df_compare['Status'].value_counts().reset_index()
            status_counts.columns = ['Status', 'Count']
            fig_bar = px.bar(status_counts, x='Status', y='Count', color='Status',
                             title='Keyword Ranking Status Counts',
                             labels={'Count': 'Number of Keywords'})
            st.plotly_chart(fig_bar, use_container_width=True)

            # Scatter plot Start Rank vs End Rank
            scatter_df = df_compare.dropna(subset=['Start Rank', 'End Rank'])
            if not scatter_df.empty:
                fig_scatter = px.scatter(
                    scatter_df,
                    x='Start Rank',
                    y='End Rank',
                    color='Status',
                    hover_data=['Keyword', 'URL', 'Volume', 'Change'],
                    title='Start Rank vs End Rank Scatter Plot',
                    labels={'Start Rank': 'Start Rank', 'End Rank': 'End Rank'},
                    height=500
                )
                st.plotly_chart(fig_scatter, use_container_width=True)
            else:
                st.info("No data available for scatter plot (both start and end rank needed).")

            # Histogram of Change
            change_df = df_compare.dropna(subset=['Change'])
            if not change_df.empty:
                fig_hist = px.histogram(change_df, x='Change', nbins=30,
                                        title='Distribution of Rank Changes',
                                        labels={'Change': 'Change in Rank'})
                st.plotly_chart(fig_hist, use_container_width=True)
            else:
                st.info("No rank change data available for histogram.")

# ---------------------- NEW TOOL: KEYWORD ANALYZER ----------------------
st.markdown("---")
st.header("🔍 Keyword Difficulty & Volume Analyzer (Independent Tool)")

file2 = st.file_uploader("Upload CSV or Excel File for Keyword Analysis", type=["csv", "xlsx"])

if file2:
    # Read file
    if file2.name.endswith(".csv"):
        df_keywords = pd.read_csv(file2)
    else:
        df_keywords = pd.read_excel(file2)

    required_cols = ["Keyword", "Difficulty", "Volume"]
    missing_cols = [col for col in required_cols if col not in df_keywords.columns]
    if missing_cols:
        st.error(f"❌ Missing required columns: {', '.join(missing_cols)}")
    else:
        st.success(f"✅ File loaded successfully! Total Keywords: {len(df_keywords)}")

        # Handle NaN values safely
        df_keywords["Difficulty"] = pd.to_numeric(df_keywords["Difficulty"], errors="coerce").fillna(0)
        df_keywords["Volume"] = pd.to_numeric(df_keywords["Volume"], errors="coerce").fillna(0)

        # Multiple keyword input
        keyword_input = st.text_input("Enter keywords (comma-separated) to filter")
        keywords_to_search = [kw.strip().lower() for kw in keyword_input.split(",") if kw.strip()]

        # Difficulty & Volume filters
        min_diff, max_diff = int(df_keywords["Difficulty"].min()), int(df_keywords["Difficulty"].max())
        min_vol, max_vol = int(df_keywords["Volume"].min()), int(df_keywords["Volume"].max())

        diff_range = st.slider("Select Difficulty Range", 0, 100, (min_diff, max_diff))
        vol_range = st.slider("Select Volume Range", min_vol, max_vol, (min_vol, max_vol))

        # Apply filters
        filtered_df = df_keywords.copy()

        # If keywords provided → filter by keywords
        if keywords_to_search:
            filtered_df = filtered_df[
                filtered_df["Keyword"].str.lower().apply(
                    lambda x: any(kw in x for kw in keywords_to_search)
                )
            ]

            # Check missing keywords
            missing_keywords = [kw for kw in keywords_to_search if not any(df_keywords["Keyword"].str.lower().str.contains(kw))]
            if missing_keywords:
                st.warning(f"⚠️ These keywords were not found: {', '.join(missing_keywords)}")

        # Always filter by difficulty & volume range
        filtered_df = filtered_df[
            (filtered_df["Difficulty"] >= diff_range[0]) &
            (filtered_df["Difficulty"] <= diff_range[1]) &
            (filtered_df["Volume"] >= vol_range[0]) &
            (filtered_df["Volume"] <= vol_range[1])
        ]

        # Show filtered results
        st.subheader(f"📊 Filtered Results ({len(filtered_df)} keywords)")
        st.dataframe(filtered_df, use_container_width=True)

        # Download filtered CSV
        csv_filtered = filtered_df.to_csv(index=False)
        st.download_button("📥 Download Filtered Results as CSV", csv_filtered, file_name="filtered_keywords.csv", mime="text/csv")
