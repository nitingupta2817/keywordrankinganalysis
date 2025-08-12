import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

st.set_page_config(page_title="Keyword Rank Tracker", layout="wide")
st.title("📈 Keyword Ranking Comparison Tool")

uploaded_file = st.file_uploader("Upload Excel File", type=["xlsx"])

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

        st.subheader("📅 Select Date Range to Compare Rankings")
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.selectbox("Start Date", ranking_cols)
        with col2:
            end_date = st.selectbox("End Date", ranking_cols, index=len(ranking_cols) - 1)

        if start_date == end_date:
            st.warning("Please select two different dates to compare.")
        else:
            df_compare = filtered_df[[keyword_col, url_col, volume_col, start_date, end_date]].copy()
            df_compare.columns = ['Keyword', 'URL', 'Volume', 'Start Rank', 'End Rank']
            df_compare['Start Rank'] = pd.to_numeric(df_compare['Start Rank'], errors='coerce')
            df_compare['End Rank'] = pd.to_numeric(df_compare['End Rank'], errors='coerce')

            # Calculate change
            df_compare['Change'] = df_compare['End Rank'] - df_compare['Start Rank']

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
            st.dataframe(improved[['Keyword', 'URL', 'Volume', 'Start Rank', 'End Rank', 'Change']].reset_index(drop=True), use_container_width=True)

            st.subheader(f"📉 Keywords Dropped (newly ranked) ({len(dropped)})")
            st.dataframe(dropped[['Keyword', 'URL', 'Volume', 'Start Rank', 'End Rank', 'Change']].reset_index(drop=True), use_container_width=True)

            st.subheader(f"📊 Keywords Rank Up ({len(up)})")
            st.dataframe(up[['Keyword', 'URL', 'Volume', 'Start Rank', 'End Rank', 'Change']].reset_index(drop=True), use_container_width=True)

            st.subheader(f"📉 Keywords Rank Down ({len(down)})")
            st.dataframe(down[['Keyword', 'URL', 'Volume', 'Start Rank', 'End Rank', 'Change']].reset_index(drop=True), use_container_width=True)

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
