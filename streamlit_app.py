import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from textblob import TextBlob
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
import numpy as np
from collections import Counter
from PIL import Image
import time
import json
from streamlit_lottie import st_lottie

# Download NLTK data
try:
    nltk.download('vader_lexicon', quiet=True)
except Exception as e:
    st.error(f"Error downloading NLTK data: {e}")

# Custom page configuration with Netflix-inspired theme
st.set_page_config(
    layout="wide", 
    page_title="Netflix Analytics Dashboard", 
    page_icon="🎬",
    initial_sidebar_state="expanded"
)

# Custom CSS for enhanced styling
st.markdown("""
<style>
    /* Main styling */
    .main {
        background-color: #0F0F0F;
        color: #FFFFFF;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background-color: #1F1F1F !important;
    }
    
    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, #E50914 0%, #B00710 100%);
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
        color: white;
        text-align: center;
        transition: transform 0.3s;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
    }
    
    .metric-title {
        font-size: 16px;
        font-weight: 300;
        margin-bottom: 5px;
    }
    
    .metric-value {
        font-size: 28px;
        font-weight: 700;
    }
    
    .metric-change {
        font-size: 14px;
        font-weight: 300;
    }
    
    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: #1F1F1F;
        color: #A0A0A0;
        padding: 12px 24px;
        border-radius: 8px 8px 0 0;
        margin-right: 4px;
        transition: all 0.3s;
    }
    
    .stTabs [aria-selected="true"] {
        background: #E50914 !important;
        color: white !important;
    }
    
    /* Button styling */
    .stButton button {
        background: linear-gradient(135deg, #E50914 0%, #B00710 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 10px 20px;
        transition: all 0.3s;
    }
    
    .stButton button:hover {
        transform: scale(1.05);
        box-shadow: 0 4px 12px rgba(229, 9, 20, 0.3);
    }
    
    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: #1F1F1F;
    }
    
    ::-webkit-scrollbar-thumb {
        background: #E50914;
        border-radius: 4px;
    }
    
    /* Custom headers */
    h1, h2, h3 {
        color: #E50914 !important;
    }
    
    /* Custom divider */
    .custom-divider {
        height: 4px;
        background: linear-gradient(90deg, #E50914 0%, transparent 100%);
        border: none;
        margin: 20px 0;
    }
    
    /* Light mode adjustments */
    .light-mode {
        background-color: #FFFFFF !important;
        color: #333333 !important;
    }
    
    .light-mode [data-testid="stSidebar"] {
        background-color: #F5F5F5 !important;
    }
    
    .light-mode h1, .light-mode h2, .light-mode h3 {
        color: #E50914 !important;
    }
</style>
""", unsafe_allow_html=True)

# Netflix logo in sidebar
def add_netflix_logo():
    netflix_logo = """
    <div style="text-align: center; margin-bottom: 30px;">
        <svg width="120" height="67" viewBox="0 0 111 30" fill="none" xmlns="http://www.w3.org/2000/svg">
            <path d="M105.062 13.281C105.062 18.708 101.124 24.026 94.6402 24.026C88.4392 24.026 84.5002 18.933 84.5002 13.281C84.5002 7.629 88.4392 2.535 94.6402 2.535C101.124 2.535 105.062 7.853 105.062 13.281ZM90.2812 13.281C90.2812 16.802 92.2842 19.34 94.6402 19.34C96.9962 19.34 98.9992 16.802 98.9992 13.281C98.9992 9.76 96.9962 7.222 94.6402 7.222C92.2842 7.222 90.2812 9.76 90.2812 13.281Z" fill="#E50914"/>
            <path d="M74.5408 13.281C74.5408 18.708 70.6018 24.026 64.1188 24.026C57.9178 24.026 53.9788 18.933 53.9788 13.281C53.9788 7.629 57.9178 2.535 64.1188 2.535C70.6018 2.535 74.5408 7.853 74.5408 13.281ZM59.7598 13.281C59.7598 16.802 61.7628 19.34 64.1188 19.34C66.4748 19.34 68.4778 16.802 68.4778 13.281C68.4778 9.76 66.4748 7.222 64.1188 7.222C61.7628 7.222 59.7598 9.76 59.7598 13.281Z" fill="#E50914"/>
            <path d="M43.7175 2.535H49.5775V23.703H43.7175V16.536C43.7175 13.374 42.1465 11.479 39.8745 11.479C37.4205 11.479 35.8495 13.556 35.8495 16.899V23.703H29.9895V2.535H35.8495V9.573C35.8495 10.709 35.6665 11.662 35.3005 12.432C36.0535 10.891 37.9685 9.207 41.3575 9.207C44.5645 9.207 43.7175 12.249 43.7175 13.281V2.535Z" fill="#E50914"/>
            <path d="M23.191 23.703H17.331V2.535H23.191V23.703Z" fill="#E50914"/>
            <path d="M7.413 23.703H1.553V2.535H7.413V23.703Z" fill="#E50914"/>
        </svg>
    </div>
    """
    st.sidebar.markdown(netflix_logo, unsafe_allow_html=True)

# Custom animated title
def animated_title():
    title_html = """
    <div style="text-align: center; margin-bottom: 30px;">
        <h1 style="color: #E50914; font-size: 42px; font-weight: 700; letter-spacing: 1px; 
                   text-shadow: 0 2px 4px rgba(0,0,0,0.3); position: relative; display: inline-block;">
            <span style="display: inline-block; animation: bounce 0.5s ease infinite alternate;">N</span>
            <span style="display: inline-block; animation: bounce 0.5s ease 0.1s infinite alternate;">E</span>
            <span style="display: inline-block; animation: bounce 0.5s ease 0.2s infinite alternate;">T</span>
            <span style="display: inline-block; animation: bounce 0.5s ease 0.3s infinite alternate;">F</span>
            <span style="display: inline-block; animation: bounce 0.5s ease 0.4s infinite alternate;">L</span>
            <span style="display: inline-block; animation: bounce 0.5s ease 0.5s infinite alternate;">I</span>
            <span style="display: inline-block; animation: bounce 0.5s ease 0.6s infinite alternate;">X</span>
            <span style="margin-left: 20px;"></span>
            <span style="display: inline-block; animation: bounce 0.5s ease 0.7s infinite alternate;">A</span>
            <span style="display: inline-block; animation: bounce 0.5s ease 0.8s infinite alternate;">N</span>
            <span style="display: inline-block; animation: bounce 0.5s ease 0.9s infinite alternate;">A</span>
            <span style="display: inline-block; animation: bounce 0.5s ease 1.0s infinite alternate;">L</span>
            <span style="display: inline-block; animation: bounce 0.5s ease 1.1s infinite alternate;">Y</span>
            <span style="display: inline-block; animation: bounce 0.5s ease 1.2s infinite alternate;">T</span>
            <span style="display: inline-block; animation: bounce 0.5s ease 1.3s infinite alternate;">I</span>
            <span style="display: inline-block; animation: bounce 0.5s ease 1.4s infinite alternate;">C</span>
            <span style="display: inline-block; animation: bounce 0.5s ease 1.5s infinite alternate;">S</span>
        </h1>
        <style>
            @keyframes bounce {
                from { transform: translateY(0); }
                to { transform: translateY(-8px); }
            }
        </style>
    </div>
    """
    st.markdown(title_html, unsafe_allow_html=True)
    st.markdown('<hr class="custom-divider">', unsafe_allow_html=True)

# Custom metric card component
def metric_card(title, value, change=None, icon="🎬"):
    change_html = ""
    if change is not None:
        change_color = "#2ECC71" if change >= 0 else "#E74C3C"
        change_symbol = "+" if change >= 0 else ""
        change_html = f'<div class="metric-change"><span style="color: {change_color}">{change_symbol}{change}</span> from total</div>'
    
    card_html = f"""
    <div class="metric-card">
        <div class="metric-title">{icon} {title}</div>
        <div class="metric-value">{value}</div>
        {change_html}
    </div>
    """
    return card_html

# Load data with progress animation
@st.cache_data
def load_data():
    try:
        with st.spinner('🍿 Loading Netflix data... Please wait...'):
            # Simulate loading delay for better UX
            time.sleep(1.5)
            
            df = pd.read_csv("netflix_titles.csv")
            
            # Data cleaning
            df['country'] = df['country'].fillna('Unknown')
            df['cast'] = df['cast'].fillna('Unknown')
            df['director'] = df['director'].fillna('Unknown')
            df['description'] = df['description'].fillna('')
            
            # Handle date conversion
            df['date_added'] = pd.to_datetime(df['date_added'], format='mixed', errors='coerce')
            df['date_added'] = df.apply(
                lambda row: pd.Timestamp(f"{row['release_year']}-01-01") if pd.isna(row['date_added']) else row['date_added'],
                axis=1
            )
            
            # Feature engineering
            df['year_added'] = df['date_added'].dt.year
            df['month_added'] = df['date_added'].dt.month
            df['genres'] = df['listed_in'].apply(lambda x: str(x).split(', '))
            
            # Extract primary country
            df['primary_country'] = df['country'].apply(lambda x: str(x).split(',')[0].strip())
            
            # Duration processing
            df['duration_minutes'] = df['duration'].apply(lambda x: 
                int(str(x).split(' ')[0]) if 'min' in str(x) else np.nan)
            df['duration_seasons'] = df['duration'].apply(lambda x: 
                int(str(x).split(' ')[0]) if 'Season' in str(x) else np.nan)
            
            # Sentiment analysis
            sid = SentimentIntensityAnalyzer()
            df['sentiment_score'] = df['description'].apply(
                lambda x: sid.polarity_scores(str(x))['compound'] if x else 0
            )
            df['sentiment_label'] = df['sentiment_score'].apply(
                lambda x: 'Positive' if x > 0.2 else 'Negative' if x < -0.2 else 'Neutral'
            )
            
            # Text analysis
            df['description_length'] = df['description'].apply(len)
            df['title_length'] = df['title'].apply(len)
            
            st.success("Data loaded successfully!")
            return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()

# Add Netflix logo to sidebar
add_netflix_logo()

# Display animated title
animated_title()

# Load data
df = load_data()

if df.empty:
    st.stop()

# Sidebar Filters with enhanced UI
st.sidebar.header("🎛️ Filter Controls")

# Dynamic filter options based on data
years = sorted([y for y in df['year_added'].dropna().unique() if not pd.isna(y)])
countries = sorted([c for c in df['primary_country'].unique() if c != 'Unknown'])
all_genres = sorted(set(g.strip() for genre_list in df['genres'] for g in genre_list if g.strip()))

# NEW: Theme toggle
theme = st.sidebar.radio("🌓 Theme Mode", ["Dark", "Light"], index=0, key="theme_toggle")
if theme == "Light":
    st.markdown("<style>.main {background-color: #FFFFFF !important; color: #333333 !important;}</style>", unsafe_allow_html=True)

# NEW: Title search
with st.sidebar.expander("🔍 Title Search", expanded=False):
    search_query = st.text_input("Search for a Title", "")

# Filter widgets with better organization
with st.sidebar.expander("⏳ Time Filters", expanded=True):
    year_range = st.select_slider(
        "Select Year Range",
        options=years,
        value=(min(years), max(years)) if years else (2020, 2021),
        format_func=lambda x: str(int(x))
    )
    
with st.sidebar.expander("🌎 Location & Content", expanded=True):
    selected_countries = st.multiselect(
        "Select Countries", 
        options=countries, 
        default=countries[:3] if len(countries) > 3 else countries,
        max_selections=10
    )
    content_type = st.selectbox(
        "Content Type", 
        options=['All', 'Movie', 'TV Show'],
        index=0
    )
    
    # NEW: Select all genres option
    select_all_genres = st.checkbox("Select All Genres", value=False, key="select_all_genres")
    if select_all_genres:
        selected_genres = all_genres
    else:
        selected_genres = st.multiselect(
            "Select Genres", 
            options=all_genres, 
            default=all_genres[:3] if len(all_genres) > 3 else all_genres,
            max_selections=5
        )

# Apply filters dynamically
def apply_filters(data):
    filtered = data.copy()
    
    # Year range filter
    filtered = filtered[
        (filtered['year_added'] >= year_range[0]) & 
        (filtered['year_added'] <= year_range[1])
    ]
    
    # Country filter
    if selected_countries:
        filtered = filtered[filtered['primary_country'].isin(selected_countries)]
    
    # Content type filter
    if content_type != 'All':
        filtered = filtered[filtered['type'] == content_type]
    
    # Genre filter
    if selected_genres:
        filtered = filtered[filtered['genres'].apply(
            lambda gs: any(g in selected_genres for g in gs)
        )]
    
    # NEW: Title search filter
    if search_query:
        filtered = filtered[filtered['title'].str.contains(search_query, case=False, na=False)]
    
    return filtered

# Apply filters
filtered_df = apply_filters(df)

# NEW: Download filtered data
csv = filtered_df.to_csv(index=False).encode('utf-8')
st.sidebar.download_button(
    "📥 Download Filtered Data", 
    csv, 
    "filtered_netflix.csv", 
    "text/csv",
    key='download_filtered_data'
)

# Display metrics with custom cards
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown(metric_card(
        "Total Titles", 
        len(filtered_df), 
        len(filtered_df) - len(df),
        icon="🎬"
    ), unsafe_allow_html=True)
with col2:
    movies_count = len(filtered_df[filtered_df['type'] == 'Movie'])
    st.markdown(metric_card(
        "Movies", 
        movies_count, 
        icon="🎥"
    ), unsafe_allow_html=True)
with col3:
    tv_count = len(filtered_df[filtered_df['type'] == 'TV Show'])
    st.markdown(metric_card(
        "TV Shows", 
        tv_count, 
        icon="📺"
    ), unsafe_allow_html=True)
with col4:
    countries_count = filtered_df['primary_country'].nunique()
    st.markdown(metric_card(
        "Countries", 
        countries_count, 
        icon="🌎"
    ), unsafe_allow_html=True)

# Reset filters button with icon
if st.sidebar.button("🔄 Reset All Filters", use_container_width=True):
    st.rerun()

st.sidebar.success(f"📊 Showing {len(filtered_df)} of {len(df)} titles")

# Main content tabs with icons
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 Overview", "🎭 Genres", "⏱️ Duration", "☁️ Text Analysis", 
    "😊 Sentiment", "📈 Trends"
])

with tab1:
    if not filtered_df.empty:
        col1, col2 = st.columns(2)
        
        with col1:
            # Dynamic content type visualization with custom colors
            type_counts = filtered_df['type'].value_counts()
            fig = px.pie(
                values=type_counts.values,
                names=type_counts.index,
                title=f"Content Distribution ({len(filtered_df)} titles)",
                color_discrete_sequence=['#E50914', '#00A8E1'],
                hole=0.4
            )
            fig.update_traces(
                textposition='inside', 
                textinfo='percent+label',
                marker=dict(line=dict(color='#1F1F1F', width=2))
            )
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white')
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Top countries with Netflix red color scale
            country_counts = filtered_df['primary_country'].value_counts().head(8)
            fig = px.bar(
                x=country_counts.values,
                y=country_counts.index,
                orientation='h',
                title="Top Countries in Selection",
                labels={'x': 'Number of Titles', 'y': 'Country'},
                color=country_counts.values,
                color_continuous_scale='reds'
            )
            fig.update_layout(
                yaxis={'categoryorder': 'total ascending', 'showgrid': False},
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white'),
                xaxis=dict(showgrid=False)
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Year-wise content addition with Netflix-style colors
        st.subheader("📅 Content Addition Timeline")
        yearly_data = filtered_df.groupby(['year_added', 'type']).size().reset_index(name='count')
        
        if not yearly_data.empty:
            fig = px.bar(
                yearly_data, 
                x='year_added', 
                y='count', 
                color='type',
                title="Content Added by Year and Type",
                labels={'year_added': 'Year', 'count': 'Number of Titles'},
                color_discrete_sequence=['#E50914', '#00A8E1']
            )
            fig.update_layout(
                xaxis_title="Year", 
                yaxis_title="Number of Titles",
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white'),
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                )
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Monthly patterns with smooth line
        if filtered_df['year_added'].nunique() > 1:
            monthly_data = filtered_df.groupby('month_added').size()
            month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                          'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
            
            fig = px.line(
                x=[month_names[i-1] for i in monthly_data.index],
                y=monthly_data.values,
                title="Seasonal Content Addition Pattern",
                labels={'x': 'Month', 'y': 'Average Titles Added'},
                line_shape='spline'
            )
            fig.update_traces(
                mode='lines+markers', 
                line_color='#E50914',
                marker=dict(size=8, color='#E50914')
            )
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white'),
                xaxis=dict(showgrid=False),
                yaxis=dict(showgrid=False)
            )
            st.plotly_chart(fig, use_container_width=True)
    
    else:
        st.warning("⚠️ No data matches the selected filters. Please adjust your selection.")

with tab2:
    st.subheader("🎭 Genre Analysis")
    
    if not filtered_df.empty:
        # Explode genres for analysis
        genre_df = filtered_df.explode('genres')
        genre_counts = genre_df['genres'].value_counts().head(15)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Top genres bar chart with Netflix red
            fig = px.bar(
                x=genre_counts.index,
                y=genre_counts.values,
                title="Most Popular Genres",
                labels={'x': 'Genre', 'y': 'Number of Titles'},
                color=genre_counts.values,
                color_continuous_scale='reds'
            )
            fig.update_layout(
                xaxis_tickangle=-45,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white'),
                xaxis=dict(showgrid=False),
                yaxis=dict(showgrid=False)
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Genre distribution by content type
            genre_type_data = genre_df.groupby(['genres', 'type']).size().reset_index(name='count')
            top_genres_list = genre_counts.head(10).index.tolist()
            genre_type_filtered = genre_type_data[genre_type_data['genres'].isin(top_genres_list)]
            
            fig = px.bar(
                genre_type_filtered,
                x='genres',
                y='count',
                color='type',
                title="Genre Distribution by Content Type",
                labels={'genres': 'Genre', 'count': 'Count'},
                color_discrete_sequence=['#E50914', '#00A8E1']
            )
            fig.update_layout(
                xaxis_tickangle=-45,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white'),
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                )
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Genre evolution over time with animation
        if len(selected_genres) > 0:
            st.subheader("📈 Selected Genres Over Time")
            genre_year_data = genre_df[genre_df['genres'].isin(selected_genres)].groupby(['year_added', 'genres']).size().reset_index(name='count')
            
            fig = px.line(
                genre_year_data,
                x='year_added',
                y='count',
                color='genres',
                title="Evolution of Selected Genres",
                labels={'year_added': 'Year', 'count': 'Number of Titles'},
                line_shape='spline',
                color_discrete_sequence=px.colors.sequential.Reds[1:]
            )
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white'),
                xaxis=dict(showgrid=False),
                yaxis=dict(showgrid=False)
            )
            st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.subheader("⏱️ Duration Analysis")
    
    if not filtered_df.empty:
        col1, col2 = st.columns(2)
        
        # Movies duration analysis
        movies_df = filtered_df[filtered_df['type'] == 'Movie'].copy()
        if not movies_df.empty and 'duration_minutes' in movies_df.columns:
            with col1:
                st.write("🎬 **Movie Durations**")
                
                # Duration distribution with custom bins
                fig = px.histogram(
                    movies_df.dropna(subset=['duration_minutes']),
                    x='duration_minutes',
                    nbins=20,
                    title="Movie Duration Distribution",
                    labels={'duration_minutes': 'Duration (minutes)', 'count': 'Number of Movies'},
                    color_discrete_sequence=['#E50914']
                )
                fig.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='white'),
                    xaxis=dict(showgrid=False),
                    yaxis=dict(showgrid=False)
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Duration stats in columns
                if not movies_df['duration_minutes'].isna().all():
                    col1a, col1b, col1c = st.columns(3)
                    with col1a:
                        avg_duration = movies_df['duration_minutes'].mean()
                        st.metric("Average", f"{avg_duration:.0f} min")
                    with col1b:
                        min_duration = movies_df['duration_minutes'].min()
                        st.metric("Shortest", f"{min_duration:.0f} min")
                    with col1c:
                        max_duration = movies_df['duration_minutes'].max()
                        st.metric("Longest", f"{max_duration:.0f} min")
        
        # TV Shows seasons analysis
        tv_df = filtered_df[filtered_df['type'] == 'TV Show'].copy()
        if not tv_df.empty and 'duration_seasons' in tv_df.columns:
            with col2:
                st.write("📺 **TV Show Seasons**")
                
                season_counts = tv_df['duration_seasons'].value_counts().head(10)
                fig = px.bar(
                    x=season_counts.index,
                    y=season_counts.values,
                    title="TV Show Season Distribution",
                    labels={'x': 'Number of Seasons', 'y': 'Number of Shows'},
                    color=season_counts.values,
                    color_continuous_scale='reds'
                )
                fig.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='white'),
                    xaxis=dict(showgrid=False),
                    yaxis=dict(showgrid=False)
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Season stats in columns
                if not tv_df['duration_seasons'].isna().all():
                    col2a, col2b, col2c = st.columns(3)
                    with col2a:
                        avg_seasons = tv_df['duration_seasons'].mean()
                        st.metric("Average", f"{avg_seasons:.1f} seasons")
                    with col2b:
                        min_seasons = tv_df['duration_seasons'].min()
                        st.metric("Shortest", f"{min_seasons:.0f} season{'s' if min_seasons != 1 else ''}")
                    with col2c:
                        max_seasons = tv_df['duration_seasons'].max()
                        st.metric("Longest", f"{max_seasons:.0f} seasons")

with tab4:
    st.subheader("☁️ Text Analysis")
    
    if not filtered_df.empty:
        col1, col2 = st.columns(2)
        
        with col1:
            # Word cloud with Netflix colors
            if not filtered_df['title'].empty:
                title_text = ' '.join(filtered_df['title'].dropna().astype(str))
                if title_text.strip():
                    wordcloud = WordCloud(
                        width=800, 
                        height=400, 
                        background_color='#0F0F0F',
                        colormap='Reds',
                        max_words=100,
                        contour_width=2,
                        contour_color='#E50914'
                    ).generate(title_text)
                    
                    fig, ax = plt.subplots(figsize=(10, 5))
                    ax.imshow(wordcloud, interpolation='bilinear')
                    ax.axis("off")
                    ax.set_title("Title Word Cloud", fontsize=16, pad=20, color='white')
                    fig.patch.set_facecolor('#0F0F0F')
                    st.pyplot(fig)
        
        with col2:
            # Description length analysis
            fig = px.histogram(
                filtered_df,
                x='description_length',
                nbins=30,
                title="Description Length Distribution",
                labels={'description_length': 'Characters', 'count': 'Number of Titles'},
                color_discrete_sequence=['#E50914']
            )
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white'),
                xaxis=dict(showgrid=False),
                yaxis=dict(showgrid=False)
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Stats in columns
            col2a, col2b = st.columns(2)
            with col2a:
                avg_desc_length = filtered_df['description_length'].mean()
                st.metric("Average Length", f"{avg_desc_length:.0f} chars")
            with col2b:
                avg_title_length = filtered_df['title_length'].mean()
                st.metric("Average Title Length", f"{avg_title_length:.0f} chars")

with tab5:
    st.subheader("😊 Sentiment Analysis")
    
    if not filtered_df.empty:
        col1, col2 = st.columns(2)
        
        with col1:
            # Sentiment distribution with Netflix colors
            sentiment_counts = filtered_df['sentiment_label'].value_counts()
            colors = {'Positive': '#2ECC71', 'Neutral': '#F39C12', 'Negative': '#E74C3C'}
            
            fig = px.pie(
                values=sentiment_counts.values,
                names=sentiment_counts.index,
                title="Sentiment Distribution",
                color=sentiment_counts.index,
                color_discrete_map=colors,
                hole=0.4
            )
            fig.update_traces(
                textposition='inside', 
                textinfo='percent+label',
                marker=dict(line=dict(color='#1F1F1F', width=2))
            )
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white')
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Sentiment score distribution
            fig = px.histogram(
                filtered_df,
                x='sentiment_score',
                nbins=30,
                title="Sentiment Score Distribution",
                labels={'sentiment_score': 'Sentiment Score', 'count': 'Count'},
                color_discrete_sequence=['#E50914']
            )
            fig.add_vline(
                x=0, 
                line_dash="dash", 
                line_color="white", 
                annotation_text="Neutral", 
                annotation_position="top"
            )
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white'),
                xaxis=dict(showgrid=False),
                yaxis=dict(showgrid=False)
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # NEW: Recommended titles based on sentiment
        st.subheader("🌟 Recommended Titles")
        recommended_titles = filtered_df[filtered_df['sentiment_label'] == 'Positive'].nlargest(5, 'sentiment_score')
        if not recommended_titles.empty:
            st.dataframe(
                recommended_titles[['title', 'type', 'primary_country', 'sentiment_score']].style
                .background_gradient(cmap='Greens', subset=['sentiment_score'])
                .format({'sentiment_score': '{:.2f}'}),
                use_container_width=True
            )
        else:
            st.info("No highly-rated titles found with current filters")

with tab6:
    st.subheader("📈 Advanced Trends")
    
    if not filtered_df.empty:
        # Content growth rate with Netflix colors
        yearly_counts = filtered_df['year_added'].value_counts().sort_index()
        if len(yearly_counts) > 1:
            fig = px.line(
                x=yearly_counts.index,
                y=yearly_counts.values,
                title="Content Growth Over Time",
                labels={'x': 'Year', 'y': 'Number of Titles Added'},
                line_shape='spline'
            )
            fig.update_traces(
                mode='lines+markers', 
                line_color='#E50914',
                marker=dict(size=8, color='#E50914')
            )
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white'),
                xaxis=dict(showgrid=False),
                yaxis=dict(showgrid=False)
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Content type trends over time
        type_trends = filtered_df.groupby(['year_added', 'type']).size().unstack().fillna(0)
        if not type_trends.empty:
            fig = px.area(
                type_trends,
                title="Content Type Trends Over Time",
                labels={'value': 'Number of Titles', 'year_added': 'Year'},
                color_discrete_sequence=['#E50914', '#00A8E1']
            )
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white'),
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                )
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Country trends
        if selected_countries:
            country_trends = filtered_df[filtered_df['primary_country'].isin(selected_countries)]
            country_trends = country_trends.groupby(['year_added', 'primary_country']).size().unstack().fillna(0)
            
            if not country_trends.empty:
                fig = px.line(
                    country_trends,
                    title="Content by Country Over Time",
                    labels={'value': 'Number of Titles', 'year_added': 'Year'},
                    color_discrete_sequence=px.colors.sequential.Reds[1:]
                )
                fig.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='white'),
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.02,
                        xanchor="right",
                        x=1
                    )
                )
                st.plotly_chart(fig, use_container_width=True)
        
        # Correlation analysis
        st.subheader("🔍 Correlation Analysis")
        
        numeric_cols = filtered_df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 1:
            corr_matrix = filtered_df[numeric_cols].corr()
            
            fig = px.imshow(
                corr_matrix,
                text_auto=True,
                aspect="auto",
                color_continuous_scale='RdBu',
                title="Correlation Matrix"
            )
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white')
            )
            st.plotly_chart(fig, use_container_width=True)

# Footer with Netflix-style branding
st.markdown("---")
st.markdown("""
<div style="text-align: center; padding: 20px; color: #A0A0A0;">
    <p>Netflix Analytics Dashboard • Powered by Streamlit • Data from Netflix</p>
    <p style="font-size: 12px;">© 2023 Netflix Clone Analytics • Not affiliated with Netflix</p>
</div>
""", unsafe_allow_html=True)