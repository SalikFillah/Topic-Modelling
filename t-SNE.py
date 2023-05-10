# -*- coding: utf-8 -*-
"""
Created on Mon May  8 18:37:00 2023

@author: salik
"""
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output
import matplotlib.colors as mcolors

# Load data
df_dominant_topic = pd.read_csv('df_dominant_topic.csv')
df_top3words = pd.read_csv('df_top3words.csv')
tsne_df = pd.read_csv('tsne_df.csv')

# Set up app
app = dash.Dash(__name__)

# Create options for dropdown
platform_options = [{'label': i, 'value': i} for i in df_dominant_topic['social_media'].unique()]

# Create layout
app.layout = html.Div([
    html.H1('t-SNE Visualization'),
    dcc.Dropdown(
        id='platform-dropdown',
        options=platform_options,
        value=df_dominant_topic['social_media'].unique()[0]
    ),
    dcc.Graph(id='tsne-plot')
])

# Create callback
@app.callback(
    dash.dependencies.Output('tsne-plot', 'figure'),
    [dash.dependencies.Input('platform-dropdown', 'value')]
)
def update_figure(selected_platform):
    # Filter data based on selected platform
    filtered_df = df_dominant_topic[df_dominant_topic['social_media'] == selected_platform]
    
    # Join with top 3 words for each topic
    merged_df = filtered_df.merge(df_top3words, left_on='Dominant_Topic', right_on='topic_id')
    
    # Filter t-SNE data
    tsne_filtered = tsne_df[tsne_df['doc_num'].isin(filtered_df['Document_No'])]
    
    # Create scatter plot
    fig = go.Figure()
    for topic in merged_df['Dominant_Topic'].unique():
        topic_df = merged_df[merged_df['Dominant_Topic'] == topic]
        fig.add_trace(go.Scatter(
            x=tsne_filtered[tsne_filtered['doc_num'].isin(topic_df['Document_No'])]['tsne_x'],
            y=tsne_filtered[tsne_filtered['doc_num'].isin(topic_df['Document_No'])]['tsne_y'],
            mode='markers',
            marker=dict(size=7),
            name=f"Topic {topic} ({', '.join(topic_df['words'].iloc[0].split()[:3])})"
        ))
    
    # Update layout
    fig.update_layout(
        xaxis_title='t-SNE X',
        yaxis_title='t-SNE Y',
        title=f"t-SNE Visualization ({selected_platform})"
    )
    return fig

if __name__ == '__main__':
    app.run_server(debug=True)
