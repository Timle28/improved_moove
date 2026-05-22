# utils/dash_utils.py
import ctypes
import dash
import numpy as np
import pandas as pd
import plotly.graph_objs as go
import socket
import string
import threading
import os
import webbrowser
import logging
from dash import dcc, html
from dash.dependencies import Input, Output, State
from flask import Flask
from wsgiref.simple_server import make_server, WSGIRequestHandler

from moove.qt_helpers import invoke_in_main_thread, show_info

logging.getLogger('werkzeug').setLevel(logging.ERROR)
logging.getLogger('flask').setLevel(logging.ERROR)
logging.getLogger('dash').setLevel(logging.ERROR)


class QuietWSGIRequestHandler(WSGIRequestHandler):
    """Custom request handler that suppresses HTTP request logging."""
    def log_message(self, format, *args):
        pass

labels = None
point_size = 5


def run_flask_server(app_state, dataset_name):
    """Create and run a Flask server for the Dash app."""

    server = Flask(__name__)
    app = dash.Dash(__name__, server=server)

    global labels, point_size
    df = pd.read_pickle(os.path.join(app_state.config['global_dir'], 'cluster_data', f'{dataset_name}.pkl'))
    if 'clustered_label' not in df.columns:
        raise KeyError("Dataset has not been clustered yet. Please cluster the dataset first.")
    labels = df['clustered_label'].values
    low_dimensional_data = df[['UMAP1', 'UMAP2']].values

    unique_labels = list(string.ascii_lowercase)
    label_mapping = {label: idx for idx, label in enumerate(unique_labels)}
    numeric_labels = [label_mapping[label] for label in labels]

    common_style = {'font-family': 'Arial', 'font-size': '14px', 'height': '15px', 'width': '30px', 'margin': '2px'}
    button_style = {'font-family': 'Arial', 'font-size': '13px', 'height': '20px', 'margin': '2px'}

    app.layout = html.Div([
        dcc.Graph(
            id='scatter-plot',
            figure={
                'data': [
                    go.Scatter(
                        x=low_dimensional_data[:, 0],
                        y=low_dimensional_data[:, 1],
                        mode='markers',
                        marker=dict(color=numeric_labels, colorscale='Jet', size=point_size),
                        selected={'marker': {'color': 'red'}},
                        unselected={'marker': {'opacity': 0.5}},
                        text=labels,
                        hoverinfo='text',
                    )
                ],
                'layout': go.Layout(
                    title=dict(
                        text='UMAP Clustering',
                        font=dict(size=24, family='Arial', color='black')
                    ),
                    clickmode='event+select',
                    margin=dict(l=20, r=20, t=40, b=20),
                    autosize=True
                )
            },
            style={'width': '100%', 'height': '100%', 'flex': '1 1 auto'}
        ),
        html.Div([
            html.Div([
                html.Label("Label for selected points:", style={**common_style, 'width': 'auto'}),
                dcc.Input(id='input-label', type='text', style={**common_style, 'margin-right': '5px'}),
                html.Button('Apply', id='apply-button', style=button_style),

                html.Label("Change all labels from:", style={**common_style, 'width': 'auto'}),
                dcc.Input(id='input-label-old', type='text', style={**common_style, 'margin-right': '5px'}),
                html.Label("to:", style={**common_style, 'width': 'auto'}),
                dcc.Input(id='input-label-new', type='text', style={**common_style, 'margin-right': '5px'}),
                html.Button('Change All', id='change-all-button', style=button_style),
            ], style={
                'display': 'flex',
                'align-items': 'center',
                'flex-wrap': 'nowrap',
                'min-width': '0'
            }),
            html.Div([
                html.Button('Save', id='save-button', style=button_style),
            ], style={
                'display': 'flex',
                'justify-content': 'center',
                'align-items': 'center',
                'min-width': '0'
            }),
            html.Div([
                html.Button('Increase Point Size', id='increase-size-button', style=button_style),
                html.Button('Decrease Point Size', id='decrease-size-button', style=button_style),
            ], style={
                'display': 'flex',
                'align-items': 'center',
                'justify-content': 'flex-end',
                'flex-wrap': 'nowrap',
                'min-width': '0'
            }),
        ], style={
            'display': 'flex',
            'justify-content': 'space-between',
            'flex-wrap': 'nowrap',
            'width': '100%',
            'margin-top': '10px',
            'padding': '0px',
            'align-items': 'center'
        }),
        html.Div(id='output-container')
    ], style={
        'display': 'flex',
        'flex-direction': 'column',
        'height': '100vh',
        'padding': '10px',
        'box-sizing': 'border-box'
    })

    @app.callback(
        Output('scatter-plot', 'figure'),
        [
            Input('change-all-button', 'n_clicks'),
            Input('apply-button', 'n_clicks'),
            Input('increase-size-button', 'n_clicks'),
            Input('decrease-size-button', 'n_clicks')
        ],
        [
            State('input-label-old', 'value'),
            State('input-label-new', 'value'),
            State('scatter-plot', 'figure'),
            State('scatter-plot', 'selectedData'),
            State('input-label', 'value')
        ],
        prevent_initial_call=True
    )
    def update_labels_and_point_size(n_clicks_change_all, n_clicks_apply, n_clicks_increase, n_clicks_decrease,
                                     old_label, new_label, figure, selectedData, input_label):
        """Update the labels of selected points, change all labels, or adjust the point size."""
        global labels, point_size

        if n_clicks_change_all and old_label and new_label:
            labels = np.where(labels == old_label, new_label, labels)

        if n_clicks_apply and selectedData and input_label:
            points_indices = [point['pointIndex'] for point in selectedData['points']]
            for idx in points_indices:
                labels[idx] = input_label

        ctx = dash.callback_context
        if ctx.triggered:
            triggered_id = ctx.triggered[0]['prop_id'].split('.')[0]
            if triggered_id == 'increase-size-button':
                point_size = min(point_size + 1, 20)
            elif triggered_id == 'decrease-size-button':
                point_size = max(point_size - 1, 1)

        numeric_labels = [label_mapping[label] for label in labels]
        trace = figure['data'][0]
        trace['marker']['color'] = numeric_labels
        trace['marker']['size'] = point_size
        hover_text = [f'Label: {label}' for label in labels]
        trace['text'] = hover_text
        return figure

    @app.callback(
        Output('output-container', 'children'),
        Input('save-button', 'n_clicks'),
        prevent_initial_call=True
    )
    def save_labels(n_clicks):
        """Save the updated labels to the dataset file and show popup notification."""
        global labels
        df['clustered_label'] = labels
        df.to_pickle(os.path.join(app_state.config['global_dir'], 'cluster_data', f'{dataset_name}.pkl'))

        try:
            invoke_in_main_thread(lambda: show_info(
                None, "Success", f"Labels saved successfully!\n\nFile: {dataset_name}.pkl"))
        except:
            pass

        return html.Div()

    app_state.logger.debug(f"Starting Flask server for dataset: {dataset_name}")
    app_state.server = make_server('localhost', 8050, server, handler_class=QuietWSGIRequestHandler)
    app_state.server_thread = threading.Thread(target=app_state.server.serve_forever, name="DashServerThread")

    app_state.add_thread(app_state.server_thread)

    app_state.server_thread.start()
    threading.Timer(1, lambda: webbrowser.open("http://127.0.0.1:8050")).start()


def start_dash_app_thread(app_state, dataset_name):
    """Start the Dash application in a separate thread using AppState."""

    if dataset_name == "Select Cluster Dataset":
        show_info(None, "Error", "Selected cluster dataset not valid! Perhaps you forgot to pick a dataset?")
    else:
        if app_state.dash_thread and app_state.dash_thread.is_alive():
            app_state.logger.debug("Dash app is already running.")
            return

        if not is_port_available(8050):
            app_state.logger.warning("Port 8050 is already in use.")
            return

        run_flask_server(app_state, dataset_name)
        app_state.dash_thread = app_state.server_thread
        app_state.logger.debug("Dash app thread started.")


def is_port_available(port):
    """Check if a specific port is available."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('localhost', port)) != 0


def kill_thread(thread):
    """Forcefully terminate the given thread."""
    if thread.is_alive():
        _async_raise(thread.ident, SystemExit)


def _async_raise(tid, exctype):
    """Raise the specified exception type in the context of the specified thread."""
    if not isinstance(tid, int):
        raise TypeError("Thread ID must be an integer")
    if not isinstance(exctype, (type, type(None))) or not issubclass(exctype, BaseException):
        raise TypeError("Exception type must be a subclass of BaseException")
    res = ctypes.pythonapi.PyThreadState_SetAsyncExc(ctypes.c_long(tid), ctypes.py_object(exctype))
    if res == 0:
        raise ValueError("Invalid thread ID")
    elif res > 1:
        ctypes.pythonapi.PyThreadState_SetAsyncExc(ctypes.c_long(tid), None)
        raise SystemError("PyThreadState_SetAsyncExc failed")


def stop_dash_app_thread(app_state):
    """Properly stop the Dash application by shutting down the server and threads."""
    app_state.logger.debug("=== STOP DASH APP CALLED ===")

    try:
        if hasattr(app_state, 'server') and app_state.server:
            app_state.logger.debug("Shutting down Flask server.")
            app_state.server.shutdown()
            app_state.server = None
            app_state.logger.debug("Flask server shutdown complete.")
        else:
            app_state.logger.debug("No Flask server found to shutdown.")

        if hasattr(app_state, 'server_thread') and app_state.server_thread and app_state.server_thread.is_alive():
            app_state.logger.debug("Stopping server thread.")
            app_state.server_thread.join(timeout=2)
            app_state.server_thread = None
            app_state.logger.debug("Server thread stopped.")
        else:
            app_state.logger.debug("No server thread found to stop.")

        if app_state.dash_thread and app_state.dash_thread.is_alive():
            app_state.logger.debug("Stopping Dash application thread.")
            try:
                app_state.dash_thread.join(timeout=2)
                app_state.logger.debug("Dash thread joined gracefully.")
            except Exception as join_error:
                app_state.logger.debug(f"Graceful join failed: {join_error}. Forcefully terminating the Dash server thread.")
                kill_thread(app_state.dash_thread)
                app_state.logger.debug("Dash thread force killed.")

            app_state.dash_thread = None
        else:
            app_state.logger.debug("No Dash thread found to stop.")

        app_state.logger.debug("=== DASH APP SUCCESSFULLY STOPPED ===")

        try:
            invoke_in_main_thread(lambda: show_info(None, "Dash GUI", "Dash GUI closed successfully!"))
        except:
            pass

    except Exception as e:
        app_state.logger.error(f"Error while stopping Dash app: {e}")
        if app_state.dash_thread and app_state.dash_thread.is_alive():
            try:
                kill_thread(app_state.dash_thread)
                app_state.dash_thread = None
                app_state.logger.debug("Fallback force kill successful.")
            except Exception as kill_error:
                app_state.logger.error(f"Failed to force kill thread: {kill_error}")
