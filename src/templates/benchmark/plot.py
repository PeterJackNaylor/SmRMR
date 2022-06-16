import pandas as pd
import sys
import plotly.graph_objects as go

table = pd.read_csv(sys.argv[1], sep="\t")

names = {
    "stg": "STG",
    "DC_lasso (L1)": "DC-Lasso (L1)",
}

colors = {}

print("group table by datasets")
metrics = []
y = []
for name in table.groupby("name"):
    fig = go.Figure()
    # shortname =
    fig.add_trace(
        go.Bar(
            x=metrics,
            y=y,
            # name='Rest of world',
            # marker_color='rgb(26, 118, 255)'
        )
    )

fig.update_layout(
    title="Comparing different methods on the simulated datasets",
    xaxis_tickfont_size=14,
    yaxis=dict(
        title="USD (millions)",
        titlefont_size=16,
        tickfont_size=14,
    ),
    legend=dict(
        x=0,
        y=1.0,
        bgcolor="rgba(255, 255, 255, 0)",
        bordercolor="rgba(255, 255, 255, 0)",
    ),
    barmode="group",
    bargap=0.15,  # gap between bars of adjacent location coordinates.
    bargroupgap=0.1,  # gap between bars of the same location coordinate.
)
fig.show()
