import pandas as pd
import plotly.graph_objects as go
import plotly.express as px


def create_compliance_gauge(score: float) -> go.Figure:
    pct = round(score * 100, 1)
    color = "#22c55e" if pct >= 90 else "#f59e0b" if pct >= 75 else "#ef4444"
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=pct,
        number={"suffix": "%", "font": {"size": 36, "color": "#0f172a"}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1},
            "bar": {"color": color},
            "steps": [
                {"range": [0, 75],   "color": "#fef2f2"},
                {"range": [75, 90],  "color": "#fefce8"},
                {"range": [90, 100], "color": "#f0fdf4"},
            ],
            "threshold": {"line": {"color": "#0f172a", "width": 3}, "value": 90},
        },
        title={"text": "Overall Compliance", "font": {"size": 14, "color": "#64748b"}},
    ))
    fig.update_layout(height=220, margin=dict(t=40, b=10, l=20, r=20), paper_bgcolor="rgba(0,0,0,0)")
    return fig


def create_zone_chart(zones_df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="Helmet Rate",
        x=zones_df["zone"],
        y=(zones_df["helmet_rate"] * 100).round(1),
        marker_color="#0ea5e9",
    ))
    fig.add_trace(go.Bar(
        name="Vest Rate",
        x=zones_df["zone"],
        y=(zones_df["vest_rate"] * 100).round(1),
        marker_color="#8b5cf6",
    ))
    fig.update_layout(
        barmode="group",
        height=300,
        margin=dict(t=20, b=80, l=40, r=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        yaxis=dict(range=[0, 110], ticksuffix="%"),
        xaxis=dict(tickangle=-20),
    )
    fig.add_hline(y=90, line_dash="dash", line_color="#ef4444", annotation_text="90% target")
    return fig


def create_trend_chart(trend_df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=trend_df["date"],
        y=trend_df["score_pct"],
        mode="lines+markers",
        line=dict(color="#0ea5e9", width=2),
        marker=dict(size=5),
        fill="tozeroy",
        fillcolor="rgba(14,165,233,0.08)",
        name="Compliance %",
    ))
    fig.add_hline(y=90, line_dash="dash", line_color="#ef4444", annotation_text="Target 90%")
    fig.update_layout(
        height=280,
        margin=dict(t=20, b=40, l=50, r=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        yaxis=dict(range=[60, 100], ticksuffix="%"),
        xaxis=dict(title="Date"),
    )
    return fig


def create_alert_table(alerts_df: pd.DataFrame) -> go.Figure:
    status = alerts_df["resolved"].map({True: "Resolved", False: "Open"})
    colors = [["#f0fdf4" if r else "#fef2f2" for r in alerts_df["resolved"]]] * 5
    fig = go.Figure(go.Table(
        header=dict(
            values=["Time", "Zone", "Violation", "Worker", "Status"],
            fill_color="#0f172a",
            font=dict(color="white", size=12),
            align="left",
        ),
        cells=dict(
            values=[
                alerts_df["time"].tolist(),
                alerts_df["zone"].tolist(),
                alerts_df["type"].tolist(),
                alerts_df["worker_id"].tolist(),
                status.tolist(),
            ],
            fill_color=colors,
            align="left",
            font=dict(size=11),
        ),
    ))
    fig.update_layout(height=260, margin=dict(t=10, b=10, l=0, r=0))
    return fig


def create_hourly_chart(hourly_df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Bar(name="Workers Detected", x=hourly_df["hour"], y=hourly_df["workers"], marker_color="#0ea5e9"))
    fig.add_trace(go.Bar(name="Violations",       x=hourly_df["hour"], y=hourly_df["violations"], marker_color="#ef4444"))
    fig.update_layout(
        barmode="overlay",
        height=260,
        margin=dict(t=20, b=40, l=40, r=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    return fig
