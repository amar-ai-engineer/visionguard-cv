import json
import os
import pandas as pd


class ComplianceEngine:
    def __init__(self, data_path: str = "data/demo_results.json"):
        self._path = data_path
        self._data = None

    def load_demo_data(self) -> dict:
        if self._data is None:
            with open(self._path) as f:
                self._data = json.load(f)
        return self._data

    def get_summary(self) -> dict:
        return self.load_demo_data()["summary"]

    def get_zone_breakdown(self) -> pd.DataFrame:
        zones = self.load_demo_data()["zone_breakdown"]
        return pd.DataFrame(zones)

    def get_compliance_trend(self) -> pd.DataFrame:
        trend = self.load_demo_data()["compliance_trend"]
        df = pd.DataFrame(trend)
        df["date"] = pd.to_datetime(df["date"])
        df["score_pct"] = (df["score"] * 100).round(1)
        return df

    def get_alerts(self) -> pd.DataFrame:
        alerts = self.load_demo_data()["alerts"]
        return pd.DataFrame(alerts)

    def get_hourly_detections(self) -> pd.DataFrame:
        hourly = self.load_demo_data()["hourly_detections"]
        return pd.DataFrame(hourly)
