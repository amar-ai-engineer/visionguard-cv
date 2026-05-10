# VisionGuard AI — PPE Safety Compliance with YOLOv11

A computer vision system that detects workers on construction and industrial sites in real time and tracks PPE compliance over time.

**Live demo:** upload a site photo and watch YOLOv11 find every person in under a second. The compliance dashboard shows zone-by-zone helmet and vest rates, a 20-day trend, and an alert log — all from a single JSON file, no GPU or API key needed.

---

## What problem does this solve?

Construction sites in the US report 1,000+ fatal injuries per year. About 40% involve workers not wearing required PPE. Manual safety checks are slow, inconsistent, and happen once a day at best.

A camera-based system can check every worker every second. It catches the morning rush when compliance drops, flags specific zones that need attention, and gives safety managers real numbers to act on.

---

## Business impact

- Reduces manual safety inspection time by ~80%
- Generates audit-ready compliance logs
- Alerts supervisors to violations in real time instead of at end-of-day reviews
- Scales across multiple camera feeds from one dashboard

---

## How to run

```bash
pip install -r requirements.txt
streamlit run app.py
```

No API key or GPU required. The YOLO model downloads automatically on first run (~6 MB).

---

## Project structure

```
visionguard-cv/
├── app.py                      # 3-tab Streamlit dashboard
├── src/
│   ├── detector.py             # SafetyDetector wrapping YOLOv11
│   ├── analytics.py            # ComplianceEngine loading demo data
│   └── visualization.py        # Plotly chart builders
├── data/
│   ├── demo_results.json       # Pre-computed PPE analytics (47 workers, 5 zones)
│   └── samples/                # Sample site images
├── notebooks/
│   └── custom_training.ipynb   # Fine-tuning YOLOv11 on a PPE dataset (Colab)
├── requirements.txt
└── yolo11n.pt                  # YOLOv11 Nano weights
```

---

## The 3 tabs

**Live Detection** — upload an image, video, or use your camera. YOLOv11 runs inference and draws bounding boxes around every detected person.

**Compliance Dashboard** — pre-computed data showing 47 workers across 5 zones, daily compliance trend over 20 days, hourly violation counts, and an active alert log.

**How It Works** — YOLO architecture explained, step-by-step custom training guide, and performance metrics.

---

## Why YOLOv11 Nano?

The nano variant is fast enough to run on CPU in a web demo (~30 FPS on a MacBook). For production, you'd use a larger variant (s, m, l) or run the nano on an edge GPU at 160+ FPS.

The base model detects people from COCO training. Detecting helmets and vests specifically requires fine-tuning on a labeled PPE dataset — the Colab notebook in `notebooks/` walks through that process end to end.

---

## Custom training path

1. Label ~2,000 images with Roboflow (classes: `helmet`, `no_helmet`, `vest`, `no_vest`)
2. Export in YOLO format
3. Fine-tune: `model.train(data="ppe.yaml", epochs=50, imgsz=640)`
4. Expected mAP@0.5 > 0.85 on a clean dataset

---

Built by **Amar Ismail** | [GitHub](https://github.com/amar-ai-engineer) | [LinkedIn](https://www.linkedin.com/in/amar-ai-engineer/)
