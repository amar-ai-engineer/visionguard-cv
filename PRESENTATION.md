# VisionGuard AI — Presentation Script (6 slides)

---

## Slide 1: The Problem

"Construction is one of the most dangerous industries in the world. In the US alone, over 1,000 workers die on job sites every year. About 40% of those deaths involve a worker who wasn't wearing the right protective equipment — helmet, vest, harness.

The challenge isn't that safety rules don't exist. It's that enforcement relies on a safety officer walking the floor a few times a day. That's one person, checking one area, at one moment in time."

---

## Slide 2: The Solution

"VisionGuard uses YOLOv11 — the same family of computer vision models used in self-driving cars and security systems — to check every camera feed, every second.

It detects workers in real time, tracks PPE compliance zone by zone, and sends alerts when violations happen — not at the end of the day in a report.

One camera replaces a full-time safety observer. A system of cameras covers the whole site."

---

## Slide 3: Live Demo

"Let me show you the live detection tab. I'll upload a site photo here — [upload image] — and in about half a second, YOLOv11 has found every person in the frame with a bounding box.

What you're seeing is the base model detecting people. In a production deployment, you'd fine-tune this on labeled PPE data — I'll show that process in the How It Works tab — and it would draw separate boxes for helmets, vests, and flag workers who are missing either."

---

## Slide 4: Compliance Dashboard

"The second tab is where the business value lives. This shows the pre-computed analytics for a 47-worker site across five zones over 20 working days.

Zone C — the scaffolding area — is consistently the lowest compliance zone at around 75-80%. That's the supervisor's first conversation to have on Monday morning.

The compliance trend shows we went from 78% to 89% over 20 days. That's the kind of measurable improvement a safety manager can show to their board."

---

## Slide 5: Technical Architecture

"Under the hood, this is a three-layer system:

The detector layer wraps YOLOv11 — it takes a frame, returns annotated output and a person count. Swapping in a custom-trained model is one line of code.

The analytics layer loads pre-computed compliance data and computes zone breakdowns, trends, and alerts. In production, this would consume live detections from the camera feeds.

The visualization layer builds Plotly charts — gauge, heatmap, trend line — from that analytics data and renders them in the Streamlit dashboard."

---

## Slide 6: Business Case and Next Steps

"The ROI here is straightforward. A safety officer costs about $60,000 a year. A camera system with this software costs a fraction of that and covers every zone simultaneously, around the clock.

For a client, the immediate deliverable would be a 30-day pilot on one site — measure baseline compliance, deploy cameras, run the dashboard, report back on trend and incident reduction.

The custom training notebook is ready. Give me a labeled dataset from your site and I can have a PPE-specific model in two weeks."
