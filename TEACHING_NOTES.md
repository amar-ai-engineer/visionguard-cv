# VisionGuard — Teaching Notes

## Core concepts

### YOLO Architecture

YOLO (You Only Look Once) frames object detection as a single regression problem. Instead of proposing regions and then classifying them, it predicts bounding boxes and class probabilities directly from the full image in one forward pass.

**Grid division:** the image is divided into an S×S grid. Each cell predicts B bounding boxes and confidence scores. Each box has 5 values: (x, y, w, h, confidence). The confidence reflects both how certain the model is that an object exists and how accurate the box is (measured by IoU with ground truth).

**Anchor boxes:** YOLO v2+ uses predefined anchor shapes. Instead of predicting raw (w, h), it predicts offsets relative to an anchor. This makes it easier to learn — the model doesn't have to predict wildly different sizes from scratch.

**Multi-scale detection:** YOLOv3+ detects at three different scales (like FPN), which improves small-object detection. Features from deeper layers have more semantic meaning; features from shallower layers have more spatial detail.

**YOLOv11 improvements over v8:**
- Better backbone (C3k2 blocks)
- Improved neck design
- More efficient anchor-free detection head
- ~10% better mAP at similar speed

### Loss function

YOLOv1 used sum-squared error which treats localization and classification equally — bad. Modern YOLO uses:

- **CIoU loss** for bounding box regression — accounts for overlap, center distance, and aspect ratio
- **Binary cross-entropy** for classification
- **Focal loss** variant for objectness — down-weights easy negatives, focuses on hard examples

### Transfer learning for PPE

The base YOLOv11n is trained on COCO (80 classes, 330K images). When fine-tuning for PPE:

1. The backbone (feature extractor) weights are frozen or trained with a small LR
2. The detection head is replaced with new layers for your classes (helmet, no_helmet, vest, no_vest)
3. The new head is trained first, then the entire network is fine-tuned end to end

Why this works: deep features from COCO (edges, shapes, textures) are universally useful. Only the final mapping to PPE-specific classes needs to be learned from scratch.

### IoU and mAP

**IoU (Intersection over Union):**
```
IoU = Area(Predicted ∩ Ground Truth) / Area(Predicted ∪ Ground Truth)
```
An IoU of 0.5 is the standard threshold — if the box overlaps at least 50% with the ground truth, it counts as a detection.

**Precision-Recall curve:** as you lower the confidence threshold, recall goes up (you detect more real objects) but precision goes down (more false positives). The area under this curve is Average Precision (AP). mAP averages AP across all classes.

**mAP@0.5:0.95** is stricter — it averages over IoU thresholds from 0.5 to 0.95. This penalizes imprecise boxes.

### Real-world deployment considerations

- **Edge vs cloud:** YOLOv11n is small enough to run on a Raspberry Pi 5 or Jetson Nano at ~10-15 FPS. For a fixed camera, that's plenty.
- **Occlusion:** workers often overlap. Using tracking (DeepSORT, ByteTrack) instead of per-frame detection avoids counting one person twice.
- **Lighting:** construction sites have harsh, variable lighting. Data augmentation during training (random brightness, saturation, noise) helps generalization.
- **False negatives in safety:** missing a real violation is more expensive than a false alarm. Tune the confidence threshold toward lower values for safety applications.

### Dataset labeling tips

- Use Roboflow or Label Studio for annotation
- Aim for at least 300-500 labeled images per class
- Include diverse conditions: day/night, different angles, partial occlusion, various PPE colors
- Augment during training: flips, rotations, mosaic (YOLO's 4-image stitch)

---

## Common questions

**Q: Why not use a classification model instead?**
Classification tells you "there's a person without a helmet in this image." Detection gives you a bounding box, so you can tell *which* worker is non-compliant and where they are in the frame — essential for alerting supervisors.

**Q: How much data do you need?**
With COCO pre-training, 500-1,000 labeled images per class can get you to mAP@0.5 > 0.8. More data always helps, especially for edge cases.

**Q: What about GDPR / worker privacy?**
In the EU, video surveillance of workers requires explicit policy, notice, and often worker consent. Anonymization (blur faces, use silhouettes) can reduce privacy concerns while still enabling compliance detection.
