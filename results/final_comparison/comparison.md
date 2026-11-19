# Baseline vs Enhanced (test split)

Same data, epochs=20, imgsz=480, batch=16, seed=42, CPU.

| Metric | Baseline YOLO11n | Enhanced YOLO11n-P2-CBAM | Change |
|---|---|---|---|
| precision | 0.7465 | 0.7431 | -0.5% |
| recall | 0.5557 | 0.5583 | +0.5% |
| mAP50 | 0.5900 | 0.5964 | +1.1% |
| mAP50_95 | 0.3109 | 0.3031 | -2.5% |
| params | 2590035 | 2754732 | |
| cbam_layers | 0 | 4 | |
| inference_ms | 30.1 | 57.5 | |
| fps_model_only | 33.3 | 17.4 | |
| train_minutes | 153.5 | 714.6 | |
