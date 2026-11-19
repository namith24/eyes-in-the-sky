# 960-px gain under simulated ESP32 capture (test split, COCO-style, IoU 0.5)

JPEG quality 70; downscale only. Median image scale: original (C2A file): 1.00, ESP32 640 x 480: 1.00, ESP32 320 x 240: 0.59

| Camera | Config | mAP@50 | AP@50 <10px | Recall@0.25 | Recall@0.25 <10px |
|---|---|---|---|---|---|
| original (C2A file) | baseline @480 | 0.588 | 0.382 | 0.588 | 0.411 |
| original (C2A file) | proposed @960 | 0.711 | 0.598 | 0.744 | 0.665 |
| ESP32 640 x 480 | baseline @480 | 0.236 | 0.078 | 0.188 | 0.060 |
| ESP32 640 x 480 | proposed @960 | 0.108 | 0.036 | 0.067 | 0.032 |
| ESP32 320 x 240 | baseline @480 | 0.087 | 0.024 | 0.037 | 0.009 |
| ESP32 320 x 240 | proposed @960 | 0.031 | 0.005 | 0.013 | 0.006 |
