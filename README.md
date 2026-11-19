# Eyes in the Sky: UAV Survivor Detection

Ground-station software for low-cost drone search and rescue. A drone streams video to a laptop, and a
small-object-aware YOLO11n detector marks people who are often only a few pixels tall. It runs on an
ordinary laptop CPU and works offline on the drone's own WiFi.

Capstone project, School of Computer Science and Engineering (SCOPE), VIT-AP University, 2026.
Team: K. Namith Surya, K. Yaswanth Kumar, R. Naveen Surya, Siva Sai Venkata Charan.
Guide: Dr. Lalitha Kumari P.

## Results (C2A test split, 300 images, 10,605 people)

| Setting | mAP@50 | Recall, people < 10 px | FPS (laptop CPU) |
|---|---|---|---|
| YOLO11n @ 480 px (baseline) | 0.588 | 41.1% | 27.3 |
| + P2 head + CBAM @ 480 px | 0.595 | 39.7% | 19.8 |
| **YOLO11n @ 960 px (proposed)** | **0.711** | **66.5%** | **11.8** |

The gain comes from inference resolution, not from extra layers. On JPEG-compressed frames like those from
the ESP32-CAM, the 960 px gain disappears (see `scripts/camera_resolution_eval.py`).

## Two ways to run it

### 1. Online demo (Streamlit Community Cloud)

Shows the dashboard, results, story and the survivor finder on a 40-image sample of the test set, and can
replay the recorded flight. **Live drone video and the webcam need the laptop version**, because a cloud
server cannot reach the drone's WiFi or your camera.

### 2. Ground-station laptop (live drone, offline)

```bash
git clone <this repository>
cd <repository folder>
pip install -r requirements.txt
```

Then double-click **`Start Eyes in the Sky.bat`**. It starts the app in a small background window and
opens it in the browser. Closing that window stops the app. You can also run
`streamlit run app/demo_app.py` directly.

To fly: connect the laptop to the drone's WiFi (`SAR-DRONE`), open **Live Drone**, and choose
*Drone WiFi stream*. The default address is `http://192.168.4.1/stream`.

## Repository layout

| Path | Contents |
|---|---|
| `app/demo_app.py` | The ground-station web app (5 pages) |
| `src/uav_survivor_detection/` | Dataset, training, evaluation and inference package (drone link, detector) |
| `runs/final_comparison/*/weights/best.pt` | Trained baseline and P2-CBAM weights |
| `results/final_comparison/` | Evaluation results that the app and report read |
| `configs/` | Training and model configs (`models/yolo11n-p2-cbam.yaml`) |
| `scripts/` | Training, evaluation and experiment scripts |
| `data/demo_test/` | 40-image sample of the C2A test split (full dataset not included) |
| `tests/` | Automated tests (`pytest`) |

## Dataset

C2A: R. A. Nihal et al., "UAV-Enhanced Combination to Application: Comprehensive Analysis and
Benchmarking of a Human Detection Dataset for Disaster Scenarios," 2024. Only a small sample is included
here; download the full dataset from its authors to retrain.
