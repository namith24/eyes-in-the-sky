# Size-stratified results (test split, COCO-style, recall/AP at IoU 0.5)

Instances: all: 10605, tiny (<10px): 4630, small (10-32px): 5885, medium+ (>=32px): 90

recall = max recall over all detections; recall@0.25 = recall at the demo's conf>=0.25 threshold

| Run | all AP50 | all recall | all recall@0.25 | tiny (<10px) AP50 | tiny (<10px) recall | tiny (<10px) recall@0.25 | small (10-32px) AP50 | small (10-32px) recall | small (10-32px) recall@0.25 | medium+ (>=32px) AP50 | medium+ (>=32px) recall | medium+ (>=32px) recall@0.25 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline@480 | 0.588 | 0.755 | 0.588 | 0.382 | 0.539 | 0.411 | 0.747 | 0.927 | 0.733 | 0.223 | 0.678 | 0.211 |
| enhanced@480 | 0.595 | 0.788 | 0.576 | 0.391 | 0.594 | 0.397 | 0.750 | 0.944 | 0.725 | 0.079 | 0.533 | 0.067 |
| baseline@800 | 0.707 | 0.876 | 0.722 | 0.565 | 0.774 | 0.621 | 0.813 | 0.962 | 0.810 | 0.232 | 0.489 | 0.200 |
| enhanced@800 | 0.691 | 0.889 | 0.705 | 0.555 | 0.803 | 0.610 | 0.790 | 0.963 | 0.788 | 0.138 | 0.433 | 0.133 |
