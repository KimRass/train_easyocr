# Metric
- Reference: https://github.com/rafaelpadilla/Object-Detection-Metrics#precision-x-recall-curve
- Reference: https://towardsdatascience.com/what-is-average-precision-in-object-detection-localization-algorithms-and-how-to-calculate-it-3f330efe697b
## True Positive, False Positive, False Negative
- **A prediction is said to be correct if the class label of the predicted bounding box and the ground truth bounding box is the same and the IoU between them is greater than a threshold value.**
  - *True Positive: The model predicted that a bounding box exists at a certain position (positive) and it was correct (true)*
  - *False Positive: The model predicted that a bounding box exists at a particular position (positive) but it was wrong (false)*
  - *False Negative: The model did not predict a bounding box at a certain position (negative) and it was wrong (false) i.e. a ground truth bounding box existed at that position.*
  - True Negative: The model did not predict a bounding box (negative) and it was correct (true). This corresponds to the background, the area without bounding boxes, and is not used to calculate the final metrics.
- ![object_detection](https://miro.medium.com/max/1400/1*mdqpx5V7TYhXRz046tm6zQ.webp)
  - Blue boxes: Predicted bounding boxes이므로 Positive이고, True Positive인 경우 그에 대응하는 Ground truth bounding box와 짝을 이룹니다. 이 경우 Bunding boxes는 2개지만 True Positive는 1개입니다.