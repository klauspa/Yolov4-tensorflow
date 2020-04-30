#greedy nms tensorflow implementation
import tensorflow as tf

CONFIDENCE_THRESHOLD = 0.45
NUM_CLASS = 80
MAX_BOX_NUM = 20

def xywh2xyxy(x):
    y = x.new(x.shape)
    y[..., 0] = x[..., 0] - x[..., 2] / 2
    y[..., 1] = x[..., 1] - x[..., 3] / 2
    y[..., 2] = x[..., 0] + x[..., 2] / 2
    y[..., 3] = x[..., 1] + x[..., 3] / 2
    return y

def nms(pred_boxes, conf_thres=0.5, nms_thres=0.4):
    pred_boxes[..., :4] = xywh2xyxy(pred_boxes[..., :4])
    output = [None for _ in range(len(pred_boxes))]

    for image_i, image_pred in enumerate(pred_boxes):
        # Filter out confidence scores below threshold
        image_pred = image_pred[image_pred[:, 4] >= conf_thres]
        # If none are remaining => process next image
        if not image_pred.size(0):
            continue
        # Object confidence times class confidence
        score = image_pred[:, 4] * image_pred[:, 5:].max(1)[0]
        # Sort by it
        image_pred = image_pred[(-score).argsort()]
        boxes = image_pred[:, :4]
        scores = image_pred[:, 4]
        selected_indices = tf.image.non_max_suppression(
                boxes, scores, MAX_BOX_NUM, nms_thres
                )
        selected_boxes = tf.gather(boxes, selected_indices)
        selected_scores = tf.gather(scores, selected_indices)