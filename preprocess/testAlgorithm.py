import ezc3d
import numpy as np
from sklearn.metrics import classification_report
from scipy.spatial.distance import cdist

def load_c3d_data_labels(filepath):
    c3d = ezc3d.c3d(filepath)
    points = c3d['data']['points'][:3]  # shape: [3, M, F]
    points = np.transpose(points, (2, 1, 0))  # to shape: [F, M, 3]
    labels = c3d['parameters']['POINT']['LABELS']['value']
    return points, labels

def match_frame_markers(gt_frame, gt_labels, pred_frame, pred_labels, threshold=30):
    y_true = []
    y_pred = []

    gt_valid_idx = np.where(~np.isnan(gt_frame[:, 0]))[0]
    pred_valid_idx = np.where(~np.isnan(pred_frame[:, 0]))[0]

    if len(gt_valid_idx) == 0 or len(pred_valid_idx) == 0:
        return y_true, y_pred

    gt_pts = gt_frame[gt_valid_idx]
    pred_pts = pred_frame[pred_valid_idx]

    distances = cdist(pred_pts, gt_pts)
    used_gt = set()
    used_pred = set()

    for i_pred, row in enumerate(distances):
        i_gt = np.argmin(row)
        dist = row[i_gt]

        if dist < threshold and i_gt not in used_gt:
            y_true.append(gt_labels[gt_valid_idx[i_gt]])
            y_pred.append(pred_labels[pred_valid_idx[i_pred]])
            used_gt.add(i_gt)
            used_pred.add(i_pred)

    # unmatched predicted
    for i_pred in range(len(pred_valid_idx)):
        if i_pred not in used_pred:
            y_pred.append(pred_labels[pred_valid_idx[i_pred]])
            y_true.append("None")

    # unmatched GT
    for i_gt in range(len(gt_valid_idx)):
        if i_gt not in used_gt:
            y_pred.append("None")
            y_true.append(gt_labels[gt_valid_idx[i_gt]])

    return y_true, y_pred

def compare_c3d_framewise(gt_path, pred_path, threshold=30):
    gt_points, gt_labels = load_c3d_data_labels(gt_path)
    pred_points, pred_labels = load_c3d_data_labels(pred_path)

    assert gt_points.shape[0] == pred_points.shape[0], "Frame counts do not match"

    all_y_true = []
    all_y_pred = []

    for f in range(gt_points.shape[0]):
        gt_frame = gt_points[f]
        pred_frame = pred_points[f]

        y_true, y_pred = match_frame_markers(gt_frame, gt_labels, pred_frame, pred_labels, threshold)
        all_y_true.extend(y_true)
        all_y_pred.extend(y_pred)

    print("\nFrame-by-frame classification report:")
    print(classification_report(all_y_true, all_y_pred, zero_division=0))

# Example usage
gt_c3d_path = '41_1_Gait_02_08.c3d'
pred_c3d_path = 'unlabeled_41_1_Gait_02_08_labelled.c3d'

compare_c3d_framewise(gt_c3d_path, pred_c3d_path, threshold=30)
