import numpy as np
from collections import defaultdict

def compute_shot_accuracies(train_labels, test_preds, test_labels):
    """
    Automatically splits classes into many/medium/few based on frequency percentiles.

    Args:
        train_labels (list or np.ndarray): Training set labels
        test_preds (list or np.ndarray): Predicted labels
        test_labels (list or np.ndarray): Ground truth labels

    Returns:
        dict: accuracy per shot category
    """
    train_labels = np.array(train_labels)
    test_preds = np.array(test_preds)
    test_labels = np.array(test_labels)

    # Count samples per class
    class_counts = defaultdict(int)
    for label in train_labels:
        class_counts[label] += 1

    # Sort classes by frequency
    sorted_classes = sorted(class_counts.items(), key=lambda x: x[1], reverse=True)
    class_ranks = [cls for cls, _ in sorted_classes]

    n_classes = len(class_ranks)
    one_third = n_classes // 3

    many_shot = set(class_ranks[:one_third])
    medium_shot = set(class_ranks[one_third:2*one_third])
    few_shot = set(class_ranks[2*one_third:])

    # Track correct predictions
    group_correct = {'many': 0, 'medium': 0, 'few': 0}
    group_total = {'many': 0, 'medium': 0, 'few': 0}

    for pred, true in zip(test_preds, test_labels):
        if true in many_shot:
            group = 'many'
        elif true in medium_shot:
            group = 'medium'
        elif true in few_shot:
            group = 'few'
        else:
            continue
        group_total[group] += 1
        if pred == true:
            group_correct[group] += 1

    acc = {}
    for group in ['many', 'medium', 'few']:
        if group_total[group] > 0:
            acc[group] = 100.0 * group_correct[group] / group_total[group]
        else:
            acc[group] = None

    return acc

