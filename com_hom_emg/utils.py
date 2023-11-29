from pathlib import Path

import numpy as np

PROJECT_PATH = Path(__file__).parent.parent


DIRECTION_GESTURES = ["Up", "Down", "Left", "Right"]
MODIFIER_GESTURES = ["Pinch", "Thumb", "Fist", "Open"]
GESTURE_NAMES = ["Up", "Thumb", "Right", "Pinch", "Down", "Fist", "Left", "Open", "Rest"]
NUM_CLASSES = len(GESTURE_NAMES)


def convert_labels_to_2d(y: np.ndarray) -> np.ndarray:
    """Given 1D labels with one or two coordinates having value "1",
    produce 2D labels, where one dimension describes direction, and other describes
    modifier gesture.

    Args:
        y (np.ndarray): shape (items, classes) - one-hot labels

    Returns:
        2d_labels (np.ndarray): shape (items, 2), where
            2d_labels[:, 0] describes direction, and
            2d_labels[:, 1] describes modifier
    """
    assert y.ndim == 2
    assert y.shape[1] == len(GESTURE_NAMES)

    direction_idx = [GESTURE_NAMES.index(d) for d in DIRECTION_GESTURES]
    modifier_idx = [GESTURE_NAMES.index(m) for m in MODIFIER_GESTURES]

    result = np.zeros((y.shape[0], 2), dtype=int)
    for item_idx, item in enumerate(y):
        direction_bits = item[direction_idx]  # Get the 4 bits for direction
        assert sum(direction_bits) in [0, 1]
        if sum(direction_bits) == 0:  # set the "None" bit for direction
            result[item_idx, 0] = 4
        else:
            result[item_idx, 0] = np.argmax(direction_bits)

        modifier_bits = item[modifier_idx]  # Get the 4 bits for modifier
        assert sum(modifier_bits) in [0, 1]
        if sum(modifier_bits) == 0:  # set the "None" bit for modifier
            result[item_idx, 1] = 4
        else:
            result[item_idx, 1] = np.argmax(modifier_bits)

        # Note that "Rest" should be handled already by the above
    return result
