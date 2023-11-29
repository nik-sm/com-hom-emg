import pickle
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
from sklearn.base import ClassifierMixin


class BaseParallelModel(ABC, ClassifierMixin):
    @abstractmethod
    def fit(self, x, y):
        """
        Args:
            x (np.ndarray): shape (items, features)
            y (np.ndarray): shape (items, 2), where first axis is a number 0-4, representing Up/Down/Left/Right/None
                    and Second axis represents Thumb/Pinch/Fist/Open/None
        """

    @abstractmethod
    def predict_proba(self, x):
        """
        Estimate class probabilities with shape (items, 2, K),
        where 2 represents probabilities for direction and modifier,
        and K is the number of possible values for each part of prediction.
        K=5 for our current case. (Up/Down/Left/Right/None) and (Thumb/Pinch/Fist/Open/None)
        Args:
            x (np.ndarray): shape (items, features)
        Returns:
            probs (np.ndarray): shape (items, 2, K)
        """

    @abstractmethod
    def predict(self, x):
        """
        Returns:
            preds (np.ndarray): shape (items, 2)
        """

    @abstractmethod
    def save(self, path):
        """Save the model to the given path."""

    @classmethod
    @abstractmethod
    def load(cls, path):
        """Load the model from the given path."""

    @abstractmethod
    def get_params(self, deep=True):
        """Return dictionary of kwargs for __init__ fn"""

    @abstractmethod
    def __repr__(self):
        ...


class ParallelA(BaseParallelModel):
    DEFAULT_SAVE_NAME = "ParallelA.pkl"

    def __init__(self, dir_clf, mod_clf):
        self.dir_clf = dir_clf
        self.mod_clf = mod_clf

    def get_params(self, deep=True):
        return {"dir_clf": self.dir_clf, "mod_clf": self.mod_clf}

    def fit(self, x, y):
        self.dir_clf.fit(x, y[:, 0])
        self.mod_clf.fit(x, y[:, 1])
        return self

    def predict_proba(self, x):
        prob0 = self.dir_clf.predict_proba(x)
        prob1 = self.mod_clf.predict_proba(x)
        return np.stack([prob0, prob1], axis=1)

    def predict(self, x):
        return self.predict_proba(x).argmax(-1)

    def save(self, save_dir: Path) -> Path:
        assert save_dir.exists() and save_dir.is_dir()
        file_path = save_dir / self.DEFAULT_SAVE_NAME
        with open(file_path, "wb") as f:
            pickle.dump(self, f)
        return file_path

    @classmethod
    def load(cls, file_path: Path) -> "ParallelA":
        with open(file_path, "rb") as f:
            return pickle.load(f)

    def __repr__(self):
        return f"{type(self).__name__}(dir_clf={self.dir_clf}, mod_clf={self.mod_clf})"


class ControlModel_RandomGuess(BaseParallelModel):
    def __init__(self, *args, **kw):
        pass

    def fit(self, x, y):
        return self

    def predict_proba(self, x):
        # Create random probs output with correct shape
        # Note that the probabilities should be normalized along the final axis
        # This is the same axis where we'll choose one prediction
        probs = np.random.rand(x.shape[0], 2, 5)
        probs /= np.sum(probs, axis=-1, keepdims=True)
        return probs

    def predict(self, x):
        return self.predict_proba(x).argmax(-1)

    def save(self, path):
        pass

    @classmethod
    def load(cls, path):
        pass

    def get_params(self, deep=True):
        return {}

    def __repr__(self):
        return f"{type(self).__name__}()"
