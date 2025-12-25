"""
Memorizer (Hash Map Learner)

Idea:
- fit(X, y): memorize every training example exactly using a dictionary
- predict(X): for each row, look it up in the dictionary and return its label

Run:
    python memorizer.py
"""

from typing import Any, List, Sequence, Tuple
import warnings
warnings.filterwarnings("ignore")


def row_to_key(row):
    """
    Convert a row into a hashable key for a dictionary.

    TODO:
    - return a tuple version of row
    """
    return tuple(row)


class Memorizer:
    """
    Memorizes training examples exactly (row -> label).
    """

    def __init__(self) -> None:
        """
        Initialize memorizer parameters.
        """
        self.table = None

    def fit(self, X, y) -> "Memorizer":
        """
        Store mapping from each row in X to the corresponding label in y.

        TODOs
        -----
        1. Validate len(X) == len(y)
        2. Initialize self.table as an empty dict
        3. For each (row, label), store self.table[row_to_key(row)] = label
        """
        if len(X)!=len(y):
            raise ValueError("X and y must have the same length")
        self.table={}
        for row,label in zip(X,y):
            key=row_to_key(row)
            self.table[key]=label
        return self
    
    def predict(self, X):
        """
        Predict by exact dictionary lookup.

        TODOs
        -----
        1. Ensure fit() has been called (self.table is not None)
        2. For each row:
           - key = row_to_key(row)
           - if key in table:
                append table[key]
           - else:
                raise KeyError(key)
        3. Return predictions list (length len(X))
        """
        if self.table is None:
            raise RuntimeError("fit() must be called before predict()")
        predictions=[]
        for row in X:
            try:
                key=row_to_key(row)
                if key in self.table:
                    predictions.append(self.table[key])
            except KeyError:
                print(f"Unseen row: {row}")
                predictions.append(None)
        return predictions 

def main():
    """
    Driver code with a small dataset.

    Expected behavior:
    - On seen rows: memorizer predicts perfectly
    - On unseen row: should raise KeyError
    """

    # -----------------------------
    # Sample training dataset
    # -----------------------------
    X_train = [
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1],
    ]
    y_train = ["A", "B", "C", "D"]

    # -----------------------------
    # Test: some seen, one unseen
    # -----------------------------
    X_test = [[0, 0], [1, 1], [9, 9]]

    mem = Memorizer()

    # fit the memorizer
    mem.fit(X_train, y_train)

    print("=== Memorizer Demo ===")
    print("Training pairs:")
    for x, y in zip(X_train, y_train):
        print(f"  {x} -> {y}")

    # run predict on test set 1
    preds = mem.predict(X_test)
    print(f"Predictions on {X_test}: {preds}")


if __name__ == "__main__":
    main()
