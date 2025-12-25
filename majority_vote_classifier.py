"""
Majority Vote (Zero-Rule) Classifier

Students must implement:
- MajorityVoteClassifier.fit
- MajorityVoteClassifier.predict

Run:
    python majority_vote_demo.py
"""

from typing import Any, List, Optional, Sequence


class MajorityVoteClassifier:
    """
    A simple baseline classifier that always predicts the
    most frequent label seen during training.
    """

    def __init__(self) -> None:
        """
        Initialize learned parameters.
        """
        # TODO: store the learned majority label here
        self.mode_label= None

    def fit(self, X, y):
        """
        Learn the most frequent label in y.

        TODOs
        -----
        1. Validate that y is non-empty
        2. Count the frequency of each label in y
        3. Find the label(s) with maximum count
        4. Break ties deterministically
        5. Store the result in self.mode_label
        """
        # TODO: raise an error if y is empty
        # TODO: create a dictionary to count label frequencies
        # TODO: compute the maximum frequency
        # TODO: select the majority label (deterministic tie-break)
        # TODO: assign it to self.mode_label
        #1
        if len(y) == 0:
            raise ValueError("y cannot be empty")
        #2
        d={}
        for label in y:
            if label in d:
                d[label]+=1
            else:
                d[label]=1
        #3
        m=max(d.values())
        #4
        candidates=[label for label, count in d.items() if count == m]
        self.mode_label=sorted(candidates)[0]


        # raise NotImplementedError

    def predict(self, X):
        """
        Predict the learned majority label for each row in X.

        TODOs
        -----
        1. Ensure fit() has been called
        2. Return a list of predictions of length len(X)
        """
        # TODO: raise an error if self.mode_label is None
        # TODO: return [self.mode_label] repeated len(X) times
        if self.mode_label is None:
            raise ValueError
        return [self.mode_label]*len(X)


def main():
    """
    Driver code with a small sample dataset.
    """

    # -----------------------------
    # Sample training dataset
    # -----------------------------
    X_train = [
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1],
        [2, 2],
    ]

    # Majority label should be 1
    y_train = [1, 1, 1, 0, 0]

    # -----------------------------
    # Sample test dataset
    # -----------------------------
    X_test = [
        [0, 0],
        [3, 3],
        [10, 10],
    ]

    # -----------------------------
    # Train and predict
    # -----------------------------
    clf = MajorityVoteClassifier()

    # call fit on training data
    clf.fit(X_train, y_train)

    # call predict on test data
    predictions = clf.predict(X_test)

    # -----------------------------
    # Display results
    # -----------------------------
    print("=== Majority Vote Classifier Demo ===")
    print(f"Training labels: {y_train}")

    # print learned majority label
    print(f"Learned majority label: {clf.mode_label}")

    # print predictions
    print(f"Predictions: {predictions}")


if __name__ == "__main__":
    main()
