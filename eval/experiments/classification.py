from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score


class NodeClassification:
    def __init__(self, x, y, train_ratio):
        self.train_X, self.test_X, self.train_Y, self.test_Y = train_test_split(
                x, y, test_size=(1-train_ratio))
        self.x = x
        self.y = y
        self.clf = LogisticRegression(
            multi_class='ovr',
            n_jobs=16
        )

    def train(self):
        self.clf.fit(self.train_X, self.train_Y)

    def test(self):
        pred = self.clf.predict(self.test_X)
        return (f1_score(pred, self.test_Y, average='macro'), 
                f1_score(pred, self.test_Y, average='micro'))
