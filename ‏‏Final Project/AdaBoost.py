import numpy as np
from knn import KNNClassifier  # ייבוא המחלקה שלך

class AdaBoost:
    def __init__(self, n_estimators=20, k=3):
        """
        אתחול AdaBoost עם מסווג KNN.

        Args:
            n_estimators (int): מספר החלשים (מודלים).
            k (int): מספר השכנים ב-KNN.
        """
        self.n_estimators = n_estimators
        self.k = k
        self.models = []
        self.alphas = []

    def train(self, X_train, y_train, X_test=None, y_test=None):
        """
        אימון מודל AdaBoost.

        Args:
            X_train (np.array): מאפיינים לאימון.
            y_train (np.array): תוויות לאימון.
            X_test (np.array): אופציונלי - מאפיינים לבדיקה.
            y_test (np.array): אופציונלי - תוויות לבדיקה.
        """
        N = X_train.shape[0]
        w = np.ones(N) / N  # משקלים התחלתיים שווים

        # המרה לתוויות בינאריות {-1, 1}
        unique_classes = np.unique(y_train)
        class_mapping = {c: i * 2 - 1 for i, c in enumerate(unique_classes)}
        y_train_mapped = np.vectorize(class_mapping.get)(y_train)

        # שמירת דיוק ראשוני
        initial_accuracy = self.evaluate(X_test, y_test) if X_test is not None and y_test is not None else None
        if initial_accuracy is not None:
            print(f"Initial accuracy: {initial_accuracy:.4f}")

        for i in range(self.n_estimators):
            # יצירת מופע חדש של KNNClassifier
            model = KNNClassifier(n_neighbors=self.k)
            model.train(X_train, y_train)

            # חיזוי התוויות של סט האימון
            y_pred = model.predict(X_train)
            y_pred_mapped = np.vectorize(class_mapping.get)(y_pred)

            # חישוב השגיאה המשוקללת
            err = np.sum(w * (y_pred_mapped != y_train_mapped)) / np.sum(w)
            err = max(err, 1e-10)  # מניעת חלוקה ב-0

            # חישוב משקל החלש (alpha)
            alpha = min(1, 0.5 * np.log((1 - err) / err))  # הגבלת alpha לערך מקסימלי

            # עדכון המשקלים
            w *= np.exp(-alpha * y_train_mapped * y_pred_mapped)
            w /= np.sum(w)  # נורמליזציה

            # שמירת המודל והמשקל שלו
            self.models.append(model)
            self.alphas.append(alpha)

        # דיוק לאחר AdaBoost
        if X_test is not None and y_test is not None:
            final_accuracy = self.evaluate(X_test, y_test)
            print(f"Final accuracy after boosting: {final_accuracy:.4f}")
            print(f"Improvement in accuracy: {final_accuracy - initial_accuracy:.4f}")

    def predict(self, X_test):
        """
        חיזוי עם המודלים שאומנו.

        Args:
            X_test (np.array): נתונים לבדיקה.

        Returns:
            np.array: תוויות חזויות.
        """
        y_pred_sum = np.zeros(X_test.shape[0])

        # חיזוי מכל המודלים ושקלול לפי alpha
        for model, alpha in zip(self.models, self.alphas):
            y_pred_sum += alpha * np.sign(model.predict(X_test))

        return np.sign(y_pred_sum)

    def evaluate(self, X_test, y_test):
        """
        חישוב דיוק המודל.

        Args:
            X_test (np.array): תכונות לבדיקה.
            y_test (np.array): תוויות אמת.

        Returns:
            float: דיוק החיזוי.
        """
        if X_test is None or y_test is None:
            return None

        y_pred = self.predict(X_test)
        accuracy = np.mean(y_pred == y_test)
        print(f"Accuracy:  {accuracy}")
        return accuracy
