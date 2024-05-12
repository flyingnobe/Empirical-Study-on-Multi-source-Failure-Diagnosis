import pandas as pd
from sklearn.model_selection import train_test_split
import time
import os
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--data", type=str, default="/data/")
params = vars(parser.parse_args())
base_route = os.path.join("parsed_data", params["data"]+'.csv')
merged_df = pd.read_csv(base_route)

if "__main__" == __name__:
    y = merged_df['label']
    x = merged_df.drop(columns=['label']) 

    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

    mnb = MultinomialNB()

    time1 = time.time()
    mnb.fit(X_train, y_train)
    time2 = time.time()


    class_order = mnb.classes_

    probabilities = mnb.predict_proba(X_test)
    y_pred = mnb.predict(X_test)
    time3 = time.time()

    print("Training time:{}, Testing time:{}".format(time2-time1, time3-time2))
    from sklearn.metrics import precision_recall_fscore_support

    def calculate_classification_metrics(y_true, y_pred):
        macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(y_true, y_pred, average='macro', zero_division=0)
        micro_precision, micro_recall, micro_f1, _ = precision_recall_fscore_support(y_true, y_pred, average='micro', zero_division=0)
        weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted', zero_division=0)
        
        metrics = {
            'Macro Precision': macro_precision,
            'Macro Recall': macro_recall,
            'Macro F1': macro_f1,
            'Micro Precision': micro_precision,
            'Micro Recall': micro_recall,
            'Micro F1': micro_f1,
            'Weighted Precision': weighted_precision,
            'Weighted Recall': weighted_recall,
            'Weighted F1': weighted_f1
        }
        
        return metrics

    ans=calculate_classification_metrics(y_test, y_pred)
    for i in ans.keys():
        print(i,':',ans[i])