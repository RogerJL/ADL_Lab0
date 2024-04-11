from sklearn import metrics
import numpy as np

labels = ['B', 'T', 'D', 'C']
def count_labels(data):
    return [[txt.count(label) for label in labels] for txt in data]


y_true = count_labels(['B', 'T', 'B', 'B', 'TD', 'TD', 'TD', 'TD'])
print("y_true =", y_true)
y_pred = count_labels(['B', 'B', 'TD', 'BD', 'BC', 'TC', 'T', 'TD'])
print("y_pred =", y_pred)

calc_methods = ['Jaccard', 'Precision', 'Recall', 'F1-measure']
print(' '.join(map(lambda m: f"{m:>10s}", [''] + calc_methods)))
for mean_method in ['micro', 'macro', 'weighted', 'samples']:
    print(f"{mean_method:>10s}", end=' ')
    for calc in calc_methods:
        if calc == 'Jaccard':
            v = metrics.jaccard_score(y_true, y_pred, average=mean_method)
        elif calc == 'Precision':
            v = metrics.precision_score(y_true, y_pred, average=mean_method)
        elif calc == 'Recall':
            v = metrics.recall_score(y_true, y_pred, average=mean_method, zero_division=0)
        elif calc == 'F1-measure':
            v = metrics.f1_score(y_true, y_pred, average=mean_method)
        else:
            raise NotImplementedError(calc)
        print(f"{v:10.3f}", end=' ')
    print()
