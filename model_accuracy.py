
from sklearn.metrics import confusion_matrix

def model_accuracy ( y_true, y_pred ):

    # Get confusion matrix parameters by creating the matrix
    # and force it into an array
    tp, fp, fn, tn = confusion_matrix(y_true, y_pred).ravel()
    # Calculate the model accuracy
    accuracy = (tp + tn) / len(y_true)
    recall = tp / (tp + fn)
    precision = tp / (tp + fp)
    # Return model accuracy
    return [accuracy, recall, precision] 

