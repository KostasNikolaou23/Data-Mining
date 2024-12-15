import pandas as pd
import numpy as np 
import seaborn as sns
import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, f1_score, recall_score, confusion_matrix
from sklearn.utils import resample


# Διαβασμα του dataset απο το αρχειο csv και φόρτωση των δεδομένων
data = pd.read_csv('Training_Dataset.csv')

# Διαχωρισμός χαρακτηριστικών και αποτελέσματος
X = data.drop(columns=['Result'])
y = data['Result']

# Διαχωρισμός δεδομένων σε training και testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=1) # 60% training, 40% testing

#  Decision Tree Classifier
dt_classifier = DecisionTreeClassifier(max_leaf_nodes = 20, random_state=1) # Μέγιστος αριθμός φύλλων 20

# Εκπαίδευση του Decision Tree
dt_classifier.fit(X_train, y_train)

# Cross-Validation = 10 για το Decision Tree
dt_scores = cross_val_score(dt_classifier, X_train, y_train, cv=10, scoring='accuracy')

# Πρόβλεψη στο test set
dt_predictions = dt_classifier.predict(X_test)

# Αξιολόγηση του Decision Tree
dt_accuracy = accuracy_score(y_test, dt_predictions) # Υπολογισμός της ακρίβειας του ποσοστού των σωστών προβλέψεων
dt_f1 = f1_score(y_test, dt_predictions, average='macro') # Υπολογισμός του F1 Score για την αξιολόγηση της απόδοσης του ταξινομητή
dt_recall = recall_score(y_test, dt_predictions, average='macro') # Υπολογισμός της αναλογίας των σωστών θετικών προβλέψεων
dt_report = classification_report(y_test, dt_predictions, target_names=['Non-Malicious', 'Malicious']) # Αναφορά ταξινόμησης


#--------------------------------------------------------------------------------------------------------------


# Κανονικοποίηση δεδομένων για k-NN
# Προεπεξεργασία των δεδομένων με την κανονικοποίηση των δεδομένων
scaler = StandardScaler() # Αφαιρεί τη μέση τιμή και κλιμακώνει τα δεδομένα στη μονάδα διακύμανσης
X_train_standard = scaler.fit_transform(X_train) # Υπολογισμός μέσης τιμής, τυπικής απόκλισης και κλιμάκωση των δεδομένων με την transform | Training set
X_test_standard = scaler.transform(X_test) # Ίδια διαδικασία με την παραπάνω για το test set

# k-NN Classifier
knn_classifier = KNeighborsClassifier(n_neighbors=5)

# Εκπαίδευση του k-NN
knn_classifier.fit(X_train, y_train)

# Cross-Validation για το k-NN
k_values = range(1, 31) # ακέραιες τιμές k από 1 έως 30

mean_accuracies = [] # Λίστα για την αποθήκευση των μέσων ακριβειών

trained_models = {} # Λεξικό για την αποθήκευση των εκπαιδευμένων μοντέλων

knn_scores = [] # Λίστα για την αποθήκευση των cross-validation scores
# Επανάληψη για κάθε τιμή του k
for k in k_values:
    # Δημιουργία του k-NN Classifier
    model = KNeighborsClassifier(n_neighbors=k)

    # Cross-Validation για το k-NN με 10-Fold

    kf = KFold(n_splits=10, shuffle=True, random_state=1) # Το dataset διαιρείται σε 10 τμήματα, shuffle=True για τυχαία ανάμειξη

    # cross_val_score: Για κάθε τμήμα, εκπαιδεύει το μοντέλο στα άλλα 9 (Training set) και το αξιολογεί στο τμήμα που αφαιρείται (Test set)
    # Επανάληψη 10 φορές, χρησιμοποιώντας κάθε φορά διαφορετικό τμήμα ως Test set
    cv_scores = cross_val_score(model, X_train_standard, y_train, cv=kf)
    
    # Υπολογισμός του μέσου όρου των ακριβειών των 10 τμημάτων
    # H mean_accuracy είναι ο μέσος όρος των ακριβειών των 10 τμημάτων
    # npmean: Υπολογίζει τον μέσο όρο των στοιχείων του πίνακα
    mean_accuracy = np.mean(cv_scores)

    # Αποθήκευση του mean_accuracy στην λίστα mean_accuracies για κάθε τιμή του k
    mean_accuracies.append(mean_accuracy)

    # Αποθήκευση των cross-validation scores για κάθε τιμή του k
    knn_scores.append(cv_scores)

    # Αποθήκευση του εκαπιδευμένου μοντέλου για μετέπειτα χρήση με k ως κλειδί
    trained_models[k] = model.fit(X_train_standard, y_train)  # Εκπαίδευση στο σύνολο των δεδομένων

print(mean_accuracies) #Εκτύπωση των μέσων ακριβειών
knn_scores = np.array(knn_scores) # Μετατροπή της λίστας σε numpy array για ευκολότερη επεξεργασία

# Πρόβλεψη στο test set
knn_predictions = knn_classifier.predict(X_test)

# Αξιολόγηση του k-NN
knn_accuracy = accuracy_score(y_test, knn_predictions)
knn_f1 = f1_score(y_test, knn_predictions, average='macro')
knn_recall = recall_score(y_test, knn_predictions, average='macro')
knn_report = classification_report(y_test, knn_predictions, target_names=['Non-Malicious', 'Malicious'])


# Εμφάνιση των αποτελεσμάτων του Decision Tree
print("Decision Tree Results:")
print(f"Accuracy: {dt_accuracy:.2f}")
print(f"F1 Score: {dt_f1:.2f}")
print(f"Recall: {dt_recall:.2f}")
print(dt_report)

# Εμφάνιση των αποτελεσμάτων k-NN
print("\nK-NN Results:")
print(f"Accuracy: {knn_accuracy:.2f}")
print(f"F1 Score: {knn_f1:.2f}")
print(f"Recall: {knn_recall:.2f}")
print(f"k-NN: {knn_scores.mean():.2f} (+/- {knn_scores.std():.2f})")
print(knn_report)

# Σύγκριση και Αξιλόγηση των μετρικών F1 scores, Accuracy, Recall μεταξύ Decision Tree και k-NN
print("\nCross-Validation Scores:")
print("Decision Tree:")
print(f" - F1 Score: {dt_scores.mean():.2f} (+/- {dt_scores.std():.2f})") # F1 Score
print(f" - Accuracy: {dt_scores.mean():.2f} (+/- {dt_scores.std():.2f})") # Accuracy
print(f" - Recall: {dt_scores.mean():.2f} (+/- {dt_scores.std():.2f})")   # Recall

print("k-NN:")
print(f" - F1 Score: {knn_scores.mean():.2f} (+/- {knn_scores.std():.2f})") # F1 Score
print(f" - Accuracy: {knn_scores.mean():.2f} (+/- {knn_scores.std():.2f})") # Accuracy
print(f" - Recall: {knn_scores.mean():.2f} (+/- {knn_scores.std():.2f})")   # Recall


#--------------------------------------------------------------------------------------------------------------


# Εμφάνιση του Decision Tree
plt.figure(figsize=(20, 10))
plot_tree(dt_classifier, feature_names=X.columns, class_names=['Non-Malicious', 'Malicious'], filled=True)
plt.title("Decision Tree Visualization")
plt.show()

# Confusion Matrices
dt_conf_matrix = confusion_matrix(y_test, dt_predictions)
knn_conf_matrix = confusion_matrix(y_test, knn_predictions)

# Πίνακας σύγχυσης για Decision Tree
plt.figure(figsize=(8, 6))
sns.heatmap(dt_conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Non-Malicious', 'Malicious'], yticklabels=['Non-Malicious', 'Malicious'])
plt.title("Confusion Matrix - Decision Tree")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# Πίνακας σύγχυσης για k-NN 
plt.figure(figsize=(8, 6))
sns.heatmap(knn_conf_matrix, annot=True, fmt='d', cmap='Greens', xticklabels=['Non-Malicious', 'Malicious'], yticklabels=['Non-Malicious', 'Malicious'])
plt.title("Confusion Matrix - k-NN")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# Πίνακας για την μέση ακρίβεια του k-NN
plt.plot(k_values, mean_accuracies, marker='o')
plt.xlabel('k (Number of Neighbors)')
plt.ylabel('Mean Accuracy')
plt.title('k-NN Accuracy vs. k Value (10-Fold Cross-Validation)')
plt.grid(True)
plt.show()

# Accuracy, F1_score and Recall
metrics = ['Accuracy', 'F1 Score', 'Recall']
dt_values = [dt_accuracy, dt_f1, dt_recall]
knn_values = [knn_accuracy, knn_f1, knn_recall]

# Πίνακας των συγκρίσεων f1_score, accuracy, recall μεταξύ Decision Tree και k-NN
x = range(len(metrics))
plt.figure(figsize=(10, 6))
plt.bar(x, dt_values, width=0.4, label='Decision Tree', color='blue', align='center')
plt.bar([p + 0.4 for p in x], knn_values, width=0.4, label='k-NN', color='green', align='center')
plt.xticks([p + 0.2 for p in x], metrics)
plt.title("Comparison of Metrics")
plt.xlabel("Metrics")
plt.ylabel("Values")
plt.legend()
plt.show()
