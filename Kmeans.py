import numpy as np 
import matplotlib as mpl
import matplotlib.cm as cm 
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans


# Συνάρτηση υπολογισμού της Ευκλείδειας απόστασης
def euclidean_distance(point, centers):
    return np.sqrt(((point - centers) ** 2).sum(axis=1)) # Υπολογισμός της Ευκλείδιας Απόστασης μεταξύ των σημείων και των κέντρων

# Συνάρτηση mykmeans
def mykmeans(data, k, epsilon=1e-3, max_iterations=300):
    # Τυχαία επιλογή K αρχικών κέντρων
    np.random.seed(42)
    # Με το replace = False δεν επιτρέπεται η επιλογή του ίδιου δείγματος παραπάνω απο μια φορά
    initial_indices = np.random.choice(data.shape[0], k, replace=False) # data.shape = ο συνολικός αριθμός των δεδομένων
    centers = data[initial_indices] # Αρχικοποίηση των κέντρων με τα τυχαία επιλεγμένα δείγματα
    
    # Ξεκινά επαναληπτική διαδικασία για την εκτέλεση του k-means
    for iteration in range(max_iterations):
        # Ανάθεση δεδομένων στα πλησιέστερα κέντρα
        distances = np.array([euclidean_distance(point, centers) for point in data])
        # Επιλογή του κέντρου με την μικρότερη απόσταση απο το σημείο δεδομένων
        labels = np.argmin(distances, axis=1) 

        # Υπολογισμός των νέων κέντρων ως τον μέσο όρο των σημείων που ανήκουν σε αυτό
        new_centers = np.array([data[labels == i].mean(axis=0) if i in labels else centers[i] for i in range(k)])

        # Έλεγχος αν η διαφορά μεταξύ των παλιών και των νέων κέντρων είναι μικρότερη απο το όριο
        if np.all(np.abs(new_centers - centers) <= epsilon):
            break

        centers = new_centers

    return centers, labels

# Δημιουργία των δεδομένων
mean1, cov1 = [4, 0], [[0.29, 0.4], [0.4, 4]] # Καθορισμός των παραμέτρων της πρώτης δισδιάστατης κανονικής κατανομής (mean και covariance) για τη δημιουργία δεδομένων. 
mean2, cov2 = [5, 7], [[0.29, 0.4], [0.4, 0.9]] 
mean3, cov3 = [7, 4], [[0.64, 0], [0, 0.64]]  
data1 = np.random.multivariate_normal(mean1, cov1, 50) # Η multivariate_normal επιστρέφει τυχαία δείγματα απο μια πολυδιάστατη κανονική κατανομή
data2 = np.random.multivariate_normal(mean2, cov2, 50) 
data3 = np.random.multivariate_normal(mean3, cov3, 50) 
data = np.vstack((data1, data2, data3)) # Συνένωση των δεδομένων σε έναν πίνακα

# Εκτέλεση του αλγορίθμου για k=3
k = 3
k_values = range(1, k + 1) # Ορισμός των τιμών του k
sse_list = [] # Λίστα για την αποθήκευση των SSE
# data: πίνακας MxN όπου N ο αριθμός των δεδομένων και M οι διαστάσεις τους
# k: Ο αριθμός των clusters που θέλουμε να δημιουργήσουμε
centers, labels = mykmeans(data,k) # Αποθήκευση των κέντρων και των labels 

# Υπολογισμός SSE για την ποιότητα των ομάδων
def compute_sse(data, labels, centers):
    # Επιστρέφει το άθροισμα των τετραγώνων των αποστάσεων των σημείων απο τα κέντρα τους
    return sum(((data[labels == i] - center) ** 2).sum() for i, center in enumerate(centers))

for i in range(1, k + 1):
    centers, labels = mykmeans(data, i) # Εκτέλεση του αλγορίθμου για k=1,2,3
    sse = compute_sse(data, labels, centers)
    sse_list.append(sse) # To SSE αποθηκεύεται στη λίστα

sse = compute_sse(data, labels, centers)
print(f"SSE: {sse}")

# Οπτικοποίηση του SSE
# Elbow Method: Εντοπισμός του σημείου όπου η μείωση του SSE είναι πολύ μικρή.
# Αυτό το σημείο είναι το βέλτιστο k
plt.figure()
plt.plot(k_values, sse_list, marker='o')
plt.title('Elbow Method for Optimal K')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('SSE')
plt.grid(True)
plt.show()


# Οπτικοποίηση των clusters μετά την εκτέλεση του k-means
colors = ['red', 'green', 'blue']
for i in range(k):
    plt.scatter(data[labels == i, 0], data[labels == i, 1], c=colors[i], label=f'Cluster {i + 1}')
center_markers = ['+', '*', 'x']
for i in range(k):
    plt.scatter(centers[i, 0], centers[i, 1], c='black', marker=center_markers[i], s=200, label=f'Center {i + 1}')
plt.title('K-means Clustering')
plt.legend()
plt.show()
