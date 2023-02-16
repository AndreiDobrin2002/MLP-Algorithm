from sklearn import neural_network
import numpy as np


#INCARCARE DATE SI IMPARTIRE IN TRAIN SI TEST

caracteristici_test = np.loadtxt('monks-1.test',usecols=[1,2,3,4,5,6])
etichete_test = np.loadtxt('monks-1.test',usecols=[0])

caracteristici_train = np.loadtxt('monks-1.train',usecols=[1,2,3,4,5,6])
etichete_train = np.loadtxt('monks-1.train',usecols=[0])


# CREARE SI ANTRENARE MLP
clf = neural_network.MLPClassifier(hidden_layer_sizes=(8,4),max_iter=2000,learning_rate_init=0.01)
clf.fit(caracteristici_train,etichete_train)

# TESTARE MLP
predictii = clf.predict(caracteristici_test) 

print (predictii)
print ('\n')
print(etichete_test)

# ACURATETE
acc=0
for i in range(len(etichete_test)):
    if etichete_test[i]==predictii[i]:
        acc=acc+1
print('Acuratetea=' + str((acc/len(etichete_test))*100) + '%')    