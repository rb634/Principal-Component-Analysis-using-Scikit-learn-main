import numpy as np
from sklearn import svm
from sklearn.decomposition import PCA

f = open(sys.argv[1])
data= np.loadtxt(f)
train = data[:,1:]
trainlabels = data[:,0]

f = open(sys.argv[2])
testdata = np.loadtxt(f)
test= testdata[:,1:]
testlabels = testdata[:,0]

#dim red train
pca = PCA(n_components = 2)
pca.fit(train)
newdata = pca.transform(train)

#dim reduction test
pcaa = PCA(n_components = 2)
pcaa.fit(test)
newtest = pcaa.transform(test)

#fit svm
clf = svm.LinearSVC()
clf.fit(newdata,trainlabels)
prediction = clf.predict(newtest)
print(prediction)

err = 0
for i in range(0,len(prediction),1):
    if(prediction[i] != testlabels[i]):
        err += 1

err = err/len(testlabels)
print (err)
accuracy=1-err
print("Accuracy is ",accuracy)
