
import emtrees
import numpy
from sklearn.model_selection import train_test_split
from sklearn import metrics, datasets

rnd = 11
digits = datasets.load_digits()
Xtrain, Xtest, ytrain, ytest = train_test_split(digits.data, digits.target, random_state=rnd)
Xtrain = (Xtrain * 2**16).astype(numpy.int32)
Xtest = (Xtest * 2**16).astype(numpy.int32)

print('Loading digits dataset. 8x8=64 features')

# 0.95+ with n_estimators=40, max_depth=20
# 0.90+ with n_estimators=10, max_depth=10
trees = 40
max_depth = 20
print('Training {} trees with max_depth {}'.format(trees, max_depth))
model = emtrees.RandomForest(n_estimators=trees, max_depth=max_depth, random_state=rnd)
model.fit(Xtrain, ytrain)

# Predict
ypred = model.predict(Xtest)
print('Accuracy on validation set {:.2f}%'.format(metrics.accuracy_score(ypred, ytest)*100))

m = numpy.max(Xtrain), numpy.min(Xtrain)

code = model.output_c('digits')
filename = 'digits.h'
with open(filename, 'w') as f:
   f.write(code)
print('Wrote C code to', filename)

port = '/dev/ttyUSB0'
print('Classify on microcontroller via', port)
import serial
device = serial.Serial(port=port, baudrate=115200, timeout=0.1) 

repetitions = 10
Y_pred = []
times = []
for idx,row in enumerate(Xtest):
   # send
   values = [idx, repetitions] + list(row)
   send = ';'.join("{}".format(v) for v in values) + '\n'
   device.write(send.encode('ascii'))
   resp = device.readline()

   # receive
   tok = resp.decode('ascii').strip().split(';')
   retvals = [int(v) for v in tok]
   (request,micros,prediction,reps),values = retvals[:4], retvals[4:]

   assert request == idx
   assert reps == repetitions
   err = numpy.array(values) - row
   assert numpy.sum(err) == 0, err

   Y_pred.append(prediction)
   times.append(micros / 1000)
   #print(idx, prediction, reps, micros)

print('Confusion matrix')
print(metrics.confusion_matrix(Y_pred, ytest))

avg = numpy.mean(times) / repetitions
stddev = numpy.std(numpy.array(times) / repetitions)
print('Time per classification (ms): {:.2f} avg, {:.2f} stdev'.format(avg, stddev))
