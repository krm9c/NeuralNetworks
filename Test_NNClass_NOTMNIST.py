import numpy as np
import tensorflow as tf
from six.moves import cPickle as pickle
from six.moves import range

from tqdm import tqdm
import Network_class
import tensorflow as tf

  ###################################################################################
def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
      assert inputs.shape[0] == targets.shape[0]
      if shuffle:
          indices = np.arange(inputs.shape[0])
          np.random.shuffle(indices)
      for start_idx in range(0, inputs.shape[0] - batchsize + 1, batchsize):
          if shuffle:
              excerpt = indices[start_idx:start_idx + batchsize]
          else:
              excerpt = slice(start_idx, start_idx + batchsize)
          yield inputs[excerpt], targets[excerpt]

  ###################################################################################
def return_dict(placeholder, List, model, batch_x, batch_y):
      S ={}
      for i, element in enumerate(List):
          S[placeholder[i]] = element
      S[model.Deep['FL_layer0']    ] = batch_x
      S[model.classifier['Target'] ] = batch_y
      return S


  #####################################################################################
def Analyse_custom_Optimizer(X_train, y_train, X_test, y_test):

      import gc
      # Lets start with creating a model and then train batch wise.
      model = Network_class.Agent()
      model = model.init_NN_custom(classes, 0.1, [inputs, 240, classes], tf.nn.relu)

      try:
          t = xrange(Train_Glob_Iterations)
          from tqdm import tqdm
          for i in tqdm(t):
              for batch in iterate_minibatches(X_train, y_train, Train_batch_size, shuffle=True):
                  batch_xs, batch_ys  = batch

                  # Gather Gradients
                  grads = model.sess.run([ model.Trainer["grads"] ],
                  feed_dict ={ model.Deep['FL_layer0'] : batch_xs, model.classifier['Target']: batch_ys })
                  List = [g for g in grads[0]]

                  # Apply gradients
                  summary, _ = model.sess.run( [ model.Summaries['merged'], model.Trainer["apply_placeholder_op"] ], \
                  feed_dict= return_dict( model.Trainer["grad_placeholder"], List, model, batch_xs, batch_ys) )

                  # model.Summaries['train_writer'].add_summary(summary, i)
              if i % 1 == 0:
                  summary, a  = model.sess.run( [model.Summaries['merged'], model.Evaluation['accuracy']], feed_dict={ model.Deep['FL_layer0'] : \
                  X_test, model.classifier['Target'] : y_test})
                  print "accuracies", a
                  # model.Summaries['test_writer'].add_summary(summary, i)
              if a > 0.99:
                  summary, pr  = model.sess.run( [ model.Summaries['merged'], model.Evaluation['prob'] ], \
                  feed_dict ={ model.Deep['FL_layer0'] : X_test, model.classifier['Target'] : y_test } )
                  tf.reset_default_graph()
                  del model
                  gc.collect()
                  return a
                  # model.Summaries['test_writer'].add_summary(summary, i)

      except Exception as e:
          print e
          print "I found an exception"
          tf.reset_default_graph()
          del model
          gc.collect()
          return 0

      tf.reset_default_graph()
      del model
      gc.collect()
      return a




pickle_file = '../data/notMNIST.pickle'

with open(pickle_file, 'rb') as f:
  save = pickle.load(f)
  train_dataset = save['train_dataset']
  train_labels = save['train_labels']
  valid_dataset = save['valid_dataset']
  valid_labels = save['valid_labels']
  test_dataset = save['test_dataset']
  test_labels = save['test_labels']
  del save  # hint to help gc free up memory
  print('Training set', train_dataset.shape, train_labels.shape)
  print('Validation set', valid_dataset.shape, valid_labels.shape)
  print('Test set', test_dataset.shape, test_labels.shape)



# Initialize the parameters, graph and the final update laws
# hyper parameter setting
image_size = 28
batch_size = 64
valid_size = test_size = 10000
num_data_input = image_size*image_size
num_hidden = 240
num_labels = 10
act_f = "relu"
init_f = "uniform"
back_init_f = "uniform"
weight_uni_range = 0.05
back_uni_range = 0.5
lr = 0.001
num_layer = 5 #should be >= 3
num_steps = 5000
Temp =[]
# Get data for running the stuff
from tqdm import tqdm
Train_batch_size = 256
Train_Glob_Iterations = 100
graph = tf.Graph()

def reformat(dataset, labels):
    dataset = dataset.reshape((-1, image_size * image_size)).astype(np.float32)
    # Map 0 to [1.0, 0.0, 0.0 ...], 1 to [0.0, 1.0, 0.0 ...]
    labels = (np.arange(num_labels) == labels[:,None]).astype(np.float32)
    return dataset, labels


train_dataset, train_labels = reformat(train_dataset, train_labels)
valid_dataset, valid_labels = reformat(valid_dataset, valid_labels)
test_dataset, test_labels = reformat(test_dataset, test_labels)
print('Training set', train_dataset.shape, train_labels.shape)
print('Validation set', valid_dataset.shape, valid_labels.shape)
print('Test set', test_dataset.shape, test_labels.shape)

X_train = train_dataset
y_train = train_labels
X_test = test_dataset
y_test = test_labels

classes = 10
# scale the data for the work
from sklearn import preprocessing
X_train = preprocessing.scale(X_train)
X_test = preprocessing.scale(X_test)
print "total number of classes", classes
print "Train, Test", X_train.shape, X_test.shape
import os,sys
# Dimension redyce the data if needed
sys.path.append('../CommonLibrariesDissertation')
from Library_Paper_two import *
X_train, Tree = initialize_calculation(T = None, Data = X_train, gsize = 2,\
par_train = 0, output_dimension = 350)
X_test, Tree = initialize_calculation(T = Tree, Data = X_test, gsize = 2,\
par_train = 1, output_dimension = 350)
print "Train, Test", X_train.shape, X_test.shape
inputs = X_train.shape[1]
print "inputs", inputs

x = input("Enter a number to continue")
import tflearn
for i in tqdm(xrange(1)):
  Temp.append(Analyse_custom_Optimizer(X_train,\
   tflearn.data_utils.to_categorical((y_train), classes),\
    X_test, tflearn.data_utils.to_categorical((y_test), classes)))
Results = np.array(Temp)
print "\n min", min(Results), "avg", Results.mean(), "max", max(Results)
