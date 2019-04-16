import numpy
import tensorflow
import keras
import theano
import scipy
f=open("version.txt","w")
string=''
string+=numpy.__version__+'  '+tensorflow.__version__+'  '+keras.__version__+'  '+theano.__version__+'  '+scipy.__version__
f.write(string)
f.close()