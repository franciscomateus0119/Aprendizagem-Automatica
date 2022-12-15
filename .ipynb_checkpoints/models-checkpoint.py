import numpy as np
import pandas as pd
import seaborn as sns
from collections import Counter
import math
      
class NaiveBayes():
  def __init__(self):
    self.classes = None
    self.mean = None
    self.std = None
    self.prioris = None

  def train(self,X,y):
    #Get the number of (unique) classes
    self.classes = np.unique(y)
    n_classes = len(self.classes)

    #Get how many samples and features there are
    n_samples = X.shape[0]
    n_features = X.shape[1]  

    # mean, variance, priori para cada classe
    self.mean = np.zeros((n_classes, n_features))
    self.std = np.zeros((n_classes, n_features))
    self.prioris = np.zeros(n_classes)

    #For every class, calculate mean, variance and priori
    for index, cls in enumerate(self.classes):
      x_classe = X[y == cls]
      self.mean[index, :] = x_classe.mean(axis=0)
      self.std[index, :] = x_classe.std(axis=0)
      self.prioris[index] = x_classe.shape[0]/(n_samples)
  
  #Calculates de Posterioris Probabilities
  def posteriori(self, x):
    posterioris = []
    
    #For every class, compute likelihood/pdf, get priori and compute posteriori
    for index, cls in enumerate(self.classes):
      likelihood = self.pdf(index,x)
      probability = self.prioris[index]
      for xy in likelihood:
        probability *= xy
      posterioris.append(probability)     
    return posterioris


  def predict(self,X):
    predictions = []
    iterator = 0
    #For every sample
    for x in X:
      #Find Posterioris
      probability = self.posteriori(x)
      
      #Label is the class with the highest probability
      predictions.append(self.classes[np.argmax(probability)])
      iterator+=1
    return np.asarray(predictions)
  
  #Probability Density Function, PDF.
  def pdf(self,class_index,x):
    mean = self.mean[class_index]
    deviation = self.std[class_index]
    deviation [deviation == 0] = 1e-10
    self.std[class_index] = deviation
    likelihood = (1/(deviation * np.sqrt(2 * np.pi))) * np.exp(-((x - mean)**2)/(2*(deviation**2)))
    return likelihood
