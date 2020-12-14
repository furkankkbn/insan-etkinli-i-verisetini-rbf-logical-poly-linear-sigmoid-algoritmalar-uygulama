from PyQt5.QtCore import pyqtSlot
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt
from PyQt5 import uic
from PyQt5.QtWidgets import QMainWindow, QLabel, QGridLayout,QDesktopWidget, QWidget,QTableWidget,QTableView,QTableWidgetItem,QHeaderView,QGraphicsScene,QGraphicsPixmapItem,QFileDialog
from Design import Ui_MainWindow
import os
import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from xlrd import open_workbook
from openpyxl.reader.excel import load_workbook
from shutil import copyfile
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from skimage import data, img_as_float,io
from skimage.measure import compare_ssim
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

class MainWindow(QWidget,Ui_MainWindow):

    veriseti_file_path = ""
    select_classes_index=0
    
    def __init__(self):
        QtWidgets.QMainWindow.__init__(self)
        self.setupUi(self)   
          
        self.model = QtGui.QStandardItemModel(self)
        self.model_x_train = QtGui.QStandardItemModel(self)
        self.btnyukle.clicked.connect(self.veriyukle)
        self.btnpcauygula.clicked.connect(self.pcauygula)
        self.btnkfold.clicked.connect(self.kfolduygula)
        self.btnuygula.clicked.connect(self.suygula)
        self.sonuclar.cellClicked.connect(self.onSelected)

    
    def onSelected(self,row,column):
        self.get_Details(column-1)
       
    x_index = []
    y_index = -1
    def onSelected_Load(self,item):
        self.x_index = []
        self.y_index = item.column()

        for i in range(len(self.dataset.iloc[0])):
            if(i != self.y_index):
                self.x_index.append(i)
        
        print("x: ",self.x_index)
        print("y: ",self.y_index)
        
        self.X = self.dataset.iloc[:, self.x_index].values
        self.y = self.dataset.iloc[:, self.y_index].values
        
        print(self.X)
        print(self.y)
    
    def get_Details(self,_column): 
        self.list_x_train.clear()
        self.list_x_test.clear()
        self.list_y_train.clear()
        self.list_y_test.clear()
        
        self.list_x_train.setColumnCount(len(self.classes_X_Train[_column][0]))
        self.list_x_train.setRowCount(len(self.classes_X_Train[_column]))
        for i,row in enumerate(self.classes_X_Train[_column]):
           for j,cell in enumerate(row):
               self.list_x_train.setItem(i,j, QTableWidgetItem(str(cell)))
        self.list_x_train.horizontalHeader().setStretchLastSection(True)
        self.list_x_train.resizeColumnsToContents()
        #self.lbl_x_train_count.setText(str(len(self.classes_X_Train[_row])))
                
        print("x_test--->",self.classes_X_Test[_column][0])
        self.list_x_test.setColumnCount(len(self.classes_X_Test[_column][0]))
        self.list_x_test.setRowCount(len(self.classes_X_Test[_column]))
        for i,row in enumerate(self.classes_X_Test[_column]):
           for j,cell in enumerate(row):
               self.list_x_test.setItem(i,j, QTableWidgetItem(str(cell)))
        self.list_x_test.horizontalHeader().setStretchLastSection(True)
        self.list_x_test.resizeColumnsToContents()
        #self.lbl_x_test_count.setText(str(len(self.classes_X_Test[_row])))


        self.list_y_train.setColumnCount(1)
        self.list_y_train.setRowCount(len(self.classes_Y_Train[_column]))
        for i,row in enumerate(self.classes_Y_Train[_column]):
            self.list_y_train.setItem(i,0, QTableWidgetItem(str(row)))
        self.list_y_train.horizontalHeader().setStretchLastSection(True)
        self.list_y_train.resizeColumnsToContents()
        #self.lbl_y_train_count.setText(str(len(self.classes_Y_Train[_row])))
        
        self.list_y_test.setColumnCount(1)
        self.list_y_test.setRowCount(len(self.classes_Y_Test[_column]))
        for i,row in enumerate(self.classes_Y_Test[_column]):
            self.list_y_test.setItem(i,0, QTableWidgetItem(str(row)))
        self.list_y_test.horizontalHeader().setStretchLastSection(True)
        self.list_y_test.resizeColumnsToContents()
    

      
    dataset = []
    X,Y=[],[]
    def veriyukle(self):
        file,_ = QFileDialog.getOpenFileName(self, 'Open file', './',"CSV files (*.csv)")
        #copyfile(file, "./"+self.dataset_file_path)
        self.dataset_file_path = file
        print(self.dataset_file_path)
        #self.dataset = pd.read_csv(self.dataset_file_path, engine='python')  
        #self.read_CSV(self.dataset_file_path)
        self.dataset = pd.read_csv(self.dataset_file_path, engine='python')
        self.dataset = self.dataset.values
        
        print("yükleme--->",len(self.dataset[0]))
        
        #print(len(self.dataset))
        self.verigoster.clear()
        self.verigoster.setColumnCount(len(self.dataset[0]))
        self.verigoster.setRowCount(len(self.dataset))
        for i,row in enumerate(self.dataset):
           for j,cell in enumerate(row):
               self.verigoster.setItem(i,j, QTableWidgetItem(str(cell)))
        self.verigoster.horizontalHeader().setStretchLastSection(True)
        self.verigoster.resizeColumnsToContents()
    
    
    def read_CSV(self,file):
        with open(file, 'r') as csvFile:
            reader = csv.reader(csvFile)
            for row in reader:
                lines=[]
                for value in row:
                    lines.append(str(round(float(value),3)))
                    
                self.dataset.append(lines)
        csvFile.close()
    
    def pcauygula(self):

        
        X = self.dataset[:, 0:(len(self.dataset[0])-1)]
        pca = PCA(n_components=5)
        pca.fit(X)
        
        features = pca.transform(X)
        print(type(features))
        
        self.verigoster2.clear()
        self.verigoster2.setColumnCount(len(features[0]))
        self.verigoster2.setRowCount(len(features))
        for i,row in enumerate(features):
           for j,cell in enumerate(row):
               self.verigoster2.setItem(i,j, QTableWidgetItem(str(cell)))
        self.verigoster2.horizontalHeader().setStretchLastSection(True)
        self.verigoster2.resizeColumnsToContents()
        

        self.X = features
        self.Y = self.dataset[:, len(self.dataset[0])-1]
        
        print(len(self.dataset))
        self.dataset = features

    
    classes_X_Train,classes_X_Test=[],[]
    classes_Y_Train,classes_Y_Test=[],[]        
    X_train, X_test, y_train, y_test=0,0,0,0
    split = 0
    def kfolduygula(self):
        
        X_train, X_test, y_train, y_test=0,0,0,0
        self.classes_X_Train,self.classes_X_Test=[],[]
        self.classes_Y_Train,self.classes_Y_Test=[],[]
        
        
        random_state = 12883823
        classes_index = 0
        
        self.split = 5
        rkf = KFold(n_splits=self.split, random_state=random_state, shuffle=True)
        for train, test in rkf.split(self.dataset):
            classes_index += 1
            
            #print("kFold--->",self.X)
            #print("kFold--->",self.Y)
            X_train, X_test = self.X[train], self.X[test]
            y_train, y_test = self.Y[train], self.Y[test]

            self.classes_X_Train.append(X_train)
            self.classes_X_Test.append(X_test)
            self.classes_Y_Train.append(y_train)
            self.classes_Y_Test.append(y_test)


    def suygula(self):
        rate = self.split
        column = rate+2
        self.sayidetay=0
        self.sonuclar.clear();
        self.sonuclar.setColumnCount(column)
        self.sonuclar.setRowCount(5)
        
        ortalama=[]
        for i in range(0,rate):
            temp_ortalama = self.rbfalgoritması(self.classes_X_Train[i],self.classes_X_Test[i],self.classes_Y_Train[i],self.classes_Y_Test[i])
            ortalama.append(temp_ortalama)

        self.Detayekleme("RDB",ortalama,column)
        self.sonuc1.setText(ortalama[0])
        self.sonuc2.setText(ortalama[1])
        self.sonuc3.setText(ortalama[2])
        self.sonuc4.setText(ortalama[3])
        self.sonuc5.setText(ortalama[4])        

            
        ortalama=[]
        for i in range(0,rate):
            temp_ortalama = self.logicalgoritma(self.classes_X_Train[i],self.classes_X_Test[i],self.classes_Y_Train[i],self.classes_Y_Test[i])
            ortalama.append(temp_ortalama) 
        self.Detayekleme("LOGIC",ortalama,column)
        self.sonuc1_2.setText(ortalama[0])
        self.sonuc2_2.setText(ortalama[1])
        self.sonuc3_2.setText(ortalama[2])
        self.sonuc4_2.setText(ortalama[3])
        self.sonuc5_2.setText(ortalama[4])
        
        
        ortalama=[]
        for i in range(0,rate):
            temp_ortalama = self.linearalgoritma(self.classes_X_Train[i],self.classes_X_Test[i],self.classes_Y_Train[i],self.classes_Y_Test[i])
            ortalama.append(temp_ortalama)
        self.Detayekleme("LINEAR",ortalama,column)
        self.sonuc1_3.setText(ortalama[0])
        self.sonuc2_3.setText(ortalama[1])
        self.sonuc3_3.setText(ortalama[2])
        self.sonuc4_3.setText(ortalama[3])
        self.sonuc5_3.setText(ortalama[4])
        
        ortalama=[]
        for i in range(0,rate):
            temp_ortalama = self.polyalgoritma(self.classes_X_Train[i],self.classes_X_Test[i],self.classes_Y_Train[i],self.classes_Y_Test[i])
            ortalama.append(temp_ortalama)
        self.Detayekleme("POLY",ortalama,column)
        self.sonuc1_4.setText(ortalama[0])
        self.sonuc2_4.setText(ortalama[1])
        self.sonuc3_4.setText(ortalama[2])
        self.sonuc4_4.setText(ortalama[3])
        self.sonuc5_4.setText(ortalama[4])

        
        ortalama=[]
        for i in range(0,rate):
            temp_ortalama = self.sigmoidalgoritma(self.classes_X_Train[i],self.classes_X_Test[i],self.classes_Y_Train[i],self.classes_Y_Test[i])
            ortalama.append(temp_ortalama)
        self.Detayekleme("SIGMOID",ortalama,column)
        self.sonuc1_5.setText(ortalama[0])
        self.sonuc2_5.setText(ortalama[1])
        self.sonuc3_5.setText(ortalama[2])
        self.sonuc4_5.setText(ortalama[3])
        self.sonuc5_5.setText(ortalama[4])
        
        self.sonucort.setText(self.sonuclar.item(0,5).text())
        self.sonucort_2.setText(self.sonuclar.item(1,5).text())
        self.sonucort_3.setText(self.sonuclar.item(2,5).text())
        self.sonucort_4.setText(self.sonuclar.item(3,5).text())
        self.sonucort_5.setText(self.sonuclar.item(4,5).text())
                
        self.sonuclar.horizontalHeader().setStretchLastSection(True)
        self.sonuclar.resizeColumnsToContents()
        self.sonuclar.setHorizontalHeaderLabels(self.datasett)
        
        #1. veriseti
        self.list_x_train.clear()
        self.list_x_train.setColumnCount(len(self.classes_X_Train[0][0]))
        self.list_x_train.setRowCount(len(self.classes_X_Train[0]))
        for i,row in enumerate(self.classes_X_Train[0]):
           for j,cell in enumerate(row):
               self.list_x_train.setItem(i,j, QTableWidgetItem(str(cell)))
        self.list_x_train.horizontalHeader().setStretchLastSection(True)
        self.list_x_train.resizeColumnsToContents()
        #self.lbl_x_train_count.setText(str(len(self.classes_X_Train[0])))
        
        self.list_x_test.clear()
        self.list_x_test.setColumnCount(len(self.classes_X_Test[0][0]))
        self.list_x_test.setRowCount(len(self.classes_X_Test[0]))
        for i,row in enumerate(self.classes_X_Test[0]):
           for j,cell in enumerate(row):
               self.list_x_test.setItem(i,j, QTableWidgetItem(str(cell)))
        self.list_x_test.horizontalHeader().setStretchLastSection(True)
        self.list_x_test.resizeColumnsToContents()
        #self.lbl_x_test_count.setText(str(len(self.classes_X_Test[0])))
        
        self.list_y_train.clear()
        self.list_y_train.setColumnCount(1)
        self.list_y_train.setRowCount(len(self.classes_Y_Train[0]))
        for i,row in enumerate(self.classes_Y_Train[0]):
            self.list_y_train.setItem(i,0, QTableWidgetItem(str(row)))
        self.list_y_train.horizontalHeader().setStretchLastSection(True)
        self.list_y_train.resizeColumnsToContents()
        #self.lbl_y_train_count.setText(str(len(self.classes_Y_Train[0])))
        
        self.list_y_test.clear()
        self.list_y_test.setColumnCount(1)
        self.list_y_test.setRowCount(len(self.classes_Y_Test[0]))
        for i,row in enumerate(self.classes_Y_Test[0]):
            self.list_y_test.setItem(i,0, QTableWidgetItem(str(row)))
        self.list_y_test.horizontalHeader().setStretchLastSection(True)
        self.list_y_test.resizeColumnsToContents()
        #self.lbl_y_test_count.setText(str(len(self.classes_Y_Test[0])))

        #2. veriseti
        self.list_x_train_2.clear()
        self.list_x_train_2.setColumnCount(len(self.classes_X_Train[1][0]))
        self.list_x_train_2.setRowCount(len(self.classes_X_Train[1]))
        for i,row in enumerate(self.classes_X_Train[1]):
           for j,cell in enumerate(row):
               self.list_x_train_2.setItem(i,j, QTableWidgetItem(str(cell)))
        self.list_x_train_2.horizontalHeader().setStretchLastSection(True)
        self.list_x_train_2.resizeColumnsToContents()
        #self.lbl_x_train_count.setText(str(len(self.classes_X_Train[0])))
        
        self.list_x_test_2.clear()
        self.list_x_test_2.setColumnCount(len(self.classes_X_Test[1][0]))
        self.list_x_test_2.setRowCount(len(self.classes_X_Test[1]))
        for i,row in enumerate(self.classes_X_Test[1]):
           for j,cell in enumerate(row):
               self.list_x_test_2.setItem(i,j, QTableWidgetItem(str(cell)))
        self.list_x_test_2.horizontalHeader().setStretchLastSection(True)
        self.list_x_test_2.resizeColumnsToContents()
        #self.lbl_x_test_count.setText(str(len(self.classes_X_Test[0])))
        
        self.list_y_train_2.clear()
        self.list_y_train_2.setColumnCount(1)
        self.list_y_train_2.setRowCount(len(self.classes_Y_Train[1]))
        for i,row in enumerate(self.classes_Y_Train[1]):
            self.list_y_train_2.setItem(i,0, QTableWidgetItem(str(row)))
        self.list_y_train_2.horizontalHeader().setStretchLastSection(True)
        self.list_y_train_2.resizeColumnsToContents()
        #self.lbl_y_train_count.setText(str(len(self.classes_Y_Train[0])))
        
        self.list_y_test_2.clear()
        self.list_y_test_2.setColumnCount(1)
        self.list_y_test_2.setRowCount(len(self.classes_Y_Test[1]))
        for i,row in enumerate(self.classes_Y_Test[1]):
            self.list_y_test_2.setItem(i,0, QTableWidgetItem(str(row)))
        self.list_y_test_2.horizontalHeader().setStretchLastSection(True)
        self.list_y_test_2.resizeColumnsToContents()
        #self.lbl_y_test_count.setText(str(len(self.classes_Y_Test[0])))

        #3. veriseti
        self.list_x_train_3.clear()
        self.list_x_train_3.setColumnCount(len(self.classes_X_Train[1][0]))
        self.list_x_train_3.setRowCount(len(self.classes_X_Train[1]))
        for i,row in enumerate(self.classes_X_Train[2]):
           for j,cell in enumerate(row):
               self.list_x_train_3.setItem(i,j, QTableWidgetItem(str(cell)))
        self.list_x_train_3.horizontalHeader().setStretchLastSection(True)
        self.list_x_train_3.resizeColumnsToContents()
        #self.lbl_x_train_count.setText(str(len(self.classes_X_Train[0])))
        
        self.list_x_test_3.clear()
        self.list_x_test_3.setColumnCount(len(self.classes_X_Test[1][0]))
        self.list_x_test_3.setRowCount(len(self.classes_X_Test[1]))
        for i,row in enumerate(self.classes_X_Test[2]):
           for j,cell in enumerate(row):
               self.list_x_test_3.setItem(i,j, QTableWidgetItem(str(cell)))
        self.list_x_test_3.horizontalHeader().setStretchLastSection(True)
        self.list_x_test_3.resizeColumnsToContents()
        #self.lbl_x_test_count.setText(str(len(self.classes_X_Test[0])))
        
        self.list_y_train_3.clear()
        self.list_y_train_3.setColumnCount(1)
        self.list_y_train_3.setRowCount(len(self.classes_Y_Train[1]))
        for i,row in enumerate(self.classes_Y_Train[2]):
            self.list_y_train_3.setItem(i,0, QTableWidgetItem(str(row)))
        self.list_y_train_3.horizontalHeader().setStretchLastSection(True)
        self.list_y_train_3.resizeColumnsToContents()
        #self.lbl_y_train_count.setText(str(len(self.classes_Y_Train[0])))
        
        self.list_y_test_3.clear()
        self.list_y_test_3.setColumnCount(1)
        self.list_y_test_3.setRowCount(len(self.classes_Y_Test[1]))
        for i,row in enumerate(self.classes_Y_Test[2]):
            self.list_y_test_3.setItem(i,0, QTableWidgetItem(str(row)))
        self.list_y_test_3.horizontalHeader().setStretchLastSection(True)
        self.list_y_test_3.resizeColumnsToContents()
        #self.lbl_y_test_count.setText(str(len(self.classes_Y_Test[0])))

        #4. veriseti
        self.list_x_train_4.clear()
        self.list_x_train_4.setColumnCount(len(self.classes_X_Train[1][0]))
        self.list_x_train_4.setRowCount(len(self.classes_X_Train[1]))
        for i,row in enumerate(self.classes_X_Train[3]):
           for j,cell in enumerate(row):
               self.list_x_train_4.setItem(i,j, QTableWidgetItem(str(cell)))
        self.list_x_train_4.horizontalHeader().setStretchLastSection(True)
        self.list_x_train_4.resizeColumnsToContents()
        #self.lbl_x_train_count.setText(str(len(self.classes_X_Train[0])))
        
        self.list_x_test_4.clear()
        self.list_x_test_4.setColumnCount(len(self.classes_X_Test[1][0]))
        self.list_x_test_4.setRowCount(len(self.classes_X_Test[1]))
        for i,row in enumerate(self.classes_X_Test[3]):
           for j,cell in enumerate(row):
               self.list_x_test_4.setItem(i,j, QTableWidgetItem(str(cell)))
        self.list_x_test_4.horizontalHeader().setStretchLastSection(True)
        self.list_x_test_4.resizeColumnsToContents()
        #self.lbl_x_test_count.setText(str(len(self.classes_X_Test[0])))
        
        self.list_y_train_4.clear()
        self.list_y_train_4.setColumnCount(1)
        self.list_y_train_4.setRowCount(len(self.classes_Y_Train[1]))
        for i,row in enumerate(self.classes_Y_Train[3]):
            self.list_y_train_4.setItem(i,0, QTableWidgetItem(str(row)))
        self.list_y_train_4.horizontalHeader().setStretchLastSection(True)
        self.list_y_train_4.resizeColumnsToContents()
        #self.lbl_y_train_count.setText(str(len(self.classes_Y_Train[0])))
        
        self.list_y_test_4.clear()
        self.list_y_test_4.setColumnCount(1)
        self.list_y_test_4.setRowCount(len(self.classes_Y_Test[1]))
        for i,row in enumerate(self.classes_Y_Test[3]):
            self.list_y_test_4.setItem(i,0, QTableWidgetItem(str(row)))
        self.list_y_test_4.horizontalHeader().setStretchLastSection(True)
        self.list_y_test_4.resizeColumnsToContents()
        #self.lbl_y_test_count.setText(str(len(self.classes_Y_Test[0])))

        #5. veriseti
        self.list_x_train_5.clear()
        self.list_x_train_5.setColumnCount(len(self.classes_X_Train[1][0]))
        self.list_x_train_5.setRowCount(len(self.classes_X_Train[1]))
        for i,row in enumerate(self.classes_X_Train[4]):
           for j,cell in enumerate(row):
               self.list_x_train_5.setItem(i,j, QTableWidgetItem(str(cell)))
        self.list_x_train_5.horizontalHeader().setStretchLastSection(True)
        self.list_x_train_5.resizeColumnsToContents()
        #self.lbl_x_train_count.setText(str(len(self.classes_X_Train[0])))
        
        self.list_x_test_5.clear()
        self.list_x_test_5.setColumnCount(len(self.classes_X_Test[1][0]))
        self.list_x_test_5.setRowCount(len(self.classes_X_Test[1]))
        for i,row in enumerate(self.classes_X_Test[4]):
           for j,cell in enumerate(row):
               self.list_x_test_5.setItem(i,j, QTableWidgetItem(str(cell)))
        self.list_x_test_5.horizontalHeader().setStretchLastSection(True)
        self.list_x_test_5.resizeColumnsToContents()
        #self.lbl_x_test_count.setText(str(len(self.classes_X_Test[0])))
        
        self.list_y_train_5.clear()
        self.list_y_train_5.setColumnCount(1)
        self.list_y_train_5.setRowCount(len(self.classes_Y_Train[1]))
        for i,row in enumerate(self.classes_Y_Train[4]):
            self.list_y_train_5.setItem(i,0, QTableWidgetItem(str(row)))
        self.list_y_train_5.horizontalHeader().setStretchLastSection(True)
        self.list_y_train_5.resizeColumnsToContents()
        #self.lbl_y_train_count.setText(str(len(self.classes_Y_Train[0])))
        
        self.list_y_test_5.clear()
        self.list_y_test_5.setColumnCount(1)
        self.list_y_test_5.setRowCount(len(self.classes_Y_Test[1]))
        for i,row in enumerate(self.classes_Y_Test[4]):
            self.list_y_test_5.setItem(i,0, QTableWidgetItem(str(row)))
        self.list_y_test_5.horizontalHeader().setStretchLastSection(True)
        self.list_y_test_5.resizeColumnsToContents()
        #self.lbl_y_test_count.setText(str(len(self.classes_Y_Test[0])))

    
    sayidetay = 0
    datasett = []
    def Detayekleme(self,algoritma,ortalamalar,rate):
        self.sonuclar.setItem(self.sayidetay,0, QTableWidgetItem(str(algoritma)))
        self.datasett.clear()
        self.datasett.append('Algorithm')
        sayidetay_column=1
        ortalama_toplam = 0
        for i,value in enumerate(ortalamalar):
            ortalama_toplam += float(value)
            self.sonuclar.setItem(self.sayidetay,sayidetay_column, QTableWidgetItem(str(value)))
            self.datasett.append(str(i+1)+'. Ortalama')
            sayidetay_column+=1
        
        self.sonuclar.setItem(self.sayidetay,sayidetay_column, QTableWidgetItem(str(ortalama_toplam/(rate-2))))
        self.datasett.append('Genel Ortalama')
        self.sayidetay+=1

    #algoritmalar
    def rbfalgoritması(self,_x_train,_x_test,_y_train,_y_test):
        sc = StandardScaler()
        X_train = sc.fit_transform(_x_train)
        X_test = sc.transform(_x_test)
        classifier = SVC(kernel = 'rbf', random_state = 0)
        classifier.fit(X_train, _y_train)
        y_pred = classifier.predict(X_test)
        cm  = confusion_matrix(_y_test, y_pred)
        sonucdeger = cross_val_score(estimator = classifier, X = X_train, y = _y_train, cv = self.split)
        return str(round(sonucdeger.mean()*100,2))
        
    def logicalgoritma(self,_x_train,_x_test,_y_train,_y_test):
        from sklearn.linear_model import LogisticRegression
        LogReg = LogisticRegression()
        LogReg.fit(_x_train, _y_train)
        y_pred = LogReg.predict(_x_test)
        cm  = confusion_matrix(_y_test, y_pred)
        accuracy = accuracy_score(_y_test, y_pred)
        return str(round(accuracy*100,2))
        
    def polyalgoritma(self,_x_train,_x_test,_y_train,_y_test):
        sc = StandardScaler()
        X_train = sc.fit_transform(_x_train)
        X_test = sc.transform(_x_test)
        classifier = SVC(kernel = 'poly', random_state = 0)
        classifier.fit(X_train, _y_train)
        y_pred = classifier.predict(X_test)
        cm  = confusion_matrix(_y_test, y_pred)
        #self.lbl_kFold_mean.setText(str(cm))
        sonucdeger = cross_val_score(estimator = classifier, X = X_train, y = _y_train, cv = self.split)
        return str(round(sonucdeger.mean()*100,2))
    
    def linearalgoritma(self,_x_train,_x_test,_y_train,_y_test):
        sc = StandardScaler()
        X_train = sc.fit_transform(_x_train)
        X_test = sc.transform(_x_test)
        classifier = SVC(kernel = 'linear', random_state = 0)
        classifier.fit(X_train, _y_train)
        y_pred = classifier.predict(X_test)
        cm  = confusion_matrix(_y_test, y_pred)
        sonucdeger = cross_val_score(estimator = classifier, X = X_train, y = _y_train, cv = self.split)
        return str(round(sonucdeger.mean()*100,2))
    
    def sigmoidalgoritma(self,_x_train,_x_test,_y_train,_y_test):
        sc = StandardScaler()
        X_train = sc.fit_transform(_x_train)
        X_test = sc.transform(_x_test)
        classifier = SVC(kernel = 'sigmoid', random_state = 0)
        classifier.fit(X_train, _y_train)
        y_pred = classifier.predict(X_test)
        cm  = confusion_matrix(_y_test, y_pred)
        sonucdeger = cross_val_score(estimator = classifier, X = X_train, y = _y_train, cv = self.split)
        return str(round(sonucdeger.mean()*100,2))

    #metodlar
    def get_CSV_READ(self,filename):
        with open(filename, "r") as fileInput:
            for row in csv.reader(fileInput):    
                items = [
                    QtGui.QStandardItem(field)
                    for field in row
                ]
                self.model.appendRow(items)
                
                
    def get_LIST_READ(self,table):
        for row in table:
            items = [
                QtGui.QStandardItem(field)
                for field in row
            ]
            self.model_x_train.appendRow(items)
    
    def set_sum(self,a,b):        
        return np.concatenate((a, b))
        
    
    def showdialog(self,window_title,title,content):
       msg = QtWidgets.QMessageBox()
       msg.setIcon(QtWidgets.QMessageBox.Information)
    
       msg.setText(title)
       msg.setInformativeText(content)
       msg.setWindowTitle(window_title)
       msg.setStandardButtons(QtWidgets.QMessageBox.Ok | QtWidgets.QMessageBox.Cancel)
       msg.exec_()