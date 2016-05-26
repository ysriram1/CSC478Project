# -*- coding: utf-8 -*-
"""
Created on Thu May 26 13:19:51 2016

@author: SYARLAG1
"""
import os
import numpy as np
import pandas as pd 
os.chdir('C:/Users/SYARLAG1/Desktop/CSC478Project')

def readindata():
    '''reads the data from the folder containing all the data files
    and returns a final nested list with all the data in it.
    '''  
    allData = []
    colNames = []
    count = 0 
    for i in range(1,102):
        if i<10: filename = '00%s.csv'%i 
        elif i < 100: filename = '0%s.csv'%i
        elif i >= 100: filename = '%s.csv'%i
    
        textLines = open('./data/%s.csv'%filename, 'r').read().split('\n')
        
        for index, line in enumerate(textLines):
            line = line.strip()
            if index in [0,1,2,23,44,65,86]: #these are the lines that we dont want to include in the data (they are columnnames, empty lines etc)
                if index == 2 and count == 0:
                    colNames.append(line.split(','))
                    count += 1
                continue
            allData.append(line.split(','))
    return allData, colNames[0]
    
data, colNames = readindata()

df = pd.DataFrame(data, columns = colNames)
df.to_csv('./fullData.csv')