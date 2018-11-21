import pandas as pd
import numpy as np
import os, pickle, heapq, math
from pathlib import Path
from scipy.spatial.distance import cosine
import clean_data
import matplotlib.pyplot as plt



def rolling_window(nearest_neighbors,loc_PM10, loc_PM25, data_2016, data_2017):
    '''
        KNN using rolling window approach
    '''
    no_of_iterations = len(data_2016)
    no_of_predictions = len(data_2017)
    predictions = {}
    for i in range(no_of_predictions):
        print('rolling_window preditcion number %d'%i)
        heap = []
        if not predictions.__contains__(data_2017[i][len(data_2017[0]) - 1]):
            predictions[data_2017[i][len(data_2017[0]) - 1]] = []
        #go through the 2016 data
        for j in range(i,no_of_iterations):
            #find euclidean distance and push it as a tuple in a heap
            distance = 0.0
            for k in range(1,len(data_2016[0]) - 1):
                if k != loc_PM10 and k != loc_PM25:
                    distance += (data_2016[j][k] - data_2017[i][k])**2
            #handle similar/different station case
            if data_2016[j][len(data_2016[0])-1] != data_2017[i][len(data_2017[0])-1]:
                distance += 1
            distance = math.sqrt(distance)
            distance = -distance
            #push distance into heap
            if len(heap) == nearest_neighbors:
                if heap[0][0] < distance:
                    heapq.heappop(heap)
                    heapq.heappush(heap,(distance,data_2016[j][loc_PM10],data_2016[j][loc_PM25]))
            elif len(heap) < nearest_neighbors:
                heapq.heappush(heap,(distance,data_2016[j][loc_PM10],data_2016[j][loc_PM25]))
        #go through 2017 data
        for j in range(0, i):
            distance = 0.0
            for k in range(1, len(data_2017[0]) - 1):
                if k != loc_PM10 and k != loc_PM25:
                    distance += (data_2017[j][k] - data_2017[i][k])**2
            #handle similar/different station case
            if data_2017[j][len(data_2017[0])-1] != data_2017[i][len(data_2017[0])-1]:
                distance += 1
            distance = math.sqrt(distance)
            distance = -distance
            #push distance into heap
            if len(heap) == nearest_neighbors:
                if distance > heap[0][0]:
                    heapq.heappop(heap)
                    heapq.heappush(heap,(distance, data_2017[j][loc_PM10], data_2017[j][loc_PM25]))
            else:
                heapq.heappush(heap,(distance, data_2017[j][loc_PM10], data_2017[j][loc_PM25]))

        #predict based on top k
        predicted_PM10 = 0.0
        predicted_PM25 = 0.0
        sum_of_weights = 0.0
        for j in range(0,nearest_neighbors):
            x = heapq.heappop(heap)
            predicted_PM10 += (1/(-x[0]))*x[1]
            predicted_PM25 += (1/(-x[0]))*x[2]
            sum_of_weights += (1/(-x[0]))
        predicted_PM10 /= sum_of_weights
        predicted_PM25 /= sum_of_weights
        #mapping to the list with station number
        predictions[data_2017[i][len(data_2017[0])-1]].append((predicted_PM10,predicted_PM25))
    return predictions



def recursive_window(nearest_neighbors,loc_PM10, loc_PM25, data_2016, data_2017):
    '''
        KNN using recursive window approach
    '''
    no_of_iterations = len(data_2016)
    no_of_predictions = len(data_2017)
    iter_2017 = 0
    predictions = {}
    for i in range(no_of_predictions):
        print('recursive_window preditcion number %d'%i)
        if not predictions.__contains__(data_2017[i][len(data_2017[0]) - 1]):
            predictions[data_2017[i][len(data_2017[0]) - 1]] = []

        heap = []
        #go through the 2016 data
        for j in range(0,no_of_iterations):
            #find euclidean distance and push it as a tuple in a heap
            distance = 0.0
            for k in range(1,len(data_2016[0]) - 1):
                if k != loc_PM10 and k != loc_PM25:
                    distance += (data_2016[j][k] - data_2017[i][k])**2
            #handle similar/different station case
            if data_2016[j][len(data_2016[0])-1] != data_2017[i][len(data_2017[0])-1]:
                distance += 1
            distance = math.sqrt(distance)
            distance = -distance
            #push distance into heap
            if len(heap) == nearest_neighbors:
                if heap[0][0] < distance:
                    heapq.heappop(heap)
                    heapq.heappush(heap,(distance,data_2016[j][loc_PM10],data_2016[j][loc_PM25]))
            elif len(heap) < nearest_neighbors:
                heapq.heappush(heap,(distance,data_2016[j][loc_PM10],data_2016[j][loc_PM25]))

        for j in range(0, i):
            distance = 0.0
            for k in range(1, len(data_2017[0]) - 1):
                if k != loc_PM10 and k != loc_PM25:
                    distance += (data_2017[j][k] - data_2017[i][k])**2
            #handle similar/different station case
            if data_2017[j][len(data_2017[0])-1] != data_2017[i][len(data_2017[0])-1]:
                distance += 1
            distance = math.sqrt(distance)
            distance = -distance
            #push distance into heap
            if len(heap) == nearest_neighbors:
                if distance > heap[0][0]:
                    heapq.heappop(heap)
                    heapq.heappush(heap,(distance, data_2017[j][loc_PM10], data_2017[j][loc_PM25]))
            else:
                heapq.heappush(heap,(distance, data_2017[j][loc_PM10], data_2017[j][loc_PM25]))


        #predict based on top k
        predicted_PM10 = 0.0
        predicted_PM25 = 0.0
        sum_of_weights = 0.0
        for j in range(0,nearest_neighbors):
            x = heapq.heappop(heap)
            predicted_PM10 += (1/(-x[0]))*x[1]
            predicted_PM25 += (1/(-x[0]))*x[2]
            sum_of_weights += (1/(-x[0]))
        predicted_PM10 /= sum_of_weights
        predicted_PM25 /= sum_of_weights
        #mapping to the list with station number
        predictions[data_2017[i][len(data_2017[0])-1]].append((predicted_PM10,predicted_PM25))
    return predictions

def normalised_rolling(nearest_neighbors,loc_PM10, loc_PM25, data_2016, data_2017):
    '''
        KNN using rolling window approach with normalisation and cosine similarity instead of euclidean distance
    '''
    no_of_iterations = len(data_2016)
    no_of_predictions = len(data_2017)
    # iter_2016 = no_of_iterations
    # iter_2017 = no_of_iterations - iter_2016
    predictions = {}
    for i in range(no_of_predictions):
        print('normalised_rolling preditcion number %d'%i)
        heap = []
        if not predictions.__contains__(data_2017[i][len(data_2017[0]) - 1]):
            predictions[data_2017[i][len(data_2017[0]) - 1]] = []
        v1 = data_2017[i][1:loc_PM10]
        v2 = data_2017[i][(loc_PM25+1):len(data_2017[0]) - 1]
        v = [*v1,*v2]
        v.append(1)
        #go through the 2016 data
        for j in range(i,no_of_iterations):
            #find euclidean distance and push it as a tuple in a heap
            distance = 0.0
            u1 = data_2016[j][1:loc_PM10]
            u2 = data_2016[j][(loc_PM25+1):len(data_2016[0]) - 1]
            u = [*u1,*u2]
            #handle similar/different station case
            if data_2016[j][len(data_2016[0])-1] != data_2017[i][len(data_2017[0])-1]:
                u.append(1)
            else:
                u.append(0)
            distance = cosine(u,v)
            distance = -distance
            #push distance into heap
            if len(heap) == nearest_neighbors:
                if heap[0][0] < distance:
                    heapq.heappop(heap)
                    heapq.heappush(heap,(distance,data_2016[j][loc_PM10],data_2016[j][loc_PM25]))
            elif len(heap) < nearest_neighbors:
                heapq.heappush(heap,(distance,data_2016[j][loc_PM10],data_2016[j][loc_PM25]))

        for j in range(0, i):
            distance = 0.0
            u1 = data_2017[j][1:loc_PM10]
            u2 = data_2017[j][(loc_PM25+1):len(data_2017[0]) - 1]
            u = [*u1,*u2]
            #handle similar/different station case
            if data_2017[j][len(data_2017[0])-1] != data_2017[i][len(data_2017[0])-1]:
                u.append(1)
            else:
                u.append(0)
            distance = cosine(u,v)
            distance = -distance
            #push distance into heap
            if len(heap) == nearest_neighbors:
                if distance > heap[0][0]:
                    heapq.heappop(heap)
                    heapq.heappush(heap,(distance, data_2017[j][loc_PM10], data_2017[j][loc_PM25]))
            else:
                heapq.heappush(heap,(distance, data_2017[j][loc_PM10], data_2017[j][loc_PM25]))

        #predict based on top k
        predicted_PM10 = 0.0
        predicted_PM25 = 0.0
        sum_of_weights = 0.0
        for j in range(0,nearest_neighbors):
            x = heapq.heappop(heap)
            predicted_PM10 += (1 + x[0])*x[1]
            predicted_PM25 += (1 + x[0])*x[2]
            sum_of_weights += (1 + x[0])
        predicted_PM10 /= sum_of_weights
        predicted_PM25 /= sum_of_weights
        #mapping to the list with station number
        predictions[data_2017[i][len(data_2017[0])-1]].append((predicted_PM10,predicted_PM25))

    return predictions

def normalised_recursive(nearest_neighbors,loc_PM10, loc_PM25, data_2016, data_2017):
    '''
        KNN using recursive window approach with normalisation and cosine similarity instead of euclidean distance
    '''
    no_of_iterations = len(data_2016)
    no_of_predictions = len(data_2017)
    iter_2017 = 0
    predictions = {}
    for i in range(no_of_predictions):
        print('normalised_recursive preditcion number %d'%i)
        heap = []
        if not predictions.__contains__(data_2017[i][len(data_2017[0]) - 1]):
            predictions[data_2017[i][len(data_2017[0]) - 1]] = []

        v1 = data_2017[i][1:loc_PM10]
        v2 = data_2017[i][(loc_PM25+1): len(data_2017[0])-1]
        v = [*v1,*v2]
        v.append(1)
        #go through the 2016 data
        for j in range(0,no_of_iterations):
            #find euclidean distance and push it as a tuple in a heap
            distance = 0.0
            u1 = data_2016[j][1:loc_PM10]
            u2 = data_2016[j][(loc_PM25+1):len(data_2016[0]) - 1]
            u = [*u1,*u2]
            #handle similar/different station case
            if data_2016[j][len(data_2016[0])-1] != data_2017[i][len(data_2017[0])-1]:
                u.append(1)
            else:
                u.append(0)
            distance = cosine(u,v)
            distance = -distance
            #push distance into heap
            if len(heap) == nearest_neighbors:
                if heap[0][0] < distance:
                    heapq.heappop(heap)
                    heapq.heappush(heap,(distance,data_2016[j][loc_PM10],data_2016[j][loc_PM25]))
            elif len(heap) < nearest_neighbors:
                heapq.heappush(heap,(distance,data_2016[j][loc_PM10],data_2016[j][loc_PM25]))

        for j in range(0, i):
            distance = 0.0
            u1 = data_2017[j][1:loc_PM10]
            u2 = data_2017[j][(loc_PM25+1):len(data_2017[0]) - 1]
            u = [*u1,*u2]
            #handle similar/different station case
            if data_2017[j][len(data_2017[0])-1] != data_2017[i][len(data_2017[0])-1]:
                u.append(1)
            else:
                u.append(0)
            distance = cosine(u,v)
            distance = -distance
            #push distance into heap
            if len(heap) == nearest_neighbors:
                if distance > heap[0][0]:
                    heapq.heappop(heap)
                    heapq.heappush(heap,(distance, data_2017[j][loc_PM10], data_2017[j][loc_PM25]))
            else:
                heapq.heappush(heap,(distance, data_2017[j][loc_PM10], data_2017[j][loc_PM25]))

        #predict based on top k
        predicted_PM10 = 0.0
        predicted_PM25 = 0.0
        sum_of_weights = 0.0
        for j in range(0,nearest_neighbors):
            x = heapq.heappop(heap)
            predicted_PM10 += (1 + x[0])*x[1]
            predicted_PM25 += (1 + x[0])*x[2]
            sum_of_weights += (1 + x[0])
        predicted_PM10 /= sum_of_weights
        predicted_PM25 /= sum_of_weights
        #mapping to the list with station number
        predictions[data_2017[i][len(data_2017[0])-1]].append((predicted_PM10,predicted_PM25))
    return predictions

def k_nearest_neighbours(nearest_neighbors,loc_PM10, loc_PM25, data_2016, data_2017,mode):
    '''
        finds and returns the predicted PM10, PM25 values for one day using k nearest neighbors algorithm, the mode decides which specific algorithm to use
    '''
    predictions = {}
    if mode == 0:
        predictions = recursive_window(nearest_neighbors,loc_PM10, loc_PM25, data_2016, data_2017)
    elif mode == 1:
        predictions = rolling_window(nearest_neighbors,loc_PM10, loc_PM25, data_2016, data_2017)
    elif mode == 2:
        predictions = normalised_recursive(nearest_neighbors,loc_PM10, loc_PM25, data_2016, data_2017)
    elif mode == 3:
        predictions = normalised_rolling(nearest_neighbors,loc_PM10, loc_PM25, data_2016, data_2017)
    return predictions


def showData(loc_PM10,loc_PM25,data_2017,predictions):
    '''
        Return the mean of mean squared error for predicted values of 24 stations for 24 hours
    '''
    values=[]
    keyset = predictions.keys()
    count = 0
    average_PM10 = 0.0
    average_PM25 = 0.0
    for j in keyset:
        pred = predictions.get(j)
        error_10 = 0.0
        error_25 = 0.0
        count2 = 0
        for i in range(count,len(data_2017),24):
            values.append((data_2017[i][0],pred[count2][0],pred[count2][1],data_2017[i][loc_PM10],data_2017[i][loc_PM25]))
            count2 += 1
        count += 1

    return values


def main():
    '''
        This function first calls the clean_data to get the data and fill the blank values and put it into the pickle files for faster retrieval. Then it plots average of mean squared error for all stations for various values of k = nearest_neighbors against k. k is scaled up by 2 in every iteration.
    '''
    loc_PM10, loc_PM25, data_2016, data_2017 = clean_data.main()

    #plot the 4 methods
    #for recursive window
    # k = 6
    # nearest_neighbors=2**k
    # predictions = k_nearest_neighbours(nearest_neighbors,loc_PM10,loc_PM25,data_2016,data_2017,0)
    # values = showData(loc_PM10,loc_PM25,data_2017,predictions)
    # with open("recursive_window_predictions.txt","w") as file:
    #     file.write("Time    Predicted_PM10      Predicted_PM25      Actual_PM10      Actual_PM25\n")
    #     for tup in values:
    #         file.write("%s\t%s\t%s\t%s\t%s\n"%(tup[0],tup[1],tup[2],tup[3],tup[4]))
            
    # #for Rolling Window
    # k = 6
    # nearest_neighbors=2**k
    # predictions = k_nearest_neighbours(nearest_neighbors,loc_PM10,loc_PM25,data_2016,data_2017,1)
    # values = showData(loc_PM10,loc_PM25,data_2017,predictions)
    # with open("recursive_window_predictions.txt","w") as file:
    #     file.write("Time    Predicted_PM10      Predicted_PM25      Actual_PM10      Actual_PM25\n")
    #     for tup in values:
    #         file.write("%s\t%s\t%s\t%s\t%s\n"%(tup[0],tup[1],tup[2],tup[3],tup[4]))

    #for Normalised Recursive Window
    k = 6
    nearest_neighbors=2**k
    predictions = k_nearest_neighbours(nearest_neighbors,loc_PM10,loc_PM25,data_2016,data_2017,2)
    values = showData(loc_PM10,loc_PM25,data_2017,predictions)
    with open("normalised_recursive_window_predictions.txt","w") as file:
        file.write("Time    Predicted_PM10      Predicted_PM25      Actual_PM10      Actual_PM25\n")
        for tup in values:
            file.write("%s\t%s\t%s\t%s\t%s\n"%(tup[0],tup[1],tup[2],tup[3],tup[4]))
   
        
    #for Normalised Rolling Window
    # k = 6
    # nearest_neighbors=2**k
    # predictions = k_nearest_neighbours(nearest_neighbors,loc_PM10,loc_PM25,data_2016,data_2017,3)
    #  values = showData(loc_PM10,loc_PM25,data_2017,predictions)
    # with open("recursive_window_predictions.txt","w") as file:
    #     file.write("Time    Predicted_PM10      Predicted_PM25      Actual_PM10      Actual_PM25\n")
    #     for tup in values:
    #         file.write("%s\t%s\t%s\t%s\t%s\n"%(tup[0],tup[1],tup[2],tup[3],tup[4]))


if __name__ == '__main__':
    main()
