import pandas as pd
from pathlib import Path
import os, math, pickle
def get_data():
    '''
        -> Read data from csv/xlsx files
        -> We have assumed that the data given is already sorted by data time, i.e. the first entry is of
        201x-01-01 01:00:00
        -> The dataset madrid_2016 and madrid_2017 has been changed a little bit to include 201x-01-01 00:00:00
        for that year and remove the final time i.e. 201(x+1)-01-01 00:00:00, as any year's dataset contained data from 201x-01-01 01:00:00 to 201(x+1)-01-01 00:00:00. The excel files now contain data from
        201x-01-01 00:00:00 to 201x-31-12 23:00:00
        -> Due to the change in the assignment, the file madrid_2017 now contains data for 24 stations of 24 hours only i.e. data of one day
    '''
    loc_PM10 = 8
    loc_PM25 = 9

    temppath1 = Path(os.getcwd()+'/data_2016.pickle')
    temppath2 = Path(os.getcwd()+'/data_2017.pickle')

    if temppath1.is_file() and temppath2.is_file():
        print("Stored files exist")
        file1 = open('data_2016.pickle','rb')
        data_2016 = pickle.load(file1)
        file1.close()
        file2 = open('data_2017.pickle','rb')
        data_2017 = pickle.load(file2)
        file2.close()
        print("Files loaded onto data structures")
    else:
        print("Stored files donot exist")
        data_2016 = pd.read_excel('madrid_2016.xlsx').values
        data_2017 = pd.read_excel('madrid_2017.xlsx').values
        data_2016, data_2017 = fill_empty_values(data_2016, data_2017)
        print("Empty Values filled")
        file1 = open('data_2016.pickle','wb')
        pickle.dump(data_2016,file1)
        file1.close()
        file2 = open('data_2017.pickle','wb')
        pickle.dump(data_2017,file2)
        file2.close()
        print("Files loaded onto data structures and saved as pickle files")

    return loc_PM10, loc_PM25, data_2016, data_2017

def fill_empty_values(data_2016, data_2017):
    '''
        Fill empty table values for 201x data with averages of the coloumn
    '''
    print("Filling empty values")
    for i in range(1,len(data_2016[0]) - 1):
        sum = 0.0
        count = 0
        for j in range(0,len(data_2016)):
            if not math.isnan(data_2016[j][i]):
                count += 1
                sum += data_2016[j][i]
        average = sum/count
        for j in range(0,len(data_2016)):
            if math.isnan(data_2016[j][i]):
                data_2016[j][i] = average

    for i in range(1,len(data_2017[0]) - 1):
        sum = 0.0
        count = 0
        for j in range(0,len(data_2017)):
            if not math.isnan(data_2017[j][i]):
                count += 1
                sum += data_2017[j][i]
        average = sum/count
        for j in range(0,len(data_2017)):
            if math.isnan(data_2017[j][i]):
                data_2017[j][i] = average
    return data_2016, data_2017


def main():
    loc_PM10, loc_PM25, data_2016, data_2017 = get_data()
    return loc_PM10, loc_PM25, data_2016, data_2017

if __name__ == '__main__':
    main()
