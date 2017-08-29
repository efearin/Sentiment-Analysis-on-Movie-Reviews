import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import io_repo


def draw(tmpdf, title, path):
    tmpsize = len(tmpdf)
    # if there is no data
    if tmpsize == 0:
        pass
    # if there is data to show
    else:
        resultFile.write(
            "\n" + "\n" + title + "\n" + "(%" + str("%.2f" % (tmpsize / size * 100)) + " of total sentiments (" + str(
                tmpsize) + " sentiment))")
        data = []
        for x in range(0, len(data_window_list) - 1):
            tmp = tmpdf.between(data_window_list[x], data_window_list[x + 1] - 0.0000000000000001)
            tmpcount = tmp.sum()
            tmppercent = tmpcount / tmpsize * 100
            data.append(tmppercent)
            # line that shows how large that data part is seen like "||||||||     "
            percent_line = ""
            for y in range(0, int(tmppercent / percent_line_divisor)):
                percent_line += "|"
            if data_window_list[x] < 0:
                resultFile.write(
                    "\n" + str(data_window_list[x]) + "/" + str(data_window_list[x + 1]) + "\t" + "%" +
                    str("%.2f" % tmppercent)+ "\t" + "(" + str(tmpcount) + ")" + "\t" + percent_line)
            else:
                resultFile.write(
                    "\n" + str(data_window_list[x]) + "/" + str(data_window_list[x + 1]) + "\t"+"\t" + "%" +
                    str("%.2f" % tmppercent)+ "\t" + "(" + str(tmpcount) + ")" + "\t" + percent_line)
        plt.plot(xlabel_list, data, 'ro')
        plt.title(title+'\n'+'error = calculated - real scores')
        plt.xlabel('error -4 to 4')
        plt.ylabel('percent in total error')
        plt.savefig(path + title + ".png")
        plt.show()
        plt.clf()

#
def get_data_window_list_and_xlabel_list (step_size):
    # datas between 2 successive data_window value will be collected
    # and assigned (discretized) as same
    data_window_list=[float(0)]
    while data_window_list[-1] < 4:
        data_window_list.append(float(data_window_list[-1]+step_size))
    while data_window_list[0] > -4:
        data_window_list = [float(data_window_list[0]-step_size)] + data_window_list
    # for the out of range parts (<-4 and >4) add infinites to both sides
    data_window_list = [-float('Inf')]+data_window_list
    data_window_list.append(float('Inf'))
    # create x label list
    # by choosing midpoints of 2 following data_windows
    # create x labels for the start and end infinity number
    # to prevent calculation and visualization error add -10 and 10 to xlabel_list
    xlabel_list=[]
    for x in range(1, len(data_window_list)-2):
        xlabel_list.append((data_window_list[x]+data_window_list[x+1])/2)
    xlabel_list = [float(-10)]+xlabel_list
    xlabel_list.append(float(10))
    return data_window_list, xlabel_list


def frange(start, stop, step):
    i = start
    while i < stop:
        yield i
        i += step

def output_results (df, step_size, path):
    global data_window_list, xlabel_list, size, resultFile, percent_line_divisor
    percent_line_divisor = 3
    # open result file at given folder
    path = path+'results/'
    io_repo.open_folder(path)
    resultFile = open(path+"result.txt", "a")
    resultFile.write('error = calculated - real scores')
    # at the reasult file all values are normalized (devided by 4)
    # so multiply the error by 4 first
    df.loc[:, 'error'] *= 4
    data_window_list, xlabel_list = get_data_window_list_and_xlabel_list(step_size)
    size = len(df)

    draw(df.error,"error", path)
    resultFile.write("\n"+df.error.describe().to_string()+"\n")
    df.error.describe()

    tmplist=df.sentiment.between(0,0.33333333333)
    tmpdf=df.error[tmplist]
    draw(tmpdf,"error of sentiment between 0-1.33 given",path)
    resultFile.write("\n"+tmpdf.describe().to_string()+"\n")
    tmpdf.describe()

    tmplist=df.sentiment.between(0.33333333334,0.66666666666)
    tmpdf=df.error[tmplist]
    draw(tmpdf,"error of sentiment between 1.33-2.66 given",path)
    resultFile.write("\n"+tmpdf.describe().to_string()+"\n")
    tmpdf.describe()

    tmplist=df.sentiment.between(0.66666666667,1)
    tmpdf=df.error[tmplist]
    draw(tmpdf,"error of sentiment between 2.66-4 given",path)
    resultFile.write("\n"+tmpdf.describe().to_string()+"\n")
    tmpdf.describe()

    resultFile.write("\n---------------------------------")

    resultFile.close()
