import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from io_repo import open_folder

def output_results (df, path):
    path = path+'results/'
    open_folder(path)
    resultFile = open(path+"result.txt", "a")
    # at the reasult file all values are normalized (devided by 4)
    # so multiply the error by 4 first
    df.loc[:, 'error'] *= 4


    size = len(df)
    def frange(start, stop, step):
        i = start
        while i < stop:
            yield i
            i += step
    def draw(tmpdf, title):
        tmpsize = len(tmpdf)
        if tmpsize==0:
            pass
        else:
            resultFile.write(
                "\n" + "\n" + title + "\n" + "(%" + str("%.2f" % (tmpsize / size * 100)) + " of total sentiments (" + str(
                    tmpsize) + " sentiment))")
            res = [0.00,0.25,0.50,0.75,1.00,1.25,1.50,1.75,2.00,2.25,2.50,2.75,3.00,3.25,3.50,3.75,4.00]
            xlabel = [0.125,0.375,0.625,0.875,1.125,1.375,1.625,1.875,2.125,2.375,2.625,2.875,3.125,3.375,3.625,3.875]
            data = []
            for x in range(0, len(res) - 1):
                tmp = tmpdf.between(res[x], res[x + 1] - 0.0000000000000001)
                tmpcount = tmp.sum()
                tmppercent = tmpcount / tmpsize * 100
                data.append(tmppercent)
                line = ""
                for y in range(0, int(tmppercent / 3)):
                    line += "|"
                resultFile.write(
                    "\n" + str(res[x]) + "-" + str(res[x + 1]) + "\t" + "%" + str("%.2f" % tmppercent) + "\t" + "(" + str(
                        tmpcount) + ")" + "\t" + line)
            plt.plot(xlabel, data)
            plt.title(title)
            plt.xlabel('error 0 to 4')
            plt.ylabel('percent in total error')
            plt.savefig(path+title+".png")
            # plt.show()
            plt.clf()


    draw(df.error,"error")
    resultFile.write("\n"+df.error.describe().to_string()+"\n")
    df.error.describe()

    tmplist=df.sentiment.between(0,0.33333333333)
    tmpdf=df.error[tmplist]
    draw(tmpdf,"error of sentiment between 0-1.33 given")
    resultFile.write("\n"+tmpdf.describe().to_string()+"\n")
    tmpdf.describe()

    tmplist=df.sentiment.between(0.33333333334,0.66666666666)
    tmpdf=df.error[tmplist]
    draw(tmpdf,"error of sentiment between 1.33-2.66 given")
    resultFile.write("\n"+tmpdf.describe().to_string()+"\n")
    tmpdf.describe()

    tmplist=df.sentiment.between(0.66666666667,1)
    tmpdf=df.error[tmplist]
    draw(tmpdf,"error of sentiment between 2.66-4 given")
    resultFile.write("\n"+tmpdf.describe().to_string()+"\n")
    tmpdf.describe()

    resultFile.write("\n---------------------------------")

    resultFile.close()
