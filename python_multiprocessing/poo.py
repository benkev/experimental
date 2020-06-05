import multiprocessing as mp
import pandas

def task(arg):
    DF = pandas.DataFrame({"hello":range(1000)}) 
    return DF

if __name__ == '__main__':

    p = mp.Pool(processes=20)
    results = p.map(task, range(20), chunksize=1)
