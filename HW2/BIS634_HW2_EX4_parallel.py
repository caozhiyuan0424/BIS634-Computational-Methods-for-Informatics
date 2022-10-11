#%%
import multiprocessing
import numpy as np
from time import perf_counter
import matplotlib.pyplot as plt

def merge(d1, d2):
    result = []
    # note: this takes the top items off the left and right piles
    left_top = next(d1)
    right_top = next(d2)
    while True:
        if left_top < right_top:
            result.append(left_top)
            try:
                left_top = next(d1)
            except StopIteration:
                # nothing remains on the left; add the right + return
                return result + [right_top] + list(d2)
        else:
            result.append(right_top)
            try:
                right_top = next(d2)
            except StopIteration:
                # nothing remains on the right; add the left + return
                return result + [left_top] + list(d1)

def alg2(data):
    if len(data) <= 1:
        return data
    else:
        split = len(data) // 2
        left = iter(alg2(data[:split]))
        right = iter(alg2(data[split:]))
        return merge(left, right)

def data1(n, sigma=10, rho=28, beta=8/3, dt=0.01, x=1, y=1, z=1):
    import numpy
    state = numpy.array([x, y, z], dtype=float)
    result = []
    for _ in range(n):
        x, y, z = state
        state += dt * numpy.array([
            sigma * (y - x),
            x * (rho - z) - y,
            x * y - beta * z
        ])
        result.append(float(state[0] + 30))
    return result

def data2(n):
    return list(range(n))

def data3(n):
    return list(range(n, 0, -1))

def para_alg2(datalist):
    if len(datalist) <= 1:
        return datalist
    else: 
        split = len(datalist) //2
        with multiprocessing.Pool() as p:
            results = p.map(alg2, [datalist[:split], datalist[split:]])
        return merge(iter(results[0]), iter(results[1]))

if __name__ == '__main__': 
    xList = np.logspace(0, 7)
    resultList1 = []
    resultList2 = []
    for x in xList:
        x = int(x)
        datalist = data1(x)
        start = perf_counter()
        para_alg2(datalist)
        end = perf_counter()
        resultList1.append(end-start)

        datalist = data1(x)
        start = perf_counter()
        alg2(datalist)
        end = perf_counter()
        resultList2.append(end-start)
    plt.loglog(xList, resultList1, label='parallel alg2')
    plt.loglog(xList, resultList2, label='normal alg2')
    plt.legend()
    plt.savefig("EX4_7.png")
    plt.show()
 
# %%
print(resultList1[-1])
print(resultList2[-1])
# %%
