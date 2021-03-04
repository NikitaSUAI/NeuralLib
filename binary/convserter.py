import numpy as np

def B2D(array):
    if array.size:
        return 2 ** (len(array)-1) * array[0] + B2D(array[1:])
    else:
        return 0

def D2B(num:int):
    res = np.zeros(8)
    count = -1
    while num != 0:
        if count < -8:
            break
        res[count] = (num % 2)
        count -= 1
        num = (num - num % 2) / 2
    return np.array(res)

if __name__=="__main__":
    print(B2D(np.array([1, 0, 1, 1])))
    print(D2B(200))