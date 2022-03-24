def getValue(pixel):
    return int(0.299 * pixel[0] + 0.587 * pixel[1] + 0.114 * pixel[2])

def getBits(n):
    n = bin(n).replace('0b','')
    temp = ''
    for i in range(len(n),8):
        temp += '0'
    temp += n
    return temp

def toDecimal(bits,k):
    result = 0
    for i in range(k):
        result += int(bits[i]) * (2 ** (7-i))
    
    return int(result)