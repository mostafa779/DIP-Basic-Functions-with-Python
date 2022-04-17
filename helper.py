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

def replicate(image, n=3):
    img = np.copy(image)
    height = img.shape[0]
    
    newImg = []
    
    n = int(n/2)
    
    col1 = img[:,0].reshape(height,1,3)
    col2 = img[:,-1].reshape(height,1,3)
    
    for i,row in enumerate(img):
        newImg.append([])
        
        for _ in range(n+1):
            newImg[i].append(col1[0][0])
        
        for pixel in row:
            newImg[i].append(pixel)
    
        for _ in range(n+1):
            newImg[i].append(col2[0][0])
            
    for _ in range(n+1):
        newImg.insert(0,newImg[0])
        newImg.insert(-1,newImg[-1])
        
    return np.array(newImg,dtype='uint8')


def convlution_sum(window,matrix):
    n = len(matrix)
    
    result = 0
    for i in range(0,n):
        for j in range(0,n):
            result += matrix[i][j] * window[i][j]
            
    return result

def getNeighbours(img, row, col, n):
    matrix = np.zeros((n,n,3))
    
    k = int((n-1)/2)
    
    r1 = row-k
    r2 = row+k+1
    
    c1 = col-k
    c2 = col+k+1

    matrix[:] = img[r1:r2,c1:c2]
    
    return matrix
