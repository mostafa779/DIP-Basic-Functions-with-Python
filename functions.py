import cv2 as cv
import numpy as np

def subSample(image,r=2):
    # 100 * 98
    rows = image.shape[0] # 1600
    columns = image.shape[1] #1568

    rowsRate = r
    columnsRate = r

    newMatrix = []

    r = 0
    for row in range(0,rows-1,rowsRate):
        newMatrix.append([])
        for column in range(0,columns-1,columnsRate):
            pixel = image[row][column]
            newMatrix[r].append(pixel)
        r += 1
    
    return np.array(newMatrix,dtype='uint8')

def reSample(image,r=2):
    
    rows = image.shape[0]

    rate = r

    newMatrix = []
    
    for row in image:
        
        temp= []

        for pixel in row:
            for x in range(rate):
                temp.append(pixel)

        for y in range(rate):
            newMatrix.append(temp)
           
    return np.array(newMatrix,dtype='uint8')

def getValue(pixel):
    return 0.299 * pixel[0] + 0.587 * pixel[1] + 0.114 * pixel[2]

def changeGrayLevel(image,rate=4):
    target_levels = 2**rate
    target_factor = 256 / target_levels
    newImage = np.floor(image/256 * target_levels) * target_factor
    return np.array(newImage,dtype='uint8')


def convertNegative(im,k=8):
    L = 2 ** k
    neg = L - 1 - im
    return neg


def powLog(im,n=4):
    c = 1
    im = c * im ** n
    return im

def convertToGrayScale(image):
    matrix = np.copy(image)
    for row in matrix:
        for pixel in row: 
            pixel[:] = 0.299 * pixel[0] + 0.587 * pixel[1] + 0.114 * pixel[2]

    return matrix

def threshold(image,k=95):

    matrix= np.copy(image)
    for row in matrix:
        for pixel in row:
            val = 0.299 * pixel[0] + 0.587 * pixel[1] + 0.114 * pixel[2]
            if val >=k: pixel[:]=255
            else: pixel[:]=0
            
    return matrix

def contrast_stretch(image,s_min=100,s_max=151):
    r_max = image.max()
    r_min = image.min()

    # s = ((s_max-s_min) / (r_max-r_min)) * (r-r_min)+s_min
    matrix = []
    for i,row in enumerate(image):
        matrix.append([])
        
        for pixel in row:
            temp = []
            
            for r in pixel:
                s = ((s_max-s_min) / (r_max-r_min)) * ((r-r_min) + s_min)
                temp.append(s)

            matrix[i].append(temp)

    return np.array(matrix,dtype='uint8')


def bitSlicing(image):
    # not implemented yet
    pass


def bits(pixel):
    # not implemented yet
    pass
