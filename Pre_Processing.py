# -*- coding: utf-8 -*-
"""
Created on Wed Oct 25 14:54:30 2023

@author: Hadiya Sajjad
"""

import numpy as np
import cv2
import skimage.color
import skimage.filters
import skimage.io
import matplotlib.pyplot as plt
#import import_ipynb
import random
from PIL import Image
import scipy ,scipy.misc 
import scipy.signal
from skimage import color, data, restoration
from skimage.util import random_noise
from skimage import exposure
from skimage.transform import resize


# Final Feature Extraction Algo
# See if there any room for improvements
# See if colour consistency make any changes for real
def FeatureExtraction(img):

    img = Ying_2017_CAIP(img)
    pil_image=to_pil(retinex((img)))
    open_cv_image = np.array(pil_image)
    open_cv_image = open_cv_image[:, :, ::-1].copy()
    hist_equalization_result = HistogramEqualization(open_cv_image)
    edgeEnhanced=edgeEnhancement(hist_equalization_result)
    gray= Convert2Grayscale(edgeEnhanced)
    return gray

def Ying_2017_CAIP(img, mu=0.5, a=-0.3293, b=1.1258):
    lamda = 0.5
    sigma = 5
    I = cv2.normalize(img.astype('float32'), None, 0.0, 1.0, cv2.NORM_MINMAX)

    # Weight matrix estimation
    t_b = np.max(I, axis=2)
    h_,w_=t_b.shape
    w_= int(w_*0.5)
    h_ =int(h_*0.5)
    #t_b_resized = np.array(Image.fromarray(t_b, mode='F').resize((40, 40), resample=Image.BICUBIC))
    #t_b_resized = cv2.resize(t_b, (w_,h_), interpolation = cv2.INTER_CUBIC)
    t_b_new= np.asarray(Image.fromarray(t_b , mode='F').resize((w_,h_), resample=Image.BICUBIC))
    t_our = cv2.resize(tsmooth(t_b_new, lamda, sigma), (t_b.shape[1], t_b.shape[0]), interpolation=cv2.INTER_AREA)
    t_our = cv2.resize(tsmooth(cv2.resize(t_b, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_CUBIC), lamda, sigma), (t_b.shape[1], t_b.shape[0]), interpolation=cv2.INTER_AREA)

    # Apply camera model with k(exposure ratio)
    isBad = t_our < 0.5
    J = maxEntropyEnhance(I, isBad)

    # W: Weight Matrix
    t = np.zeros((t_our.shape[0], t_our.shape[1], I.shape[2]))
    for i in range(I.shape[2]):
        t[:,:,i] = t_our
    W = t**mu

    I2 = I*W
    J2 = J*(1-W)

    result = I2 + J2
    result = result * 255
    result[result > 255] = 255
    result[result<0] = 0
    return result.astype(np.uint8)

def HistogramEqualization(img):
    img_to_yuv = cv2.cvtColor(img,cv2.COLOR_RGB2YUV)
    img_to_yuv[:,:,0] = cv2.equalizeHist(img_to_yuv[:,:,0])
    return cv2.cvtColor(img_to_yuv, cv2.COLOR_YUV2RGB)

# Edge Enhancement
def edgeEnhancement(img):
    kernel = np.array([[-1,-1,-1,-1,-1],
                               [-1,2,2,2,-1],
                               [-1,2,8,2,-1],
                               [-2,2,2,2,-1],
                               [-1,-1,-1,-1,-1]])/8.0
    img = cv2.filter2D(img, -1, kernel)
    image=cv2.filter2D(img, -1, kernel)
    return image

def Convert2Grayscale(img):
    return skimage.color.rgb2gray(img)

def motion_blur(img):

  size = 15
  # generating the kernel
  kernel_motion_blur = np.zeros((size, size))
  kernel_motion_blur[int((size-1)/2), :] = np.ones(size)
  kernel_motion_blur = kernel_motion_blur / size

  # applying the kernel to the input image
  output = cv2.filter2D(img, -1, kernel_motion_blur)
  return output

def motion_blur_horizontal_vertical(img):


  kernel_size = 30

  # Create the vertical kernel.
  kernel_v = np.zeros((kernel_size, kernel_size))

  # Create a copy of the same for creating the horizontal kernel.
  kernel_h = np.copy(kernel_v)

  # Fill the middle row with ones.
  kernel_v[:, int((kernel_size - 1)/2)] = np.ones(kernel_size)
  kernel_h[int((kernel_size - 1)/2), :] = np.ones(kernel_size)

  # Normalize.
  kernel_v /= kernel_size
  kernel_h /= kernel_size

  # Apply the vertical kernel.
  vertical_mb = cv2.filter2D(img, -1, kernel_v)

  # Apply the horizontal kernel.
  horizonal_mb = cv2.filter2D(vertical_mb, -1, kernel_h)
  return horizonal_mb

def computeTextureWeights(fin, sigma, sharpness):
    dt0_v = np.vstack((np.diff(fin, n=1, axis=0), fin[0,:]-fin[-1,:]))
    dt0_h = np.vstack((np.diff(fin, n=1, axis=1).conj().T, fin[:,0].conj().T-fin[:,-1].conj().T)).conj().T

    gauker_h = scipy.signal.convolve2d(dt0_h, np.ones((1,sigma)), mode='same')
    gauker_v = scipy.signal.convolve2d(dt0_v, np.ones((sigma,1)), mode='same')

    W_h = 1/(np.abs(gauker_h)*np.abs(dt0_h)+sharpness)
    W_v = 1/(np.abs(gauker_v)*np.abs(dt0_v)+sharpness)

    return  W_h, W_v

def solveLinearEquation(IN, wx, wy, lamda):
    [r, c] = IN.shape
    k = r * c
    dx =  -lamda * wx.flatten('F')
    dy =  -lamda * wy.flatten('F')
    tempx = np.roll(wx, 1, axis=1)
    tempy = np.roll(wy, 1, axis=0)
    dxa = -lamda *tempx.flatten('F')
    dya = -lamda *tempy.flatten('F')
    tmp = wx[:,-1]
    tempx = np.concatenate((tmp[:,None], np.zeros((r,c-1))), axis=1)
    tmp = wy[-1,:]
    tempy = np.concatenate((tmp[None,:], np.zeros((r-1,c))), axis=0)
    dxd1 = -lamda * tempx.flatten('F')
    dyd1 = -lamda * tempy.flatten('F')

    wx[:,-1] = 0
    wy[-1,:] = 0
    dxd2 = -lamda * wx.flatten('F')
    dyd2 = -lamda * wy.flatten('F')

    Ax = scipy.sparse.spdiags(np.concatenate((dxd1[:,None], dxd2[:,None]), axis=1).T, np.array([-k+r,-r]), k, k)
    Ay = scipy.sparse.spdiags(np.concatenate((dyd1[None,:], dyd2[None,:]), axis=0), np.array([-r+1,-1]), k, k)
    D = 1 - ( dx + dy + dxa + dya)
    A = ((Ax+Ay) + (Ax+Ay).conj().T + scipy.sparse.spdiags(D, 0, k, k)).T

    tin = IN[:,:]
    tout = scipy.sparse.linalg.spsolve(A, tin.flatten('F'))
    OUT = np.reshape(tout, (r, c), order='F')

    return OUT

def tsmooth(img, lamda=0.01, sigma=3.0, sharpness=0.001):
    I = cv2.normalize(img.astype('float64'), None, 0.0, 1.0, cv2.NORM_MINMAX)
    x = np.copy(I)
    wx, wy = computeTextureWeights(x, sigma, sharpness)
    S = solveLinearEquation(I, wx, wy, lamda)
    return S

def rgb2gm(I):
    if (I.shape[2] == 3):
        I = cv2.normalize(I.astype('float64'), None, 0.0, 1.0, cv2.NORM_MINMAX)
        I = np.abs((I[:,:,0]*I[:,:,1]*I[:,:,2]))**(1/3)

    return I

def applyK(I, k, a=-0.3293, b=1.1258):
    f = lambda x: np.exp((1-x**a)*b)
    beta = f(k)
    gamma = k**a
    J = (I**gamma)*beta
    return J

def entropy(X):
    tmp = X * 255
    tmp[tmp > 255] = 255
    tmp[tmp<0] = 0
    tmp = tmp.astype(np.uint8)
    _, counts = np.unique(tmp, return_counts=True)
    pk = np.asarray(counts)
    pk = 1.0*pk / np.sum(pk, axis=0)
    S = -np.sum(pk * np.log2(pk), axis=0)
    return S

def maxEntropyEnhance(I, isBad, a=-0.3293, b=1.1258):
    # Esatimate k
    tmp = cv2.resize(I, (50,50), interpolation=cv2.INTER_AREA)
    tmp[tmp<0] = 0
    tmp = tmp.real
    Y = rgb2gm(tmp)

    isBad = isBad * 1
    # isBad = scipy.misc.imresize(isBad, (50,50), interp='bicubic', mode='F')
    isBad = np.array(Image.fromarray(isBad,mode='F').resize((50, 50), resample=Image.BICUBIC))
    # isBad =  cv2.resize(isBad, (50,50), interpolation = cv2.INTER_CUBIC)
    isBad[isBad<0.5] = 0
    isBad[isBad>=0.5] = 1
    Y = Y[isBad==1]

    if Y.size == 0:
       J = I
       return J

    f = lambda k: -entropy(applyK(Y, k))
    opt_k = scipy.optimize.fminbound(f, 1, 7)

    # Apply k
    J = applyK(I, opt_k, a, b) - 0.01
    return J

## Read image in path
def readImage(path):
    return cv2.imread(path)

## Sharpen Image
def sharpen(img):
    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    im = cv2.filter2D(img, -1, kernel)
    plt.imshow(im)
    return im

# Excessive sharpening image
def excessive(img):
    kernel = np.array([[1,1,1], [1,-7,1], [1,1,1]])
    im = cv2.filter2D(img, -1, kernel)
    return im

# Blur of images
def blur(img):
    blur =  cv2.medianBlur(img,5)
    return blur

# Weight image with another image
def addWeightedImg(img,blur):
    result = cv2.addWeighted(img, 1, blur, -0.5, 0)
    return result

# Contrast Enhancement
def ContrastEnhancement(img):
    hsvImg = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    hsvImg[...,1] = hsvImg[...,1]*1.6
    hsvImg[...,2] = hsvImg[...,2]*0.8
    img=cv2.cvtColor(hsvImg,cv2.COLOR_HSV2RGB)

# Find Contours
def FindContours(edged):
    contours, hierarchy=cv2.findContours(edged,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(img,contours,-1,(0,255,0),1)
    ret =40
    img[img>ret]=255
    img[img<=ret]=0
    img =cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)

def RemoveNoiseColouredImg(img):
    dst = cv2.fastNlMeansDenoisingColored(img,None,10,10,7,21)

def CalculateContrast(img):
    # convert to LAB color space
    lab = cv2.cvtColor(img,cv2.COLOR_RGB2LAB)

    # separate channels
    L,A,B=cv2.split(lab)

    # compute minimum and maximum in 5x5 region using erode and dilate
    kernel = np.ones((5,5),np.uint8)
    min = cv2.erode(L,kernel,iterations = 1)
    max = cv2.dilate(L,kernel,iterations = 1)

    # convert min and max to floats
    min = min.astype(np.float64)
    max = max.astype(np.float64)

    # compute local contrast
    contrast = (max-min)/(max+min)

    # get average across whole image
    average_contrast = 100*np.mean(contrast)

    print(str(average_contrast)+"%")

def from_pil(pimg):
    pimg = pimg.convert(mode='RGB')
    nimg = np.asarray(pimg)
    nimg.flags.writeable = True
    return nimg

def to_pil(nimg):
    return Image.fromarray(np.uint8(nimg))

def stretch_pre(nimg):
    """
    from 'Applicability Of White-Balancing Algorithms to Restoring Faded Colour Slides: An Empirical Evaluation'
    """
    nimg = nimg.transpose(2, 0, 1)
    nimg[0] = np.maximum(nimg[0]-nimg[0].min(),0)
    nimg[1] = np.maximum(nimg[1]-nimg[1].min(),0)
    nimg[2] = np.maximum(nimg[2]-nimg[2].min(),0)
    return nimg.transpose(1, 2, 0)

def grey_world(nimg):
    nimg = nimg.transpose(2, 0, 1).astype(np.uint32)
    mu_g = np.average(nimg[1])
    nimg[0] = np.minimum(nimg[0]*(mu_g/np.average(nimg[0])),255)
    nimg[2] = np.minimum(nimg[2]*(mu_g/np.average(nimg[2])),255)
    return  nimg.transpose(1, 2, 0).astype(np.uint8)

def max_white(nimg):
    if nimg.dtype==np.uint8:
        brightest=float(2**8)
    elif nimg.dtype==np.uint16:
        brightest=float(2**16)
    elif nimg.dtype==np.uint32:
        brightest=float(2**32)
    else:
        brightest==float(2**8)
    nimg = nimg.transpose(2, 0, 1)
    nimg = nimg.astype(np.int32)
    nimg[0] = np.minimum(nimg[0] * (brightest/float(nimg[0].max())),255)
    nimg[1] = np.minimum(nimg[1] * (brightest/float(nimg[1].max())),255)
    nimg[2] = np.minimum(nimg[2] * (brightest/float(nimg[2].max())),255)
    return nimg.transpose(1, 2, 0).astype(np.uint8)

def stretch(nimg):
    return max_white(stretch_pre(nimg))

def retinex(nimg):
    nimg = nimg.transpose(2, 0, 1).astype(np.uint32)
    mu_g = nimg[1].max()
    nimg[0] = np.minimum(nimg[0]*(mu_g/float(nimg[0].max())),255)
    nimg[2] = np.minimum(nimg[2]*(mu_g/float(nimg[2].max())),255)
    return nimg.transpose(1, 2, 0).astype(np.uint8)

def retinex_adjust(nimg):
    """
    from 'Combining Gray World and Retinex Theory for Automatic White Balance in Digital Photography'
    """
    nimg = nimg.transpose(2, 0, 1).astype(np.uint32)
    sum_r = np.sum(nimg[0])
    sum_r2 = np.sum(nimg[0]**2)
    max_r = nimg[0].max()
    max_r2 = max_r**2
    sum_g = np.sum(nimg[1])
    max_g = nimg[1].max()
    coefficient = np.linalg.solve(np.array([[sum_r2,sum_r],[max_r2,max_r]]),
                                  np.array([sum_g,max_g]))
    nimg[0] = np.minimum((nimg[0]**2)*coefficient[0] + nimg[0]*coefficient[1],255)
    sum_b = np.sum(nimg[1])
    sum_b2 = np.sum(nimg[1]**2)
    max_b = nimg[1].max()
    max_b2 = max_r**2
    coefficient = np.linalg.solve(np.array([[sum_b2,sum_b],[max_b2,max_b]]),
                                             np.array([sum_g,max_g]))
    nimg[1] = np.minimum((nimg[1]**2)*coefficient[0] + nimg[1]*coefficient[1],255)
    return nimg.transpose(1, 2, 0).astype(np.uint8)

def retinex_with_adjust(nimg):
    return retinex_adjust(retinex(nimg))

def standard_deviation_weighted_grey_world(nimg,subwidth,subheight):
    """
    This function does not work correctly
    """
    nimg = nimg.astype(np.uint32)
    height, width,ch = nimg.shape
    strides = nimg.itemsize*np.array([width*subheight,subwidth,width,3,1])
    shape = (height/subheight, width/subwidth, subheight, subwidth,3)
    blocks = np.lib.stride_tricks.as_strided(nimg, shape=shape, strides=strides)
    y,x = blocks.shape[:2]
    std_r = np.zeros([y,x],dtype=np.float16)
    std_g = np.zeros([y,x],dtype=np.float16)
    std_b = np.zeros([y,x],dtype=np.float16)
    std_r_sum = 0.0
    std_g_sum = 0.0
    std_b_sum = 0.0
    for i in xrange(y):
        for j in xrange(x):
            subblock = blocks[i,j]
            subb = subblock.transpose(2, 0, 1)
            std_r[i,j]=np.std(subb[0])
            std_g[i,j]=np.std(subb[1])
            std_b[i,j]=np.std(subb[2])
            std_r_sum += std_r[i,j]
            std_g_sum += std_g[i,j]
            std_b_sum += std_b[i,j]
    sdwa_r = 0.0
    sdwa_g = 0.0
    sdwa_b = 0.0
    for i in xrange(y):
        for j in xrange(x):
            subblock = blocks[i,j]
            subb = subblock.transpose(2, 0, 1)
            mean_r=np.mean(subb[0])
            mean_g=np.mean(subb[1])
            mean_b=np.mean(subb[2])
            sdwa_r += (std_r[i,j]/std_r_sum)*mean_r
            sdwa_g += (std_g[i,j]/std_g_sum)*mean_g
            sdwa_b += (std_b[i,j]/std_b_sum)*mean_b
    sdwa_avg = (sdwa_r+sdwa_g+sdwa_b)/3
    gain_r = sdwa_avg/sdwa_r
    gain_g = sdwa_avg/sdwa_g
    gain_b = sdwa_avg/sdwa_b
    nimg = nimg.transpose(2, 0, 1)
    nimg[0] = np.minimum(nimg[0]*gain_r,255)
    nimg[1] = np.minimum(nimg[1]*gain_g,255)
    nimg[2] = np.minimum(nimg[2]*gain_b,255)
    return nimg.transpose(1, 2, 0).astype(np.uint8)

def ProposedSegmentationAlgoWithDisplay(img):
    
    img = Ying_2017_CAIP(img)
    pil_image=to_pil(retinex((img)))
    open_cv_image = np.array(pil_image)
    open_cv_image = open_cv_image[:, :, ::-1].copy()
    hist_equalization_result = HistogramEqualization(open_cv_image)
    edgeEnhanced=edgeEnhancement(hist_equalization_result)
    gray= Convert2Grayscale(edgeEnhanced)
    fig, ax = plt.subplots(nrows=1, ncols=4)
    ax[0].imshow(img)
    ax[1].imshow(hist_equalization_result)
    ax[2].imshow(edgeEnhanced)
    ax[3].imshow(gray)
    plt.show()

def rBrightness(image,ratio):
    hsv=cv2.cvtColor(image,cv2.COLOR_RGB2HSV)
    brightness=np.float64(hsv[:, :, 2])
    brightness=brightness*(1.0+np.random.uniform(-ratio,ratio))
    brightness[brightness>255]=255
    brightness[brightness<0]=0
    hsv[:, :, 2]=brightness
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
def rRotation(image, angle):
    if angle==0:
        return image
    angle=np.random.uniform(-angle,angle)
    rows,cols=image.shape[:2]
    size=rows,cols
    center=rows/2,cols/2
    scale=1.0
    rotation=cv2.getRotationMatrix2D(center,angle,scale)
    return cv2.warpAffine(image,rotation,size)
def rTranslation(image, translation):
    if translation==0:
        return 0
    rows,cols=image.shape[:2]
    size=rows,cols
    x=np.random.uniform(-translation,translation)
    y=np.random.uniform(-translation,translation)
    trans=np.float32([[1,0,x],[0,1,y]])
    return cv2.warpAffine(image,trans,size)
def rShear(image, shear):
    if shear==0:
        return image
    rows,cols=image.shape[:2]
    size=rows,cols
    left,right,top,bottom=shear,cols-shear,shear,rows-shear
    dx=np.random.uniform(-shear,shear)
    dy=np.random.uniform(-shear,shear)
    p1=np.float32([[left,top],[right,top],[left,bottom]])
    p2=np.float32([[left+dx,top],[right+dx,top+dy],[left,bottom+dy]])
    move=cv2.getAffineTransform(p1,p2)
    return cv2.warpAffine(image,move,size)
def augment_image(image,brightness,angle,translation,shear):
    image=rBrightness(image,brightness)
    image=rRotation(image,angle)
    image=rTranslation(image,translation)
    image=rShear(image,shear)
    return image

def cv2_clipped_zoom(img, zoom_factor):
    """
    Center zoom in/out of the given image and returning an enlarged/shrinked view of
    the image without changing dimensions
    Args:
        img : Image array
        zoom_factor : amount of zoom as a ratio (0 to Inf)
    """
    height, width = img.shape[:2] # It's also the final desired shape
    new_height, new_width = int(height * zoom_factor), int(width * zoom_factor)

    ### Crop only the part that will remain in the result (more efficient)
    # Centered bbox of the final desired size in resized (larger/smaller) image coordinates
    y1, x1 = max(0, new_height - height) // 2, max(0, new_width - width) // 2
    y2, x2 = y1 + height, x1 + width
    bbox = np.array([y1,x1,y2,x2])
    # Map back to original image coordinates
    bbox = (bbox / zoom_factor).astype(np.int)
    y1, x1, y2, x2 = bbox
    cropped_img = img[y1:y2, x1:x2]

    # Handle padding when downscaling
    resize_height, resize_width = min(new_height, height), min(new_width, width)
    pad_height1, pad_width1 = (height - resize_height) // 2, (width - resize_width) //2
    pad_height2, pad_width2 = (height - resize_height) - pad_height1, (width - resize_width) - pad_width1
    pad_spec = [(pad_height1, pad_height2), (pad_width1, pad_width2)] + [(0,0)] * (img.ndim - 2)

    result = cv2.resize(cropped_img, (resize_width, resize_height))
    result = np.pad(result, pad_spec, mode='constant')
    assert result.shape[0] == height and result.shape[1] == width
    return result

def another_augment(img):
  # image_with_random_noise = random_noise(img)
  #play with the contrast
  v_min, v_max = np.percentile(img, (0.2, 99.8))
  better_contrast = exposure.rescale_intensity(img, in_range=(v_min, v_max))
  gamma_img=exposure.adjust_gamma(better_contrast, gamma=0.4, gain=0.9)
  horizontal_flip = gamma_img[:, ::-1]
  vertical_flip = horizontal_flip[::-1, :]
  img_final=cv2_clipped_zoom(vertical_flip,1.5)
  return img_final

def augmenation_combined(img):
  random_number_probability=random.uniform(0, 1)
  if random_number_probability>=0.5:
    img=augment_image(img,0.7,11,5,2)
    img=motion_blur(img)
  else:
    img=another_augment(img)
    img=motion_blur_horizontal_vertical(img)
  return img