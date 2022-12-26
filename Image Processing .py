#!/usr/bin/env python
# coding: utf-8

# # Importing Libraries 

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import sys
from PIL import Image, ImageFilter
import warnings
get_ipython().run_line_magic('matplotlib', 'inline')
warnings.filterwarnings("ignore")


# # Image Segmentation 

# In[2]:


def image_segmentation(img_names):
    
    # kernels 
    #Line Detection
    horizontal = np.array([[-1,-1,-1], [2,2,2], [-1,-1,-1]])
    vertical = np.array([[-1,2,-1], [-1, 2, -1], [-1, 2, -1]])
    diagonal_45 = np.array([[-1,-1,2], [-1,2,-1], [2,-1,-1]])
    diagonal_45_min = np.array([[2,-1,-1], [-1, 2, -1], [-1,-1,2]])
    
    for i in img_names:
        img1 = cv2.imread(i)
        img  = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        
        
        line_horizontal = cv2.filter2D(src=img,ddepth=-1,kernel = horizontal)
        line_vertical = cv2.filter2D(src=img,ddepth=-1,kernel = vertical)
        line_diagonal_45 = cv2.filter2D(src=img,ddepth=-1,kernel = diagonal_45)
        line_diagonal_45_min = cv2.filter2D(src=img,ddepth=-1,kernel = diagonal_45_min)
        
        print('Image Segmentation On: '+i)
        
        plt.figure(figsize=(12,6))
    
        plt.subplot(231).set_title('Original Image'), plt.imshow(img1)
        plt.subplot(232).set_title('Grey Image'), plt.imshow(img, cmap="gray")
        plt.subplot(233).set_title('line_horizontal'), plt.imshow(line_horizontal)
        plt.subplot(234).set_title(' line_vertical'), plt.imshow(line_vertical)
        plt.subplot(235).set_title('line_diagonal_45'), plt.imshow(line_diagonal_45)
        plt.subplot(236).set_title('line_diagonal_45_min'), plt.imshow(line_diagonal_45_min)
        plt.show()
 


# In[3]:


img_name = ['img1.png', 'img3.jpg', 'img4.jpg', 'img5.jpg', 'img6.jpg']
image_segmentation(img_name)


# # Canny Edge Detection 

# In[4]:


originalImage = cv2.imread('img5.jpg')
img = cv2.cvtColor(originalImage, cv2.COLOR_BGR2RGB)

def CannyEdge(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    blur = cv2.GaussianBlur(gray, (3, 3), 0) # help of GussianBlur
    edge = cv2.Canny(blur, 10, 100)
    output=plt.imshow(edge)
   


# In[5]:


CannyEdge(img)


# # Harris Corner

# In[6]:


input_img = 'img6.jpg'

originalImage = cv2.imread(input_img)
image = cv2.imread(input_img)

gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
gray = np.float32(gray)

dst = cv2.cornerHarris(gray,2,3,0.04)

# img - Input image. It should be grayscale and float32 type.
# 2 - blockSize - It is the size of neighbourhood considered for corner detection
# 3 - ksize - Aperture parameter of the Sobel derivative used.
# 0.04 - Harris detector free parameter in the equation.

# dst = cv2.dilate(dst,None)
# print(dst)

image[dst>0.01*dst.max()]=[0,0,255]
# image[dst>0.001*dst.max()]=[0,0,255]

# print(image)
plt.figure(figsize=(15,15))

plt.subplot(121),
plt.imshow(originalImage),
plt.title("Original Image")

plt.subplot(122),
plt.imshow(image),
plt.title("Harris Cornered Image")


# In[7]:


def HarrisCorner(img):
    operatedImage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    operatedImage = np.float32(operatedImage)
    dest = cv2.cornerHarris(operatedImage, 2, 5, 0.07)
    dest = cv2.dilate(dest, None)
    img[dest > 0.01 * dest.max()]=[0, 0, 255]
    output=plt.imshow(dest, cmap='gray')
    harris_output=img
   


# In[8]:


HarrisCorner(originalImage)


# # Unsharp Masking

# In[9]:


#Scratch 

original_image = cv2.imread("img6.jpg")
img = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

gaussignBlur = cv2.GaussianBlur(img,(5,5),0)  # applying gaussian blur with kernel size 5 x 5 and standard deviation as 0

UnsharpMask = img - gaussignBlur * 2

plt.figure(figsize=(15,15))

plt.subplot(221),
plt.imshow(img),
plt.title("Original Image")

plt.subplot(222),
plt.imshow(gaussignBlur),
plt.title("Gaussian Filter")

plt.subplot(223),
plt.imshow(UnsharpMask),
plt.title("Unsharp Filter")


# In[10]:


#Using Library

image = cv2.imread('img6.jpg')
image = Image.fromarray(image.astype('uint8'))

new_image = image.filter(ImageFilter.UnsharpMask(radius=2, percent=150))


plt.figure(figsize=(15,15))

plt.subplot(221),
plt.imshow(image),
plt.title("Original Image")

plt.subplot(222),
plt.imshow(new_image),
plt.title("Unsharp Filter")


plt.show()


# # Sobel Operator

# In[11]:


originalImage = cv2.imread('img5.jpg')

img = cv2.cvtColor(originalImage, cv2.COLOR_BGR2RGB)

vertical_kernel = np.array([[-1, 0, 1],
                            [-2, 0, 2],
                            [-1, 0, 1]])

horizontal_kernel = np.array([[1, 2, 1],
                              [0, 0, 0],
                              [-1, -2, -1]])

vertical_output = cv2.filter2D(src=img,ddepth=-1,kernel = vertical_kernel)
horizontal_output = cv2.filter2D(src=img,ddepth=-1,kernel = horizontal_kernel)  


plt.figure(figsize=(15,15))

plt.subplot(321),
plt.imshow(img),
plt.title("Original Image")

plt.subplot(322),
plt.imshow(horizontal_output),
plt.title("Horizontal Features")

plt.subplot(323),
plt.imshow(vertical_output),
plt.title("Vertical Features")


# In[12]:


# Remove the negative values taking the absolute

horizontal_output = np.absolute(horizontal_output)
vertical_output = np.absolute(vertical_output)

sobel = horizontal_output + vertical_output

plt.figure(figsize=(25,15))


plt.subplot(2,2,1),plt.imshow(img,cmap = 'gray')
plt.title('Original'), plt.xticks([]), plt.yticks([])
plt.subplot(2,2,2),plt.imshow(horizontal_output,cmap = 'gray')
plt.title('horizontal features - sobel-x'), plt.xticks([]), plt.yticks([])
plt.subplot(2,2,3),plt.imshow(vertical_output,cmap = 'gray')
plt.title('vertical features - sobel-y'), plt.xticks([]), plt.yticks([])
plt.subplot(2,2,4),plt.imshow(sobel,cmap = 'gray')
plt.title('vertical + vertical features - sobel= x+y'), plt.xticks([]), plt.yticks([])

plt.show()


# In[13]:


def sobel(img_names):
    
    # kernels 
    sobelx = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
    sobely = np.array ([[1,2,1],[0,0,0],[-1,-2,-1]])
    sobel = sobelx + sobely
    
    
    for i in img_names:
        img2 = cv2.imread(i)
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        
        
        v_feature = cv2.filter2D(img2, -1, kernel = sobelx)
        h_feature = cv2.filter2D(img2, -1, kernel = sobely)
        c_feature = cv2.filter2D(img2, -1, kernel = sobel)
        c_feature2 = v_feature + h_feature
        
        print('sobel on'+i)
        
        plt.figure(figsize=(12,6))
        
      
        plt.subplot(231).set_title('original'), plt.imshow(img2, cmap = 'gray')
        plt.subplot(232).set_title('vertical features'), plt.imshow(v_feature, cmap = 'gray')
        plt.subplot(233).set_title(' horizontal features'), plt.imshow(h_feature, cmap = 'gray')
        plt.subplot(234).set_title('both features (added kernel)'), plt.imshow(c_feature, cmap = 'gray')
        plt.subplot(235).set_title('both features (output images added)'), plt.imshow(c_feature2, cmap = 'gray')
        plt.show()
    
img_name2 = ['img6.jpg', 'img5.jpg']
sobel(img_name2)


# # Laplacian 

# In[14]:


originalImage = cv2.imread('img5.jpg',0)
img = cv2.cvtColor(originalImage, cv2.COLOR_BGR2RGB)

kernel_1 = np.array([[0, 1, 0],
                    [1, -4, 1],
                    [0, 1, 0]])

kernel_2 = np.array([[0, -1, 0],
                    [-1, 4, -1],
                    [0, -1, 0]])


kernel_3 = np.array([[0, 1, 0],
                    [1, -8, 1],
                    [0, 1, 0]])


kernel_4 = np.array([[0, -1, 0],
                    [-1, 8, -1],
                    [0, -1, 0]])


sharpen_kernel = np.array([[0, -1, 0],
                          [-1, 5, -1],
                          [0, -1, 0]])

laplacian_1 = cv2.filter2D(src=img,ddepth=-1,kernel = kernel_1)  
laplacian_2 = cv2.filter2D(src=img,ddepth=-1,kernel = kernel_2)
laplacian_3 = cv2.filter2D(src=img,ddepth=-1,kernel = kernel_3)
laplacian_4 = cv2.filter2D(src=img,ddepth=-1,kernel = kernel_4)
laplacian_5 = cv2.filter2D(src=img,ddepth=-1,kernel = sharpen_kernel)

plt.figure(figsize=(15,15))

plt.subplot(421),
plt.imshow(originalImage),
plt.title("Original Image")

plt.subplot(422),
plt.imshow(img),
plt.title("GreyScale Image")

plt.subplot(423),
plt.imshow(laplacian_1),
plt.title("Laplacian Filter 3x3")

plt.subplot(424),
plt.imshow(laplacian_2),
plt.title("Laplacian Filter 3x3")

plt.subplot(425),
plt.imshow(laplacian_3),
plt.title("Laplacian Filter 3x3")


plt.subplot(426),
plt.imshow(laplacian_4),
plt.title("Laplacian Filter 3x3")

plt.subplot(427),
plt.imshow(laplacian_4),
plt.title("Laplacian Filter 3x3")


# # Gaussian 

# In[15]:


#  𝐺𝑎𝑢𝑠𝑠𝑖𝑎𝑛𝐹𝑖𝑙𝑡𝑒𝑟   #########################################################

img=cv2.imread('img6.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

#Gaussian 3x3 matrix
#k = np.array([[1, 2, 1], [2, 4, 2],[1, 2, 1]], dtype= int)
#Gaussian 5x5 matrix
# k = np.array([[1, 4, 7, 4, 1], [4, 16, 26, 16, 4], [7, 26, 41, 26, 7],[4, 16, 26, 16, 4],[1, 4, 7, 4, 1]])
#Gaussian 7x7 matrix
#mask = np.array([[1, 4, 7, 9, 7, 4, 1], [4, 16, 26, 36, 26, 16, 4], [7, 26, 41, 55, 41, 26, 7],[4, 16, 26, 36, 26, 16, 4],[1, 4, 7, 9, 7, 4, 1]])

k = np.array([[1, 2, 1], [2, 4, 2],[1, 2, 1]], dtype= int)
print(k)
kernel = k/16
Gaussian_blur = cv2.filter2D(img,0,kernel)
plt.figure(figsize=(5,5))
plt.imshow(Gaussian_blur)
plt.title('Gaussian blur')


# In[16]:


#3*3 Kernel
mask = np.array([[1, 2, 1], 
                 [2, 4, 2], 
                 [1, 2, 1]])

weight = mask.sum()

kernel = mask/weight

#5*5 Kernel
mask = np.array([[1, 4, 7, 4, 1], 
                 [4, 16, 26, 16, 4], 
                 [7, 26, 41, 26, 7],
                 [4, 16, 26, 16, 4],
                 [1, 4, 7, 4, 1]])

weight = mask.sum()

kernel_1 = mask/weight

#7*7 Kernel
mask = np.array([[1, 4, 7, 9, 7, 4, 1], 
                 [4, 16, 26, 36, 26, 16, 4], 
                 [7, 26, 41, 55, 41, 26, 7],
                 [4, 16, 26, 36, 26, 16, 4],
                 [1, 4, 7, 9, 7, 4, 1]])

weight = mask.sum()

kernel_2 = mask/weight

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

gaussignBlur = cv2.GaussianBlur(img,kernel.shape,0)  # applying gaussian blur with kernel size 5 x 5 and standard deviation as 0
gaussignBlur_1 = cv2.GaussianBlur(img,kernel_1.shape,0)  # applying gaussian blur with kernel size 5 x 5 and standard deviation as 0
gaussignBlur_2 = cv2.GaussianBlur(img,kernel_2.shape,0)  # applying gaussian blur with kernel size 7 x 7 and standard deviation as 0

plt.figure(figsize=(15,15))

plt.subplot(221),
plt.imshow(img),
plt.title("Original Image")

plt.subplot(222),
plt.imshow(gaussignBlur),
plt.title("Gaussian Filter 3x3")

plt.subplot(223),
plt.imshow(gaussignBlur_1),
plt.title("Gaussian Filter 5x5")

plt.subplot(224),
plt.imshow(gaussignBlur_2),
plt.title("Gaussian Filter 7x7")


# In[17]:


img = cv2.imread("img6.jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

gaussignBlur = cv2.GaussianBlur(img,(5,5),0)  # applying gaussian blur with kernel size 5 x 5 and standard deviation as 0

plt.figure(figsize=(15,15))

plt.subplot(221),
plt.imshow(img),
plt.title("Original Image")

plt.subplot(222),
plt.imshow(gaussignBlur),
plt.title("Gaussian Filter")


# # Bilateral Filter 

# In[18]:


img = cv2.imread('img6.jpg')

output_1 = cv2.bilateralFilter(img, 5,  50, 50) 
output_2 = cv2.bilateralFilter(img, 7,  60, 60) 
output_3 = cv2.bilateralFilter(img, 10, 70, 70) 
output_4 = cv2.bilateralFilter(img, 12, 75, 75) 
output_5 = cv2.bilateralFilter(img, 15, 85, 80) 

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


plt.figure(figsize=(15,15))

plt.subplot(231),
plt.imshow(img),
plt.title("Original Image")

plt.subplot(232),
plt.imshow(output_1),
plt.title("Output - 1")

plt.subplot(233),
plt.imshow(output_2),
plt.title("Output - 2")

plt.subplot(234),
plt.imshow(output_3),
plt.title("Output - 3")

plt.subplot(235),
plt.imshow(output_4),
plt.title("Output - 4")

plt.subplot(236),
plt.imshow(output_5),
plt.title("Output - 5")


# # Minimum Maximum

# In[19]:


#From Scratch
def min_func(img, cmap='gray'):
    height = img.shape[1]
    width = img.shape[0]
    # print(img.shape)

    new_img = img.copy()

    for k in range(3):
        for i in range(1,width-1):
            for j in range(1,height-1):
                new_img[i, j, k] = np.min(
                    [img[i-1, j-1, k],
                    img[i, j-1, k],
                    img[i+1, j-1, k],
                    img[i-1, j, k],
                    img[i, j, k],
                    img[i+1, j, k],
                    img[i-1, j+1, k],
                    img[i, j+1, k],
                    img[i+1, j+1, k],]
                )

    plt.figure(figsize=(12, 8))
    plt.subplot(121)
    plt.imshow(img, cmap = cmap)
    plt.subplot(121).set_title("Real")

    plt.subplot(122)
    plt.imshow(new_img, cmap = cmap)
    plt.subplot(122).set_title("Minimum")
    
img = cv2.imread('img6.jpg')
min_func(img)


# In[20]:


#From Scratch
def max_func(img, cmap='gray'):
    height = img.shape[1]
    width = img.shape[0]
    # print(img.shape)

    new_img = img.copy()

    for k in range(3):
        for i in range(1,width-1):
            for j in range(1,height-1):
                new_img[i, j, k] = np.max(
                    [img[i-1, j-1, k],
                    img[i, j-1, k],
                    img[i+1, j-1, k],
                    img[i-1, j, k],
                    img[i, j, k],
                    img[i+1, j, k],
                    img[i-1, j+1, k],
                    img[i, j+1, k],
                    img[i+1, j+1, k],]
                )

    plt.figure(figsize=(12, 8))
    plt.subplot(121)
    plt.imshow(img, cmap = cmap)
    plt.subplot(121).set_title("Real")

    plt.subplot(122)
    plt.imshow(new_img, cmap = cmap)
    plt.subplot(122).set_title("Maximum")
    
img = cv2.imread('img6.jpg')
max_func(img)


# In[21]:


def min_max():
    img = Image.open('img6.jpg')

    max_img = img.filter(ImageFilter.MaxFilter())
    min_img = img.filter(ImageFilter.MinFilter())

    plt.figure(figsize=(12,6))
    plt.subplot(231).set_title('original'), plt.imshow(img, cmap = 'gray')
    plt.subplot(232).set_title('max'), plt.imshow(max_img, cmap = 'gray')
    plt.subplot(233).set_title(' min'), plt.imshow(min_img, cmap = 'gray')
    plt.show()
    
min_max()


# # Median Filter 

# In[22]:


#From Scratch
image = cv2.imread('img6.jpg')
new_img = image.copy()

def med_fil(img):
    height = img.shape[1]
    width = img.shape[0]
    # print(img.shape)

    new_img = img.copy()
    for k in range(3):
        for i in range(1,width-1):
            for j in range(1,height-1):
                new_img[i, j, k] = np.sort(
                    [img[i-1, j-1, k],
                    img[i, j-1, k],
                    img[i+1, j-1, k],
                    img[i-1, j, k],
                    img[i, j, k],
                    img[i+1, j, k],
                    img[i-1, j+1, k],
                    img[i, j+1, k],
                    img[i+1, j+1, k],]
                )[4]
    return new_img

med_fil(image)
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 6))

ax[0].imshow(image)
ax[0].set_title("Original".title(), fontsize=14)

ax[1].imshow(new_img)
ax[1].set_title("median filter".title(), fontsize=14)


# In[23]:


#Using Library

img = cv2.imread('img6.jpg')
img.shape
median = cv2.medianBlur(img,5) # 5 window size in image => 5*5
plt.figure(figsize=(15,15))

plt.subplot(221),
plt.imshow(img),
plt.title("Original Image")

plt.subplot(222),
plt.imshow(median),
plt.title("Median Filtered Image")


# # Mean or Average 

# In[24]:


#From Scratch

img = cv2.imread('img6.jpg')
resize = cv2.resize(img,(0, 0), fx = 0.05, fy = 0.05)
print(resize.shape)
print(resize)
print("R ==> \n",resize[:,:,0])
print("G ==> \n",resize[:,:,1])
print("B ==> \n",resize[:,:,2])

resize = cv2.cvtColor(resize, cv2.COLOR_BGR2RGB)

print(resize.shape)
m, n, o = resize.shape
print(m)
print(n)
print(o)

plt.figure(figsize=(10,10))

plt.subplot(221),
plt.imshow(img),
plt.title("Original Image")

plt.subplot(222),
plt.imshow(resize),
plt.title("Resized Filter")

mask = np.ones([3, 3], dtype = int)
print("Filter ==> \n",mask)
mask = mask / 9
print("Mean Filter ==> \n",mask)

img_new = np.zeros([m, n, o])
# img_new

for k in range(3):
    for i in range(0,m):
        for j in range(0,n):
            print("row : ",i,"Column : ",j, "channel",k,"ImageValue",resize[i,j,k])
            
for i in range(1,m-1):
    for j in range(1,n-1):
        print("i : ", i, " : j : ",j)
        temp_img =  resize[i-1,j-1] * mask[0,0] + resize[i-1,j] * mask[0,1] + resize[i-1,j+1] * mask[0,2] + resize[i,j-1] * mask[1,0] + resize[i,j] * mask[1,1] + resize[i,j+1] * mask[1,2] +  resize[i + 1, j-1]*mask[2, 0] + resize[i + 1, j]* mask[2, 1] + resize[i + 1, j + 1]*mask[2, 2]                  
#         print(temp_img)
        img_new[i, j]= temp_img
        
img_new = img_new.astype(np.uint8)
print("R ==> \n",img_new[:,:,0])
print("G ==> \n",img_new[:,:,1])
print("B ==> \n",img_new[:,:,2])

plt.figure(figsize=(15,15))

plt.subplot(221),
plt.imshow(resize),
plt.title("Resized Image")

plt.subplot(222),
plt.imshow(img_new),
plt.title("Average Filter")


# In[25]:


#From Libraries 

blur = cv2.blur(resize,mask.shape)
plt.figure(figsize=(15,15))

plt.subplot(221),
plt.imshow(resize),
plt.title("Resized Image")

plt.subplot(222),
plt.imshow(blur),
plt.title("Average Filter")

plt.subplot(223),
plt.imshow(img),
plt.title("Original Image")

blur_full_img = cv2.blur(img,mask.shape)

plt.subplot(224),
plt.imshow(blur_full_img),
plt.title("Average Filter on Full Image")

blur = cv2.boxFilter(resize, -1, mask.shape)
plt.figure(figsize=(15,15))

plt.subplot(221),
plt.imshow(resize),
plt.title("Resized Image")

plt.subplot(222),
plt.imshow(blur),
plt.title("Average Filter")

plt.subplot(223),
plt.imshow(img),
plt.title("Original Image")

blur_full_img = cv2.boxFilter(img, -1, mask.shape)

plt.subplot(224),
plt.imshow(blur_full_img),
plt.title("Average Filter on Full Image")


# # Weighted Average

# In[26]:


img = cv2.imread('img6.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
print(img.shape)
m, n, o = img.shape

mask = np.matrix([[1, 2, 1], [2, 4, 2],[1, 2, 1]],dtype = int)
mask

weight = mask.sum()
weight

kernel = mask/weight
print(kernel)

# Convolve the 3X3 mask over the image
img_new = np.zeros([m, n, o])

for i in range(1, m-1):
    for j in range(1, n-1):
        temp = img[i-1, j-1]*kernel[0, 0]+img[i-1, j]*kernel[0, 1]+img[i-1, j + 1]*kernel[0, 2]+img[i, j-1]*kernel[1, 0]+ img[i, j]*kernel[1, 1]+img[i, j + 1]*kernel[1, 2]+img[i + 1, j-1]*kernel[2, 0]+img[i + 1, j]*kernel[2, 1]+img[i + 1, j + 1]*kernel[2, 2]

        img_new[i, j]= temp

img_new = img_new.astype(np.uint8)

plt.figure(figsize=(15,15))

plt.subplot(221),
plt.imshow(img),
plt.title("Original Image")

plt.subplot(222),
plt.imshow(img_new),
plt.title("Weighted Average Filter")


# In[27]:


#Using Libraries

# Ddepth – Depth of the output image [ -1 will give the output image depth as same as the input image]
output = cv2.filter2D(img,-1,kernel) 

plt.figure(figsize=(15,15))

plt.subplot(221),
plt.imshow(img),
plt.title("Original Image")

plt.subplot(222),
plt.imshow(output),
plt.title("weighted average")


# # Smoothing

# In[28]:


# Gaussian Filter Using openCV

img = cv2.imread("img6.jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

gaussignBlur = cv2.GaussianBlur(img,(5,5),0)  # applying gaussian blur with kernel size 5 x 5 and standard deviation as 0

plt.figure(figsize=(15,15))

plt.subplot(221),
plt.imshow(img),
plt.title("Original Image")

plt.subplot(222),
plt.imshow(gaussignBlur),
plt.title("Gaussian Filter")


# In[29]:


# Opening the image
image = Image.open(r"img6.jpg")

# Blurring image by sending the ImageFilter.
# GaussianBlur predefined kernel argument
image = image.filter(ImageFilter.GaussianBlur)

# image.show()

plt.figure(figsize=(15,15))

plt.subplot(221),
plt.imshow(img),
plt.title("Original Image")

plt.subplot(222),
plt.imshow(image),
plt.title("Gaussian Filter")


# In[30]:


# To blur specific portion of image

image = Image.open(r"img6.jpg")
original_image = Image.open(r"img6.jpg")

# # Cropping the image 
smol_image = image.crop((0, 0, 150, 150))
  
# Blurring on the cropped image
blurred_image = smol_image.filter(ImageFilter.GaussianBlur)
  
image.paste(blurred_image, (0,0))
# image.save('output.png')
# image.show()

plt.figure(figsize=(15,15))

plt.subplot(131),
plt.imshow(original_image),
plt.title("Original Image")

plt.subplot(132),
plt.imshow(blurred_image),
plt.title("Gaussian Blur Portion")

plt.subplot(133),
plt.imshow(image),
plt.title("Specific Portion/Regiion using Gaussian Blur")


# In[31]:


#𝑀𝑒𝑑𝑖𝑎𝑛𝐹𝑖𝑙𝑡𝑒𝑟(𝑆𝑚𝑜𝑜𝑡ℎ𝑖𝑛𝑔𝑁𝑜𝑛−𝐿𝑖𝑛𝑒𝑎𝑟𝐹𝑖𝑙𝑡𝑒𝑟𝑠) 

img = cv2.imread("img6.jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

gaussignBlur = cv2.medianBlur(img,5) # 5 means 50% noise added 5 => 5*5 == ksize×ksize

plt.figure(figsize=(15,15))

plt.subplot(221),
plt.imshow(img),
plt.title("Original Image")

plt.subplot(222),
plt.imshow(gaussignBlur),
plt.title("median blur")


# In[32]:


# Read the image
img = cv2.imread('img6.jpg', 0)

m, n = img.shape

# Traverse the image. For every 3X3 area,
# find the median of the pixels and
# replace the center pixel by the median
img_new1 = np.zeros([m, n])
print(img_new1.shape)
print(img_new1)


for i in range(1, m-1):
    for j in range(1, n-1):
        temp = [img[i-1, j-1],
            img[i-1, j],
            img[i-1, j + 1],
            img[i, j-1],
            img[i, j],
            img[i, j + 1],
            img[i + 1, j-1],
            img[i + 1, j],
            img[i + 1, j + 1]]

        temp = sorted(temp)
        img_new1[i, j]= temp[4]

img_new1 = img_new1.astype(np.uint8)
# cv2.imwrite('new_median_filtered.png', img_new1)

plt.figure(figsize=(15,15))

plt.subplot(221),
plt.imshow(img),
plt.title("Original Image")

plt.subplot(222),
plt.imshow(gaussignBlur),
plt.title("Median Filter")


# In[33]:


#Average filtering using Low Pass kernal

img = cv2.imread('img6.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
print(img.shape)
m, n, o = img.shape

# Create Averaging filter(3, 3) mask
mask = np.ones([3, 3], dtype = int)
print(mask)
mask = mask / 9
print(mask)

# Convolve the 3X3 mask over the image
img_new = np.zeros([m, n, o])

for i in range(1, m-1):
    for j in range(1, n-1):
        temp = img[i-1, j-1]*mask[0, 0]+img[i-1, j]*mask[0, 1]+img[i-1, j + 1]*mask[0, 2]+img[i, j-1]*mask[1, 0]+ img[i, j]*mask[1, 1]+img[i, j + 1]*mask[1, 2]+img[i + 1, j-1]*mask[2, 0]+img[i + 1, j]*mask[2, 1]+img[i + 1, j + 1]*mask[2, 2]

        img_new[i, j]= temp

img_new = img_new.astype(np.uint8)
# cv2.imwrite('blurred.tif', img_new)

plt.figure(figsize=(15,15))

plt.subplot(221),
plt.imshow(img),
plt.title("Original Image")

plt.subplot(222),
plt.imshow(img_new),
plt.title("Average Filter")


# In[34]:


# Average filtering using High Pass kernal

img = cv2.imread('img6.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
print(img.shape)
m, n, o = img.shape

mask = np.matrix([[-1, -1, -1], [-1, 8, -1],[-1, -1, -1]],dtype = int)
print(mask)
mask = mask / 9
print(mask)

# Convolve the 3X3 mask over the image
img_new = np.zeros([m, n, o])

for i in range(1, m-1):
    for j in range(1, n-1):
        temp = img[i-1, j-1]*mask[0, 0]+img[i-1, j]*mask[0, 1]+img[i-1, j + 1]*mask[0, 2]+img[i, j-1]*mask[1, 0]+ img[i, j]*mask[1, 1]+img[i, j + 1]*mask[1, 2]+img[i + 1, j-1]*mask[2, 0]+img[i + 1, j]*mask[2, 1]+img[i + 1, j + 1]*mask[2, 2]

        img_new[i, j]= temp

img_new = img_new.astype(np.uint8)

plt.figure(figsize=(15,15))

plt.subplot(221),
plt.imshow(img),
plt.title("Original Image")

plt.subplot(222),
plt.imshow(img_new),
plt.title("Average Filter")


# # Grey Image Level - Quantization

# In[35]:


img = cv2.imread("img6.jpg", 1)

#we need to transform this in order that Matplotlib reads it correctly
fix_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.imshow(fix_img)
# fix_img

#Let's extract the three channels
R, G, B = fix_img[:,:,0], fix_img[:,:,1],fix_img[:,:,2]

grayscale_weighted_img = 0.299 * R + 0.587 * G + 0.114 * B
# print(grayscale_weighted_img.shape)

plt.imshow(grayscale_weighted_img, cmap='gray')
plt.savefig('image_average_method.png')


# In[36]:


im_gray = cv2.imread('image_average_method.png', cv2.IMREAD_GRAYSCALE)

thresh = 200
img_1 = cv2.threshold(im_gray, thresh, 100, cv2.THRESH_BINARY)[1]
# cv2.imwrite('img_1.png', im_bw)

thresh = 100
img_2 = cv2.threshold(im_gray, thresh, 150, cv2.THRESH_BINARY)[1]
# cv2.imwrite('img_1.png', im_bw)

thresh = 200
img_3 = cv2.threshold(im_gray, thresh, 256, cv2.THRESH_BINARY)[1]
# cv2.imwrite('img_1.png', im_bw)

thresh = 50
img_4 = cv2.threshold(im_gray, thresh, 500, cv2.THRESH_BINARY)[1]
# cv2.imwrite('img_1.png', im_bw)

plt.subplot(221),plt.imshow(img_1, cmap="gray"),plt.title("Grey Image - 1")
plt.xticks([]), plt.yticks([])

plt.subplot(222),plt.imshow(img_2, cmap="gray"),plt.title("Grey Image - 2")
plt.xticks([]), plt.yticks([])

plt.subplot(223),plt.imshow(img_3, cmap="gray"),plt.title("Grey Image - 3")
plt.xticks([]), plt.yticks([])

plt.subplot(224),plt.imshow(img_4, cmap="gray"),plt.title("Grey Image - 4")
plt.xticks([]), plt.yticks([])

plt.show()


# # Morphology

# In[37]:


img = cv2.imread("img6.jpg")

img1 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

grayimage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
grayimage

kernel = np.ones((5,5),np.uint8)
kernel

erosion = cv2.erode(img1, kernel, iterations=1)
dilation = cv2.dilate(img1, kernel, iterations=1)
opening = cv2.morphologyEx(img1, cv2.MORPH_OPEN, kernel)
closing = cv2.morphologyEx(img1, cv2.MORPH_CLOSE, kernel)
gradiant = cv2.morphologyEx(img1, cv2.MORPH_GRADIENT, kernel)
tophat  = cv2.morphologyEx(img1, cv2.MORPH_TOPHAT, kernel)
blackhat = cv2.morphologyEx(img1, cv2.MORPH_BLACKHAT, kernel)

plt.figure(figsize=(12,6))

plt.subplot(241),plt.imshow(img1),plt.title("Original Image")
plt.xticks([]), plt.yticks([])

plt.subplot(242),plt.imshow(grayimage, cmap="gray"),plt.title("Gray Image")
plt.xticks([]), plt.yticks([])

plt.subplot(243),plt.imshow(erosion, cmap="gray"),plt.title("Erosion")
plt.xticks([]), plt.yticks([])

plt.subplot(244),plt.imshow(dilation, cmap="gray"),plt.title("Dilation")
plt.xticks([]), plt.yticks([])

plt.subplot(245),plt.imshow(opening),plt.title("Opening")
plt.xticks([]), plt.yticks([])

plt.subplot(246),plt.imshow(closing),plt.title("Closing")
plt.xticks([]), plt.yticks([])

plt.subplot(247),plt.imshow(tophat, cmap="gray"),plt.title("Tophat")
plt.xticks([]), plt.yticks([])

plt.subplot(248),plt.imshow(blackhat, cmap="gray"),plt.title("Blackhat")
plt.xticks([]), plt.yticks([])

plt.show()


# In[38]:


# Morphology 


# Read function
def readImage(img):
    return cv2.imread(img)



# Kernel Function
def kernelFunc(size):
    return np.ones((size, size), np.uint8)


# Orignal Image
def orignalImage(img):
    rgb_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return rgb_image


# Erosion Matrix Function
def matrixEro(img, size_of_kernel=3, no_of_iterations=5):
    orignal_image = orignalImage(img)
    erosion = cv2.erode(orignal_image, kernel=kernelFunc(size_of_kernel), iterations=no_of_iterations)
    return erosion


# Dilation Matrix Function
def matrixDil(img, size_of_kernel=3, no_of_iterations=5):
    orignal_image = orignalImage(img)
    dilation = cv2.dilate(orignal_image, kernel=kernelFunc(size_of_kernel), iterations=no_of_iterations)
    return dilation


# Gradient Matrix Function
def matrixGrad(img, size_of_kernel=3, no_of_iterations=5):
    gradient = matrixEro(img, size_of_kernel, no_of_iterations) - matrixDil(img, size_of_kernel, no_of_iterations)
    return gradient


# Opening  Matrix Function
def matrixOpening(img, size_of_kernel=3, no_of_iterations=5):
    erosion_image = matrixEro(img, size_of_kernel, no_of_iterations)
    opening_image = cv2.dilate(erosion_image, kernel=kernelFunc(size_of_kernel), iterations=no_of_iterations)
    return opening_image


# Closing Matrix Function
def matrixClosing(img, size_of_kernel=3, no_of_iterations=5):
    orignal_image = orignalImage(img)
    dilation_image = matrixDil(img, size_of_kernel, no_of_iterations)
    closing_image = cv2.erode(dilation_image, kernel=kernelFunc(size_of_kernel), iterations=no_of_iterations)
    return closing_image


# Black Hat Matrix Function
def matrixBlackHat(img, size_of_kernel=3, no_of_iterations=5):
    orignal_image = orignalImage(img)
    closing_img = matrixClosing(img, size_of_kernel, no_of_iterations)
    black_hat = closing_img - orignal_image
    return black_hat


# Top Hat Matrix Function
def matrixTopHat(img, no_of_iterations=5, size_of_kernel=3, no_tophat_iterations=6):
    orignal_image = orignalImage(img)
    opening_image = matrixOpening(img, no_of_iterations, size_of_kernel)
    for i in range(0, no_tophat_iterations):
        top_hat = orignal_image - opening_image
        top_hat
    return top_hat


# Plot All in one Function
def plotAll(img, size_of_kernel=3, no_of_iterations=5, no_tophat_iterations=6):
    orig_img = orignalImage(img)
    ero_img_1 = matrixEro(img, size_of_kernel, no_of_iterations)
    dil_img_2 = matrixDil(img, size_of_kernel, no_of_iterations)
    gra_img_3 = matrixGrad(img, size_of_kernel, no_of_iterations)
    ope_img_4 = matrixOpening(img, size_of_kernel, no_of_iterations)
    clo_img_5 = matrixClosing(img, size_of_kernel, no_of_iterations)
    bla_img_6 = matrixBlackHat(img, size_of_kernel, no_of_iterations)
    top_img_7 = matrixTopHat(img, no_of_iterations, size_of_kernel, no_tophat_iterations)
    imglist = [orig_img, ero_img_1, dil_img_2, gra_img_3, ope_img_4, clo_img_5, bla_img_6, top_img_7]

    fig, ax = plt.subplots(4, 2, figsize=(8, 8), facecolor="w", tight_layout=True)
    plt.subplots_adjust(left=0.01,
                        bottom=0.01,
                        right=0.9,
                        top=0.9,
                        wspace=0.4,
                        hspace=0.4)

    ax[0, 0].imshow(imglist[0])
    ax[0, 0].set_title("Orignal Image")
    ax[0, 0].set_xticklabels(labels="", rotation=90, c="w", fontweight="bold", fontsize=12)
    ax[0, 0].set_yticklabels(labels="", rotation=360, c="w", fontweight="bold", fontsize=12)

    ax[0, 1].imshow(imglist[1])
    ax[0, 1].set_title("Erosion Image")
    ax[0, 1].set_xticklabels(labels="", rotation=90, c="w", fontweight="bold", fontsize=12)
    ax[0, 1].set_yticklabels(labels="", rotation=360, c="w", fontweight="bold", fontsize=12)

    ax[1, 0].imshow(imglist[2])
    ax[1, 0].set_title("Dilation Image")
    ax[1, 0].set_xticklabels(labels="", rotation=90, c="w", fontweight="bold", fontsize=12)
    ax[1, 0].set_yticklabels(labels="", rotation=360, c="w", fontweight="bold", fontsize=12)

    ax[1, 1].imshow(imglist[3])
    ax[1, 1].set_title("Gradient Image")
    ax[1, 1].set_xticklabels(labels="", rotation=90, c="w", fontweight="bold", fontsize=12)
    ax[1, 1].set_yticklabels(labels="", rotation=360, c="w", fontweight="bold", fontsize=12)

    ax[2, 0].imshow(imglist[4])
    ax[2, 0].set_title("Opening Image")
    ax[2, 0].set_xticklabels(labels="", rotation=90, c="w", fontweight="bold", fontsize=12)
    ax[2, 0].set_yticklabels(labels="", rotation=360, c="w", fontweight="bold", fontsize=12)

    ax[2, 1].imshow(imglist[5])
    ax[2, 1].set_xticklabels(labels="", rotation=90, c="w", fontweight="bold", fontsize=12)
    ax[2, 1].set_yticklabels(labels="", rotation=360, c="w", fontweight="bold", fontsize=12)
    ax[2, 1].set_title("Closing Image")

    ax[3, 0].imshow(imglist[6])
    ax[3, 0].set_xticklabels(labels="", rotation=90, c="w", fontweight="bold", fontsize=12)
    ax[3, 0].set_yticklabels(labels="", rotation=360, c="w", fontweight="bold", fontsize=12)
    ax[3, 0].set_title("Black Hat Image")

    ax[3, 1].imshow(imglist[7])
    ax[3, 1].set_xticklabels(labels="", rotation=90, c="w", fontweight="bold", fontsize=12)
    ax[3, 1].set_yticklabels(labels="", rotation=360, c="w", fontweight="bold", fontsize=12)
    ax[3, 1].set_title("Top Hat Image")

    plt.savefig("abc.png", dpi=1000, bbox_inches='tight')
    img_rtr = cv2.imread("abc.png")
    img_rtr = cv2.cvtColor(img_rtr, cv2.COLOR_BGR2RGB)
    #os.remove("abc.png")

    return img_rtr

img = cv2.imread("img6.jpg")
plotAll(img, size_of_kernel=3, no_of_iterations=5, no_tophat_iterations=6)


# In[39]:


# img=cv2.imread('img6.jpg')

# # opening
# def opening(img):
#     eroded = cv2.erode(img, (3, 3))
#     # erode
#     dilated = cv2.dilate(eroded, (3, 3))
#     # dilate
#     return dilated

# plt.imshow(opening(image))

# # closing
# def closing(img):
#     dilated = cv2.dilate(img, (3, 3))
#     # erode
#     eroded = cv2.erode(dilated, (3, 3))
#     # dilate
#     return eroded

# plt.imshow(closing(image))

# # gradient
# eroded = cv2.erode(image, (3, 3))
# dilated = cv2.dilate(image, (3, 3))

# gradient = dilated - eroded

# plt.imshow(gradient)

# # tophat
# tophat = image - opening(image)

# plt.imshow(tophat)

# # blackhat
# blackhat = closing(image) - image

# plt.imshow(blackhat)


# # RGB to grey and Negative 

# In[40]:


img = cv2.imread("img6.jpg", 1)

#we need to transform this in order that Matplotlib reads it correctly
fix_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.imshow(fix_img)
# fix_img


# In[41]:


#Average Method

#Let's extract the three channels
R, G, B = fix_img[:,:,0], fix_img[:,:,1],fix_img[:,:,2]

grayscale_average_img = np.mean(fix_img, axis=2) # axis=2 means 2 dimension - 2D

plt.imshow(grayscale_average_img, cmap='gray')
plt.savefig('image_average_method.png')


# In[42]:


# Weighted average

grayscale_weighted_img = 0.299 * R + 0.587 * G + 0.114 * B

plt.imshow(grayscale_average_img, cmap='gray')


# In[43]:


# Using OpenCV

grayimage = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
nagativeimg = 255 - fix_img

plt.subplot(231),plt.imshow(fix_img),plt.title("RGB Image")
plt.xticks([]), plt.yticks([])

plt.subplot(232),plt.imshow(grayimage, cmap="gray"),plt.title("Gray Image")
plt.xticks([]), plt.yticks([])

plt.subplot(233),plt.imshow(nagativeimg),plt.title("Negative Image")
plt.xticks([]), plt.yticks([])

plt.show()

# cv2.imshow('image',img)
# cv2.imshow('image1',grayimage)
# cv2.imshow('image2',nagativeimg)

# cv2.waitKey(0)
# cv2.destroyAllWindows()


# # Robert Cross

# In[3]:


img3 = cv2.imread("img6.jpg")


roberts_cross_v = np.array( [[1, 0 ],
                             [0,-1 ]] )
  
roberts_cross_h = np.array( [[ 0, 1 ],
                             [ -1, 0 ]] )


vertical = cv2.filter2D( img3,-1,roberts_cross_v )
horizontal = cv2.filter2D( img3,-1, roberts_cross_h )
  
edged_img = ( np.square(horizontal) + np.square(vertical))

plt.imshow(edged_img)


# In[ ]:




