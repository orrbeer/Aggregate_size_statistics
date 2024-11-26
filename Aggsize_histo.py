from scipy import ndimage
from skimage import measure
from skimage import filters
import scipy
import numpy as np
import skimage
import cv2
import numpy as np
from matplotlib_scalebar.scalebar import ScaleBar
from matplotlib_scalebar.scalebar import SI_LENGTH_RECIPROCAL
import matplotlib.pyplot as plt
#############################################################################
SMALL_SIZE = 14
MEDIUM_SIZE = 16
BIGGER_SIZE = 20
plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
plt.rcParams["font.family"] = "Times New Roman"
##############################################################################

folder =r'Images'
files = ["Sample 1_01", "Sample 2_03", "Sample 3_04", "Sample 4_02", "Sample 5_11"]
plt.close()
fig = plt.figure()#(figsize=(5, 8))

it = 0
for ff in files:
    img1 = cv2.imread("{}\\{}.tif".format(folder, ff),0)
    ret,thresh_img = cv2.threshold(img1,120,255,cv2.THRESH_BINARY)
    thresh_img = scipy.ndimage.median_filter(thresh_img, size=3)
    thresh_img = thresh_img[50:1400,:]
    blobs = thresh_img > 0.5*thresh_img.mean()
    all_labels = measure.label(blobs)
    blobs_labels = measure.label(blobs, background=0)
    # find contours in the binary image
    contours, hierarchy = cv2.findContours(thresh_img,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    histdata = np.zeros(len(contours))
    itt = 0
    for c in contours:
       # calculate moments for each contour
        M = cv2.moments(c)
       # calculate x,y coordinate of center
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
        else:
            cX, cY = 0, 0
        if np.count_nonzero(all_labels==blobs_labels[cY, cX]) < img1.shape[0]*img1.shape[1]/2:
            histdata[itt] = np.count_nonzero(all_labels==blobs_labels[cY, cX])*190 #  nm**2/pixel
            cv2.putText(thresh_img, "{:.2f}".format(histdata[itt]), 
            (cX - 25, cY - 25),cv2.FONT_HERSHEY_SIMPLEX, 0.6, (175, 175, 175), 2)
        itt+=1
    
    ### Main figure
    # print(img1.shape[0]*img1.shape[1]) #This is useful for debugging
    # print(max(histdata))               #This is useful for debugging
    # grid = plt.GridSpec(8, 6)#, hspace=0.2, wspace=0.2)
    # ax1 = fig.add_subplot(grid[:6,:6])#[:6,it*3:3+it*3])
    # ax1.imshow(img1[50:1400,:], cmap='gray', aspect = 'equal')
    # ax1.tick_params(labelbottom=False)
    # plt.axis('off')
    # scalebar =  ScaleBar(13.77e-3, 'um') # 1 pixel = 0.2 1/cm
    # plt.gca().add_artist(scalebar)
    # ax2 = fig.add_subplot(grid[6:,:6])#[6:,it*3:3+it*3])
    ax2 = fig.add_subplot(2,3,it+1)
    hx_range = [3*190, 2e5]
    histrange=(hx_range[0], hx_range[1])
    kwargs = dict(range=histrange, histtype='stepfilled', alpha=0.8, bins=20)# 
    ax2.hist(histdata,**kwargs)
    # ax2.set_yscale('log')
    ax2.axis([hx_range[0], hx_range[1],1,330])#([5*182.615, 1050*182.615,1,1e3])
    ax2.set_xlabel("Aggregate size (nm$^2$)")
    if it == 0:
        ax2.set_ylabel("Counts")
    # plt.yticks([1, 10, 100, 1e3])
    
    plt.text(2e4,65,"Average = {0:.2e} \nVariance = {1:.2e}".format( np.average(histdata),np.var(histdata)))
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))

    it+=1
fig.subplots_adjust(hspace=0,bottom=0.35)
# plt.tight_layout()
plt.show()
# plt.savefig("C:\\Users\\orrbeer\\Pictures\\{}".format(files[0]))