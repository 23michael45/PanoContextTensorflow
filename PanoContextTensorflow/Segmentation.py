import CoordsTransform
import numpy as np
from scipy import signal
from scipy.ndimage.filters import gaussian_filter
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.misc import imread, imsave, imresize
from PythonCallC import segmentGraphEdge
from sklearn.neighbors import NearestNeighbors
import PIL.Image
def gbPanoSegment( img, sigma, k, minSz ):
    #GBPANOSEGMENT Graph-based image segmentation on panorama
    #   Similar as Pedro's algorithm, only the graph is built on sphere, so
    #   left and right side are considered as attached.
    #   img: should be in uint8 [0~256]
    #   sigma, k, minSz: same parameters as in Pedro's algorithm

    [height, width, _] = img.shape;
    #img_smooth = smooth(img, sigma);

    #sigma = 10;
    img_smooth = np.zeros([height,width,3])
    img_smooth[:,:,0] = gaussian_filter(img[:,:,0], sigma);
    img_smooth[:,:,1] = gaussian_filter(img[:,:,1], sigma);
    img_smooth[:,:,2] = gaussian_filter(img[:,:,2], sigma);


    
    #SrcImage = './data/rotImg_smooth.mat'
    #dict = loadmat(SrcImage)
    #img_smooth = dict['img_smooth']
    #plt.subplot(2, 1, 1)
    #plt.imshow(np.uint8(img))

    #plt.subplot(2, 1, 2)
    #plt.imshow(np.uint8(img_smooth))
    #plt.show()
    



    ## uniformly sample vectors on sphere and segment, test later
    dict = loadmat('./data/uniformvector_lvl8.mat');
    coor = dict['coor'];
    tri = dict['tri']
    #[coor, tri] = getUniformVector(8);

    # [ E ] = getSketchTokenEdgemap( img );
    # [EE, Ix, Iy] = dt2(double(E), 0.1, 0, 0.1, 0 );
    EE = np.zeros([height, width]);

    xySubs = CoordsTransform.uv2coords(CoordsTransform.xyz2uvN(coor,0), width, height,0);
    xySubs = np.int32(xySubs);

    SrcImage = './data/xySubs.mat'
    dict = loadmat(SrcImage)
    xySubs = dict['xySubs']
    xySubs = xySubs -1;
    
    idx = np.where(xySubs[:,1] < 0)
    xySubs[idx,1] = 0
    
    idx = np.where(xySubs[:,1] >= 512)
    xySubs[idx,1] = 511

    
    idx = np.where(xySubs[:,0] >= 1024)
    xySubs[idx,0] = 1023

    SubXY = np.array([xySubs[:,1], xySubs[:,0]]).T

    #xyinds = np.ravel_multi_index(SubXY,(height ,width));
    #offset = width*height;


    tri = tri - 1
    e0 = np.array([tri[:,0],tri[:,1]]).T
    e1 = np.array([tri[:,1],tri[:,2]]).T
    e2 = np.array([tri[:,2],tri[:,0]]).T

    edges = np.row_stack((e0,e1,e2))
    invert = edges[:,1]<edges[:,0];
    edges[invert,:] = edges[invert,1::-1];

    
    uniEdges,_ = np.unique(edges,return_inverse=True ,axis = 0);

    #eid0 = np.unravel_index(xyinds[uniEdges[:,0]],(height,width,3))
    #eid1 = np.unravel_index(xyinds[uniEdges[:,1]],(height,width,3))

    eid0 = SubXY[uniEdges[:,0],:].T
    eid1 = SubXY[uniEdges[:,1],:].T
    
    
    #eid0offset =  np.unravel_index(xyinds[uniEdges[:,0]] + offset,(height,width,3))
    #eid1offset =  np.unravel_index(xyinds[uniEdges[:,1]] + offset,(height,width,3))

    
    #eid0offset2 =  np.unravel_index(xyinds[uniEdges[:,0]] + 2 * offset,(height,width,3))
    #eid1offset2 =  np.unravel_index(xyinds[uniEdges[:,1]] + 2 * offset,(height,width,3))
    
    if any(eid0[:,0] >= 512):
        print('nono')

    weight = (img_smooth[eid0[0],eid0[1],0]-img_smooth[eid1[0],eid1[1],0])**2  + (img_smooth[eid0[0],eid0[1],1]-img_smooth[eid1[0],eid1[1],1])**2  + (img_smooth[eid0[0],eid0[1],2]-img_smooth[eid1[0],eid1[1],2])**2;
    gdweight = (EE[eid0[0],eid0[1]]+EE[eid0[0],eid0[1]])/2;
    panoEdge = np.array([uniEdges[:,0] , uniEdges[:,1] , np.sqrt(np.double(weight))+10*np.double(gdweight)]);

    maxID = coor.shape[0];
    num = uniEdges.shape[0];

    edgeLabel = segmentGraphEdge((maxID, num, panoEdge, k, minSz));

    L = np.unique(edgeLabel);
    temp = np.zeros(len(edgeLabel));

    [gridX, gridY] = np.meshgrid(np.arange(width), np.arange(height)) ;

    for i in np.arange(len(L)):
        temp[edgeLabel==L[i]] = i + 1;
    


    pixelvector = CoordsTransform.uv2xyzN(CoordsTransform.coords2uv([gridX[:] + 1, gridY[:] + 1], width, height),0);

    # k = 1;
    # [nnidx, dists] = annsearch( coor', pixelvector', k);
    #[nnidx, dists] = knnsearch( coor, pixelvector);


    nbrs = NearestNeighbors(n_neighbors=1, algorithm='auto').fit(coor)
    dists, nnidx = nbrs.kneighbors(pixelvector)

    #from scipy import spatial
    #tree = spatial.KDTree(coor);
    #dists, nnidx  = tree.query(pixelvector,k=1);

    
    #SrcImage = './data/temp.mat'
    #dict = loadmat(SrcImage)
    #tempm = dict['temp']

    #SrcImage = './data/nnidx.mat'
    #dict = loadmat(SrcImage)
    #nnidxm = dict['nnidx']
    #nnidxm = nnidxm - 1
    panoSegment = np.reshape( temp[nnidx] , [width, height]).T;

    #pimg = PIL.Image.fromarray(np.uint8(panoSegment))
    ##pimggray = pimg.convert('L')
    
    #pimg.save('./data/segmentation.jpg');

    return panoSegment 

def fgaussian(size, sigma):
     m,n = size 
     h, k = m//2, n//2
     x, y = np.mgrid[-h:h, -k:k]
     return np.exp(-(x**2 + y**2)/(2*sigma**2))

def smooth( img, sigma ):
    #SMOOTH Summary of this function goes here
    #   Detailed explanation goes here
    if img.dtype == 'uint8':
        img = np.double(img);
    elif img.dtype == 'double':
        img = np.double(img*255);
    

    WIDTH = 4;
    sigma = max(sigma, 0.01);
    len = np.int(2*np.ceil(sigma*WIDTH)+1);

    padsz = np.int((len-1)/2);
    img = np.pad(img, [padsz, padsz], 'edge');

    #h = fspecial('gaussian', [1 ,len], sigma);

    h = fgaussian([1,len],sigma)

    output = cat(3, conv2(h.T, h, img[:,:,1],'same'), 
                    conv2(h.T, h, img[:,:,2],'same'), 
                    conv2(h.T, h, img[:,:,3],'same'));

    output = output[padsz+1:end-padsz, padsz+1:end-padsz, :];
    # h = fspecial('gaussian', len, sigma);
    # output = filter2(h,img);

    return output

