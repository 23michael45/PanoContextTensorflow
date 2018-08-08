import numpy as np
from scipy import interpolate
from scipy import ndimage
import matplotlib.pyplot as plt
from scipy.misc.pilutil import imshow
from scipy.misc import imread, imsave, imresize
import CoordsTransform
class Scene:
    img = []
    vx = []
    vy = []
    fov = 0
    sz = []

def separatePano( panoImg, fov, x, y, imgSize):
    #SEPARATEPANO project a panorama image into several perspective views
    # panoImg: panorama image; fov: field of view;
    # x,y: view direction of center of perspective views, in UV expression
    # imgSize: size of perspective views
    # saveDir: give if you want save result to disk

    if len(x) != len(y):
        printf('x and y must be the same size.\n');
        return;
  
    
    # if fov>pi
    #     fprintf('I guess the fov is in deg, convert to rad :)\n');
    #     fov = fov * pi / 180;
    # end
    # if ~isdouble('panoImg')
    #     fprintf('Image is not double, convert to double and scale to 1 :)');
    #     panoImg = double(panoImg)./255;
    # end

    numScene = len(x);

    #numScene = 1;

    # imgSize = 2*f*tan(fov/2);
    # sepScene = zeros(imgSize, imgSize, 3, numScene);


    sepScene = []
    for i in np.arange(numScene):
        scene = Scene()

        warped_image = imgLookAt(panoImg, x[i], y[i], imgSize, fov );
        scene.img = warped_image;
        scene.vx = x[i];
        scene.vy = y[i];
        scene.fov = fov;
        scene.sz = imgSize;
        
        sepScene = np.append(sepScene,scene)

        print('separatePano  : ',i,'/',numScene)
        
    return sepScene
    '''
    if exist('saveDir', 'var'):
        if !exist(saveDir, 'dir'):
            mkdir(saveDir);
            for i = 1:numScene
                imwrite(sepScene(i).img, sprintf('#s\#02d.pgm', saveDir, i), 'pgm');
    '''

def combineViews( Imgs, width, height ):
    #COMBINEVIEWS Combine separate views to panorama
    #   Imgs: same format as separatePano

    panoout = np.zeros([height, width, Imgs[0].img.shape[2]]);
    panowei = np.zeros([height, width, Imgs[0].img.shape[2]]);
    imgNum = len(Imgs);
    for i in np.arange(imgNum):
        [sphereImg, validMap] = CoordsTransform.im2Sphere( Imgs[i].img, Imgs[i].fov, width, height, Imgs[i].vx[0,0], Imgs[i].vy[0,0]);
        sphereImg[~validMap]= 0;   
        panoout = panoout + sphereImg;
        panowei = panowei + validMap;
    
    panoout[panowei==0] = 0;
    panowei[panowei==0] = 1;
    panoout = panoout/np.double(panowei);

    return panoout




def imgLookAt(im, CENTERx, CENTERy, new_imgH, fov ):

    '''
    Citation:
    J. Xiao, K. A. Ehinger, A. Oliva and A. Torralba.
    Recognizing Scene Viewpoint using Panoramic Place Representation.
    Proceedings of 25th IEEE Conference on Computer Vision and Pattern Recognition, 2012.
    http://sun360.mit.edu
    '''

    shape = im.shape

    sphereH = shape[0];  
    sphereW = shape[1];

 
    warped_im = np.zeros([new_imgH,new_imgH,3]);
       
   

    [TXwarp, TYwarp] = np.meshgrid(np.arange(new_imgH), np.arange(new_imgH));
    TX = TXwarp[:]; 
    TY = TYwarp[:];
    TX = (TX - 0.5 - new_imgH/2);    
    TY = (TY -0.5 - new_imgH/2);
    # new_imgH = tan(fov/2) * R * 2
    # TX = tan(ang/2) * R
    r = new_imgH/2 / np.tan(fov/2);


    
    

    # convert to 3D
    R = np.sqrt( np.power(TY,2) + np.power(r, 2));
    ANGy = np.arctan(- TY/r);
    ANGy = ANGy + CENTERy;

    
    X = np.sin(ANGy) * R;
    Y = - np.cos(ANGy) * R;
    Z = TX;

    INDn = np.where(np.abs(ANGy) > np.pi/2); 
    
    
    #project back to sphere

    ANGx = np.arctan(Z / -Y);

    
    RZY = np.sqrt(np.power(Z,2) + np.power(Y,2));
    ANGy = np.arctan(X / RZY);
    # INDn = np.where(abs(ANGy) > pi/2); 

    ANGx[INDn] = ANGx[INDn]+np.pi; # if ANGy>pi/2, connect to +pi
    ANGx = ANGx + CENTERx;

    INDy = np.where(ANGy < -np.pi/2);  
    ANGy[INDy] = - np.pi - ANGy[INDy] ;
    ANGx[INDy] = ANGx[INDy] + np.pi;

    INDx = np.where(ANGx <= -np.pi);     ANGx[INDx] = ANGx[INDx] + 2 * np.pi;
    INDx = np.where(ANGx > np.pi);     ANGx[INDx] = ANGx[INDx] - 2 * np.pi;
    INDx = np.where(ANGx > np.pi);     ANGx[INDx] = ANGx[INDx] - 2 * np.pi;
    INDx = np.where(ANGx > np.pi);     ANGx[INDx] = ANGx[INDx] - 2 * np.pi;


    

    # debug
    # X: [-pi pi]
    # Y: [-pi/2 pi/2]

    Px = (ANGx+np.pi) / (2*np.pi) * sphereW + 0.5;
    Py = ((- ANGy) + np.pi/2) / np.pi * sphereH + 0.5;

    #Px(INDn)=1;
    #Py(INDn)=1;

    INDxx = np.where(Px<1);
    Px[INDxx] = Px[INDxx] + sphereW;


    pix2 = np.arange(2)
    #im[:,pix2 + sphereW,:] = im[:,pix2,:];

    im = np.append(im,im[:,pix2,:],1)
    # debug
    # hold on
    # plot(Px, Py, 'r.');


    Px = np.reshape(Px, [new_imgH, new_imgH]);
    Py = np.reshape(Py, [new_imgH, new_imgH]);

    # finally warp image
    warped_im = warpImageFast(im, Px, Py);

    return warped_im

    
def warpImageFast(im,XXdense, YYdense):

    '''
    Citation:
    J. Xiao, K. A. Ehinger, A. Oliva and A. Torralba.
    Recognizing Scene Viewpoint using Panoramic Place Representation.
    Proceedings of 25th IEEE Conference on Computer Vision and Pattern Recognition, 2012.
    http://sun360.mit.edu
    '''

    m = np.min(np.min(XXdense))
    m = np.floor(m)-1
    minX = np.maximum(1.0,m);
    minY = np.maximum(1.0,np.floor(np.min(np.min(YYdense)))-1);
    
    shape = im.shape

    sphereH = shape[0];  
    sphereW = shape[1];

    maxX = np.minimum(sphereW,np.ceil(np.max(np.max(XXdense)))+1);
    maxY = np.minimum(sphereH,np.ceil(np.max(np.max(YYdense)))+1);

    minX = np.int32(minX)
    minY = np.int32(minY)
    maxX = np.int32(maxX)
    maxY = np.int32(maxY)

    im = im[minY:maxY,minX:maxX,:];

    shape = XXdense.shape
    im_warp =np.zeros([shape[0],shape[1],3])

    
    xcoord = XXdense-minX+1
    ycoord = YYdense-minY+1

    for c in np.arange(3):
        # im_warp(:,:,c) = uint8(interp2(double(im(:,:,c)), XXdense-minX+1, YYdense-minY+1,'*cubic'));
        # im_warp(:,:,c) = interp2(im(:,:,c), XXdense-minX+1, YYdense-minY+1,'*cubic');


        '''
        xmincoord = np.min(np.min(xcoord))
        xmaxcoord = np.max(np.max(xcoord))

        
        ymincoord = np.min(np.min(ycoord))
        ymaxcoord = np.max(np.max(ycoord))

        xaxis = np.arange(xmincoord,xmaxcoord,(xmaxcoord - xmincoord) / im.shape[1])
        yaxis = np.arange(ymincoord,ymaxcoord,(ymaxcoord - ymincoord) / im.shape[0])
        [xcoordorg ,ycoordorg]= np.meshgrid(xaxis,yaxis) 
        f = interpolate.interp2d(xaxis,yaxis,im[:,:,c], kind ='linear');

        
        xaxis = np.arange(xmincoord,xmaxcoord,(xmaxcoord - xmincoord) / xcoord.shape[1])
        yaxis = np.arange(ymincoord,ymaxcoord,(ymaxcoord - ymincoord) / xcoord.shape[0])
        im_warp[:,:,c] = f(xaxis,yaxis)
        '''

        #这里用map_coordinates 对应matlab里的interp2
        im_warp[:,:,c] = ndimage.map_coordinates(im[:,:,c],[ycoord.ravel(),xcoord.ravel()],order = 3,mode = 'nearest').reshape([shape[0],shape[1]])

        #im_warp(:,:,c) = interp2(im(:,:,c), XXdense-minX+1, YYdense-minY+1,'*nearest');
        #im_warp(:,:,c) = interp2(im(:,:,c), XXdense-minX+1, YYdense-minY+1,'linear');


    #im_warp = imresize(im,[shape[0],shape[1]])
    return im_warp
    
