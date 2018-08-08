import numpy as np
import Projection
from numpy.matlib import repmat
import matplotlib.pyplot as plt
def xyz2uvN( xyz, planeID ):
    #   XYZ2UVN Convert XYZ expression to UV expression in 3D
    #   change 3-dim unit vector XYZ expression to UV in 3D
    #   planeID: base plane for UV, 1=XY
    #   Check uv2xyzN for opposite mapping
    if 'planeID' in locals() == False:
        planeID = 0;
    

    ID1 =  np.int32((planeID+0)%3);
    ID2 =  np.int32((planeID+1)%3);
    ID3 =  np.int32((planeID+2)%3);

    normXY = np.sqrt( xyz[:,ID1]**2+xyz[:,ID2]**2);
    normXY[normXY<0.000001] = 0.000001;
    normXYZ = np.sqrt( xyz[:,ID1]**2+xyz[:,ID2]**2+xyz[:,ID3]**2);
    # 
    v = np.arcsin(xyz[:,ID3]/normXYZ);

    u = np.arcsin(xyz[:,ID1]/normXY);
    valid = np.logical_and(xyz[:,ID2]<0 ,u>=0);
    u[valid] = np.pi-u[valid];
    valid = np.logical_and(xyz[:,ID2]<0 , u<=0);
    u[valid] = -np.pi-u[valid];

    uv = np.row_stack((u ,v));
    uv[np.isnan(uv[:,1]),1] = 0;
    
    return uv
def coords2uv( coords, width, height ):
    #COORDS2UV Image coordinates (xy) to uv
    #   Convert pixel location on panorama image to UV expression in 3D
    #   width and height are size of panorama image, width = 2 x height
    #   the output UV take XY plane for U
    #   Check uv2coords for opposite mapping
    middleX = width/2+0.5;
    middleY = height/2+0.5;

    coords0 = coords[0].T.flatten()
    coords1 = coords[1].T.flatten()

    uv = np.array([(coords0-middleX)/width*2*np.pi ,-(coords1-middleY)/height*np.pi]).T;

    return uv 


def  uv2coords( uv, width, height, planeID ):
    
    #UV2COORDS Convert UV to image coordinates on panorama
    #   Convert UV expression in 3D to pixel location on panorama image.
    #   width and height are size of panorama image, width = 2 x height
    #   planeID: the base plane for U, 1=XY
    #   Check coords2uv for opposite mapping
   


    if 'planeID' in locals() == False:
        planeID = 0;

    if (planeID != 0):
        uv = xyz2uvN(uv2xyzN(uv, planeID), 0);
   

    uvcoord = np.zeros([uv.shape[1],2]);
    uvcoord[:,0] = np.minimum(np.round((uv[0,:]+np.pi)/2/np.pi*width+0.5), width);
    uvcoord[:,1] = np.minimum(np.round((np.pi/2-uv[1,:])/np.pi*height+0.5), height);
    

    return uvcoord

def computeUVN( n, s, planeID ):
    #COMPUTEUVN compute v given u and normal.
    #   A point on unit sphere can be represented as (u,v). Sometimes we only
    #   have u and the normal direction of the great circle that point locates
    #   on, and want to to get v.
    #   planeID: which plane we choose for uv expression. planeID=1 means u is
    #   in XY plane. planeID=2 means u is in YZ plane.
       
    if planeID==1:
        n = [n[1], n[2], n[0]];
    
    if planeID==2:
        n = [n[2], n[0], n[1]];
    
    bc = n[0]*np.sin(s) + n[1]*np.cos(s);
    bs = n[2];
    out = np.arctan(-bc/bs);
    return out

def  uv2xyzN( uv, planeID ):
    #UV2XYZN Convert UV expression to XYZ expression in 3D
    #   change UV expression to 3-dim unit vector in 3D
    #   planeID: base plane for UV, 1=XY
    #   Check xyz2uvN for opposite mapping
    
    if 'planeID' in locals() == False:
        planeID = 0;

    ID1 =  np.int32((planeID+0)%3);
    ID2 =  np.int32((planeID+1)%3);
    ID3 =  np.int32((planeID+2)%3);

    xyz = np.zeros([uv.shape[0],3]);
    xyz[:,ID1] = np.cos(uv[:,1])*np.sin(uv[:,0]);
    xyz[:,ID2] = np.cos(uv[:,1])*np.cos(uv[:,0]);
    xyz[:,ID3] = np.sin(uv[:,1]);

    return xyz


def im2Sphere(im, imHoriFOV, sphereW, sphereH, x, y):
    # perfect warping from separate image to panorama
    # Work for x in [-pi,+pi], y in [-pi/2,+pi/2], and proper FOV
    # For other (x,y,fov), it should also work, but depends on trigonometric 
    # functions process in matlab, and not tested.

    # map pixel in panorama to viewing direction
    [TX,TY] = np.meshgrid(np.arange(sphereW), np.arange(sphereH));
    TX = TX.T.flatten();
    TY = TY.T.flatten();
    ANGx = (TX- sphereW/2 -0.5)/sphereW * np.pi *2 ;
    ANGy = -(TY- sphereH/2 -0.5)/sphereH * np.pi;

    # compute the radius of ball
    [imH, imW, _] = im.shape;
    R = (imW/2) / np.tan(imHoriFOV/2);

    # im is the tangent plane, contacting with ball at [x0 y0 z0]
    x0 = R * np.cos(y) * np.sin(x);
    y0 = R * np.cos(y) * np.cos(x);
    z0 = R * np.sin(y);

    # plane function: x0(x-x0)+y0(y-y0)+z0(z-z0)=0
    # view line: x/alpha=y/belta=z/gamma
    # alpha=cos(phi)sin(theta);  belta=cos(phi)cos(theta);  gamma=sin(phi)
    alpha = np.cos(ANGy)*np.sin(ANGx);
    belta = np.cos(ANGy)*np.cos(ANGx);
    gamma = np.sin(ANGy);

    # solve for intersection of plane and viewing line: [x1 y1 z1]
    division = x0*alpha + y0*belta + z0*gamma;
    x1 = R*R*alpha/division;
    y1 = R*R*belta/division;
    z1 = R*R*gamma/division;

    # vector in plane: [x1-x0 y1-y0 z1-z0]
    # positive x vector: vecposX = [cos(x) -sin(x) 0]
    # positive y vector: vecposY = [x0 y0 z0] x vecposX
    vec = np.concatenate((x1-x0, y1-y0, z1-z0)).T;
    vecposX = np.array([[np.cos(x), -np.sin(x), 0]]);
    deltaX = np.dot(vecposX ,vec.T) / np.sqrt(np.sum(vecposX * vecposX));
    vecposY = np.cross(np.concatenate((x0 ,y0 ,z0)).T, vecposX);
    deltaY = np.dot(vecposY,vec.T) / np.sqrt(np.sum(vecposY * vecposY));

    # convert to im coordinates
    Px = deltaX.reshape( [sphereW, sphereH]).T + (imW+1)/2;
    Py = deltaY.reshape( [sphereW, sphereH]).T + (imH+1)/2;

    # warp image
    sphereImg = Projection.warpImageFast(im, Px, Py);
    #validMap = ~np.isnan(sphereImg[:,:,0]);

    validMap =np.logical_and(np.logical_and (Px > 0 , Px < imW) ,np.logical_and (Py > 0 , Py < imH))


    # view direction: [alpha belta gamma]
    # contacting point direction: [x0 y0 z0]
    # so division>0 are valid region
    division = division.reshape( [sphereW, sphereH]).T
    validMap[division<0] = False;

    validMap = np.stack((validMap,)*3, -1)
    print(validMap.shape)
   
    return sphereImg, validMap

