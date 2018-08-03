import numpy as np


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



