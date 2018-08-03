import numpy as np
import CoordsTransform
import Projection
def rotatePanorama( img, vp, R ):
    #ROTATEPANORAMA Rotate panorama
    #   if R is given, vp (vanishing point) will be overlooked
    #   otherwise R is computed from vp

    [sphereH, sphereW, C] = img.shape;
    # rotImg = zeros( sphereH, sphereW, C);

    ## new uv coordinates
    [TX, TY] = np.meshgrid(np.arange(sphereW), np.arange(sphereH));
    TX = TX.T.flatten() + 1;
    TY = TY.T.flatten() + 1;
    ANGx = (TX- sphereW/2 -0.5)/sphereW * np.pi *2 ;
    ANGy = -(TY- sphereH/2 -0.5)/sphereH * np.pi;
    uvNew = np.column_stack((ANGx, ANGy))
    xyzNew = CoordsTransform.uv2xyzN(uvNew,0);

    ## rotation matrix
    if R == None:
        R = np.dot(np.diag([1,1, 1]), np.linalg.pinv(vp.T));
    

    xyzOld = np.dot(np.linalg.pinv(R) , xyzNew.T).T;
    uvOld = CoordsTransform.xyz2uvN(xyzOld, 0).T;

    # Px = uvOld(:,1)/2/pi*sphereW + 0.5 + sphereW/2;
    # Py = -uvOld(:,2)/pi*sphereH + 0.5 + sphereH/2;
    Px = (uvOld[:,0]+np.pi) / (2*np.pi) * sphereW + 0.5;
    Py = (-uvOld[:,1] + np.pi/2) / np.pi * sphereH + 0.5;

    Px = np.reshape(Px, [sphereW, sphereH]).T;
    Py = np.reshape(Py, [sphereW, sphereH]).T;

    # boundary
    imgNew = np.double(np.zeros([sphereH+2, sphereW+2, C]));
    imgNew[1:-1, 1:-1, :] = img;
    imgNew[1:-1,0,:] = img[:,-1,:];
    imgNew[1:-1,-1,:] = img[:,0,:];

    halfW = np.int(sphereW/2)

    imgNew[0,1:halfW + 1,:] = img[0,sphereW:halfW -1:-1,:];
    imgNew[0,halfW+1:-1,:] = img[0,halfW-1::-1,:];
    imgNew[-1,1:halfW+1,:] = img[-1,sphereW:halfW-1 :-1,:];
    imgNew[-1,halfW+1:-1,:] = img[0,halfW:0:-1,:];
    imgNew[0,0,:] = img[0,0,:];
    imgNew[-1,-1,:] = img[-1,-1,:];
    imgNew[0,-1,:] = img[0,-1,:];
    imgNew[-1,0,:] = img[-1,0,:];

    rotImg = Projection.warpImageFast(imgNew, Px+1, Py+1);
    # rotImg = warpImageFast(img, Px, Py);

    return rotImg, R 


def rotatePoint( p, R ):
    #ROTATEPOINT Rotate points
    #   p is point in 3D, R is rotation matrix
    op = np.dot(R , p.T).T;
    return  op 



def rotateLines( lines, R ):
    #ROTATELINES Rotate lines on panorama
    #   lines: parameterized lines, R: rotation matrix

    [numLine, dimLine] = lines.shape;
    lines_N = np.zeros([numLine, dimLine]);
    for i in np.arange(numLine):
        n = lines[i,0:3];
        sid = lines[i,4]*2*np.pi-np.pi;
        eid = lines[i,5]*2*np.pi-np.pi;
        u = np.row_stack((sid,eid));
        v = CoordsTransform.computeUVN(n, u, lines[i,3]);
        xyz = CoordsTransform.uv2xyzN(np.column_stack((u, v)), lines[i,3]);

        n_N = np.dot(R,n.T).T; 
        n_N = n_N/np.linalg.norm(n_N,2);
        xyz_N = np.dot(R,xyz.T).T;
        lines_N[i,3] = np.argmax(np.abs(n_N[[2, 0, 1]]));
        uv_N = CoordsTransform.xyz2uvN(xyz_N, lines_N[i,3]).T;
        umax = max(uv_N[:,0])+np.pi;
        umin = min(uv_N[:,0])+np.pi;
        if umax-umin>np.pi:
            lines_N[i,4:6] = np.array([umax ,umin])/2/np.pi;
        else:
            lines_N[i,4:6] = np.array([umin ,umax])/2/np.pi;
        
    
        lines_N[i,0:3] = n_N;   
        #     lines_N(i,5:6) = (uv_N(:,1)'+pi)/2/pi;
        if dimLine>=7:
            lines_N[i,6] = np.arccos(np.sum(xyz_N[0,:] * xyz_N[1,:])/(np.linalg.norm(xyz_N[0,:],2)*np.linalg.norm(xyz_N[1,:],2)));
            # lines_N(i,7) = lines(i,7); # this should be ok as well
        
        if dimLine>=8:
            lines_N[i,7] = lines[i,7];
        
    
    

    return lines_N




