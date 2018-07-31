import numpy as np
import CoordsTransform
import Projection
def rotatePanorama( img, vp, R ):
    #ROTATEPANORAMA Rotate panorama
    #   if R is given, vp (vanishing point) will be overlooked
    #   otherwise R is computed from vp

    [sphereH, sphereW, C] = size(img);
    # rotImg = zeros( sphereH, sphereW, C);

    ## new uv coordinates
    [TX, TY] = np.meshgrid(np.arange(sphereW), np.arange(sphereH));
    TX = TX[:];
    TY = TY[:];
    ANGx = (TX- sphereW/2 -0.5)/sphereW * np.pi *2 ;
    ANGy = -(TY- sphereH/2 -0.5)/sphereH * np.pi;
    uvNew = [ANGx ,ANGy];
    xyzNew = CoordsTransform.uv2xyzN(uvNew,1);

    ## rotation matrix
    if nargin<3:
        R = np.diag([1,1, 1])/(vp.T);
    

    xyzOld = (np.linalg.pinv(R) * xyzNew.T).T;
    uvOld = CoordsTransform.xyz2uvN(xyzOld, 1);

    # Px = uvOld(:,1)/2/pi*sphereW + 0.5 + sphereW/2;
    # Py = -uvOld(:,2)/pi*sphereH + 0.5 + sphereH/2;
    Px = (uvOld[:,0]+np.pi) / (2*no.pi) * sphereW + 0.5;
    Py = (-uvOld[:,1] + np.pi/2) / np.pi * sphereH + 0.5;

    Px = reshape(Px, [sphereH, sphereW]);
    Py = reshape(Py, [sphereH, sphereW]);

    # boundary
    imgNew = np.double(np.zeros(sphereH+2, sphereW+2, C));
    imgNew[2:end-1, 2:end-1, :] = img;
    imgNew[2:end-1,1,:] = img[:,end,:];
    imgNew[2:end-1,end,:] = img[:,1,:];
    imgNew[1,2:sphereW/2+1,:] = img[1,sphereW:-1:sphereW/2+1,:];
    imgNew[1,sphereW/2+2:end-1,:] = img[1,sphereW/2:-1:1,:];
    imgNew[end,2:sphereW/2+1,:] = img[end,sphereW:-1:sphereW/2+1,:];
    imgNew[end,sphereW/2+2:end-1,:] = img[1,sphereW/2:-1:1,:];
    imgNew[1,1,:] = img[1,1,:];
    imgNew[end,end,:] = img[end,end,:];
    imgNew[1,end,:] = img[1,end,:];
    imgNew[end,1,:] = img[end,1,:];

    rotImg = Projection.warpImageFast(imgNew, Px+1, Py+1);
    # rotImg = warpImageFast(img, Px, Py);

    return rotImg, R 


def rotatePoint( p, R ):
    #ROTATEPOINT Rotate points
    #   p is point in 3D, R is rotation matrix
    op = (R * p.T).T;
    return  op 



def rotateLines( lines, R ):
    #ROTATELINES Rotate lines on panorama
    #   lines: parameterized lines, R: rotation matrix

    [numLine, dimLine] = size(lines);
    lines_N = zeros(numLine, dimLine);
    for i in np.arange(numLine):
        n = lines[i,0:3];
        sid = lines[i,4]*2*np.pi-np.pi;
        eid = lines[i,5]*2*np.pi-np.pi;
        u = np.row_stack((sid,eid));
        v = CoordsTransform.computeUVN(n, u, lines[i,3]);
        xyz = CoordsTransform.uv2xyzN(np.row_stack((u, v)), lines[i,3]);

        n_N = (R*n.T).T; n_N = n_N/np.linalg.norm(n_N,2);
        xyz_N = (R*xyz.T).T;
        lines_N[i,3] = np.argmax(np.abs(n_N([3, 1, 2])));
        uv_N = CoordsTransform.xyz2uvN(xyz_N, lines_N[i,3]);
        umax = max(uv_N[:,0])+np.pi;
        umin = min(uv_N[:,0])+np.pi;
        if umax-umin>np.pi:
            lines_N[i,4:6] = [umax ,umin]/2/np.pi;
        else:
            lines_N[i,4:6] = [umin ,umax]/2/np.pi;
        
    
        lines_N[i,0:3] = n_N;   
        #     lines_N(i,5:6) = (uv_N(:,1)'+pi)/2/pi;
        if dimLine>=7:
            lines_N[i,6] = np.arccos(np.sum(xyz_N[0,:] * xyz_N[1,:], 1)/(np.linalg.norm(xyz_N[0,:],2)*np.linalg.norm(xyz_N[1,:],2)));
            # lines_N(i,7) = lines(i,7); # this should be ok as well
        
        if dimLine>=8:
            lines_N[i,7] = lines[i,7];
        
    
    

    return lines_N




