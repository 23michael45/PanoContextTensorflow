import numpy as np
from numpy.matlib import repmat
import CoordsTransform
def LineFaceIntersection( faceX, faceN, lineX, lineD ):
    #LINEFACEINTERSECTION Intersection of a plane and a line
    #   faceX: a point on the plane
    #   faceN: the normal direction of the plane
    #   lineX: a point on the line
    #   lineD: direction of the line
    #

    # A = dot(faceN,lineD,2)/lineD(1);
    # B = -dot(faceX,faceN,2) + (dot(lineX(2:3),faceN(2:3),2)) - lineX(1)/lineD(1)*(dot(lineD(2:3),faceN(2:3),2));
    A = np.sum(faceN*lineD)/lineD[0];
    B = -np.sum(faceX*faceN) + (np.sum(lineX[1:3]*faceN[1:3])) - lineX[0]/lineD[0]*(sum(lineD[1:3]*faceN[1:3]));

    x = -B/A;

    y = (x-lineX[0])/lineD[0]*lineD[1] + lineX[1];
    z = (x-lineX[0])/lineD[0]*lineD[2] + lineX[2];
    X = np.array([x, y, z]);
    return X 

def lineFromTwoPoint( pt1, pt2 ):
    #LINEFROMTWOPOINT Generate line segment based on two points on panorama
    #   pt1, pt2: two points on panorama
    #   lines: 
    #       1~3-th dim: normal of the line
    #       4-th dim: the projection dimension ID
    #       5~6-th dim: the u of line segment endpoints in projection plane
    #   use paintParameterLine to visualize

    numLine =  pt1.shape[0];
    lines = np.zeros([numLine, 6]);
    n = np.cross( pt1, pt2);
    n = n/repmat( np.sqrt(np.sum(n**2,1)), 3, 1).T;
    lines[:,0:3] = n;

    areaXY = np.abs(np.sum(n*repmat(np.array([0 ,0 ,1]), numLine, 1),1));
    areaYZ = np.abs(np.sum(n*repmat(np.array([1, 0, 0]), numLine, 1),1));
    areaZX = np.abs(np.sum(n*repmat(np.array([0, 1, 0]), numLine, 1),1));
    planeIDs = np.argmax(np.array([areaXY, areaYZ, areaZX]),0); # 1:XY 2:YZ 3:ZX
    lines[:,3] = planeIDs;

    for i in np.arange(numLine):
        uv = CoordsTransform.xyz2uvN(np.array([pt1[i,:], pt2[i,:]]), lines[i,3]).T;
        umax = np.max(uv[:, 0])+np.pi;
        umin = np.min(uv[:,0])+np.pi;
        if umax-umin>np.pi:
            lines[i,4:6] = np.array([umax, umin])/2/np.pi;
        else:
            lines[i,4:6] = np.array([umin ,umax])/2/np.pi;
        
   

    return  lines 


