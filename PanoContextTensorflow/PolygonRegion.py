from numpy.matlib import repmat
import numpy as np

def sortXYZ( xyz ):
    #SORTXYZ Sort 3D point in clockwise order
    #   The definition of clockwise order for 3D points is ill-posed. To make it well defined,
    #   we must first define a viewing direction. This code treats the center
    #   of the set of points as the viewing direction (from original point).
    ori_xyz = xyz;
    xyz = xyz/repmat(np.sum(xyz**2,1),3,1).T;
    center = np.mean(xyz,0);
    center = center/np.linalg.norm(center,2);
    # set up z axis at center
    z = center;
    x = [-center[1], center[0], 0]; x = x/np.linalg.norm(x,2);
    y = np.cross(z,x);
    R = np.dot(np.diag([1 ,1 ,1]) , np.linalg.pinv(np.array([x ,y ,z]).T));

    newXYZ = np.dot(R,(xyz.T));
    A = np.arctan2(newXYZ[1,:], newXYZ[0,:]);
    I = np.argsort(A);

    s_xyz = ori_xyz[I,:];


    return [ s_xyz, I ]


def insideCone( ccwCones, vc, tol ):
    #INSIDECONE Check if vectors are in a cone, in 3D space
    #   Cone is formed by ccwCones, vectors should be counter-clockwise viewing
    #   from original point; vc is the vector for checking; tol is a tolerance
    #   value, set tol=0 for exact judgement.
    if tol == None:
        tol = 0;
    

    ## get outward normal of cone surface
    numEdge = ccwCones.shape[0];
    normal = np.zeros([numEdge, 3]);
    for i in np.arange(numEdge-1):
        normal[i,:] = np.cross( ccwCones[i,:], ccwCones[i+1,:]);
    
    normal[numEdge-1,:] = np.cross( ccwCones[numEdge-1,:], ccwCones[0,:]);

    normal = normal/repmat( np.sqrt(np.sum(normal**2, 1)), 3, 1).T;

    ## negative to all outward normal
    numVC = vc.shape[0];
    inside = np.ones(numVC, dtype=np.bool);
    dotprods= np.zeros([numVC, numEdge]);
    for i in np.arange(numEdge):
        dotprods[:,i] = np.sum( vc * repmat(normal[i,:], numVC, 1), 1);
        valid = dotprods[:,i] < np.cos((90-tol)*np.pi/180);
        inside = inside & valid;
    

    B = np.max(dotprods,1);
    I = np.argmax(dotprods,1)
    
    max_intrude_CT = np.pi/2-np.arccos(B);
    max_intrude_ID = I;
    return [ inside, max_intrude_CT, max_intrude_ID ] 

