import numpy as np
from numpy.matlib import repmat

def icosahedron2sphere(level):

    # copyright by Jianxiong Xiao http://mit.edu/jxiao
    # this function use a icosahedron to sample uniformly on a sphere
    '''
    Please cite this paper if you use this code in your publication:
    J. Xiao, T. Fang, P. Zhao, M. Lhuillier, and L. Quan
    Image-based Street-side City Modeling
    ACM Transaction on Graphics (TOG), Volume 28, Number 5
    Proceedings of ACM SIGGRAPH Asia 2009
    '''


    a= 2/(1+np.sqrt(5));
    M=[
         0,a,-1,a,1,0,-a,1,0
        ,0,a,1,-a,1,0,a,1,0
        ,0,a,1,0,-a,1,-1,0,a
        ,0,a,1,1,0,a,0,-a,1
        ,0,a,-1,0,-a,-1,1,0,-a
        ,0,a,-1,-1,0,-a,0,-a,-1
        ,0,-a,1,a,-1,0,-a,-1,0
        ,0,-a,-1,-a,-1,0,a,-1,0
        ,-a,1,0,-1,0,a,-1,0,-a
        ,-a,-1,0,-1,0,-a,-1,0,a
        ,a,1,0,1,0,-a,1,0,a
        ,a,-1,0,1,0,a,1,0,-a
        ,0,a,1,-1,0,a,-a,1,0
        ,0,a,1,a,1,0,1,0,a
        ,0,a,-1,-a,1,0,-1,0,-a
        ,0,a,-1,1,0,-a,a,1,0
        ,0,-a,-1,-1,0,-a,-a,-1,0
        ,0,-a,-1,a,-1,0,1,0,-a
        ,0,-a,1,-a,-1,0,-1,0,a
        ,0,-a,1,1,0,a,a,-1,0
        ];

    coor = np.reshape(M,[60,3]);
    #[M(:,[1 2 3]); M(:,[4 5 6]); M(:,[7 8 9])];


    coor ,idx = np.unique(coor, return_inverse=True , axis = 0);

    tri = np.reshape(idx,[20,3]);

    '''
    for i in np.arange(tri.shape[0]):
        x(1)=coor(tri(i,1),1);
        x(2)=coor(tri(i,2),1);
        x(3)=coor(tri(i,3),1);
        y(1)=coor(tri(i,1),2);
        y(2)=coor(tri(i,2),2);
        y(3)=coor(tri(i,3),2);
        z(1)=coor(tri(i,1),3);
        z(2)=coor(tri(i,2),3);
        z(3)=coor(tri(i,3),3);
        patch(x,y,z,'r');
    end

    axis equal
    axis tight
    '''

    # extrude
    sqrtvalue = np.sqrt(np.sum(coor * coor,1));
    coor = coor / repmat(sqrtvalue,3,1).T;

   
    for i in np.arange(level):
        m = 0;

        triN= [];
        for t in np.arange(tri.shape[0]):
            n = coor.shape[0];
            c1 =  (coor[tri[t,0],:] + coor[tri[t,1],:]) / 2;
            coor = np.row_stack((coor,c1))

            c2 =  (coor[tri[t,1],:] + coor[tri[t,2],:]) / 2;
            coor = np.row_stack((coor,c2))

            c3 =  (coor[tri[t,2],:] + coor[tri[t,0],:]) / 2;
            coor = np.row_stack((coor,c3))

        
            triV = [n+0  ,   tri[t,0] ,   n+2];
           
            if(len(triN) == 0):
                triN = triV
            else:
                triN = np.row_stack((triN,triV))

            #triN = np.row_stack((triN,t))
            triV = [n+0  ,   tri[t,1]  ,  n+1];
            triN = np.row_stack((triN,triV))
            triV = [n+1  ,   tri[t,2]  ,  n+2];
            triN = np.row_stack((triN,triV))
            triV = [n+0 ,    n+1       ,  n+2];
            triN = np.row_stack((triN,triV))
        
            n = n+3;
            m = m+4;
        
        
        tri = triN;
    
        # uniquefy
        coor, idx = np.unique(coor,return_inverse=True ,axis = 0);
        tri = idx[tri];
    
        # extrude
        coor = coor / repmat(np.sqrt(np.sum(coor * coor,1)),3, 1).T;
    

    # vertex number: 12  42  162  642
    return  [coor,tri] 