from scipy.misc import imread, imsave, imresize
from scipy.misc.pilutil import imshow
from scipy.io import loadmat
import numpy as np
import Projection
import VpEstimation
import Visualization
import CoordsTransform
import PIL.Image
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import Rotation
import Segmentation
import PolygonRegion
import LineFaceIntersection
def process1():

    SrcImage = './data/pano_room.jpg'

    panoImg = imread(SrcImage, mode='RGB')
    panoImg = np.double(panoImg / 255)

    pi = np.pi

    #project it to multiple perspective views
    cutSize = 320; # size of perspective views
    fov = pi/3; # horizontal field of view of perspective views

    xh = np.arange(-pi,5/6*pi,pi/6);


    yh = np.zeros([1, len(xh)]);

    xp = [-3/3,-2/3,-1/3,+0/3,+1/3,+2/3,-3/3,-2/3,-1/3,+0/3,+1/3,+2/3]
    xp =np.multiply( xp , pi);


    yp = [1/4,1/4,1/4,1/4,1/4,1/4,-1/4,-1/4,-1/4,-1/4,-1/4,-1/4]
    yp = np.multiply(yp,pi);
    x = np.append(xh, xp)
    x = np.append(x, 0)
    x = np.append(x, 0)
    y =np.append(yh, yp)
    y = np.append(y,+pi/2)
    y = np.append(y,-pi/2) # viewing direction of perspective views



    sepScene = Projection.separatePano( panoImg, fov, x, y, cutSize);

    #figure 1
    #plt.imshow(np.uint8(sepScene[0].img* 255))
    #plt.show()

    ## Line segment detection on panorama: first on perspective views and project back to panorama
    numScene = len(sepScene);
    edges = [];
    for i in np.arange(numScene):
        [ edgeMap, edgeList ] = VpEstimation.lsdWrap( sepScene[i].img, 0.7);

        edge = VpEstimation.Edge()
        edge.img = edgeMap;
        edge.edgeLst = edgeList;
        edge.fov = sepScene[i].fov;
        edge.vx = sepScene[i].vx;
        edge.vy = sepScene[i].vy;
        edge.panoLst = VpEstimation.edgeFromImg2Pano( edge );

        edges = np.append(edges,edge)
    
    [lines,olines] = VpEstimation.combineEdgesN( edges); # combine line segments from views
    panoEdge = Visualization.paintParameterLine( lines, 1024, 512,None); # paint parameterized line segments

    
    
    ##plt.subplot(3, 1, 1)
    #plt.imshow(np.uint8(panoEdge),cmap='gray')   
    #plt.show()


    ##plt.subplot(3, 1, 2)
    #plt.imshow(np.uint8(sepScene[0].img * 255))
    #plt.show()
    
    ##plt.subplot(3, 1, 3)
    #plt.imshow(np.uint8(edges[0].img) * 255,cmap='gray')
    #plt.show()



    # estimating vanishing point: Hough 
    [ olines, mainDirect] = VpEstimation.vpEstimationPano( lines ); # mainDirect is vanishing point, in xyz format


    vpCoords = CoordsTransform.uv2coords(CoordsTransform.xyz2uvN(mainDirect,0), 1024, 512,0); # transfer to uv format, then image coords

    imgres = imresize(panoImg, [512, 1024]);
    panoEdge1r = Visualization.paintParameterLine( olines[0].line, 1024, 512, imgres);
    panoEdge2r = Visualization.paintParameterLine( olines[1].line, 1024, 512, imgres);
    panoEdge3r = Visualization.paintParameterLine( olines[2].line, 1024, 512, imgres);
    panoEdgeVP = np.array([panoEdge1r, panoEdge2r, panoEdge3r]);
   

    panoEdgeVP = np.zeros((512,1024,3), 'uint8')
    panoEdgeVP[:,:, 0] = panoEdge1r
    panoEdgeVP[:,:, 1] = panoEdge2r
    panoEdgeVP[:,:, 2] = panoEdge3r

    #panoEdgeVP = np.reshape(panoEdgeVP,[512,1024,3])
    plt.imshow(panoEdgeVP); 
  
    
    color = 'rgb';
    for i in np.arange(3):
        plt.scatter(vpCoords[i,0], vpCoords[i,1], 100, color[i],'o');
        plt.scatter(vpCoords[i+3,0], vpCoords[i+3,1], 100, color[i],'o');

    plt.title('Vanishing points and assigned line segments');
    #plt.show()   

    # rotate panorama to coordinates spanned by vanishing directions
    vp = mainDirect[2::-1,:];
    [ rotImg, R ] = Rotation.rotatePanorama( imgres, vp ,None );
    newMainDirect  = Rotation.rotatePoint( mainDirect, R );
    panoEdge1r =  Visualization.paintParameterLine( Rotation.rotateLines(olines[0].line, R), 1024, 512, rotImg);
    panoEdge2r =  Visualization.paintParameterLine( Rotation.rotateLines(olines[1].line, R), 1024, 512, rotImg);
    panoEdge3r =  Visualization.paintParameterLine( Rotation.rotateLines(olines[2].line, R), 1024, 512, rotImg);
   
    newPanoEdgeVP = np.zeros((512,1024,3), 'uint8')
    newPanoEdgeVP[:,:, 0] = panoEdge1r
    newPanoEdgeVP[:,:, 1] = panoEdge2r
    newPanoEdgeVP[:,:, 2] = panoEdge3r

    plt.imshow(newPanoEdgeVP); 
    for i in np.arange(3):
        plt.scatter(vpCoords[i,0], vpCoords[i,1], 100, color[i],'o');
        plt.scatter(vpCoords[i+3,0], vpCoords[i+3,1], 100, color[i],'o');
    
    plt.title('Original image');  
    plt.show()

   
    plt.imshow(newPanoEdgeVP); 
    newVpCoords = CoordsTransform.uv2coords(CoordsTransform.xyz2uvN(newMainDirect,0), 1024, 512,0);

    for i in np.arange(3):
        plt.scatter(newVpCoords[i,0], newVpCoords[i,1], 100, color[i],'o');
        plt.scatter(newVpCoords[i+3,0], newVpCoords[i+3,1], 100, color[i],'o');
    
    plt.title('Rotated image');
    plt.show()

    return newPanoEdgeVP;
    
def process2(rotImg):
    #SrcImage = './data/rotImg.jpg'
    #rotImg = imread(SrcImage, mode='RGB')
    #rotImg = np.uint8(rotImg)
    

    ## image segmentation: 
    panoSegment  = Segmentation.gbPanoSegment(rotImg, 0.5, 200, 50 );
    


    #SrcImage = './data/panoSegment.mat'
    #dict = loadmat(SrcImage)
    #panoSegment = dict['panoSegment']

    plt.imshow(panoSegment,cmap='gray');
    plt.title('Segmentation: left and right are connected');
    plt.show()

    ## Get region inside a polygon
    dict = loadmat('./data/points.mat'); # load room corner
    points = dict['points']

    dict = loadmat('./data/uniformvector_lvl8.mat');
    coor = dict['coor'];
    tri = dict['tri']


    vcs = CoordsTransform.uv2coords(CoordsTransform.xyz2uvN(coor,0), 1024, 512,0); # transfer vectors to image coordinates
    coords = CoordsTransform.uv2coords(CoordsTransform.xyz2uvN(points,0), 1024, 512,0);

    [ s_xyz, _] = PolygonRegion.sortXYZ( points[0:4,:] ); # define a region with 4 vertices
    [ inside, _, _ ] = PolygonRegion.insideCone( s_xyz[-1::-1,:], coor, 0 ); # test which vectors are in region

    #figure(8); 
    plt.imshow(rotImg); 
    #hold on
    for i in np.arange(4):
        plt.scatter(coords[i,0], coords[i,1], 100,'r','s');
    
    for i in np.where(inside):
        plt.scatter(vcs[i,0], vcs[i,1], 1, 'g','o');
    

    [ s_xyz, I ] = PolygonRegion.sortXYZ( points[4:8,:] );
    [ inside, _, _ ] = PolygonRegion.insideCone( s_xyz[-1::-1,:], coor, 0 );
    for i in np.arange(4,8):
        plt.scatter(coords[i,0], coords[i,1], 100, 'r','s');
    
    for i in np.where(inside):
        plt.scatter(vcs[i,0], vcs[i,1], 1, 'b','o');
    
    plt.title('Display of two wall regions');
    plt.show()

    ## Reconstruct a box, assuming perfect upperright cuboid
    D3point = np.zeros([8,3]);
    pointUV = CoordsTransform.xyz2uvN(points,0).T;
    floor = -160;

    floorPtID = np.array([2,3,6,7,2]) -1 ;
    ceilPtID = np.array([1,4,5,8,1]) -1 ;
    for i in np.arange(4):
        D3point[floorPtID[i],:] = LineFaceIntersection.LineFaceIntersection( np.array([0, 0, floor]), np.array([0, 0 ,1]), np.array([0, 0 ,0]), points[floorPtID[i],:] );
        D3point[ceilPtID[i],2] = D3point[floorPtID[i],2]/np.tan(pointUV[floorPtID[i],1])*np.tan(pointUV[ceilPtID[i],1]);
    
    ceiling = np.mean(D3point[ceilPtID,2]);
    for i in np.arange(4):
        D3point[ceilPtID[i],:] = LineFaceIntersection.LineFaceIntersection( np.array([0, 0, ceiling]), np.array([0, 0, 1]), np.array([0, 0, 0]), points[ceilPtID[i],:] );
    


    #figure(9);
    #fig = plt.figure()
    #ax = fig.add_subplot(111, projection='3d')
    #ax.scatter(D3point[floorPtID,0], D3point[floorPtID,1], D3point[floorPtID,2]);
    #hold on
    #ax.scatter(D3point[ceilPtID,0], D3point[ceilPtID,1], D3point[ceilPtID,2]);
    
    #for i in np.arange(4):
        #ax.scatter(D3point[[floorPtID[i], ceilPtID[i]],0], D3point[[floorPtID[i], ceilPtID[i]],1], D3point[[floorPtID[i], ceilPtID[i]],2]);
    
    #plt.title('Basic 3D reconstruction');
    #plt.show()
    #figure(10); 
    firstID = np.array([1,4,5,8,2,3,6,7,1,4,5,8]) -1 ;
    secndID = np.array([4,5,8,1,3,6,7,2,2,3,6,7]) - 1;
    lines = LineFaceIntersection.lineFromTwoPoint(points[firstID,:], points[secndID,:]);

    
    plt.imshow(Visualization.paintParameterLine(lines, 1024, 512, rotImg)); 
    #hold on
   
    for i in np.arange(8):
        plt.scatter(coords[i,0], coords[i,1], 100, 'r','s');
    
    plt.title('Get lines by two points');
    plt.show();

