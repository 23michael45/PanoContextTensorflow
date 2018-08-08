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
import Rotation
import Segmentation
import PolygonRegion
import LineFaceIntersection

def panoEdgeDetection( img, viewSize, qError ):
    #PANOEDGEDETECTION line detection on panorama
    #   INPUT:
    #   img: image waiting for detection, double type, range 0~1
    #   viewSize: image size of croped views
    #   qError: set smaller if more line segment wanted
    #   OUTPUT:
    #   oLines: detected line segments
    #   vp: vanishing point
    #   views: separate views of panorama
    #   edges: original detection of line segments in separate views
    #   panoEdge: image for visualize line segments
    
    pi = np.pi
    cutSize = viewSize;
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

    ## line segment detection
    # [eout,thresh] = edge_m(rgb2gray(img), 'canny');
    sepScene = Projection.separatePano( img, fov, x, y, cutSize);
   
  
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
    #panoEdge = Visualization.paintParameterLine( lines, 1024, 512,None); # paint parameterized line segments


 
    # estimating vanishing point: Hough 
    [ olines, mainDirect,score,angle] = VpEstimation.vpEstimationPano( lines ); # mainDirect is vanishing point, in xyz format




    imgres = imresize(img, [512, 1024]);
    panoEdge1r = Visualization.paintParameterLine( olines[0].line, 1024, 512, imgres);
    panoEdge2r = Visualization.paintParameterLine( olines[1].line, 1024, 512, imgres);
    panoEdge3r = Visualization.paintParameterLine( olines[2].line, 1024, 512, imgres);
  
    panoEdgeVP = np.zeros((512,1024,3), 'uint8')
    panoEdgeVP[:,:, 0] = panoEdge1r
    panoEdgeVP[:,:, 1] = panoEdge2r
    panoEdgeVP[:,:, 2] = panoEdge3r

    ## output
    olines = olines;
    vp = mainDirect;

    views = sepScene;
    edges = edge;
    panoEdge = panoEdgeVP;
    return  [ olines, vp, views, edges, panoEdge, score, angle]

