from scipy.misc import imread, imsave, imresize
from scipy.misc.pilutil import imshow
import numpy as np
import Projection
import VpEstimation
import Visualization
import CoordsTransform
import PIL.Image
import matplotlib.pyplot as plt
import Rotation
def process():

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
    plt.show()   

    # rotate panorama to coordinates spanned by vanishing directions
    vp = mainDirect[2:-1:0,:];
    [ rotImg, R ] = Rotation.rotatePanorama( imgres, vp );
    [ newMainDirect ] = Rotation.rotatePoint( mainDirect, R );
    panoEdge1r =  Visualization.paintParameterLine( rotateLines(olines(1).line, R), 1024, 512, rotImg);
    panoEdge2r =  Visualization.paintParameterLine( rotateLines(olines(2).line, R), 1024, 512, rotImg);
    panoEdge3r =  Visualization.paintParameterLine( rotateLines(olines(3).line, R), 1024, 512, rotImg);
    newPanoEdgeVP = cat(3, panoEdge1r, panoEdge2r, panoEdge3r);


    plt.imshow(panoEdgeVP); 
    for i in np.arange(3):
        plt.scatter(vpCoords[i,0], vpCoords[i,1], 100, color(i),,'o');
        plt.scatter(vpCoords[i+3,0], vpCoords[i+3,1], 100, color(i),,'o');
    
    plt.title('Original image');  
    plt.show()

   
    plt.imshow(newPanoEdgeVP); 
    newVpCoords = CoordsTransform.uv2coords(CoordsTransform.xyz2uvN(newMainDirect), 1024, 512);

    for i in np.arange(3):
        scatter(newVpCoords[i,0], newVpCoords[i,1], 100, color(i),'o');
        scatter(newVpCoords[i+3,0], newVpCoords[i+3,1], 100, color(i),'o');
    
    plt.title('Rotated image');
    plt.show()
    
'''
%% image segmentation: 
[ panoSegment ] = gbPanoSegment( im2uint8(rotImg), 0.5, 200, 50 );
figure(7);
imshow(panoSegment,[]);
title('Segmentation: left and right are connected');

%% Get region inside a polygon
load('./data/points.mat'); % load room corner
load('./icosahedron2sphere/uniformvector_lvl8.mat'); % load vectors uniformly on sphere
vcs = uv2coords(xyz2uvN(coor), 1024, 512); % transfer vectors to image coordinates
coords = uv2coords(xyz2uvN(points), 1024, 512);

[ s_xyz, ~] = sortXYZ( points(1:4,:) ); % define a region with 4 vertices
[ inside, ~, ~ ] = insideCone( s_xyz(end:-1:1,:), coor, 0 ); % test which vectors are in region

figure(8); imshow(rotImg); hold on
for i = 1:4
    scatter(coords(i,1), coords(i,2), 100, [1 0 0],'fill','s');
end
for i = find(inside)
    scatter(vcs(i,1), vcs(i,2), 1, [0 1 0],'fill','o');
end

[ s_xyz, I ] = sortXYZ( points(5:8,:) );
[ inside, ~, ~ ] = insideCone( s_xyz(end:-1:1,:), coor, 0 );
for i = 5:8
    scatter(coords(i,1), coords(i,2), 100, [1 0 0],'fill','s');
end
for i = find(inside)
    scatter(vcs(i,1), vcs(i,2), 1, [0 0 1],'fill','o');
end
title('Display of two wall regions');

%% Reconstruct a box, assuming perfect upperright cuboid
D3point = zeros(8,3);
pointUV = xyz2uvN(points);
floor = -160;

floorPtID = [2 3 6 7 2];
ceilPtID = [1 4 5 8 1];
for i = 1:4
    D3point(floorPtID(i),:) = LineFaceIntersection( [0 0 floor], [0 0 1], [0 0 0], points(floorPtID(i),:) );
    D3point(ceilPtID(i),3) = D3point(floorPtID(i),3)/tan(pointUV(floorPtID(i),2))*tan(pointUV(ceilPtID(i),2));
end
ceiling = mean(D3point(ceilPtID,3));
for i = 1:4
    D3point(ceilPtID(i),:) = LineFaceIntersection( [0 0 ceiling], [0 0 1], [0 0 0], points(ceilPtID(i),:) );
end
figure(9);
plot3(D3point(floorPtID,1), D3point(floorPtID,2), D3point(floorPtID,3)); hold on
plot3(D3point(ceilPtID,1), D3point(ceilPtID,2), D3point(ceilPtID,3));
for i = 1:4
    plot3(D3point([floorPtID(i) ceilPtID(i)],1), D3point([floorPtID(i) ceilPtID(i)],2), D3point([floorPtID(i) ceilPtID(i)],3));
end
title('Basic 3D reconstruction');

figure(10); 
firstID = [1 4 5 8 2 3 6 7 1 4 5 8];
secndID = [4 5 8 1 3 6 7 2 2 3 6 7];
lines = lineFromTwoPoint(points(firstID,:), points(secndID,:));
imshow(paintParameterLine(lines, 1024, 512, rotImg)); hold on
for i = 1:8
    scatter(coords(i,1), coords(i,2), 100, [1 0 0],'fill','s');
end
title('Get lines by two points');

'''




