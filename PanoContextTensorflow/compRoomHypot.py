import Rotation
import PanoEdgeDetection
from scipy.misc import imread, imsave, imresize
import numpy as np
from scipy.io import loadmat
import RoomHypothesisSampling
import VpEstimation
import matplotlib.pyplot as plt
def compRoomHypot( SrcImage ):
    #compRoomHypot compute line segment detection, vanishing point, room layout
    #hypothesis, feature like OM, GC, CN are also computed because they rely on
    #line segments.
    #   To be consistent with previous buffer data, take care of GC
    '''
    panoImg = imread(SrcImage, mode='RGB')
    panoImg = np.double(panoImg / 255)
    
    imgSize = 320;
    qError = 0.7;
    [ olines, vp, views, edges, panoEdge, score, angle] = PanoEdgeDetection.panoEdgeDetection( panoImg, imgSize, qError); 
  

    Img_small = imresize(Img, [1024, 2048]);
    [rotImg, R] = Rotation.rotatePanorama(Img_small, vp[3::-1,:]);
    '''
  
   
     ## Get region inside a polygon
    dict = loadmat('./data/panoEdgeDetection.mat'); # load room corner
    views = dict['views']
    edges = dict['edges']
    vp = dict['vp']

    edgesObjs = []
    for i in np.arange(edges.shape[1]):
      
        edge = VpEstimation.Edge()
        edge.img = edges[0,i][0];
        edge.edgeLst = edges[0,i][1];
        edge.fov = edges[0,i][4];
        edge.vx = edges[0,i][2];
        edge.vy = edges[0,i][3];
        edge.panoLst = edges[0,i][5];

        edgesObjs = np.append(edgesObjs,edge)

    
    ##
    [ _, panoOmap ] = RoomHypothesisSampling.computePanoOmap( views, edgesObjs, vp );
    #parsave([bufname 'computePanoOmap.mat'], 'panoOmap', panoOmap);
    
    plt.imshow( np.uint8(panoOmap) * 255)
    plt.show()

    '''


    #save([bufname 'compRoomHypot_checkpoint1.mat'])

    figure(1);
    imshow(panoOmap); hold on

    panoOmap_rot = rotatePanorama(panoOmap, [], R);   
    [ ~, wallPanoNormal] = compSurfaceLabel( rotImg );


    #save([bufname 'compRoomHypot_checkpoint2.mat'])

    wallPanoNormal_rot = rotatePanorama(wallPanoNormal, [], R);

    orientation_ori = panoOmap; #C.orientation_ori;
    surfacelabel = rotatePanorama(wallPanoNormal, inv(R)); #rotatePanorama(C.surfacelabel_ori, inv(R));
    hyps = generateHypsB(olines, vp, 3, orientation_ori, surfacelabel);


    #save([bufname 'compRoomHypot_checkpoint3.mat'])

    hyps_rot = rotateHyps( hyps, R);

    rotImg = min(max(rotImg,0),1);
    colorName_rot = im2cM(double(rotImg));
    colorName = rotatePanorama(colorName_rot, inv(R));

    if config.UPDATE_TO_DISK
        parsave([bufname config.roomModelFile], ...
            'orientation_ori', panoOmap, 'surfacelabel_ori', wallPanoNormal, ...
            'orientation', panoOmap_rot, 'surfacelabel', wallPanoNormal_rot, ...
            'colorName', colorName, 'colorName_rot', colorName_rot);
        parsave([bufname config.roomHypoFile], ...
            'hypothesis_ori', hyps, 'hypothesis', hyps_rot);
    end

    #save([bufname 'compRoomHypot_checkpoint4.mat'])


    load([bufname 'compRoomHypot_checkpoint4.mat'])

    '''

