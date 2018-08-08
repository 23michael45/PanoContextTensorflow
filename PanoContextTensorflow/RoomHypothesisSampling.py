import CoordsTransform
import numpy as np
import VanishingPoint.VanishingPoint as VVPR
import VanishingPoint.OMPrivate as VOPR
import VanishingPoint.OrientMap as VORM
import VpEstimation
import matplotlib.pyplot as plt
import Projection
class Line:
    point1 = []
    point2 = []
    length = []
    lineclass1 = []
    lineclass2 = []
    lineclass3 = []
    above_horizon = []


def computePanoOmap( scene, edges, xyz ):


    
    #COMPUTEPANOOMAP compute orientation map
    #   edge: line segments in separate views
    #   xyz: vanishing point
    #   OUTPUT:
    #   omap: orientation map of separate views
    #   panoOmap: project to panorama theta phi coordinate

    uv = CoordsTransform.xyz2uvN(xyz,0).T;
    omap = edges;
    numViews = len(edges);

    
    [H,W] = edges[0].img.shape;
    for i in np.arange(numViews):
        edgeLst = edges[i].edgeLst;
        if edgeLst.shape[0]==0:
            omap[i].img = np.zeros(H,W,3);
            continue;
        
        
        edge = VpEstimation.Edge()
        lines = [];
        for j in np.arange(edgeLst.shape[0]):
            
            line = Line()
            line.point1 = edgeLst[j,0:2];
            line.point2 = edgeLst[j,2:4];
            line.length = np.sqrt(np.sum((edgeLst[j,0:2]-edgeLst[j,2:4])**2));

            
            lines = np.append(lines,line)
        

    
        x = edges[i].vx; y = edges[i].vy;
        fov = edges[i].fov;
        ANGx = uv[:,0]; ANGy = uv[:,1];
        # compute the radius of ball
        [imH, imW] = edges[i].img.shape;
        R = (imW/2) / np.tan(fov/2);

        # im is the tangent plane, contacting with ball at [x0 y0 z0]
        x0 = R * np.cos(y) * np.sin(x);
        y0 = R * np.cos(y) * np.cos(x);
        z0 = R * np.sin(y);

        # plane function: x0(x-x0)+y0(y-y0)+z0(z-z0)=0
        # view line: x/alpha=y/belta=z/gamma
        # alpha=cos(phi)sin(theta);  belta=cos(phi)cos(theta);  gamma=sin(phi)
        alpha = np.cos(ANGy)*np.sin(ANGx);
        belta = np.cos(ANGy)*np.cos(ANGx);
        gamma = np.sin(ANGy);

        # solve for intersection of plane and viewing line: [x1 y1 z1]
        division = x0*alpha + y0*belta + z0*gamma;
        x1 = R*R*alpha/division;
        y1 = R*R*belta/division;
        z1 = R*R*gamma/division;

        # vector in plane: [x1-x0 y1-y0 z1-z0]
        # positive x vector: vecposX = [cos(x) -sin(x) 0]
        # positive y vector: vecposY = [x0 y0 z0] x vecposX
        vec = np.row_stack([x1-x0, y1-y0, z1-z0]).T;
        vecposX = np.row_stack([np.cos(x) ,-np.sin(x), 0]).T;
        deltaX = np.dot(vecposX,vec.T) / np.sqrt(np.dot(vecposX,vecposX.T)) + (imW+1)/2;
        vecposY = np.cross(np.column_stack([x0, y0, z0]), vecposX);
        deltaY = np.dot(vecposY,vec.T) /np.sqrt(np.dot(vecposY,vecposY.T)) + (imH+1)/2;

        deltaX = deltaX.flatten()
        deltaY = deltaY.reshape([-1])

    
        vp = np.zeros([3]).tolist()
        vp[0] = np.array([deltaX[0], deltaY[0]]);
        vp[1] = np.array([deltaX[1], deltaY[1]]);
        vp[2] = np.array([deltaX[2], deltaY[2]]);
    
        lines_orig = lines; 
        [lines, lines_ex] = VVPR.taglinesvp(vp, lines_orig);
        [omapmore, OMAP_FACTOR] = VORM.compute_omap(lines, vp, [H, W ,3]);
        
        #plt.imshow( np.uint8(omapmore) * 255)
        #plt.show()


        omap[i].img = np.double(omapmore);
        omap[i].lines_orig = lines_orig;
        omap[i].lines = lines;
        omap[i].vp = vp;
        linesImg = np.zeros([H, W, 3]);
        for j in np.arange(lines.shape[0]):
            lineclass = lines[j].lineclass;
            if lineclass==0:
                continue;
            
            x = np.linspace( lines[j].point1[0]+1, lines[j].point2[0]+1, 1000);
            y = np.linspace( lines[j].point1[1]+1, lines[j].point2[1]+1, 1000);
            xx = np.maximum( np.minimum( np.round(x), W-1), 0);
            yy = np.maximum( np.minimum( np.round(y), H-1), 0);
            #index = sub2ind( [H W], yy, xx);        
            #linesImg(H*W*(lineclass-1)+index) = 1;
        
        omap[i].linesImg = linesImg;
    
    #     roomhyp = sample_roomhyp(1000, lines_ex, vp, [H W 3]);
    #     omap(i).roomhyp = roomhyp;
    
    #     [ bestHyp ] = evaluateRoomHyp( omap(i) );
    #     disp_room(roomhyp(randsample(length(roomhyp),10)), scene(i).img, 1); # display some
    
    #     cuboidhyp_omap = generate_cuboid_from_omap(omapmore, vp, OMAP_FACTOR);
    #     disp_cubes(cuboidhyp_omap, scene(i).img, 1); # display all
    ## original
    #     img = edge(i).img;
    #     [lines linesmore] = compute_lines(img);
    #     if length(lines)<=3
    #         omap(i).img = zeros(H,W,3);
    #         continue;
    #     end
    #     
    #     [vp f] = compute_vp(lines, [H W 3]);
    #     lines_orig = lines; 
    #     [lines lines_ex] = taglinesvp(vp, lines_orig);
    #     linesmore_orig = linesmore; 
    #     [linesmore linesmore_ex] = taglinesvp(vp, linesmore_orig);
    #     [omapmore, OMAP_FACTOR] = compute_omap(lines, vp, [H W 3]);
    #     omap(i).img = double(omapmore);

    '''
    from scipy.io import loadmat
    dict = loadmat('./data/var.mat'); # load room corner
    omap = dict['omap']


    edgesObjs = []
    for i in np.arange(omap.shape[1]):
      
        edge = VpEstimation.Edge()
        edge.img = omap[0,i][0];
        edge.edgeLst = omap[0,i][1];
        edge.fov = omap[0,i][4];
        edge.vx = omap[0,i][2];
        edge.vy = omap[0,i][3];
        edge.panoLst = omap[0,i][5];
        edge.lines_orig = omap[0,i][6];
        edge.lines = omap[0,i][7];
        edge.vp = omap[0,i][8];
        edge.linesImg = omap[0,i][9];
        edgesObjs = np.append(edgesObjs,edge)
    '''

    panoOmap = Projection.combineViews( omap, 2048, 1024 );

    return [ omap, panoOmap ] 

