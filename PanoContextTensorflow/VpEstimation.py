import sys, string, os
import time
import numpy as np
import datetime
from pathlib import Path
import PIL.Image
#from scipy.misc import imread, imsave, imresize
from matplotlib.image import imsave
import subprocess
import cv2
import array
import random
import sys
from numpy.matlib import repmat
import CoordsTransform
import icosahedron2sphere
class Edge:
    img = []
    edgeLst = []
    vx = []
    vy = []
    fov = 0
    panoLst = []
class OLLine:
    line = []

def lsdWrap( img, qerror ):
    #LSDWRAPPER Matlab wrapper of LSD: Call external LSD for line segment detection
    #   img: input image, cmdline: command line for executable LSD
    #   edgeList: output of LSD, [xmin,ymin,xmax,ymax]
    #   edgeMap: plot line segments on image

    lsdLocation = 'tools\\lsd';
    bufferFolder = './data/';

    # rng('shuffle');
    t = datetime.datetime.now()
    
    
    randomID = '{:0>2d}_{:0>2d}_{:0>2d}_{:0>8d}'.format(np.random.randint(0,100), t.hour, t.minute, np.int32(t.second*1000000 + t.microsecond));
    imgbuf = '{}buf_{}.pgm'.format(bufferFolder ,randomID);
    edgbuf ='{}buf_result_{}.txt'.format(bufferFolder ,randomID);
    
    
    ## line detection
    while os.path.isfile(imgbuf) or os.path.isfile(edgbuf):
        print('buf name conflict\n');
        pause(0.1);
        
    pimg = PIL.Image.fromarray(np.uint8(img* 255))
    pimggray = pimg.convert('L')
    
    pimggray.save(imgbuf);

    #cv2.imwrite(imgbuf,np.uint8(img* 255))


    
    #imgbuf = './data/buf_test.pgm'
    #edgbuf = './data/buf_result_test.txt'

    os.system('{} -q {} {} {}'.format(lsdLocation ,qerror, imgbuf , edgbuf));

    edgeList = np.loadtxt( edgbuf);

    edgeNum = edgeList.shape[0]

    ## draw on image

    
    shape = img.shape

    imgH = shape[0];  
    imgW = shape[1];

    edgeMap = np.zeros([imgH, imgW]);

    for i in np.arange(edgeNum):

        x = np.linspace( edgeList[i,0]+1, edgeList[i,2]+1, 1000);
        y = np.linspace( edgeList[i,1]+1, edgeList[i,3]+1, 1000);

        xx = np.maximum( np.minimum( np.round(x), imgW-1), 0);
        yy = np.maximum( np.minimum( np.round(y), imgH-1), 0);

        pos = [np.int32(yy), np.int32(xx)]
        size = (imgH ,imgW)
        #index = np.ravel_multi_index(pos,size);
        edgeMap[pos] = 1;#rand(1);
    

    ## delete buffer files
    #os.remove( imgbuf );
    #os.remove( edgbuf );
    

    return  [ edgeMap, edgeList ]



def edgeFromImg2Pano( edges ):
    #EDGEFROMIMG2PANO Convert line segment from perspective views to panorama
    #   edges: a structure contains vx, vy, fov of perspective views and line
    #   segments and edge map of each views.
    #   panoList: output, in format [normal u1 u2 score]
    edgeList = edges.edgeLst;
    if len(edgeList) == 0:
        panoList = [];
        return;
    

    vx = edges.vx;
    vy = edges.vy;
    fov = edges.fov;
    [imH, imW] = edges.img.shape;

    R = (imW/2) / np.tan(fov/2);

    # im is the tangent plane, contacting with ball at [x0 y0 z0]
    x0 = R * np.cos(vy) * np.sin(vx);
    y0 = R * np.cos(vy) * np.cos(vx);
    z0 = R * np.sin(vy);
    vecposX = [np.cos(vx), -np.sin(vx), 0];
    vecposY = np.cross([x0 ,y0, z0], vecposX);

    vY13 = vecposY.reshape(1,-1)
    vY31 = vecposY.reshape(-1,1)
    v = np.dot(vY13,vY31)

    vecposY = vecposY / np.sqrt(v);
    Xc = (0 + imW-1)/2;
    Yc = (0 + imH-1)/2;

    vecx1 = edgeList[:,0]-Xc;
    vecy1 = edgeList[:,1]-Yc;
    vecx2 = edgeList[:,2]-Xc;
    vecy2 = edgeList[:,3]-Yc;

    vec1 = repmat(vecx1.reshape(-1,1), 1, 3) * repmat(vecposX, vecx1.shape[0], 1) + repmat(vecy1.reshape(-1,1), 1, 3) * repmat(vecposY, vecy1.shape[0], 1);
    vec2 = repmat(vecx2.reshape(-1,1), 1, 3) * repmat(vecposX, vecx2.shape[0] ,1) + repmat(vecy2.reshape(-1,1), 1, 3) * repmat(vecposY, vecy2.shape[0], 1);
    coord1 = repmat([x0, y0 ,z0],vec1.shape[0], 1) + vec1;
    coord2 = repmat([x0 ,y0 ,z0],vec2.shape[0], 1) + vec2;

    normal = np.cross( coord1, coord2, 1);
    n = np.sqrt( normal[:,0]**2 + normal[:,1]**2 + normal[:,2]**2);
    normal = normal / repmat(n.reshape(-1,1), 1, 3);



    #
    panoList = np.column_stack((normal,coord1,coord2,edgeList[:,-1]))

    ## lines



    

    return panoList 






def combineEdgesN(  edges ):
    #COMBINEEDGES Combine some small line segments, should be very conservative
    #   lines: combined line segments
    #   ori_lines: original line segments
    #   line format: [nx ny nz projectPlaneID umin umax LSfov score]
    arcList = [];
    for i in np.arange(edges.shape[0]):
        panoLst = edges[i].panoLst;
        if panoLst[0].shape[0] == 0:
            continue;
        
        if  (len(arcList)) == 0:
            arcList = panoLst
        else:
            arcList = np.row_stack( (arcList,panoLst));
    

    ## ori lines
    numLine = arcList.shape[0];
    ori_lines = np.zeros((numLine,8));
    areaXY = np.abs(np.sum(arcList[:,0:3]*repmat([0, 0, 1], arcList.shape[0], 1),1));
    areaYZ = np.abs(np.sum(arcList[:,0:3]*repmat([1, 0 ,0], arcList.shape[0], 1),1));
    areaZX = np.abs(np.sum(arcList[:,0:3]*repmat([0 ,1, 0], arcList.shape[0], 1),1));

    vec = [areaXY, areaYZ, areaZX]

    #[_, planeIDs] = np.max(vec,  1); # 1:XY 2:YZ 3:ZX
    planeIDs = np.argmax(vec,0);
   
    for i in np.arange(numLine):
        ori_lines[i,0:3] = arcList[i,0:3];
        ori_lines[i,3] = planeIDs[i];
        coord1 = arcList[i,3:6];
        coord2 = arcList[i,6:9];

        uv = CoordsTransform.xyz2uvN(np.row_stack((coord1, coord2)), planeIDs[i]);
        umax = np.max(uv[0,:])+np.pi;
        umin = np.min(uv[0,:])+np.pi;
        if umax-umin>np.pi:
            ori_lines[i,4:6] = np.column_stack((umax ,umin))/2/np.pi;
    #         ori_lines(i,7) = umin + 1 - umax;
        else:
            ori_lines[i,4:6] = np.column_stack((umin ,umax))/2/np.pi;
    #         ori_lines(i,7) = umax - umin;
        
        ori_lines[i,6] = np.arccos(np.sum(coord1*coord2)/(np.linalg.norm(coord1,2)*np.linalg.norm(coord2,2)));
        ori_lines[i,7] = arcList[i,9];
  
   
    # valid = ori_lines(:,3)<0;
    # ori_lines(valid,1:3) = -ori_lines(valid,1:3);


    ## additive combination
    lines = ori_lines;
    # panoEdge = paintParameterLine( lines, 1024, 512);
    # figure; imshow(panoEdge);
    for iter  in  np.arange(3):
        numLine = lines.shape[0];
        valid_line = np.ones([numLine],dtype = bool);
        for i in  np.arange(numLine):
    #         fprintf('#d/#d\n', i, numLine);
            if valid_line[i] == False:
                continue;
            
            dotProd = np.sum(lines[:,0:3]* repmat(lines[i,0:3],numLine, 1), 1);
            valid_curr =  (np.abs(dotProd) > np.cos(1*np.pi/180)) & valid_line;
            valid_curr[i] = False;
            valid_ang = np.where(valid_curr);
            for j in valid_ang[0]:
                range1 = lines[i,4:6];
                range2 = lines[j,4:6];
                valid_rag = intersection(range1, range2);
                if valid_rag==False:
                    continue;
                

                # combine   
                I = np.argmax(np.abs(lines[i,0:3]));
                if lines[i,I]*lines[j,I]>0:
                    nc = lines[i,0:3]*lines[i, 6] + lines[j,0:3]*lines[j, 6];
                else:
                    nc = lines[i,0:3]*lines[i, 6] - lines[j,0:3]*lines[j, 6];
                
                nc = nc / np.linalg.norm(nc,2);

                if insideRange(range1[0], range2):
                    nrmin = range2[0] ;
                else:
                    nrmin = range1[0];
                
                if insideRange(range1[1], range2):
                    nrmax = range2[1];
                else:
                    nrmax = range1[1];
                

                u = np.array([nrmin,nrmax])*2*np.pi - np.pi;
                v = CoordsTransform.computeUVN( nc, u, lines[i,3]);
                xyz = CoordsTransform.uv2xyzN(np.column_stack((u, v)), lines[i,3]);
                length = np.arccos(np.sum(xyz[0,:]* xyz[1,:]));
                scr = (lines[i,6]*lines[i,7] + lines[j,6]*lines[j,7])/(lines[i,6]+lines[j,6]);
            

                nc = np.append(nc,lines[i,3])
                nc = np.append(nc,nrmin)
                nc = np.append(nc,nrmax)
                nc = np.append(nc,length)
                nc = np.append(nc,scr)
                newLine =nc
                lines[i,:] = newLine;
                valid_line[j] = False;
            
        
        lines =  lines[valid_line,:] 
        print('iter: #d, before: #d, after: #d\n', iter, len(valid_line), sum(valid_line));
        
    return [ lines, ori_lines ]
    ''' 
    #     panoEdge = paintParameterLine( lines, 1024, 512);
    #     figure; imshow(panoEdge);
    
    ## previous method, bin voting
    # ## build up voting space
    # numDivision = 15;#round(max(width, height)/4);
    # 
    # normal = arcList(:,1:3);
    # valid = normal(:,3)<0;
    # normal(valid,:) = -normal(valid,:);
    # uv = xyz2uvN(normal, 1);
    # thetas = uv(:,1);   phis = uv(:,2);
    # 
    # uBinSize = 2*pi/(4*numDivision);
    # vBinSize = pi/2/(numDivision);
    # m = min(floor( (thetas-(-pi))/uBinSize) + 1, numDivision*4);
    # n = min(floor( phis/vBinSize) + 1, numDivision);
    # normalInd = sub2ind([4*numDivision numDivision], m, n);
    # 
    # uniqueNormal = unique(normalInd);
    # # decide voting dimension
    # [m,n] = ind2sub([4*numDivision numDivision], uniqueNormal);
    # u = -pi + (m-1)*uBinSize + uBinSize/2;
    # v = 0   + (n-1)*vBinSize + vBinSize/2;
    # uniNormal = [cos(v).*sin(u) cos(v).*cos(u) sin(v)];
    # areaXY = abs(sum(uniNormal.*repmat([0 0 1], [size(uniNormal,1) 1]),2));
    # areaYZ = abs(sum(uniNormal.*repmat([1 0 0], [size(uniNormal,1) 1]),2));
    # areaZX = abs(sum(uniNormal.*repmat([0 1 0], [size(uniNormal,1) 1]),2));
    # [~, planeIDs] = max([areaXY areaYZ areaZX], [], 2); # 1:XY 2:YZ 3:ZX
    # 
    # subVoteBinNum = 1024;
    # uvBin = false(length(uniqueNormal), subVoteBinNum);
    # uBinSize = 2*pi/subVoteBinNum;
    # for i = 1:length(uniqueNormal)
    #     normIDs = find(normalInd==uniqueNormal(i));
    #     norms = normal(normIDs,:);
    #     rectNorm = sum(norms,1);
    #     uniNormal(i,:) = rectNorm./norm(rectNorm);
    #     dimIndc = planeIDs(i);
    #     for j = normIDs'
    #         coord1 = arcList(j,4:6);
    #         coord2 = arcList(j,7:9);
    #         xx = linspace(coord1(1),coord2(1),1000);
    #         yy = linspace(coord1(2),coord2(2),1000);
    #         zz = linspace(coord1(3),coord2(3),1000);
    #         uv = xyz2uvN([xx' yy' zz'], dimIndc);  
    # 
    #         m = min(floor( (uv(:,1)-(-pi))/uBinSize) + 1, subVoteBinNum);
    #         uvBin(i,m) = true;
    #     end
    # end
    # 
    # ## extract line segments
    # # numLines = 0;
    # lines = [];
    # for i = 1:length(uniqueNormal)
    # #     fprintf('#d\n',i);
    #     bins = int32(uvBin(i,:));
    #     changePt(2:length(bins)) = bins(2:end) - bins(1:end-1);
    #     changePt(1) = bins(1) - bins(end);
    #     startPt = find(changePt==1);
    #     endPt   = find(changePt==-1) - 1;
    #     endPt = rem(endPt + subVoteBinNum+1, subVoteBinNum+1);
    #     
    #     if endPt(1)>=startPt(1)
    #         mtEndPt = endPt;
    #     else
    #         mtEndPt = [endPt(2:end) endPt(1)]; 
    #     end
    #     
    #     lines(end+1:end+length(startPt),:) = ...
    #         [repmat([uniNormal(i,:) planeIDs(i)],[length(startPt) 1]) ...
    #         (startPt'-1)/subVoteBinNum mtEndPt'/subVoteBinNum];
    # end

'''
def intersection(range1, range2):
    if range1[1]<range1[0]:
        range11 = [range1[0] ,1];
        range12 = [0, range1[1]];
    else:
        range11 = range1;
        range12 = [0, 0];
    
    if range2[1]<range2[0]:
        range21 = [range2[0], 1];
        range22 = [0, range2[1]];
    else:
        range21 = range2;
        range22 = [0, 0];
    
    b = max(range11[0],range21[0])<min(range11[1],range21[1]);
    if b:
        return b;
    
    b2 = max(range12[0],range22[0])<min(range12[1],range22[1]);
    b = b | b2;
    return b

def insideRange(pt, range):
    if range[1]>range[0]:
        b = pt>=range[0] and pt<=range[1];
    else:
        b1 = pt>=range[0] and pt<=1;
        b2 = pt>=0 and pt<=range[1];
        b = b1 or b2;
    return b



def vpEstimationPano( lines ):
    #VPESTIMATION Estimate vanishing points via lines
    #   lines: all lines in format of [nx ny nz projectPlaneID umin umax LSfov score]
    #   olines: line segments in three directions
    #   mainDirect: vanishing points

    clines = lines;
    for iter in np.arange(3):
        print('*************#d-th iteration:****************\n', iter);
        [mainDirect, score, angle] = findMainDirectionEMA( clines );    # search for main directions
    
        [ type, typeCost ] = assignVanishingType( lines, mainDirect[0:3,:], 0.1, 10 ); # assign directions to line segments
        lines1 = lines[type==0,:];
        lines2 = lines[type==1,:];
        lines3 = lines[type==2,:];
    
        # slightly change line segment to fit vanishing direction.
        # the last parameter controls strenght of fitting, here 0 means no
        # fitting, inf means line segments are forced to vp. Sometimes, fitting
        # could be helpful when big noise in line segment estimation.
        lines1rB = refitLineSegmentB(lines1, mainDirect[0,:], 0); 
        lines2rB = refitLineSegmentB(lines2, mainDirect[1,:], 0);
        lines3rB = refitLineSegmentB(lines3, mainDirect[2,:], 0);
    
        clines = np.row_stack((lines1rB,lines2rB,lines3rB));


    [ type, typeCost ] = assignVanishingType( lines, mainDirect[0:3,:], 0.1, 10 );
    lines1rB = lines[type==0,:];
    lines2rB = lines[type==1,:];
    lines3rB = lines[type==2,:];
    # clines = [lines1rB;lines2rB;lines3rB];

    l1 = OLLine()
    l1.line = lines1rB

    l2 = OLLine()
    l2.line = lines2rB

    l3 = OLLine()
    l3.line = lines3rB

    
    olines = [l1,l2,l3]
    return [ olines, mainDirect] 


def assignVanishingType( lines, vp, tol, area ):
    #ASSIGNVANISHINGTYPE Summary of this function goes here
    #   Detailed explanation goes here
    if area < 0:
        area = 10;
    

    numLine = lines.shape[0];
    numVP = vp.shape[0];
    typeCost = np.zeros([numLine, numVP]);
    # perpendicular 
    for vid in np.arange(numVP):
        cosint = np.sum( lines[:,0:3] * repmat( vp[vid,:], numLine ,1), 1);
        typeCost[:,vid] = np.arcsin(np.abs(cosint));
    
    # infinity
    for vid in np.arange(numVP):
        valid = np.ones([numLine,1] , dtype = bool);
       
        for i in np.arange(numLine):
            us = lines[i,4];
            ue = lines[i,5];
            u = np.array([us,ue])*2.0*np.pi-np.pi;
            v = CoordsTransform.computeUVN(lines[i,0:3], u, lines[i,3]);
            xyz = CoordsTransform.uv2xyzN(np.row_stack((u, v)).T, lines[i,3]);
            x = np.linspace(xyz[0,0],xyz[1,0],100);
            y = np.linspace(xyz[0,1],xyz[1,1],100);
            z = np.linspace(xyz[0,2],xyz[1,2],100);
            xyz = np.column_stack((x ,y ,z));
            xyz = xyz / repmat(np.sqrt(np.sum(xyz**2,1)),3, 1).T;
            ang = np.arccos( np.abs(np.sum(xyz * repmat(vp[vid,:], 100, 1), 1)));
            valid[i] = not any(ang<area*np.pi/180);
        
        typeCost[~valid[:,0],vid] = 100;
    

    I = np.min(typeCost,1);
    type = np.argmin(typeCost,1)
    type[I>tol] = numVP;

    return [ type, typeCost ] 


def findMainDirectionEMA( lines ):
    #FINDMAINDIRECTION compute vp from set of lines
    #   Detailed explanation goes here
    print('Computing vanishing point:\n');

    # arcList = [];
    # for i = 1:length(edge)
    #     panoLst = edge(i).panoLst;
    #     if size(panoLst,1) == 0
    #         continue;
    #     end
    #     arcList = [arcList; panoLst];
    # end

    ## initial guess
    segNormal = lines[:,0:3];
    segLength = lines[:,6];
    segScores = np.ones([lines.shape[0],1]);#lines(:,8);

    shortSegValid = segLength < 5*np.pi/180;
    segNormal = segNormal[~shortSegValid,:];
    segLength = segLength[~shortSegValid];
    segScores = segScores[~shortSegValid];

    numLinesg = segNormal.shape[0];
    [candiSet, tri] = icosahedron2sphere.icosahedron2sphere(3);
    ang = np.arccos(np.sum(candiSet[tri[0,0],:] * candiSet[tri[0,1],:])) / np.pi * 180;
    binRadius = ang/2;
    [ initXYZ, score, angle] = sphereHoughVote( segNormal, segLength, segScores, 2*binRadius, 2, candiSet,True );

    if  len(initXYZ) == 0:
        print('Initial Failed\n');
        mainDirect = [];
        return;
    

    print('Initial Computation: #d candidates, #d line segments\n', candiSet.shape[0], numLinesg);
    print('direction 1: #f #f #f\ndirection 2: #f #f #f\ndirection 3: #f #f #f\n', 
            initXYZ[0,0], initXYZ[0,1], initXYZ[0,2], initXYZ[1,0], initXYZ[1,1], initXYZ[1,2], initXYZ[2,0], initXYZ[2,1], initXYZ[2,2]);
    ## iterative refine
    iter_max = 3;
    [candiSet, tri] = icosahedron2sphere.icosahedron2sphere(5);
    numCandi = candiSet.shape[0];
    angD = np.arccos(np.sum(candiSet[tri[0,0],:] * candiSet[tri[0,1],:])) / np.pi * 180;
    binRadiusD = angD/2;
    curXYZ = initXYZ;
    tol = np.linspace(4*binRadius, 4*binRadiusD, iter_max); # shrink down #ls and #candi
    for iter in np.arange(iter_max):
        dot1 = np.abs(np.sum( segNormal* repmat(curXYZ[0,:], numLinesg, 1), 1));
        dot2 = np.abs(np.sum( segNormal* repmat(curXYZ[1,:], numLinesg, 1), 1));
        dot3 = np.abs(np.sum( segNormal* repmat(curXYZ[2,:], numLinesg, 1), 1));
        valid1 = dot1<np.cos((90-tol[iter])*np.pi/180);
        valid2 = dot2<np.cos((90-tol[iter])*np.pi/180);
        valid3 = dot3<np.cos((90-tol[iter])*np.pi/180);
        valid = valid1 | valid2 | valid3;
    
        if(sum(valid)==0):
            print('ZERO line segment for voting\n');
            break;
        
    
        subSegNormal = segNormal[valid,:];
        subSegLength = segLength[valid];
        subSegScores = segScores[valid];
    
        dot1 = np.abs(np.sum( candiSet* repmat(curXYZ[0,:], numCandi, 1), 1));
        dot2 = np.abs(np.sum( candiSet* repmat(curXYZ[1,:], numCandi, 1), 1));
        dot3 = np.abs(np.sum( candiSet* repmat(curXYZ[2,:], numCandi, 1), 1));
        valid1 = dot1>np.cos(tol[iter]*np.pi/180);
        valid2 = dot2>np.cos(tol[iter]*np.pi/180);
        valid3 = dot3>np.cos(tol[iter]*np.pi/180);
        valid = valid1 | valid2 | valid3;
    
        if(sum(valid)==0):
            print('ZERO candidate for voting\n');
            break;
        
       
        subCandiSet = candiSet[valid,:];
    
        [tcurXYZ,_,_]  = sphereHoughVote( subSegNormal, subSegLength, subSegScores, 2*binRadiusD, 2, subCandiSet ,True);
    
        if(len(tcurXYZ) == 0):
            print('NO answer found!\n');
            break;
        
        curXYZ = tcurXYZ;

        print('#d-th iteration: #d candidates, #d line segments\n', iter, subCandiSet.shape[0], len(subSegScores));

    
    print('direction 1: #f #f #f\ndirection 2: #f #f #f\ndirection 3: #f #f #f\n', 
            curXYZ[0,0], curXYZ[0,1], curXYZ[0,2], 
            curXYZ[1,0], curXYZ[1,1], curXYZ[1,2], 
            curXYZ[2,0], curXYZ[2,1], curXYZ[2,2]);
    mainDirect = curXYZ;

    mainDirect[0,:] = mainDirect[0,:]*np.sign(mainDirect[0,2]);
    mainDirect[1,:] = mainDirect[1,:]*np.sign(mainDirect[1,2]);
    mainDirect[2,:] = mainDirect[2,:]*np.sign(mainDirect[2,2]);

    uv = CoordsTransform.xyz2uvN(mainDirect,0);
    I1 = np.argmax(uv[1,:]);
    J = np.setdiff1d([0,1,2,], I1);
    I2 = np.argmin(np.abs(np.sin(uv[0,J])));
    I2 = J[I2];
    I3 = np.setdiff1d([0,1,2], [I1, I2]);
    mainDirect = np.row_stack((mainDirect[I1,:], mainDirect[I2,:], mainDirect[I3,:]));

    mainDirect[0,:] = mainDirect[0,:]*np.sign(mainDirect[0,2]);
    mainDirect[1,:] = mainDirect[1,:]*np.sign(mainDirect[1,1]);
    mainDirect[2,:] = mainDirect[2,:]*np.sign(mainDirect[2,0]);

    mainDirect = np.row_stack((mainDirect, -mainDirect));


    # score = 0;

    
    return  [ mainDirect, score, angle]




def sphereHoughVote( segNormal, segLength, segScores, binRadius, orthTolerance, candiSet, force_unempty ):
    #SPHEREHOUGHVOTE Summary of this function goes here
    #   Detailed explanation goes here


    if 'force_unempty' in locals() == False:
        force_unempty = true;

    ## initial guess
    # segNormal = arcList(:,1:3);
    # segLength = sqrt( sum((arcList(:,4:6)-arcList(:,7:9)).^2, 2));
    # segScores = arcList(:,end);
    numLinesg = segNormal.shape[0];

    # [voteBinPoints tri] = icosahedron2sphere(level);
    voteBinPoints = candiSet;
    voteBinPoints = voteBinPoints[voteBinPoints[:,2]>=0,:]; 
    reversValid = segNormal[:,2]<0;
    segNormal[reversValid,:] = -segNormal[reversValid,:];

    voteBinUV = CoordsTransform.xyz2uvN(voteBinPoints,0);
    numVoteBin = len(voteBinPoints);
    voteBinValues = np.zeros([numVoteBin,1]);
    for i in np.arange(numLinesg):
        tempNorm = segNormal[i,:];
        tempDots = np.sum(voteBinPoints * repmat(tempNorm, numVoteBin ,1),1);
    
    #     tempAngs = acos(abs(tempDots));
    #     voteBinValues = voteBinValues + normpdf(tempAngs, 0, 0.5*binRadius*pi/180)*segScores(i)*segLength(i);
    #     voteBinValues = voteBinValues + max(0, (2*binRadius*pi/180-tempAngs)./(2*binRadius*pi/180))*segScores(i)*segLength(i);
    
    
        valid = np.abs(tempDots)<np.cos((90-binRadius)*np.pi/180);

        voteBinValues[valid] = voteBinValues[valid] + segScores[i]*segLength[i];

    

    checkIDs1 = np.where(voteBinUV[1,:]>np.pi/3);
    checkIDs1 = checkIDs1[0]

    voteMax = 0;
    checkID1Max = 0;
    checkID2Max = 0;
    checkID3Max = 0;

    for j in np.arange(len(checkIDs1)):
    #     fprintf('#d/#d\n', j, length(checkIDs1));
        checkID1 = checkIDs1[j]; 
        vote1 = voteBinValues[checkID1];
        if voteBinValues[checkID1]==0 and force_unempty:
            continue;
        
        checkNormal = voteBinPoints[checkID1,:];
        dotProduct = np.sum(voteBinPoints * repmat(checkNormal, voteBinPoints.shape[0] , 1), 1);
        checkIDs2 = np.where(np.abs(dotProduct)<np.cos((90-orthTolerance)*np.pi/180));
        checkIDs2 = checkIDs2[0]
        for i in np.arange(len(checkIDs2)):
            checkID2 = checkIDs2[i];
            if voteBinValues[checkID2]==0 and force_unempty:
                continue;
            
            vote2 = vote1 + voteBinValues[checkID2];
            cpv = np.cross(voteBinPoints[checkID1,:], voteBinPoints[checkID2,:]);
            cpn = np.sqrt(np.sum(cpv**2));
            dotProduct = np.sum(voteBinPoints * repmat(cpv, voteBinPoints.shape[0] ,1), 1)/cpn;
            checkIDs3 = np.where(abs(dotProduct)>np.cos(orthTolerance*np.pi/180));  
            checkIDs3 = checkIDs3[0]

            for k in np.arange(len(checkIDs3)):
                checkID3 = checkIDs3[k]; 
                if voteBinValues[checkID3]==0 and force_unempty:
                    continue;
                
                vote3 = vote2 + voteBinValues[checkID3];
                if vote3>voteMax:
    #               print('#f\n', vote3);
                    lastStepCost = vote3-voteMax;
                    if voteMax != 0:
                        tmp = np.sum(voteBinPoints[[checkID1Max, checkID2Max, checkID3Max],:] * 
                                  voteBinPoints[[checkID1, checkID2, checkID3],:], 1);
                        lastStepAngle = np.arccos(tmp);
                    else:
                        lastStepAngle = [0 ,0, 0];
                    
                                     
                    checkID1Max = checkID1;
                    checkID2Max = checkID2;
                    checkID3Max = checkID3;               
                                           
                    voteMax = vote3;
                
    #             voteBins(checkID1, checkID2, checkID3) = true;
            
        
    

    if checkID1Max==0:
        print('Warning: No orthogonal voting exist!!!\n');
        refiXYZ = [];
        lastStepCost = 0;
        lastStepAngle = 0;
        return;
    
    initXYZ = voteBinPoints[[checkID1Max, checkID2Max, checkID3Max],:];

    ## refine
    # binRadius = binRadius/2;

    refiXYZ = np.zeros([3,3]);
    dotprod = np.sum(segNormal * repmat(initXYZ[0,:], segNormal.shape[0], 1), 1);
    valid = np.abs(dotprod)<np.cos((90-binRadius)*np.pi/180);
    validNm = segNormal[valid,:];

    segL = segLength[valid]
    validWt = np.reshape(segL,[segL.shape[0],1])*segScores[valid];
    validWt = validWt/np.max(validWt);
    _,refiNM = curveFitting(validNm, validWt);
    refiXYZ[0,:] = refiNM;

    dotprod = np.sum(segNormal * repmat(initXYZ[1,:], segNormal.shape[0], 1), 1);
    valid = np.abs(dotprod)<np.cos((90-binRadius)*np.pi/180);
    validNm = segNormal[valid,:];

    segl = segLength[valid]
    validWt = np.reshape(segl,[segl.shape[0],1])*segScores[valid];
    validWt = validWt/max(validWt);

    validNm = np.row_stack((validNm,refiXYZ[0,:]))
    validWt = np.row_stack((validWt,sum(validWt)*0.1))

    _,refiNM = curveFitting(validNm, validWt);
    refiXYZ[1,:] = refiNM;

    refiNM = np.cross(refiXYZ[0,:], refiXYZ[1,:]);
    refiXYZ[2,:] = refiNM/np.linalg.norm(refiNM,2);



    # [~,refiNM] = curveFitting(validNm, validWt);
    # refiXYZ(i,:) = refiNM;
    # 
    # 
    # 
    # for i = 1:3
    #     dotprod = dot(segNormal, repmat(initXYZ(i,:), [size(segNormal,1) 1]), 2);
    #     valid = abs(dotprod)<cos((90-binRadius)*pi/180);
    #     validNm = segNormal(valid,:);
    #     validWt = segLength(valid).*segScores(valid);
    #     [~,refiNM] = curveFitting(validNm, validWt);
    #     refiXYZ(i,:) = refiNM;
    # end
    ## output 
    # # [voteBinPoints tri] = icosahedron2sphere(level);
    # OBJ.vertices = voteBinPoints;
    # # OBJ.vertices_normal = zeros(size(voteBinPoints));
    # # OBJ.vertices_normal(:,1) = 1;
    # OBJ.vertices_normal = voteBinPoints;
    # uv = xyz2uvN(voteBinPoints);
    # OBJ.vertices_texture = [(uv(:,1)+pi)/2/pi (uv(:,2)+pi/2)/pi];
    # 
    # OBJ.objects.type = 'f';
    # OBJ.objects.data.vertices = tri;
    # OBJ.objects.data.texture = tri;
    # OBJ.objects.data.normal = tri;
    # 
    # # check boundary
    # newVTSID = size(OBJ.vertices_texture,1);
    # newFace = 0;
    # newAddVT = zeros(1000,2);
    # for i = 1:size(OBJ.objects.data.vertices,1)
    #     texture = OBJ.objects.data.texture(i,:);
    #     vt = OBJ.vertices_texture(texture,:);
    #     v = OBJ.vertices(texture,:);
    #     if (std(vt(:,1)))<0.3
    #         continue;
    #     end
    #     
    #     newFace = newFace + 1;
    #     
    #     modify = (vt(1,1)-vt(:,1))>0.5;
    #     vt(modify,1) = vt(modify,1)+1;
    #     modify = (vt(1,1)-vt(:,1))<-0.5;
    #     vt(modify,1) = vt(modify,1)-1;
    #     
    #     newAddVT((newFace-1)*3+1:newFace*3,:) = vt;
    #     OBJ.objects.data.texture(i,:) = [(newFace-1)*3+1:newFace*3] + newVTSID;
    #     
    #     if newFace>300
    #         fprintf('Warning: pre-assign more memory!/n');
    #     end
    # end
    # OBJ.vertices_texture = [OBJ.vertices_texture;newAddVT];
    # 
    # material(1).type='newmtl';
    # material(1).data='Textured';
    # material(2).type='Ka';
    # material(2).data=[1.0 1.0 1.0];
    # material(3).type='Kd';
    # material(3).data=[1.0 1.0 1.0];
    # material(4).type='Ks';
    # material(4).data=[1.0 1.0 1.0];
    # material(5).type='illum';
    # material(5).data=2;
    # material(6).type='map_Kd';
    # material(6).data='pano_hotel_2.jpg';
    # OBJ.material = material;
    # 
    # write_wobj(OBJ, 'exciting_notext.obj');

    # uv = xyz2uvN(voteBinPoints);
    # textureMap = zeros(512, 1024);
    # x = min(round((uv(:,1)+pi)/2/pi*1024+1), 1024);
    # y = 512 - min(round((uv(:,2)+pi/2)/pi*512+1), 512) + 1;
    # value = voteBinValues./max(voteBinValues);
    # value = value.^3;
    # # [gridX, gridY] = meshgrid(x,y);
    # for i = 1:length(x)
    #     sx = max(1, x(i)-3);
    #     ex = min(1024, x(i)+3);
    #     sy = max(1, y(i)-3);
    #     ey = min(512, y(i)+3);
    #     textureMap(sy:ey, sx:ex) = value(i);
    # end
    # 
    # # ind = sub2ind([512 1024], y, x);
    # # textureMap(ind) = voteBinValues;
    # imwrite(textureMap, 'exciting.png');

    

    return [ refiXYZ, lastStepCost, lastStepAngle ] 



def curveFitting( inputXYZ, weight ):
    #CURVEFITTING Summary of this function goes here
    #   Detailed explanation goes here
    len = np.sqrt(np.sum(inputXYZ**2,1));
    inputXYZ = inputXYZ / repmat(len,3 ,1).T;
    weightXYZ = inputXYZ * repmat(weight,1, 3);
    XX = sum(weightXYZ[:,0]**2);
    YY = sum(weightXYZ[:,1]**2);
    ZZ = sum(weightXYZ[:,2]**2);
    XY = sum(weightXYZ[:,0]*weightXYZ[:,1]);
    YZ = sum(weightXYZ[:,1]*weightXYZ[:,2]);
    ZX = sum(weightXYZ[:,2]*weightXYZ[:,0]);

    A = [[XX,XY,ZX],
        [XY,YY,YZ],
        [ZX,YZ,ZZ]];
    V,s,v =  np.linalg.svd(A);
    outputNM = V[:,-1].T;
    outputNM = outputNM/np.linalg.norm(outputNM,2);
    outputXYZ = [];

    
    return [ outputXYZ, outputNM ] 





def refitLineSegmentB( lines, vp, vpweight ):
    #REFITLINESEGMENTA refit direction of line segments 
    #   lines: original line segments
    #   vp: vannishing point
    #   vpweight: if set to 0, lines will not change; if set to inf, lines will
    #   be forced to pass vp
    if vpweight == None:
        vpweight = 0.1;
   

    numSample = 100;
    numLine = lines.shape[0];
    xyz = np.zeros([numSample+1,3]);
    wei = np.ones([numSample+1,1]); wei[numSample] = vpweight*numSample;
    lines_ali = lines;
    for i in np.arange(numLine):
        n = lines[i,0:3];
        sid = lines[i,4]*2*np.pi;
        eid = lines[i,5]*2*np.pi;
        if eid<sid:
            x = np.linspace(sid,eid+2*np.pi,numSample);
            x = (x % (2*np.pi));
    #         x = sid-1:(eid-1+numBins);
    #         x = rem(x,numBins) + 1;
        else:
            x = np.linspace(sid,eid,numSample);
        
    #     u = -pi + (x'-1)*uBinSize + uBinSize/2; 
        u = -np.pi + x.T;
        v = CoordsTransform.computeUVN(n, u, lines[i,3]);
        xyz[0:numSample,:] = CoordsTransform.uv2xyzN(np.column_stack((u ,v)), lines[i,3]);
        xyz[numSample,:] = vp;
        _, outputNM = curveFitting( xyz, wei );
        lines_ali[i,0:3] = outputNM;
    

    return lines_ali



