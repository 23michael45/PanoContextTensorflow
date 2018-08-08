import numpy as np
from numpy.matlib import repmat
from matplotlib import path
import matplotlib.pyplot as plt
class Sample:
    n_sample = []
    sample = []
    lineclass = []
def orient_from_lines(lines, vp, imgwidth, imgheight):


    ##
    ls = sample_line(lines);

    linesamples = [];
    linesampleclass = [];
    for i in np.arange(len(ls)):
        if i==0:
            linesamples = np.array(ls[i].sample).T
            linesampleclass = np.array(ls[i].lineclass);       
        else:
            linesamples = np.concatenate((linesamples,np.array(ls[i].sample).T),axis = 0);
            linesampleclass = np.concatenate((linesampleclass,np.array(ls[i].lineclass)));



    ##
    # lineextimg = cell(3,3);
    # for i=1:9, lineextimg{i} = zeros(imgheight,imgwidth); end
    lineextimg = []
    for i in np.arange(18):
        lineextimg.append( np.zeros([np.int(imgheight),np.int(imgwidth)]))

    lineextimg = np.array(lineextimg).reshape(3,3,2,np.int(imgheight),np.int(imgwidth))
    ##
    # poly = extend_line(line, vp{1}, stoppinglines_sample, imgwidth, imgheight);
    for i in np.arange(len(lines)):
        lc = lines[i].lineclass;
        if lc != 0:
            for extdir in np.setdiff1d(np.arange(1,4), lc):
                targetdir = np.setdiff1d(np.arange(1,4), np.array([lc, extdir]));
    
        #         poly = extend_line_old(lines(i), vp{extdir}, linesamples(linesampleclass==targetdir,:), imgwidth, imgheight);
        #         lineextimg{lc,extdir} = lineextimg{lc,extdir} + poly2mask(poly(:,1), poly(:,2), imgheight, imgwidth);
        
                trueSampleLine = linesamples[np.where(linesampleclass.flatten()==targetdir)[0],:]
                poly = extend_line(lines[i], vp[extdir-1], 1,trueSampleLine , imgwidth, imgheight);

                #arr = np.array([lc-1,extdir-1,0])
                #idx = np.ravel_multi_index(arr, (3,3,2),order='F')
                

                lineextimg[lc-1,extdir-1,0] = lineextimg[lc-1,extdir-1,0] + poly2mask(poly, imgheight, imgwidth);
                




                poly = extend_line(lines[i], vp[extdir-1], -1,trueSampleLine, imgwidth, imgheight);

                lineextimg[lc-1,extdir-1,1] = lineextimg[lc-1,extdir-1,1] + poly2mask(poly, imgheight, imgwidth);
            
            
        
    return lineextimg 

def poly2mask(poly,height,width):
    # Create vertex coordinates for each grid cell...
    # (<0,0> is at the top left of the grid in this system)
    x, y = np.meshgrid(np.arange(height), np.arange(width))
    x, y = x.flatten(), y.flatten()

    points = np.vstack((x,y)).T

    
    poly_path = path.Path(poly)
    grid = poly_path.contains_points(points);

    grid = grid.reshape([np.int(width),np.int(height)])

    mask = np.zeros([np.int(width),np.int(height)])
    mask = np.where(grid,1,0)

    return mask

##
def extend_line(line, vp, towards_or_away, stoppinglines_sample, imgwidth, imgheight):
    # towards_or_away: 1 or -1

    p1 = line.point1;
    p2 = line.point2;

    curp1 = p1; curp2 = p2;
    move_amount = 128;
    while move_amount>=1:
        [newp1, newp2, atvp] = move_line_towards_vp(curp1, curp2, vp, towards_or_away * move_amount);
    
        failed = 0;
        if atvp==1:
    #         move_amount = 0; # exit now.
            failed = 1;
        
        elif (newp1[0]>imgwidth or newp1[0]<1 or newp1[1]>imgheight or newp1[1]<1) and (newp2[0]>imgwidth or newp2[0]<1 or newp2[1]>imgheight or newp2[1]<1):
            failed = 1;
        
        else:

            poly_path = path.Path([p1, p2, newp2, newp1, p1])
            isstop = poly_path.contains_points(stoppinglines_sample);
        
            if any(isstop):
                failed = 1;
            
        
    
        if failed:
            move_amount = move_amount/2;
        else:
            curp1 = newp1;
            curp2 = newp2;
        
    
    # poly = [curp1(:)'; curp2(:)'];
    poly = [p1[:].T, p2[:].T, curp2[:].T, curp1[:].T];

    # curp1 = p1; curp2 = p2;
    # move_amount = 32;
    # while move_amount>=1
    #     [newp1 newp2 atvp] = move_line_towards_vp(curp1, curp2, vp, -move_amount);
    #     
    #     failed = 0;
    #     if atvp==1
    #         move_amount = 0; # exit now.
    #         
    #     elseif (newp1(1)>imgwidth or newp1(1)<1 or newp1(2)>imgheight or newp1(2)<1) && ...
    #        (newp2(1)>imgwidth or newp2(1)<1 or newp2(2)>imgheight or newp2(2)<1)
    #         failed = 1;
    #         
    #     else
    #         isstop = inpolygon(stoppinglines_sample(:,1), stoppinglines_sample(:,2), ...
    #             [p1(1) p2(1) newp2(1) newp1(1) p1(1)], [p1(2) p2(2) newp2(2) newp1(2) p1(2)]);
    #         
    #         if any(isstop)
    #             failed = 1;
    #         end
    #     end
    #     
    #     if failed
    #         move_amount = move_amount/2;
    #     else
    #         curp1 = newp1;
    #         curp2 = newp2;
    #     end
    # end
    # poly = [poly; curp2(:)'; curp1(:)'];
    # poly = [poly; poly(1,:)];

    ##

    ##
    return poly

def move_line_towards_vp(linep1, linep2, vp, amount):

    # d = dist_line_to_point(linep1, linep2, vp);
    # r = amount / d;
    n1 = np.linalg.norm(vp-linep1,2);
    n2 = np.linalg.norm(vp-linep2);
    dir1 = (vp - linep1) / n1;
    dir2 = (vp - linep2) / n2;
    ratio21 = n2 / n1;

    # if n1>amount && n2<amount
    #     fprintf('check');
    # end

    if n1 < amount:
        newp1 = linep1;
        newp2 = linep2;
        atvp = 1;
    else:
        newp1 = linep1 + dir1 * amount;
        newp2 = linep2 + dir2 * amount * ratio21;
        atvp = 0;
    return newp1 ,newp2 ,atvp

##
def move_line_towards_vp_old(linep1, linep2, vp, amount):

    d = dist_line_to_point(linep1, linep2, vp);
    r = amount / d;
    if r > 1:
        newp1 = vp;
        newp2 = vp;
        atvp = 1;
    else:
        newp1 = linep1 + (vp-linep1)*r;
        newp2 = linep2 + (vp-linep2)*r;
        atvp = 0;
    return newp1 ,newp2 ,atvp
def sample_line(lines):

    sample_rate = 1; # sample every 5 pixel on line

    # build intermediate datastructure ls

    linenum = len(lines)
    ls = np.empty([linenum],dtype=np.object);
    for i in np.arange(linenum):
        ls[i] = Sample()

    for i in np.arange(len(lines)):
        n_sample = np.ceil( np.linalg.norm(lines[i].point1-lines[i].point2,2) / sample_rate );
        ls[i].n_sample = n_sample;

        ls[i].sample = [ np.linspace(lines[i].point1[0], lines[i].point2[0], n_sample).T,
		    np.linspace(lines[i].point1[1], lines[i].point2[1], n_sample).T ];

        ls[i].lineclass = repmat(lines[i].lineclass, np.int(n_sample), 1);
    return ls
