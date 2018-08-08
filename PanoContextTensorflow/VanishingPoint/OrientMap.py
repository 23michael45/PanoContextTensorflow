import numpy as np
import VanishingPoint.OMPrivate as VOPR
import matplotlib.pyplot as plt
def compute_omap(lines, vp, imgsize):

    # global OMAP_FACTOR; 
    # OMAP_FACTOR = 250 / norm(imgsize(1:2));
    OMAP_FACTOR = 1;

    [omap, omapstrict1 ,omapstrict2] = compute_orientationmap(lines, vp, imgsize[0:2], OMAP_FACTOR);
    # omapstrict1, omapstrict2 are more conservative estimates.
    # They should be more reliable but have more uncertain regions.

    return  omap, OMAP_FACTOR

def compute_orientationmap(lines, vp, imgsize, OMAP_FACTOR):


    ## resize lines
    lines_rsz = lines;
    for i in np.arange(len(lines)):
        lines_rsz[i].point1 = lines_rsz[i].point1 * OMAP_FACTOR;
        lines_rsz[i].point2 = lines_rsz[i].point2 * OMAP_FACTOR;
        lines_rsz[i].lineeq = lines_rsz[i].lineeq * OMAP_FACTOR;
    
    vp[0] = vp[0] * OMAP_FACTOR;
    vp[1] = vp[1] * OMAP_FACTOR;
    vp[2] = vp[2] * OMAP_FACTOR;
    omapsize = np.ceil(imgsize * OMAP_FACTOR);

    ##
    lineextimg = VOPR.orient_from_lines(lines_rsz, vp, omapsize[1], omapsize[0]);

                
  

    ##
    ao23 = (lineextimg[1,2,0] + lineextimg[1,2,1]);
    ao32 = (lineextimg[2,1,0] + lineextimg[2,1,1]);
    ao13 = (lineextimg[0,2,0] + lineextimg[0,2,1]);
    ao31 = (lineextimg[2,0,0] + lineextimg[2,0,1]);
    ao12 = (lineextimg[0,1,0] + lineextimg[0,1,1]);
    ao21 = (lineextimg[1,0,0] + lineextimg[1,0,1]);
    aa23 = (lineextimg[1,2,0] * lineextimg[1,2,1]);
    aa32 = (lineextimg[2,1,0] * lineextimg[2,1,1]);
    aa13 = (lineextimg[0,2,0] * lineextimg[0,2,1]);
    aa31 = (lineextimg[2,0,0] * lineextimg[2,0,1]);
    aa12 = (lineextimg[0,1,0] * lineextimg[0,1,1]);
    aa21 = (lineextimg[1,0,0] * lineextimg[1,0,1]);

    ## regular
    # a{1} = (lineextimg{2,3,1} or lineextimg{2,3,2}) and (lineextimg{3,2,1} or lineextimg{3,2,2});
    # a{2} = (lineextimg{1,3,1} or lineextimg{1,3,2}) and (lineextimg{3,1,1} or lineextimg{3,1,2});
    # a{3} = (lineextimg{1,2,1} or lineextimg{1,2,2}) and (lineextimg{2,1,1} or lineextimg{2,1,2});
    a = np.zeros([3,lineextimg.shape[3],lineextimg.shape[4]])
    a[0] = ao23 * ao32;
    a[1] = ao13 * ao31;
    a[2] = ao12 * ao21;

    b = np.zeros([lineextimg.shape[3],lineextimg.shape[4],3])
    b[:,:,0] = a[0] * -(a[1]-1) * -(a[2]-1);
    b[:,:,1] = -(a[0]-1) * a[1] * -(a[2]-1);
    b[:,:,2] = -(a[0]-1) * -(a[1]-1) * a[2];

    omap = b;

   
    ## AND thing
    # a[1] = lineextimg[2,3,1] and lineextimg[2,3,2] and lineextimg[3,2,1] and lineextimg[3,2,2];
    # a[2] = lineextimg[1,3,1] and lineextimg[1,3,2] and lineextimg[3,1,1] and lineextimg[3,1,2];
    # a[3] = lineextimg[1,2,1] and lineextimg[1,2,2] and lineextimg[2,1,1] and lineextimg[2,1,2];
    a = np.zeros([3,lineextimg.shape[3],lineextimg.shape[4]])
    a[0] = aa23 * aa32;
    a[1] = aa13 * aa31;
    a[2] = aa12 * aa21;
    
    b = np.zeros([lineextimg.shape[3],lineextimg.shape[4],3])
    b[:,:,0] = a[0] * -(a[1]-1) * -(a[2]-1);
    b[:,:,1] = -(a[0]-1) * a[1] * -(a[2]-1);
    b[:,:,2] = -(a[0]-1) * -(a[1]-1) * a[2];

    omapstrict2 = b

    ## between
    a = np.zeros([3,lineextimg.shape[3],lineextimg.shape[4]])
    a[0] = (ao23 * aa32) + (aa23 * ao32);
    a[1] = (ao13 * aa31) + (aa13 * ao31);
    a[2] = (ao12 * aa21) + (aa12 * ao21);
    
    b = np.zeros([lineextimg.shape[3],lineextimg.shape[4],3])
    b[:,:,0] = a[0] * -(a[1]-1) * -(a[2]-1);
    b[:,:,1] = -(a[0]-1) * a[1] * -(a[2]-1);
    b[:,:,2] = -(a[0]-1) * -(a[1] * a[2]-1);

    omapstrict1 = b


    return omap, omapstrict1, omapstrict2