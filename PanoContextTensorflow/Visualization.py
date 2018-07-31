from scipy.misc import imread, imsave, imresize
import CoordsTransform
import numpy as np
import PIL.Image
def paintParameterLine(parameterLine, width, height, img ):
    #PAINTPARAMETERLINE Paint parameterized line
    #   parameterLine: [n1,n2,n3,planeID,u1,u2], first 3 dims are normal
    #   direction, 4th dim is the base plane ID for U (1=XY), 5-6th dims are U
    #   of start and end point.
    #   width, height: the size of output panorama, width = height x 2
    #   img: the image on which lines will be drawn. If no img,
    #   img=zeros(height,width).
    lines = parameterLine;

    if img is None:
        panoEdgeC = np.zeros([height, width]);
    else:
        img = np.double(img);
        panoEdgeC = imresize(img, [height, width]);
        if img.shape[2]==3:
            
            pimg = PIL.Image.fromarray(np.uint8(panoEdgeC))
            panoEdgeC = pimg.convert('L')
            
            panoEdgeC = np.array(panoEdgeC) * 0.5
        
    
    # valid = true(size(lines,1),1);
    # uv_vp = xyz2vp([vp;-vp]);
    # vpm = min(floor( (uv_vp(:,1)-(-pi)) /(2*pi)*width)+1, width);
    # vpn = min(floor( ((pi/2)-uv_vp(:,2))/(pi)*height )+1, height);
    # valid = lines(:,4)==1;
    # lines = lines(~valid,:); #horizontal
    # lines = lines(valid,:); #vertical
    num_sample = max(height,width);
    for i in np.arange(lines.shape[0]):
    #     fprintf('#d\n',i);
        n = lines[i,0:3];
        sid = lines[i,4]*2*np.pi;
        eid = lines[i,5]*2*np.pi;
        if eid<sid:
            x = np.linspace(sid,eid+2*np.pi,num_sample);
            x = x%(2*np.pi);
    #         x = sid-1:(eid-1+numBins);
    #         x = rem(x,numBins) + 1;
        else:
            x = np.linspace(sid,eid,num_sample);
        
    #     u = -pi + (x'-1)*uBinSize + uBinSize/2; 
        u = -np.pi + x.T;
        v = CoordsTransform.computeUVN(n, u, lines[i,3]);
        xyz = CoordsTransform.uv2xyzN(np.column_stack((u, v)), lines[i,3]);
        uv = CoordsTransform.xyz2uvN( xyz, 0);
    
        uv = uv.T

        m = np.minimum(np.floor( (uv[:,0]-(-np.pi)) /(2*np.pi)*width), width - 1);
        n = np.minimum(np.floor( ((np.pi/2)-uv[:,1])/(np.pi)*height ), height - 1);
        #drawId = sub2ind([height, width], n, m);

        panoEdgeC[np.int32(n),np.int32(m)] = 255;
    

    

    return panoEdgeC

