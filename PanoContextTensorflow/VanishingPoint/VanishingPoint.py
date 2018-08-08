import VanishingPoint.VPPrivate as VVPR
import numpy as np
import VanishingPoint.Geometry as VGEO
import RoomHypothesisSampling
def taglinesvp(vp, lines):

    lines = assign_lineclass(lines, vp);

    lines = compute_line_attributes(lines, vp);

    lines_ex = expand_ambiguous_lineclass(lines);

    lines_ex = compute_line_attributes(lines_ex, vp);
    return [lines, lines_ex]

def assign_lineclass(lines, vp):

    THRES_THETA = 10;
    lineclass1 = [];
    lineclass2 = [];
    lineclass3 = [];
    if len(vp)>=1 and any(vp[0]) != None:
        lineclass1,_ = VVPR.line_belongto_vp(lines, vp[0], THRES_THETA);
    else:
        lineclass1 = np.zeros([1, len(lines)]);
    
    if len(vp)>=2 and any(vp[1]) != None:
        lineclass2,_ = VVPR.line_belongto_vp(lines, vp[1], THRES_THETA);
    else:
        lineclass2 = zeros(1, length(lines));
    
    if len(vp)>=3 and any(vp[2]) != None:
        lineclass3,_ = VVPR.line_belongto_vp(lines, vp[2], THRES_THETA);
    else:
        lineclass3 = np.zeros([1, len(lines)]);
    


    for i in np.arange(len(lines)):
        lines[i].lineclass1 = lineclass1[i];
        lines[i].lineclass2 = lineclass2[i];
        lines[i].lineclass3 = lineclass3[i];
        
        add = np.where(lineclass1[i] ,1,0) + np.where(lineclass2[i] ,1,0) + np.where(lineclass3[i] ,1,0)

        if add == 1:
            lines[i].lineclass = 1*lineclass1[i] + 2*lineclass2[i] + 3*lineclass3[i];
        else:
            lines[i].lineclass = 0;

    return lines 

def compute_line_attributes(lines, vp):

    #vp = np.array(np);
    ## determine if lines are above the horizon or below the horizon
    [lines, horizon] = VVPR.is_line_above_horizon(lines, vp);
    # disp_lines(rectimg, lines, [lines(:).above_horizon]);

    ## determine if lines are left or right of the center vanishing point
    lines = VVPR.is_line_leftorright(lines, vp);
    # disp_lines(rectimg, lines, [lines(:).leftorright]);

    ## let lines carry their own ID
    for i in np.arange(lines.shape[0]):
       lines[i].id = i; 

    ## compute 2D line equations for all lines
    lines = VVPR.compute_lineeq(lines);

    ##
    # lines = is_vertline_outside(lines);
    return  lines 
def expand_ambiguous_lineclass(lines):
    # when lines belong to more than 1 class,
    # hallucinate line for each class.

    ##
    num_lines = len(lines);

    lines_expand = np.empty([num_lines * 2],dtype=np.object);
    for i in np.arange(num_lines*2):
        lines_expand[i] = RoomHypothesisSampling.Line()
        lines_expand[i].point1 = np.array([0 ,0]);
        lines_expand[i].point2 = np.array([0 ,0]);
        lines_expand[i].lineclass = 0;
    ##
     
   


    count = 0;

    ##
    for i in np.arange(num_lines):
        if lines[i].lineclass1 == 1:
            count = count + 1;
            lines_expand[count].point1 = lines[i].point1;
            lines_expand[count].point2 = lines[i].point2;
            lines_expand[count].lineclass = 1;
        
        if lines[i].lineclass2 == 1:
            count = count + 1;
            lines_expand[count].point1 = lines[i].point1;
            lines_expand[count].point2 = lines[i].point2;
            lines_expand[count].lineclass = 2;
        
        if lines[i].lineclass3 == 1:
            count = count + 1;
            lines_expand[count].point1 = lines[i].point1;
            lines_expand[count].point2 = lines[i].point2;
            lines_expand[count].lineclass = 3;
        
        if lines[i].lineclass1==0 and lines[i].lineclass2==0 and lines[i].lineclass3==0:
            count = count + 1;
            lines_expand[count].point1 = lines[i].point1;
            lines_expand[count].point2 = lines[i].point2;
            lines_expand[count].lineclass = 0;
        
    

    lines_expand = lines_expand[0:count];
    return lines_expand 
