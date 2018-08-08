import numpy as np
import VanishingPoint.Geometry as VGE
def line_belongto_vp(lines=None,vp=None,THRES_THETA=None):
   
    # find lines that belong to the vanishing point
    
    angle=np.zeros([len(lines)])
    for k in np.arange(len(lines)):
        angle[k]=anglebetween(lines[k],vp)
    
    lineclass=angle < THRES_THETA
    return lineclass,angle
    
def anglebetween(line=None,targetpoint=None):

    # get angle between targetpoint & line direction
    
    if not any(np.isreal(targetpoint)):
        INFDIST=10000000.0
        targetpoint=(INFDIST * targetpoint) / i
    
    
    midpoint=(line.point1 + line.point2) / 2
    v1=targetpoint - midpoint
    v2=line.point2 - midpoint

    theta=180 / np.pi * np.real(np.arccos(np.sum(v1*v2.T) / np.linalg.norm(v1,2) / np.linalg.norm(v2,2)))

    if theta > 90:
        theta=180 - theta
    return theta


def is_line_above_horizon(lines, vp):
    #
    # adds lines(i).above_horizon:  1 if above, -1 if below, 0 if neither
    # horizon: [3x1] (a,b,c) parameters of the equation of the horizon in 2D where ax+by+c=0

    #% horizon is the line connecting vp{2} and vp{3}
    horizon = VGE.line_equation_from_two_points(vp[1], vp[2])

    if VGE.distance_of_point_to_line(horizon, np.row_stack([0 , -10000000])) < 0:# if a really high point is not above horizon
        horizon = -horizon# flip sign of horizon
    

    #% test all lines
    for i in np.arange(len(lines)):
        l1 = np.sign(VGE.distance_of_point_to_line(horizon, lines[i].point1))
        l2 = np.sign(VGE.distance_of_point_to_line(horizon, lines[i].point2))
        if l1 == 1 and l2 == 1:
            lines[i].above_horizon = 1
        elif l1 == -1 and l2 == -1:
            lines[i].above_horizon = -1
        else:
            lines[i].above_horizon = 0
        
    return lines,horizon


def  is_line_leftorright(lines, vp):
    #
    # left or right of vp{3}.. vp{3} is the vp close to image center
    # adds lines(i).leftorright:  1 if right, -1 if left, 0 if neither
    # horizon: [3x1] (a,b,c) parameters of the equation of the horizon in 2D where ax+by+c=0

    #% centerline is the line connecting vp{1} and vp{3}
    centerline = VGE.line_equation_from_two_points(vp[0], vp[2])

    if VGE.distance_of_point_to_line(centerline, np.row_stack([10000000, 0])) < 0:# if a really right point is negative
        centerline = -centerline# flip sign of horizon
    

    #% test all lines
    for i in np.arange(len(lines)):
        l1 = np.sign(VGE.distance_of_point_to_line(centerline, lines[i].point1))
        l2 = np.sign(VGE.distance_of_point_to_line(centerline, lines[i].point2))
        if l1 == 1 and l2 == 1:
            lines[i].leftorright = 1
        elif l1 == -1 and l2 == -1:
            lines[i].leftorright = -1
        else:
            lines[i].leftorright = 0
        
    return lines    

def compute_lineeq(lines):

    for i in np.arange(len(lines)):
	
	    lineeq = VGE.line_equation_from_two_points(lines[i].point1, lines[i].point2);
	
	    if lines[i].lineclass == 1:
		    if VGE.distance_of_point_to_line(lineeq, np.array([0, -10000000])) < 0:
			    lineeq = -lineeq;
		    
	    else:
		    if VGE.distance_of_point_to_line(lineeq, np.array([10000000 ,0])) < 0:
			    lineeq = -lineeq;
		    
	    
	
	    lines[i].lineeq = lineeq;
    return lines 