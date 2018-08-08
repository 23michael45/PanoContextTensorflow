import numpy as np
def line_equation_from_two_points(p1=None, p2=None):
    # function lineeq = line_equation_from_two_points(p1, p2)
    # returns line parameters(ax+by+c=0) that passes through p1 and p2
    # p1, p2: [2x1] (x,y)
    # lineeq: [3x1] (a,b,c)

    lineeq = np.row_stack([p2[1] - p1[1], p1[0] - p2[0], - p1[0] * p2[1] + p2[0] * p1[1]])
    lineeq = lineeq / np.linalg.norm(lineeq[0:2],2)

    # normalize sign to be consistent... (0,0) should have a positive distance
    distzero = np.dot(np.array([0, 0, 1]) , lineeq)
    if np.abs(distzero) > 1e-10:
        lineeq = lineeq * np.sign(distzero)
    return lineeq

def distance_of_point_to_line(line=None, x=None):
    # function d = distance_of_point_to_line(line, x)
    # 
    # returns d = ax+by+c
    # line: [3x1] (a,b,c)
    # x: [2x1] (x,y)
    x = x.flatten();
    xh = np.append(x, 1)
    d = np.dot(line[:].T , xh)
    return d

def line_intersect(p1, p2, p3, p4):
    # function pt = line_intersect(p1, p2, p3, p4)
    # intersection point of line defined by p1 & p2 and line defined by p3 & p4
    # http://local.wasp.uwa.edu.au/~pbourke/geometry/lineline2d/
    x1 = p1[0]
    y1 = p1[1]; 

    x2 = p2[0]
    y2 = p2[1]; 

    x3 = p3[0]
    y3 = p3[1];

    x4 = p4[0]
    y4 = p4[1]; 


    if (y4 - y3) * (x2 - x1) - (x4 - x3) * (y2 - y1) == 0:
        #       warning('line_intersect.m degenerate --dclee');
        #       pt = (p1 + p2 + p3 + p4)/4;
        pt = []
        degen = 1
        return
    

    pt = np.row_stack([x1 + (x2 - x1) * ((x4 - x3) * (y1 - y3) - (y4 - y3) * (x1 - x3)) / ((y4 - y3) * (x2 - x1) - (x4 - x3) * (y2 - y1)),
              y1 + (y2 - y1) * ((x4 - x3) * (y1 - y3) - (y4 - y3) * (x1 - x3)) / ((y4 - y3) * (x2 - x1) - (x4 - x3) * (y2 - y1))])
    degen = 0
    return pt,degen
def dist_line_to_point(linep1=None, linep2=None, p=None):
    # compute distance of point to line
    # line is defined by 2 points

    n = norm(linep1 - linep2)

    if n < 1e-10:
        warning(mstring('dist_line_to_point.m: degenerate input'))
    

    d = np.abs((linep1[0] - linep2[0]) * (linep1[1] - p[1]) - (linep1[1] - linep2[1]) * (linep1[0] - p[0])) / n

    return d