
import numpy as np
import matplotlib.pyplot as plt
from scipy.misc import imread, imsave, imresize
import PIL.Image


if __name__ == '__main__':
   
   
    '''
    a = [[1,2,3,4],
         [5,6,7,8],
         [9,10,11,12],
         [13,14,15,16]]
    print(np.linalg.norm(a,2))
    print(np.linalg.norm(a))

    x1 = np.arange(4).reshape((4, 1))
    x2 = np.arange(4).reshape((1, 4))
    m44 = np.dot(x1, x2)
    m1 = np.dot(x2,x1)
    print(m44)
    print(m1)


    x = [np.pi /4,2,3,4]
    x = np.tan(x)
    
    y = x[1]

    from scipy import interpolate
    x = np.arange(-5.01, 5.01, 0.25)
    y = np.arange(-5.01, 5.01, 0.25)
    xx, yy = np.meshgrid(x, y)
    z = np.sin(xx**2+yy**2)
    f = interpolate.interp2d(x, y, z, kind='cubic')


    xnew = np.arange(-5.01, 5.01, 1e-2)
    ynew = np.arange(-5.01, 5.01, 1e-2)
    znew = f(xnew, ynew)
    
    #plt.plot(x, z[:, 0], 'ro-', xnew, znew[:, 0], 'b-')
    #plt.show()

  
    N = 10
    p = 0.5
    l1 = np.random.choice(a = [False,True],size = N,p=[p, 1-p]) 
    l2 = np.random.choice(a = [False,True],size = (N,1),p=[p, 1-p])

    l = l1 & l2
    print(l)


        

    
    SrcImage = './data/pano_room.jpg'
    testimg = imread(SrcImage, mode='RGB')
    testimggray = PIL.Image.fromarray(testimg)
    testimggray = testimggray.convert('L')
    testimggray = np.array(testimggray)
    
    plt.subplot(3, 1, 1)
    plt.imshow(testimggray, cmap='gray')


    
    plt.subplot(3, 1, 2)
    testimg = np.ones([512, 1024]);
    testimg = np.uint8(testimg)*255

    testimg[:,:] = testimggray[0:testimg.shape[0],0:testimg.shape[1]]

    plt.imshow(testimg,cmap='gray')

    
    plt.subplot(3, 1, 3)
    testimg[0:256,0:256] = np.random.randint(0,1,[256,256])
    
    testimg = np.ones([512, 1024]) * 255;
    testimg[0,0] = 254
    #testimg[0:1,0:1] = np.random.randint(0,1,[1,1])
    #testimg = np.random.randint(0,149,[512,1024])
    plt.imshow(testimg,cmap='gray')

    
    #testimggray = PIL.Image.fromarray(testimg)
    imsave('./data/save.jpg',testimg)

    plt.show()
    '''

    mat = np.zeros([16,16])

    x = [1,2,5,6]
    y = [3,4,5,6]
    z = np.array([x,y])

    mat[[x,y]] = 99;

    print(mat)
    
    print(x[3:0:-1])
    print(x[::-1])
    print(z[...,::-1])
    print(z[::-1,...])

    plt.scatter(x,y)
    
    plt.scatter(x,y, 100, 'r','o');
    #plt.show()
    import PanoContextTensorflow
    PanoContextTensorflow.process()