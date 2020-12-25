from cv2 import cv2
import numpy as np
import matplotlib.pyplot as plt

def tem(path):
    '''
    该函数功能是生成一个用于matchshape的轮廓，返回值是一个螺丝轮廓。
    参数说明，
    path：模板图片，要求只有一个螺丝目标，背景无干扰，不然影响后续匹配精度
    '''
    tem = cv2.imread(path)
    #  灰度化
    gray_tem = cv2.cvtColor(tem,cv2.COLOR_BGR2GRAY)
    # mask = np.zeros(gray_tem.shape,np.uint8)
    #  定义一个101尺寸的矩形核，用于减弱光照不均匀的影响。这个核的尺寸要足够大，小了不行
    k = np.ones((101,101),np.uint8)
    #  定义一个15尺寸的矩形核，用于滤波
    k1 = np.ones((15,15),np.uint8)
    #  对灰度图进行开操作，将局部很亮的区域扩大，很暗的地方基本不变。这类似在二值图上进行开操作，只不过在这里是在灰度图上
    #  操作，效果就是扩大亮区域，对于暗区域影响不大
    opend_tem = cv2.morphologyEx(gray_tem,cv2.MORPH_OPEN,k)
    #  将灰度图减去开操作之后的图，相当于亮的减亮的，暗的减暗的，减完之后整张图灰度值就均匀了，为了相减之后灰度值不至于太小
    #  加一个偏量100
    add_tem = cv2.addWeighted(gray_tem,1,opend_tem,-1,100)
    #  阈值化
    _,binary_tem =cv2.threshold(add_tem,180,255,cv2.THRESH_BINARY)
    #  用开操作去噪
    opening_tem = cv2.morphologyEx(binary_tem,cv2.MORPH_OPEN,k1)
    _,contours,_ = cv2.findContours(opening_tem,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)

    n = len(contours)
    s = []
    #  将轮廓面积值放入列表s中
    for i in range(n):
        s.append(cv2.contourArea(contours[i]))
    #  为了防止去噪后依然有一些小块干扰，通过一个循环只返回轮廓面积最大的那个作为模板
    for i in range(n):
        if s[i] == max(s):
            #  这一步是为了查看轮廓图片，可不要
            # mask = cv2.drawContours(mask,contours,i,(255,255,255),-1)
            return contours[i]
    

def detect(gray,o,tem):
    '''
    该函数用于将待匹配图片与模板图片进行比对，返回检测成功的目标图片和目标的最小矩形。
    参数说明，
    gray，待匹配图片的灰度图像
    o，原图
    tem，模板轮廓
    '''
    #  二值化
    _,binary = cv2.threshold(gray,210,255,cv2.THRESH_BINARY)
    #  用开操作滤波
    k = np.ones((8,8),np.uint8)
    opening = cv2.morphologyEx(binary,cv2.MORPH_OPEN,k)
    #  找图片中轮廓
    _,contours,_ = cv2.findContours(opening,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    n = len(contours)
    #  存储形状匹配成功的轮廓面积
    s = []
    #  存储形状匹配成功轮廓的索引值
    b = []
    #  存储最终匹配成功的目标最小矩形
    rects = []
    #  存储轮廓的所有面积值
    s1 = []
    rets = []
    for i in range(n):
        s1.append(cv2.contourArea(contours[i]))

    for i in range(n):
        #  形状匹配
        ret = cv2.matchShapes(tem,contours[i],1,0)
        rets.append(ret)
        # print('第%d个轮廓形状匹配度：%f\n'%(i,ret))
        # print('第%d个轮廓形状面积：%f\n'%(i,cv2.contourArea(contours[i])))
        #  设置一个阈值，小于该阈值则认为与模板形状相似，可用于下一步比较。该阈值需要通过实验得到。这一步是粗过滤
        #  可以将那些形状明显不相符的目标过滤掉
        if ret < 5.4:
            s.append(cv2.contourArea(contours[i]))
            b.append(i)
    print('看一看过滤之后的轮廓索引和面积：\n')
    print(b)
    print(s)
    print('看一看过滤之前的轮廓匹配度和面积：\n')
    print(rets)
    print(s1)    

    #  设置面积判别的上下限。该上下限的值也是需要实验得到
    maxarea = max(s)
    area1 = maxarea-4200
    area2 = maxarea+2000
    #  精过滤，利用面积来筛选目标。因为上面已经粗过滤了一次，所以这里循环的次数就是s的长度
    for i in range(len(s)):
        #  轮廓在面积上下限之间的才被认为是目标
        if s1[b[i]] > area1 and s1[b[i]] < area2:
            rect = cv2.minAreaRect(contours[b[i]])
            #  rect格式不是cv2.drawContours要求的格式，所以需要用cv2.boxPoints将其转换一下
            points = cv2.boxPoints(rect)
            #  将其转化为整型，不然报错
            points = np.int0(points)
            o = cv2.drawContours(o,[points],0,(0,0,255),4)
            rects.append(rect)
    #  显示二值化之后的图
    # cv2.namedWindow('binary',cv2.WINDOW_GUI_NORMAL)
    # cv2.imshow('binary',binary)
    return rects,o


if __name__ == '__main__':
    #  用于匹配的图
    img = cv2.imread(r"C:\Users\Administrator\Desktop\boltimg\9.jpg")
    #  生成模板轮廓
    template = tem(r"C:\Users\Administrator\Desktop\boltimg\tem.jpg")
    o = img.copy()
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    # _,binaryori = cv2.threshold(gray,210,255,cv2.THRESH_BINARY)
    #  减弱光照不均匀的影响
    k = np.ones((101,101),np.uint8)
    opend = cv2.morphologyEx(gray,cv2.MORPH_OPEN,k)
    addimg = cv2.addWeighted(gray,1,opend,-1,100)
    #  调用detect函数检测匹配图中的螺丝
    rects,objimg = detect(addimg,o,template)
    #  打印检测出的目标形心和旋转角度
    for i in range(len(rects)):
        print('第%s个目标中心\n' % str(i+1))
        print(rects[i][0])
        print('\n')
        print('第%s个目标旋转角度\n' % str(i+1))
        print(rects[i][2])
    #  显示原图
    cv2.namedWindow('original',cv2.WINDOW_GUI_NORMAL)
    cv2.imshow('original',img)
    #  显示检测成功图
    cv2.namedWindow('objimg',cv2.WINDOW_GUI_NORMAL)
    cv2.imshow('objimg',objimg)
    
    #  显示由灰度图直接二值化后的二值图，调试时需要
    # cv2.namedWindow('binaryori',cv2.WINDOW_GUI_NORMAL)
    # cv2.imshow('binaryori',binaryori)
    #  显示开操作之后的图像
    # cv2.namedWindow('open',cv2.WINDOW_GUI_NORMAL)
    # cv2.imshow('open',opend)
    # #  显示光照均匀化之后的图像
    # cv2.namedWindow('uniform',cv2.WINDOW_GUI_NORMAL)
    # cv2.imshow('uniform',addimg)
    
    cv2.waitKey()
    cv2.destroyAllWindows()
