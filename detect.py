import cv2
import numpy as np

import matplotlib.pyplot as plt
class Detectpose:
    
    def __init__(self,file_name):
        self.file_name=file_name
        self.img = cv2.imread(file_name, cv2.IMREAD_GRAYSCALE)
        #self.img=cv2.blur(self.img,(3,3))
        self.img = cv2.equalizeHist(self.img)
        self.img=cv2.resize(self.img, (150, 150))
        self.surf = cv2.xfeatures2d.SURF_create()
        self.keypoints, self.descriptors = self.surf.detectAndCompute(self.img, None)
    

    def hist_match(self,source):
        """
        Adjust the pixel values of a grayscale image such that its histogram
        matches that of a target image

        Arguments:
        -----------
            source: np.ndarray
                Image to transform; the histogram is computed over the flattened
                array
            template: np.ndarray
                Template image; can have different dimensions to source
        Returns:
        -----------
            matched: np.ndarray
                The transformed output image
        """
        template=self.img
        oldshape = source.shape
        source = source.ravel()
        template = template.ravel()

        # get the set of unique pixel values and their corresponding indices and
        # counts
        s_values, bin_idx, s_counts = np.unique(source, return_inverse=True,
                                                return_counts=True)
        t_values, t_counts = np.unique(template, return_counts=True)

        # take the cumsum of the counts and normalize by the number of pixels to
        # get the empirical cumulative distribution functions for the source and
        # template images (maps pixel value --> quantile)
        s_quantiles = np.cumsum(s_counts).astype(np.float64)
        s_quantiles /= s_quantiles[-1]
        t_quantiles = np.cumsum(t_counts).astype(np.float64)
        t_quantiles /= t_quantiles[-1]

        # interpolate linearly to find the pixel values in the template image
        # that correspond most closely to the quantiles in the source image
        interp_t_values = np.interp(s_quantiles, t_quantiles, t_values)
        return interp_t_values[bin_idx].reshape(oldshape)
        
    
    '''def matcher(self,img):
        surf = cv2.xfeatures2d.SURF_create()
       # img=self.hist_match(img)
        keypoints,descriptors = surf.detectAndCompute(img, None)
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(self.descriptors,descriptors,k=2)
        good = []
        for m,n in matches:
            if m.distance < 0.6*n.distance:
                good.append([m])
        print(self.file_name,len(good),m.distance)
        img3 = cv2.drawMatchesKnn(self.img,self.keypoints,img,keypoints,good,None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        cv2.imshow("img",img3)
        cv2.waitKey(1)
    '''
    
    def flann_matcher(self,img):
        surf = cv2.xfeatures2d.SURF_create()
        img = cv2.equalizeHist(img)
        keypoints,descriptors = surf.detectAndCompute(img, None)
        print(len(descriptors))
        # FLANN parameters
        FLANN_INDEX_KDTREE = 0
        index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 3)
        search_params = dict(checks=20)   # or pass empty dictionary
        flann = cv2.FlannBasedMatcher(index_params,search_params)

        matches = flann.knnMatch(self.descriptors,descriptors,k=2)
        # Need to draw only good matches, so create a mask
        matchesMask = [[0,0] for i in range(len(matches))]
        # ratio test as per Lowe's paper
        c=0
        s=0
        val={"walking_segmented.png":0.6,"jogging_segmented.png":0.65,"running_segmented.png":0.61}
        for i,(m,n) in enumerate(matches):
            if m.distance < val[self.file_name]*n.distance:
                c+=1
                matchesMask[i]=[1,0]
                s+=n.distance
     #   if c is not 0:
      #      print(self.file_name,c,s/c)
        draw_params = dict(matchColor = (0,255,0),
                   singlePointColor = (255,0,0),
                   matchesMask = matchesMask,
                   flags = 0)

       # img3 = cv2.drawMatchesKnn(self.img,self.keypoints,img,keypoints,matches,None,**draw_params)
        
        return c

    def flann_matcher_2(self,img):
        surf = cv2.xfeatures2d.SURF_create()
        keypoints,descriptors = surf.detectAndCompute(img, None)
        # FLANN parameters
        FLANN_INDEX_KDTREE = 0
        index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        search_params = dict(checks=50)   # or pass empty dictionary
        dist_l2  = cv2.norm(descriptors,self.descriptors);      
        dist_ham = cv2.norm(descriptors,self.descriptors,cv2.NORM_HAMMING); 
       # print(dist_ham)
       # print(dist_l2)


walking=Detectpose("walking_segmented.png")
#jogging=Detectpose("jogging_segmented.png")
running=Detectpose("running_segmented.png")

cap=cv2.VideoCapture(0)
#walking_3
cap = cv2.VideoCapture('human_action_dataset/chew/AMADEUS_chew_h_nm_np1_fr_goo_7.avi')
while(True):
    ret,frame =cap.read()
    if ret:
        frame=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cv2.imshow("jogging",frame)
        cv2.waitKey(1)
        frame=cv2.resize(frame, (150, 150))
      #  frame=cv2.blur(frame,(3,3))
        c1=walking.flann_matcher(frame)
       # c2=jogging.flann_matcher(frame)
        c3=running.flann_matcher(frame)
        m=max(c1,c3)
        
     #   if m==c1 and c1>0:
     #       print("walking")
       # elif m==c2 and c2>0:
        #    print("jogging")
      #  elif c3>0:
       #     print("running")


       # print(c1)#,c2,c3)

cap.release()
cv2.destroyAllWindows()