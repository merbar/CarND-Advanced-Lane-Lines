import glob
import numpy as np
import cv2
import pickle
import advLaneDetectUtil as laneUtil
import sys
from moviepy.editor import VideoFileClip

# GLOBALS
warpedImgSize = (800, 1000)
#warpedImgSize = (img.shape[1], img.shape[0])
# Define conversions in x and y from pixels space to meters
ym_per_pix = 30/warpedImgSize[1] # meters per pixel in y dimension
xm_per_pix = 5/warpedImgSize[0] # meters per pixel in x dimension
slidingWindow_margin = int(warpedImgSize[0]/10)
slidingWindow_windows = 9
marginSearch_margin = int(warpedImgSize[0]/10)
# Visualize
fontFace = cv2.FONT_HERSHEY_SIMPLEX
fontScale = 0.7
thickness = 2
imgNumber = 1

# Define a class to receive the characteristics of each line detection
class Line:
    def __init__(self, warpedImgSize):
        # was the line detected in the last iteration?
        self.detected = False 
        self.filterSize = 5
        # x values of the last n fits of the line
        self.recent_xfitted = [] 
        #average x values of the fitted line over the last n iterations
        self.bestx = None     
        #polynomial coefficients averaged over the last n iterations
        self.best_fit = None
        self.best_fit_f = None
        #polynomial coefficients for the most recent fits
        self.recent_fit = []  
        #radius of curvature of the line in some units
        self.radius_of_curvature = None 
        #distance in meters of vehicle center from the line
        self.line_base_pos = None 
        #difference in fit coefficients between last and new fits
        self.diffs = np.array([0,0,0], dtype='float') 
        #x values for detected line pixels
        self.allx = None
        #y values for detected line pixels
        self.ally = None
        # prep y values for warped image space
        self.fity = np.linspace(0, warpedImgSize[1]-1, warpedImgSize[1])

    def updateData(self, fit):
        self.detected = True
        self.recent_fit.append(fit)
        if len(self.recent_fit) > self.filterSize:
            self.recent_fit = self.recent_fit[1:]
        self.best_fit = np.average(self.recent_fit, axis=0)
        #self.best_fit = fit
        self.best_fit_f = np.poly1d(self.best_fit)
        self.radius_of_curvature = laneUtil.getCurveRadius(self.best_fit_f, warpedImgSize[1], xm_per_pix, ym_per_pix)
        return self.best_fit, self.radius_of_curvature

lineRight = Line(warpedImgSize)
lineLeft = Line(warpedImgSize)


def process_image(img):
    global imgNumber
    debugFolder = 'vid_debug'
    img_size = (img.shape[1], img.shape[0])
    # Undistort
    dstImg = cv2.undistort(img, mtx, dist, None, mtx)
    
    # get useful greyscale channels
    hls_s = laneUtil.makeGrayImg(dstImg, colorspace='hls', useChannel=2)
    hsv_v = laneUtil.makeGrayImg(dstImg, colorspace='hsv', useChannel=2)
    hls_l = laneUtil.makeGrayImg(dstImg, colorspace='hls', useChannel=1)
    
    # histogram equalization test
    hsv_v_equi = cv2.equalizeHist(hsv_v)
    
    # good general result, deals well with low contrast
    bin_hls_s_thresh = laneUtil.makeBinaryImg(hls_s, threshold=(150,255), mode='simple')
    # generally better result than hls_s_thresh
    # deals VERY poorly with low contrast road surface, but deals GREAT with shadows
    bin_hsv_v_thresh = laneUtil.makeBinaryImg(hsv_v_equi, threshold=(210,255), mode='simple')
    # better in all aspects than sobelX or dir. Using non-equalized hsv_v because it has more contrast
    bin_hsv_v_mag_equi = laneUtil.makeBinaryImg(hsv_v_equi, threshold=(90,255), mode='mag')
    # simple sobelX to fill in
    bin_hsv_v_sobelX = laneUtil.makeBinaryImg(hsv_v, threshold=(20,100), mode='sobelX')
    
    # two different binary images. binLowContrast can deal with low contrast better while binHighContrast is much less noisy
    binHighContrast = np.zeros_like(hls_s)
    binHighContrast[(bin_hls_s_thresh == 1) | (bin_hsv_v_mag_equi == 1)] = 1 
    binLowContrast = np.zeros_like(hls_s)
    binLowContrast[(bin_hls_s_thresh == 1) | (bin_hsv_v_sobelX == 1)] = 1
    binShadowSpec = np.zeros_like(hls_s)
    binShadowSpec[(bin_hsv_v_mag_equi == 1) | (bin_hsv_v_thresh == 1)] = 1
    
    # Perspective transform
    src = np.float32(
        [[30, 670],
         [1250, 670],
         [710, 448],
         [570, 448]])
    dst = np.float32(
        [[0, warpedImgSize[1]],
         [warpedImgSize[0], warpedImgSize[1]],
         [warpedImgSize[0], 0],
         [0, 0]])
    
    srcArr = np.array( [src], dtype=np.int32 )
    roiOverlay = np.copy(dstImg)
    roiOverlay = cv2.fillPoly(roiOverlay, srcArr, 255)
    
    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)
    
    warpedHC = cv2.warpPerspective(binHighContrast, M, warpedImgSize, flags=cv2.INTER_LINEAR)
    warpedLC = cv2.warpPerspective(binLowContrast, M, warpedImgSize, flags=cv2.INTER_LINEAR)
    warpedOrig = cv2.warpPerspective(dstImg, M, warpedImgSize, flags=cv2.INTER_LINEAR)
   
    # Identify lanes
    # Left lane
    if not lineLeft.detected:
        leftx_base, rightx_base = laneUtil.findLaneBases(warpedLC)
        ret_left, fit_left, data_img = laneUtil.slidingWindowFit(warpedLC, leftx_base, lanePxColor=(0,0,220), nwindows=slidingWindow_windows, margin=slidingWindow_margin)
        warpedLC_data = np.dstack((warpedLC, warpedLC, warpedLC))*255
        # Want data to be fully opaque. Mask out area to fill with data with black first, then overlay
        img2gray = cv2.cvtColor(data_img,cv2.COLOR_BGR2GRAY)
        warpedLC_data[(img2gray != 0)] = (0,0,0)
        warpedLC_data = cv2.add(warpedLC_data, data_img)
        if not ret_left:       
            cv2.putText(warpedLC_data, 'left lane detection failed', (100, warpedImgSize[1]-20), fontFace, fontScale,(0,0,255), thickness)
        else:
            best_fit_left, leftCrvRad = lineLeft.updateData(fit_left)
            fit_f_left = np.poly1d(best_fit_left)
            # plot lane
            # Generate x and y values for plotting
            fity = np.linspace(0, warpedLC.shape[0]-1, warpedLC.shape[0])
            fit_x_left = fit_f_left(fity)
            #fit_x_left = fit[0]*fity**2 + fit[1]*fity + fit[2]
            for y in range(len(fity)):
                visCircleRad = 3
                cv2.circle(warpedLC_data, (int(fit_x_left[y]), int(fity[y])), visCircleRad, (0,170,170), thickness=-1)
    else:
        ret_leftMargin, fit_left, data_img = laneUtil.marginSearch(warpedLC, lineLeft.best_fit_f, lanePxColor=(0,0,220), margin=marginSearch_margin)
        warpedLC_data = np.dstack((warpedLC, warpedLC, warpedLC))*255
        # Want data to be fully opaque. Mask out area to fill with data with black first, then overlay
        img2gray = cv2.cvtColor(data_img,cv2.COLOR_BGR2GRAY)
        warpedLC_data[(img2gray != 0)] = (0,0,0)
        warpedLC_data = cv2.add(warpedLC_data, data_img)
        #warpedLC_data = cv2.addWeighted(warpedLC_data, 1., data_img, 0.5, 0.)
        if not ret_leftMargin:
            cv2.putText(warpedLC_data, 'left lane margin search failed', (100, warpedImgSize[1]-20), fontFace, fontScale,(0,0,255), thickness)
        else:
            best_fit_left, leftCrvRad = lineLeft.updateData(fit_left)
            fit_f_left = np.poly1d(best_fit_left)
            # plot lane
            # Generate x and y values for plotting
            fity = np.linspace(0, warpedLC.shape[0]-1, warpedLC.shape[0] )
            fit_x_leftMargin = fit_f_left(fity)
            for y in range(len(fity)):
                visCircleRad = 3
                cv2.circle(warpedLC_data, (int(fit_x_leftMargin[y]), int(fity[y])), visCircleRad, (0,170,170), thickness=-1)
    # Right lane
    if not lineRight.detected:
        ret_right, fit_right, data_img = laneUtil.slidingWindowFit(warpedLC, rightx_base, lanePxColor=(220,0,0), nwindows=slidingWindow_windows, margin=slidingWindow_margin)
        fit_f_right = np.poly1d(fit_right)
        img2gray = cv2.cvtColor(data_img,cv2.COLOR_BGR2GRAY)
        warpedLC_data[(img2gray != 0)] = (0,0,0)
        warpedLC_data = cv2.add(warpedLC_data, data_img)
        if not ret_right:
            cv2.putText(warpedLC_data, 'right lane detection failed', (int(warpedImgSize[0]/2)+100, warpedImgSize[1]-20), fontFace, fontScale,(0,0,255), thickness)
        else:
            best_fit_right, rightCrvRad = lineRight.updateData(fit_right)
            fit_f_right = np.poly1d(best_fit_right)
            # plot lane
            # Generate x and y values for plotting
            fity = np.linspace(0, warpedLC.shape[0]-1, warpedLC.shape[0] )
            fit_x_right = fit_f_right(fity)
            for y in range(len(fity)):
                visCircleRad = 3
                cv2.circle(warpedLC_data, (int(fit_x_right[y]), int(fity[y])), visCircleRad, (0,170,170), thickness=-1)
    else:
        ret_rightMargin, fit_right, data_img = laneUtil.marginSearch(warpedLC, lineRight.best_fit_f, lanePxColor=(0,0,220), margin=marginSearch_margin)
        warpedLC_data = np.dstack((warpedLC, warpedLC, warpedLC))*255
        # Want data to be fully opaque. Mask out area to fill with data with black first, then overlay
        img2gray = cv2.cvtColor(data_img,cv2.COLOR_BGR2GRAY)
        warpedLC_data[(img2gray != 0)] = (0,0,0)
        warpedLC_data = cv2.add(warpedLC_data, data_img)
        #warpedLC_data = cv2.addWeighted(warpedLC_data, 1., data_img, 0.5, 0.)
        if not ret_rightMargin:
            cv2.putText(warpedLC_data, 'right lane margin search failed', (int(warpedImgSize[0]/2)+100, warpedImgSize[1]-20), fontFace, fontScale,(0,0,255), thickness)
        else:
            best_fit_right, rightCrvRad = lineRight.updateData(fit_right)
            fit_f_right = np.poly1d(best_fit_right)
            # plot lane
            # Generate x and y values for plotting
            fity = np.linspace(0, warpedLC.shape[0]-1, warpedLC.shape[0] )
            fit_x_rightMargin = fit_f_left(fity)
            for y in range(len(fity)):
                visCircleRad = 3
                cv2.circle(warpedLC_data, (int(fit_x_rightMargin[y]), int(fity[y])), visCircleRad, (0,170,170), thickness=-1)
    
    outFile = '%s/%s_debug.jpg' % (debugFolder, imgNumber)
    #laneUtil.writeImg(warpedLC_data, outFile, binary=False)
    
    # Calculate radius of curvature (meters)
    #leftCrvRad = laneUtil.getCurveRadius(fit_f_left, warpedImgSize[1], xm_per_pix, ym_per_pix)
    cv2.putText(warpedLC_data, 'left crv rad: {:.0f}m'.format(leftCrvRad), (100, warpedImgSize[1]-20), fontFace, fontScale,(0,255,0), thickness)
    #rightCrvRad = laneUtil.getCurveRadius(fit_f_right, warpedImgSize[1], xm_per_pix, ym_per_pix)
    cv2.putText(warpedLC_data, 'right crv rad: {:.0f}m'.format(rightCrvRad), (int(warpedImgSize[0]/2)+100, warpedImgSize[1]-20), fontFace, fontScale,(0,255,0), thickness)
    
    # Calculate how far off-center the vehicle is (meters)
    offset = laneUtil.getCarPositionOffCenter(fit_f_left, fit_f_right, warpedImgSize[0], warpedImgSize[1], xm_per_pix)
    cv2.putText(warpedLC_data, 'off center: {:.1f}m'.format(offset), (int(warpedImgSize[0]/2)-50, warpedImgSize[1]-60), fontFace, fontScale,(0,255,0), thickness)

    # Plot lane and warp back to original image
    curvature = (leftCrvRad+rightCrvRad) / 2
    finalImg = laneUtil.makeFinalImage(img, fit_f_left, fit_f_right, Minv, curvature, offset, warpedImgSize[0], warpedImgSize[1])

    imgNumber += 1

    return finalImg


def main():
    # load camera calibration data
    global mtx, dist, rvecs, tvecs
    with open('cameraCalibration.pickle', 'rb') as fp:
        mtx, dist, rvecs, tvecs = pickle.load(fp)

    videoFile = sys.argv[1]
    clip = VideoFileClip(videoFile).subclip(0,3)
    #clip = VideoFileClip(videoFile)
    proc_clip = clip.fl_image(process_image)
    proc_output = '{}_proc.mp4'.format(videoFile.split('.')[0])
    proc_clip.write_videofile(proc_output, audio=False)


if __name__ == '__main__':
    main()