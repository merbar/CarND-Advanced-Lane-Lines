import glob
import sys
import numpy as np
import cv2
import pickle
import advLaneDetectUtil as laneUtil
from moviepy.editor import VideoFileClip
from advLaneDetect_line import Line

# GLOBALS
warpedImgSize = (500, 1500)
#warpedImgSize = (img.shape[1], img.shape[0])
# Define conversions in x and y from pixels space to meters
ym_per_pix = 30/warpedImgSize[1] # meters per pixel in y dimension
xm_per_pix = 6.0/warpedImgSize[0] # meters per pixel in x dimension for 15m projection
#xm_per_pix = 5.3/warpedImgSize[0] # meters per pixel in x dimension for 20m projection
#xm_per_pix = 6.2/warpedImgSize[0] # meters per pixel in x dimension for 30m projection
slidingWindow_margin = int(warpedImgSize[0]/10)
slidingWindow_windows = 9
marginSearch_margin = int(warpedImgSize[0]/10)
# max amount of frames a line can fail to detect until it abandons marginSearch and retries search from scratch with sliding windows
failCountMax = 5
# Visualize
fontFace = cv2.FONT_HERSHEY_DUPLEX
fontScale = 0.7
thickness = 1
imgNumber = 1

lineRight = Line(warpedImgSize, xm_per_pix, ym_per_pix, 'right')
lineLeft = Line(warpedImgSize, xm_per_pix, ym_per_pix, 'left')
lineRight.addOtherLine(lineLeft)
lineLeft.addOtherLine(lineRight)

def process_image(img):
    global imgNumber
    img_size = (img.shape[1], img.shape[0])
    ############################### UNDISTORT ###############################
    dstImg = cv2.undistort(img, mtx, dist, None, mtx)
    dstImg = cv2.cvtColor(dstImg, cv2.COLOR_RGB2BGR)

    ################################ PERSPECTIVE TRANSFORM ###############################
    # projection for 30 meters out
    src30m = np.float32(
        [[0, 670],
         [1280, 670],
         [710, 448],
         [570, 448]])
    # projection for 20 meters out
    src20m = np.float32(
        [[30, 700],
         [1250, 700],
         [732, 462],
         [548, 462]])
    src15m = np.float32(
        [[30, 700],
         [1250, 700],
         [748, 475],
         [532, 475]])
    src = src15m
    dst = np.float32(
        [[0, warpedImgSize[1]],
         [warpedImgSize[0], warpedImgSize[1]],
         [warpedImgSize[0], 0],
         [0, 0]])
    '''
    srcArr = np.array( [src], dtype=np.int32 )
    roiOverlay = np.copy(dstImg)
    roiOverlay = cv2.fillPoly(roiOverlay, srcArr, 255)
    '''
    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)
    warped_dst = cv2.warpPerspective(dstImg, M, warpedImgSize, flags=cv2.INTER_LINEAR)
      
    ############################### IMAGE CREATION ###############################
    kernel_size = 19  
    warped_dstImgBlur = cv2.GaussianBlur(warped_dst, (kernel_size, kernel_size), 0)
    # get useful greyscale channels
    hls_s = laneUtil.makeGrayImg(warped_dst, colorspace='hls', useChannel=2)
    hls_s_blur = laneUtil.makeGrayImg(warped_dstImgBlur, colorspace='hls', useChannel=2)
    hsv_v = laneUtil.makeGrayImg(warped_dst, colorspace='hsv', useChannel=2)
    hsv_v_blur = laneUtil.makeGrayImg(warped_dstImgBlur, colorspace='hsv', useChannel=2)
    hls_l = laneUtil.makeGrayImg(warped_dst, colorspace='hls', useChannel=1)
    grayBlur = cv2.cvtColor(warped_dstImgBlur, cv2.COLOR_RGB2GRAY)

    # Isolate white and yellow
    hsv = cv2.cvtColor(warped_dst, cv2.COLOR_BGR2HSV)
    lower_yellow = np.array([10,80,100])
    upper_yellow = np.array([70,255,255])
    lower_white = np.array([0,0,200])
    upper_white = np.array([255,30,255])
    maskYellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
    maskWhite = cv2.inRange(hsv, lower_white, upper_white)
    maskYellowWhite = cv2.bitwise_or(maskYellow, maskWhite)
    bin_color_tresh = np.zeros_like(hls_s)
    bin_color_tresh[maskYellowWhite > 0] = 1
    # cut out noise on car hood
    bin_color_tresh[int(warpedImgSize[1]-35):int(warpedImgSize[1]):] = 0  
    
    bin_hls_s_thresh = laneUtil.makeBinaryImg(hls_s, threshold=(120,254), mode='simple')
    bin_hsv_v_thresh = laneUtil.makeBinaryImg(hsv_v, threshold=(200,255), mode='simple')
    #bin_hsv_v_mag = laneUtil.makeBinaryImg(hsv_v_blur, threshold=(50,255), mode='mag')
    bin_hsv_v_sobelX = laneUtil.makeBinaryImg(hsv_v_blur, threshold=(60,120), sobel_kernel=3, mode='sobelX')
    bin_hls_s_sobelX =laneUtil.makeBinaryImg(hls_s, threshold=(60,120), sobel_kernel=3, mode='sobelX')
    bin_sobelX = np.zeros_like(hls_s)
    bin_sobelX[(bin_hsv_v_sobelX > 0) | (bin_hsv_v_sobelX > 0)] = 1
    # denoise sobelX. Can produce fine-grained noise across the image, so liberally remove it.
    bin_sobelX = laneUtil.denoiseBinary(bin_sobelX, noiseColumnThresh=55)

    bin_mag = laneUtil.makeBinaryImg(grayBlur, threshold=(100,255), sobel_kernel=3, mode='mag')
    # cut out noise on car hood
    bin_mag[int(warpedImgSize[1]-35):int(warpedImgSize[1]):] = 0

    # denoise color threshold images before combining
    bin_hsv_v_thresh = laneUtil.denoiseBinary(bin_hsv_v_thresh)
    bin_hls_s_thresh = laneUtil.denoiseBinary(bin_hls_s_thresh)
    bin_color_tresh = laneUtil.denoiseBinary(bin_color_tresh)
    colorComb_bin = np.zeros_like(hls_s)
    colorComb_bin[(bin_color_tresh > 0) | (bin_hsv_v_thresh > 0) | (bin_hls_s_thresh > 0)] = 1
    # merge with edge detection images
    warped_bin = np.zeros_like(hls_s)
    warped_bin[(colorComb_bin > 0) | (bin_mag > 0) | (bin_sobelX > 0)] = 1

    ############################### RAW IMAGE OUTPUT ###############################
    binFolder = 'vid_debug/bin'
    outFile = '{}/{}_dst.jpg'.format(binFolder, imgNumber)
    laneUtil.writeImg(dstImg, outFile, binary=False)
    outFile = '{}/{}_hsv_v.jpg'.format(binFolder, imgNumber)
    laneUtil.writeImg(hsv_v, outFile, binary=False)
    outFile = '{}/{}_hls_s.jpg'.format(binFolder, imgNumber)
    laneUtil.writeImg(hls_s, outFile, binary=False)
    outFile = '{}/{}_bin_mag.jpg'.format(binFolder, imgNumber)
    laneUtil.writeImg(bin_mag, outFile, binary=True)
    outFile = '{}/{}_bin_sobelX.jpg'.format(binFolder, imgNumber)
    laneUtil.writeImg(bin_sobelX, outFile, binary=True)
    outFile = '{}/{}_bin_color_tresh.jpg'.format(binFolder, imgNumber)
    laneUtil.writeImg(bin_color_tresh, outFile, binary=True)
    outFile = '{}/{}_bin_hls_s_thresh.jpg'.format(binFolder, imgNumber)
    laneUtil.writeImg(bin_hls_s_thresh, outFile, binary=True)
    outFile = '{}/{}_bin_hsv_v_thresh.jpg'.format(binFolder, imgNumber)
    laneUtil.writeImg(bin_hsv_v_thresh, outFile, binary=True)    
    outFile = '{}/{}_bin_colorComb.jpg'.format(binFolder, imgNumber)
    laneUtil.writeImg(colorComb_bin, outFile, binary=True)
    outFile = '{}/{}_bin_final.jpg'.format(binFolder, imgNumber)
    laneUtil.writeImg(warped_bin, outFile, binary=True)
    outFile = '{}/{}_warped_dst.jpg'.format(binFolder, imgNumber)
    laneUtil.writeImg(warped_dst, outFile, binary=False)

    ############################### IDENTIFY LINES ###############################
    leftCrvRad = 0
    rightCrvRad = 0
    leftFailCount = 0
    rightFailCount = 0
    # Left lane
    fit_left = None
    fit_right = None
    data_imgLeft = None
    data_imgRight = None
    if not lineLeft.detected:
        leftx_base, rightx_base = laneUtil.findLaneBases(warped_bin, xm_per_pix)
        if leftx_base:
            fit_left, data_imgLeft, leftConfidence = laneUtil.slidingWindowFit(warped_bin, leftx_base, lanePxColor=(0,0,220), nwindows=slidingWindow_windows, margin=slidingWindow_margin)
            best_fit_left, leftCrvRad, leftFailCount = lineLeft.updateData(fit_left, leftConfidence)
    else:
        fit_left, data_imgLeft, leftConfidence = laneUtil.marginSearch(warped_bin, lineLeft.best_fit_f, lanePxColor=(0,0,220), margin=marginSearch_margin)
        best_fit_left, leftCrvRad, leftFailCount = lineLeft.updateData(fit_left, leftConfidence)
    # Right lane
    if not lineRight.detected:
        leftx_base, rightx_base = laneUtil.findLaneBases(warped_bin, xm_per_pix)
        if rightx_base:
            fit_right, data_imgRight, rightConfidence = laneUtil.slidingWindowFit(warped_bin, rightx_base, lanePxColor=(220,0,0), nwindows=slidingWindow_windows, margin=slidingWindow_margin)
            best_fit_right, rightCrvRad, rightFailCount = lineRight.updateData(fit_right, rightConfidence)
    else:
        fit_right, data_imgRight, rightConfidence = laneUtil.marginSearch(warped_bin, lineRight.best_fit_f, lanePxColor=(220,0,0), margin=marginSearch_margin)
        best_fit_right, rightCrvRad, rightFailCount = lineRight.updateData(fit_right, rightConfidence)

    ############################### CREATE (and write) DIAGNOSTICE IMAGES ###############################
    debugFolder = 'vid_debug'
    diagImg = laneUtil.makeDiagnosticsImage(warped_bin, lineLeft, lineRight, debugFolder, imgNumber, xm_per_pix, data_imgLeft, data_imgRight)
    ############################### CREATE FINAL OUTPUT IMAGE ###############################
    finalImg = laneUtil.makeFinalLaneImage(img, lineLeft, lineRight, Minv, warpedImgSize[0], warpedImgSize[1], xm_per_pix)
    imgNumber += 1
    return finalImg


def main():
    # load camera calibration data
    global mtx, dist, rvecs, tvecs
    with open('cameraCalibration.pickle', 'rb') as fp:
        mtx, dist, rvecs, tvecs = pickle.load(fp)

    videoFile = sys.argv[1]
    # straight road
    #clip = VideoFileClip(videoFile).subclip('00:00:16.15','00:00:16.15')
    # low contrast!
    #clip = VideoFileClip(videoFile).subclip('00:00:40.00','00:00:40.00')
    #shadows and contrast!
    #clip = VideoFileClip(videoFile).subclip('00:00:41.38','00:00:41.38')
    #clip = VideoFileClip(videoFile).subclip(19,26)
    #clip = VideoFileClip(videoFile).subclip(37,43)
    clip = VideoFileClip(videoFile)
    proc_clip = clip.fl_image(process_image)
    proc_output = '{}_proc.mp4'.format(videoFile.split('.')[0])
    proc_clip.write_videofile(proc_output, audio=False)


if __name__ == '__main__':
    main()