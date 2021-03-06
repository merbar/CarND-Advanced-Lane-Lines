import os
import sys
import glob
import numpy as np
import cv2
import pickle
import advLaneDetectUtil as laneUtil
from moviepy.editor import VideoFileClip, ImageSequenceClip
from advLaneDetect_line import Line

# GLOBALS
warpedImgSize = (500, 1500)
# Define conversions in x and y from pixels space to meters
ym_per_pix = 30/warpedImgSize[1] # meters per pixel in y dimension
xm_per_pix = 5.4/warpedImgSize[0] # meters per pixel in x dimension for 15m projection
slidingWindow_margin = int(warpedImgSize[0]/10)
slidingWindow_windows = 9
marginSearch_margin = int(warpedImgSize[0]/15)
defaultLaneWidth = 3.7 / xm_per_pix
laneWidths = [defaultLaneWidth]
# Visualize
fontFace = cv2.FONT_HERSHEY_DUPLEX
fontScale = 0.7
thickness = 1
img_i = 1
imgNumberPadding = 4
binaryMethod = 'colorOnly'

lineRight = Line(warpedImgSize, xm_per_pix, ym_per_pix, 'right')
lineLeft = Line(warpedImgSize, xm_per_pix, ym_per_pix, 'left')
lineRight.addOtherLine(lineLeft)
lineLeft.addOtherLine(lineRight)

outputEverything = False
outputCtrlCenterMov = True
debugFolder = 'vid_debug'
if not os.path.exists(debugFolder):
    os.makedirs(debugFolder)

def process_image(img):
    global img_i, laneWidths
    # take care of padding image numbers for cleaner output
    imgNumber = '0'*(imgNumberPadding-len(str(img_i))) + str(img_i)
    img_size = (img.shape[1], img.shape[0])

    ############################### UNDISTORT ###############################
    dstImg = cv2.undistort(img, mtx, dist, None, mtx)
    dstImg = cv2.cvtColor(dstImg, cv2.COLOR_RGB2BGR)

    ################################ PERSPECTIVE TRANSFORM ###############################
    src = np.float32(
        [[30, 700],
         [1250, 700],
         [748, 475],
         [532, 475]])
    dst = np.float32(
        [[0, warpedImgSize[1]],
         [warpedImgSize[0], warpedImgSize[1]],
         [warpedImgSize[0], 0],
         [0, 0]])
    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)
    warped_dst = cv2.warpPerspective(dstImg, M, warpedImgSize, flags=cv2.INTER_LINEAR)
      
    ############################### IMAGE CREATION ###############################
    # get useful greyscale channels
    hls_s = laneUtil.makeGrayImg(warped_dst, colorspace='hls', useChannel=2)    
    hsv_v = laneUtil.makeGrayImg(warped_dst, colorspace='hsv', useChannel=2)    
    hls_l = laneUtil.makeGrayImg(warped_dst, colorspace='hls', useChannel=1)
    # Isolate white and yellow
    # enhance and equalize contrast of image to use with white threshold
    lab_l = laneUtil.makeGrayImg(warped_dst, colorspace='lab', useChannel=0)
    lab_a = laneUtil.makeGrayImg(warped_dst, colorspace='lab', useChannel=1)
    lab_b = laneUtil.makeGrayImg(warped_dst, colorspace='lab', useChannel=2)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    lab_l_eq = clahe.apply(lab_l)
    lab_merge = cv2.merge((lab_l_eq,lab_a,lab_b))
    warped_dst_eq = cv2.cvtColor(lab_merge, cv2.COLOR_LAB2BGR)
    # now do thresholds. Yellow on regular hsv image, white on contrast-equalized hsv
    hsv_equi = cv2.cvtColor(warped_dst_eq, cv2.COLOR_BGR2HSV)
    hsv = cv2.cvtColor(warped_dst, cv2.COLOR_BGR2HSV)
    # default color thresholds
    lower_yellow = np.array([10,40,100])
    upper_yellow = np.array([70,255,255])
    lower_white = np.array([0,0,200])
    upper_white = np.array([255,30,255])
    maskYellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
    maskWhite = cv2.inRange(hsv_equi, lower_white, upper_white)
    maskYellowWhite = cv2.bitwise_or(maskYellow, maskWhite)
    bin_color_thresh = np.zeros_like(hls_s)
    bin_color_thresh[maskYellowWhite > 0] = 1
    # cut out noise on car hood
    bin_color_thresh[int(warpedImgSize[1]-35):int(warpedImgSize[1]):] = 0
    # less sensitive color thresholds (less noise, good for perfect conditions)
    lower_yellow = np.array([10,80,100])
    upper_yellow = np.array([70,255,255])
    lower_white = np.array([0,0,200])
    upper_white = np.array([255,30,255])
    maskYellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
    maskWhite = cv2.inRange(hsv, lower_white, upper_white)
    maskYellowWhite = cv2.bitwise_or(maskYellow, maskWhite)
    bin_colorLessSensitive_tresh = np.zeros_like(hls_s)
    bin_colorLessSensitive_tresh[maskYellowWhite > 0] = 1
    bin_colorLessSensitive_tresh[int(warpedImgSize[1]-35):int(warpedImgSize[1]):] = 0
    # more sensitive color thresholds (detection of features in low-light areas, but much more noise)
    lower_yellow = np.array([10,80,100])
    upper_yellow = np.array([70,255,255])
    lower_white = np.array([0,0,100])
    upper_white = np.array([255,60,255])
    maskYellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
    maskWhite = cv2.inRange(hsv_equi, lower_white, upper_white)
    maskYellowWhite = cv2.bitwise_or(maskYellow, maskWhite)
    bin_colorMoreSensitive_tresh = np.zeros_like(hls_s)
    bin_colorMoreSensitive_tresh[maskYellowWhite > 0] = 1
    bin_colorMoreSensitive_tresh[int(warpedImgSize[1]-35):int(warpedImgSize[1]):] = 0
        # color images only for binary image creation
    # denoise color threshold images before combining
    bin_color_thresh = laneUtil.denoiseBinary(bin_color_thresh)
    bin_colorLessSensitive_tresh = laneUtil.denoiseBinary(bin_colorLessSensitive_tresh)
    bin_colorMoreSensitive_tresh = laneUtil.denoiseBinary(bin_colorMoreSensitive_tresh)
    bin_color_threshComb = np.add(bin_color_thresh, bin_colorLessSensitive_tresh) # fills in previously denoised areas
    bin_color_threshComb = np.add(bin_color_threshComb, bin_colorMoreSensitive_tresh)
    # combine final binary image
    warped_bin = np.zeros_like(hls_s)
    warped_bin[(bin_color_threshComb > 0)] = 1

    ############################### RAW IMAGE OUTPUT ###############################
    binFolder = 'vid_debug/bin'
    if outputEverything:
        outFile = '{}/dst.{}.jpg'.format(binFolder, imgNumber)
        laneUtil.writeImg(dstImg, outFile, binary=False)
        outFile = '{}/hsv_v.{}.jpg'.format(binFolder, imgNumber)
        laneUtil.writeImg(hsv_v, outFile, binary=False)
        outFile = '{}/hls_s.{}.jpg'.format(binFolder, imgNumber)
        laneUtil.writeImg(hls_s, outFile, binary=False)
        outFile = '{}/bin_color_thresh_default.{}.jpg'.format(binFolder, imgNumber)
        laneUtil.writeImg(bin_color_thresh, outFile, binary=True)
        outFile = '{}/bin_color_thresh_lessSensitive.{}.jpg'.format(binFolder, imgNumber)
        laneUtil.writeImg(bin_colorLessSensitive_tresh, outFile, binary=True)
        outFile = '{}/bin_color_thresh_moreSensitive.{}.jpg'.format(binFolder, imgNumber)
        laneUtil.writeImg(bin_colorMoreSensitive_tresh, outFile, binary=True)
        outFile = '{}/warped_dst.{}.jpg'.format(binFolder, imgNumber)
        laneUtil.writeImg(warped_dst, outFile, binary=False)
        outFile = '{}/warped_dst_equi.{}.jpg'.format(binFolder, imgNumber)
        laneUtil.writeImg(warped_dst_eq, outFile, binary=False)
        outFile = '{}/bin_final.{}.jpg'.format(binFolder, imgNumber)
        laneUtil.writeImg(warped_bin, outFile, binary=True)


    ############################### IDENTIFY LINES ###############################
    leftCrvRad = 0
    rightCrvRad = 0
    # Left lane
    fit_left = None
    fit_right = None
    data_imgLeft = None
    data_imgRight = None
    laneWidthAvg = np.average(laneWidths)
    lineLeft.setLaneWidth(laneWidthAvg)
    lineRight.setLaneWidth(laneWidthAvg)

    if not lineLeft.detected:
        leftx_base, rightx_base = laneUtil.findLaneBases(warped_bin, xm_per_pix, laneWidthAvg)
        if leftx_base:
            fit_left, data_imgLeft, leftConfidence = laneUtil.slidingWindowFit(warped_bin, leftx_base, lanePxColor=(0,0,220), nwindows=slidingWindow_windows, margin=slidingWindow_margin)
            best_fit_left, leftCrvRad = lineLeft.updateData(fit_left, leftConfidence)
        else:
            # try to mirror other line
            best_fit_left, leftCrvRad = lineLeft.updateData(fit_left, 0.)
            if best_fit_left == None:
                # total failure
                data_imgLeft = None
                lineLeft.best_fit = None
    else:
        fit_left, data_imgLeft, leftConfidence = laneUtil.marginSearch(warped_bin, lineLeft.best_fit_f, lanePxColor=(0,0,220), margin=marginSearch_margin)
        best_fit_left, leftCrvRad = lineLeft.updateData(fit_left, leftConfidence)
    # Right lane
    if not lineRight.detected:
        leftx_base, rightx_base = laneUtil.findLaneBases(warped_bin, xm_per_pix, laneWidthAvg)
        if rightx_base:
            fit_right, data_imgRight, rightConfidence = laneUtil.slidingWindowFit(warped_bin, rightx_base, lanePxColor=(220,0,0), nwindows=slidingWindow_windows, margin=slidingWindow_margin)
            best_fit_right, rightCrvRad = lineRight.updateData(fit_right, rightConfidence)
        else:
            # try to mirror other line
            best_fit_left, leftCrvRad = lineLeft.updateData(fit_left, 0.)
            if best_fit_left == None:
                # total failure
                data_imgRight = None
                lineRight.best_fit = None
    else:
        fit_right, data_imgRight, rightConfidence = laneUtil.marginSearch(warped_bin, lineRight.best_fit_f, lanePxColor=(220,0,0), margin=marginSearch_margin)
        best_fit_right, rightCrvRad = lineRight.updateData(fit_right, rightConfidence)

    # update lane width in each line. This is to account for excessively thinning/widening roads and is used when projecting one line onto the other
    if (lineLeft.best_fit != None) & (lineRight.best_fit != None):
        laneWidthPx, laneWidthMeters = laneUtil.getLaneWidth(lineLeft.best_fit_f, lineRight.best_fit_f, warpedImgSize[0], warpedImgSize[1], xm_per_pix)
        laneWidths.append(laneWidthPx)
        if len(laneWidths) > 10:
            laneWidths = laneWidths[1:]
    ############################### CREATE (and write) DIAGNOSTICE IMAGES ###############################
    diagImg = laneUtil.makeDiagnosticsImage(warped_bin, lineLeft, lineRight, xm_per_pix, data_imgLeft, data_imgRight)
    outFile = '{}/diag.{}.jpg'.format(debugFolder, imgNumber)
    laneUtil.writeImg(diagImg, outFile, binary=False)

    if outputCtrlCenterMov:
        textDataImg = laneUtil.makeTextDataImage(warped_bin, lineLeft, lineRight, xm_per_pix)
        outFile = '{}/textData.{}.jpg'.format(debugFolder, imgNumber)
        laneUtil.writeImg(textDataImg, outFile, binary=False)

    ############################### CREATE FINAL OUTPUT IMAGE ###############################
    finalImg = laneUtil.makeFinalLaneImage(img, lineLeft, lineRight, Minv, warpedImgSize[0], warpedImgSize[1], xm_per_pix, featherEdge=False)
    finalImg = cv2.cvtColor(finalImg, cv2.COLOR_BGR2RGB)
    outFile = '{}/final.{}.jpg'.format(debugFolder, imgNumber)
    laneUtil.writeImg(finalImg, outFile, binary=False)

    if outputCtrlCenterMov:
        ############################### CREATE "CONTROL CENTER" IMAGE ###############################
        ctrlImg = laneUtil.makeCtrlImg(finalImg, textDataImg, diagImg, warped_bin)
        outFile = '{}/ctrl.{}.jpg'.format(debugFolder, imgNumber)
        laneUtil.writeImg(ctrlImg, outFile, binary=False)

    img_i += 1    
    return finalImg


def main():
    # load camera calibration data
    global mtx, dist, rvecs, tvecs
    with open('cameraCalibration.pickle', 'rb') as fp:
        mtx, dist, rvecs, tvecs = pickle.load(fp)

    videoFile = sys.argv[1]
    # stuff from project video
    # straight road
    #clip = VideoFileClip(videoFile).subclip('00:00:16.15','00:00:16.15')
    # low contrast!
    #clip = VideoFileClip(videoFile).subclip('00:00:22.30','00:00:22.30')
    #shadows and contrast!
    #clip = VideoFileClip(videoFile).subclip('00:00:41.38','00:00:41.38')
    #clip = VideoFileClip(videoFile).subclip(19,26)
    #clip = VideoFileClip(videoFile).subclip(37,43)
    
    # stuff from challenge video
    #clip = VideoFileClip(videoFile).subclip('00:00:15.00','00:00:19.00')
    #clip = VideoFileClip(videoFile).subclip('00:00:05.00','00:00:06.00')
    #clip = VideoFileClip(videoFile).subclip('00:00:05.00','00:00:05.50')
    
    clip = VideoFileClip(videoFile)

    proc_clip = clip.fl_image(process_image)
    proc_output = '{}_proc.mp4'.format(videoFile.split('.')[0])
    proc_clip.write_videofile(proc_output, audio=False)

    if outputCtrlCenterMov:
        images = glob.glob('vid_debug\ctrl.*.jpg')
        clip = ImageSequenceClip(images, fps=25)
        proc_output = '{}_ctrl.mp4'.format(videoFile.split('.')[0])
        clip.write_videofile(proc_output, audio=False)

if __name__ == '__main__':
    main()