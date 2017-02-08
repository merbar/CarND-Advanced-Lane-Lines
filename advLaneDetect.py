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
xm_per_pix = 5.3/warpedImgSize[0] # meters per pixel in x dimension
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

lineRight = Line(warpedImgSize, xm_per_pix, ym_per_pix)
lineLeft = Line(warpedImgSize, xm_per_pix, ym_per_pix)
lineRight.addOtherLine(lineLeft)
lineLeft.addOtherLine(lineRight)

def process_image(img):
    global imgNumber
    debugFolder = 'vid_debug'
    img_size = (img.shape[1], img.shape[0])
    # Undistort
    dstImg = cv2.undistort(img, mtx, dist, None, mtx)

    # VISUALIZATION OUTPUT
    binFolder = 'vid_debug/bin'
    dstImg = cv2.cvtColor(dstImg, cv2.COLOR_RGB2BGR)
    outFile = '{}/{}_dst.jpg'.format(binFolder, imgNumber)
    laneUtil.writeImg(dstImg, outFile, binary=False)

    # Perspective transform    
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
    src = src20m
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

    warped_dst = cv2.warpPerspective(dstImg, M, warpedImgSize, flags=cv2.INTER_LINEAR)
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

    # VISUALIZATION OUTPUT
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

    leftCrvRad = 0
    rightCrvRad = 0
    leftFailCount = 0
    rightFailCount = 0

    # prep data overlay
    warped_bin_data = np.dstack((warped_bin, warped_bin, warped_bin))*255
    # Identify lanes
    # Left lane
    if not lineLeft.detected:
        leftx_base, rightx_base = laneUtil.findLaneBases(warped_bin)
        ret_left, fit_left, data_img, confidence_left = laneUtil.slidingWindowFit(warped_bin, leftx_base, lanePxColor=(0,0,220), nwindows=slidingWindow_windows, margin=slidingWindow_margin)
        # Want data to be fully opaque. Mask out area to fill with data with black first, then overlay
        img2gray = cv2.cvtColor(data_img,cv2.COLOR_RGB2GRAY)
        warped_bin_data[(img2gray != 0)] = (0,0,0)
        warped_bin_data = cv2.add(warped_bin_data, data_img)
        if not ret_left:       
            cv2.putText(warped_bin_data, 'left lane detection failed', (100, warpedImgSize[1]-20), fontFace, fontScale,(0,0,255), thickness)
            best_fit_left, leftCrvRad, leftFailCount = lineLeft.updateData(fit_left,0.)
        else:
            best_fit_left, leftCrvRad, leftFailCount = lineLeft.updateData(fit_left,confidence_left)
            fit_f_left = np.poly1d(best_fit_left)
            # plot lane
            # Generate x and y values for plotting
            fity = lineLeft.fity
            fit_x_left = fit_f_left(fity)
            #fit_x_left = fit[0]*fity**2 + fit[1]*fity + fit[2]
            for y in range(len(fity)):
                visCircleRad = 3
                cv2.circle(warped_bin_data, (int(fit_x_left[y]), int(fity[y])), visCircleRad, (0,220,220), thickness=-1)
    else:
        ret_leftMargin, fit_left, data_img, confidence_left = laneUtil.marginSearch(warped_bin, lineLeft.best_fit_f, lanePxColor=(0,0,220), margin=marginSearch_margin)
        # Want data to be fully opaque. Mask out area to fill with data with black first, then overlay
        img2gray = cv2.cvtColor(data_img,cv2.COLOR_RGB2GRAY)
        warped_bin_data[(img2gray != 0)] = (0,0,0)
        warped_bin_data = cv2.add(warped_bin_data, data_img)
        #warped_bin_data = cv2.addWeighted(warped_bin_data, 1., data_img, 0.5, 0.)
        if not ret_leftMargin:
            cv2.putText(warped_bin_data, 'left lane margin search failed', (100, warpedImgSize[1]-20), fontFace, fontScale,(0,0,255), thickness)
            best_fit_left, leftCrvRad, leftFailCount = lineLeft.updateData(fit_left, 0.)
        else:
            best_fit_left, leftCrvRad, leftFailCount = lineLeft.updateData(fit_left, confidence_left)
            fit_f_left = np.poly1d(best_fit_left)
            # plot lane
            # Generate x and y values for plotting
            fity = lineLeft.fity
            fit_x_leftMargin = fit_f_left(fity)
            for y in range(len(fity)):
                visCircleRad = 3
                cv2.circle(warped_bin_data, (int(fit_x_leftMargin[y]), int(fity[y])), visCircleRad, (0,220,220), thickness=-1)

    # Right lane
    if not lineRight.detected:
        ret_right, fit_right, data_img, confidence_right = laneUtil.slidingWindowFit(warped_bin, rightx_base, lanePxColor=(220,0,0), nwindows=slidingWindow_windows, margin=slidingWindow_margin)
        img2gray = cv2.cvtColor(data_img,cv2.COLOR_RGB2GRAY)
        warped_bin_data[(img2gray != 0)] = (0,0,0)
        warped_bin_data = cv2.add(warped_bin_data, data_img)
        if not ret_right:
            cv2.putText(warped_bin_data, 'right lane detection failed', (int(warpedImgSize[0]/2)+100, warpedImgSize[1]-20), fontFace, fontScale,(0,0,255), thickness)
            best_fit_right, rightCrvRad, rightFailCount, leftMirrored = lineRight.updateData(fit_right, 0.)
        else:
            best_fit_right, rightCrvRad, rightFailCount, leftMirrored = lineRight.updateData(fit_right, confidence_right)
            fit_f_right = np.poly1d(best_fit_right)
            # plot lane
            # Generate x and y values for plotting
            fity = lineRight.fity
            fit_x_right = fit_f_right(fity)
            for y in range(len(fity)):
                visCircleRad = 3
                cv2.circle(warped_bin_data, (int(fit_x_right[y]), int(fity[y])), visCircleRad, (0,220,220), thickness=-1)
    else:
        ret_rightMargin, fit_right, data_img, confidence_right = laneUtil.marginSearch(warped_bin, lineRight.best_fit_f, lanePxColor=(220,0,0), margin=marginSearch_margin)
        # Want data to be fully opaque. Mask out area to fill with data with black first, then overlay
        img2gray = cv2.cvtColor(data_img,cv2.COLOR_RGB2GRAY)
        warped_bin_data[(img2gray != 0)] = (0,0,0)
        warped_bin_data = cv2.add(warped_bin_data, data_img)
        if not ret_rightMargin:
            cv2.putText(warped_bin_data, 'right lane margin search failed', (int(warpedImgSize[0]/2)+100, warpedImgSize[1]-20), fontFace, fontScale,(0,0,255), thickness)
            best_fit_right, rightCrvRad, rightFailCount, rightMirrored = lineRight.updateData(fit_right, 0.)
        else:
            best_fit_right, rightCrvRad, rightFailCount, rightMirrored = lineRight.updateData(fit_right, confidence_right)
            fit_f_right = np.poly1d(best_fit_right)
            # plot lane
            # Generate x and y values for plotting
            fity = lineRight.fity
            fit_x_rightMargin = fit_f_right(fity)
            for y in range(len(fity)):
                visCircleRad = 3
                cv2.circle(warped_bin_data, (int(fit_x_rightMargin[y]), int(fity[y])), visCircleRad, (0,220,220), thickness=-1)
    
    # Calculate radius of curvature (meters)
    cv2.putText(warped_bin_data, 'left crv rad: {:.0f}m'.format(lineLeft.radius_of_curvature), (int(warpedImgSize[0]/2)-100, warpedImgSize[1]-60), fontFace, fontScale,(0,255,0), thickness)
    cv2.putText(warped_bin_data, 'right crv rad: {:.0f}m'.format(lineRight.radius_of_curvature), (int(warpedImgSize[0]/2)-100, warpedImgSize[1]-30), fontFace, fontScale,(0,255,0), thickness)

    if rightFailCount > 0:
        cv2.putText(warped_bin_data, 'reused', (int(warpedImgSize[0]/2)+50, warpedImgSize[1]-50), fontFace, fontScale,(255,0,0), thickness)
    if leftFailCount > 0:
        cv2.putText(warped_bin_data, 'reused', (50, warpedImgSize[1]-50), fontFace, fontScale,(255,0,0), thickness)
    if rightMirrored > 0:
        cv2.putText(warped_bin_data, 'mirrored', (int(warpedImgSize[0]/2)+50, warpedImgSize[1]-80), fontFace, fontScale,(255,0,0), thickness)
    if leftMirrored > 0:
        cv2.putText(warped_bin_data, 'mirrored', (50, warpedImgSize[1]-80), fontFace, fontScale,(255,0,0), thickness)

    # poly coefficients
    cv2.putText(warped_bin_data, 'l coeff: {:.7f}, {:.2f}, {:.2f}'.format(lineLeft.best_fit[0], lineLeft.best_fit[1], lineLeft.best_fit[2]), (20, 100), fontFace, fontScale,(255,255,255), thickness)
    cv2.putText(warped_bin_data, 'r coeff: {:.7f}, {:.2f}, {:.2f}'.format(lineRight.best_fit[0], lineRight.best_fit[1], lineRight.best_fit[2]), (20, 130), fontFace, fontScale,(255,255,255), thickness)

    # Calculate how far off-center the vehicle is (meters)
    offset = laneUtil.getCarPositionOffCenter(lineLeft.best_fit_f, lineRight.best_fit_f, warpedImgSize[0], warpedImgSize[1], xm_per_pix)
    cv2.putText(warped_bin_data, 'off center: {:.1f}m'.format(offset), (int(warpedImgSize[0]/2)-100, warpedImgSize[1]-90), fontFace, fontScale,(0,255,0), thickness)

    # Calculate lane width
    laneWidthPx, laneWidthMeters = laneUtil.getLaneWidth(lineLeft.best_fit_f, lineRight.best_fit_f, warpedImgSize[0], warpedImgSize[1], xm_per_pix)
    cv2.putText(warped_bin_data, 'lane width: {}px = {:.1f}m'.format(laneWidthPx, laneWidthMeters), (int(warpedImgSize[0]/2)-100, warpedImgSize[1]-120), fontFace, fontScale,(0,255,0), thickness)

    # lane confidence
    cv2.putText(warped_bin_data, 'l conf: {:.2f}'.format(lineLeft.confidence), (20, 160), fontFace, fontScale,(255,255,255), thickness)
    cv2.putText(warped_bin_data, 'r conf: {:.2f}'.format(lineRight.confidence), (20, 190), fontFace, fontScale,(255,255,255), thickness)

    outFile = '{}/debug.{}.jpg'.format(debugFolder, imgNumber)
    laneUtil.writeImg(warped_bin_data, outFile, binary=False)

    # Plot lane and warp back to original image
    curvature = (leftCrvRad+rightCrvRad) / 2
    finalImg = laneUtil.makeFinalDataImage(img, fit_f_left, fit_f_right, Minv, curvature, offset, warpedImgSize[0], warpedImgSize[1])

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
    clip = VideoFileClip(videoFile).subclip(37,43)
    #clip = VideoFileClip(videoFile)
    proc_clip = clip.fl_image(process_image)
    proc_output = '{}_proc.mp4'.format(videoFile.split('.')[0])
    proc_clip.write_videofile(proc_output, audio=False)


if __name__ == '__main__':
    main()