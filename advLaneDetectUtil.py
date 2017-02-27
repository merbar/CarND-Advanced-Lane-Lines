import numpy as np
import cv2

curvature_ar = []

# Create thresholded binary image
def makeGrayImg(img, mask=None, colorspace='rgb', useChannel=0):
    '''
    Returns a grey image based on the following inputs
    - mask
    - choice of color space
    - choice of channel(s) to use
    '''
    # color space conversion
    if colorspace == 'gray':
        cvt_img = cv2.cvtColor(img, cv2.COLOR_BGRGRAY)
    elif colorspace == 'hsv':
        cvt_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    elif colorspace == 'hls':
        cvt_img = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    elif colorspace == 'lab':
        cvt_img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    elif colorspace == 'luv':
        cvt_img = cv2.cvtColor(img, cv2.COLOR_BGR2LUV)
    elif colorspace == 'yuv':
        cvt_img = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    else: 
        cvt_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # isolate channel
    if colorspace != 'gray':
        cvt_img = cvt_img[:,:,useChannel]     

    # apply image mask
    if mask is not None:
        imgMask = np.zeros_like(cvt_img)    
        ignore_mask_color = 255
        # filling pixels inside the polygon defined by "vertices" with the fill color    
        cv2.fillPoly(imgMask, mask, ignore_mask_color)
        # returning the image only where mask pixels are nonzero
        cvt_img = cv2.bitwise_and(cvt_img, imgMask)
    return cvt_img
                
def makeBinaryImg(img, threshold=(0,255), mode='simple', sobel_kernel=7):
    '''
    Returns a binary image based on the following inputs
    - threshold
    - threshold mode
    -- 'dir' requires a non-8 bit threshold
    '''
    binary = np.zeros_like(img)
    if mode == 'sobelX':
        sobelX_abs = np.absolute(cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=sobel_kernel))
        scaled_sobelX = np.uint8(255*sobelX_abs/np.max(sobelX_abs))
        binary[(scaled_sobelX >= threshold[0]) & (scaled_sobelX <= threshold[1])] = 1
    elif mode == 'sobelY':
        sobelY_abs = np.absolute(cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=sobel_kernel))
        scaled_sobelY = np.uint8(255*sobelY_abs/np.max(sobelY_abs))
        binary[(scaled_sobelY >= threshold[0]) & (scaled_sobelY <= threshold[1])] = 1
    elif mode == 'mag':
        sobelX = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
        sobelY = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
        # calculate the magnitude
        sobel_mag = np.sqrt(np.square(sobelX) + np.square(sobelY))
        # scale to 8-bit (0 - 255) and convert to type = np.uint8
        sobel_scale = np.uint8(255*sobel_mag/np.max(sobel_mag))
        # create a binary mask where mag thresholds are met
        binary[(sobel_scale >= threshold[0]) & (sobel_scale <= threshold[1])] = 1
    elif mode == 'dir':
        sobelX_abs = np.abs(cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=sobel_kernel))
        #scaled_sobelX = np.uint8(255*sobelX_abs/np.max(sobelX_abs))
        sobelY_abs = np.abs(cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=sobel_kernel))
        #scaled_sobelY = np.uint8(255*sobelY_abs/np.max(sobelY_abs))
        # use np.arctan2(abs_sobely, abs_sobelx) to calculate the direction of the gradient 
        gradients = np.arctan2(sobelY_abs, sobelX_abs)
        binary[(gradients >= threshold[0]) & (gradients <= threshold[1])] = 1
    else:
        binary[(img >= threshold[0]) & (img <= threshold[1])] = 1  
    return binary

def binaryToGray(img):
    '''
    returns gray image from binary input
    '''
    return np.uint8(255*img/np.max(img))


def denoiseBinary(binImg, stepSize=50, noiseColumnThresh = 100):
    '''
    Goes through the image in chunks to detect areas of noise and replaces them with zero. If too many chunks are noisy, image is discarded.
    Noise is defined as lots of positive values on the x-axis
    inputs: image to remove noise from
            [chunks of the image in y that get processed]
            [number of columns with a positive value to use as noise threshold]
    '''
    pixelNumThres = noiseColumnThresh
    img_size = (binImg.shape[1], binImg.shape[0])
    empty_img = np.zeros_like(binImg)
    out_img = binImg.copy()
    noiseCount = 0
    noiseCountTresh = (img_size[1]/stepSize)*0.5
    for y in np.arange(0, img_size[1], stepSize):
        topRange = y+stepSize
        histBinImg = np.sum(binImg[y:topRange:], axis=0)
        nonzeroX_histBin = histBinImg.nonzero()[0]
        if (len(set(np.unique(nonzeroX_histBin))) > noiseColumnThresh) & (len(nonzeroX_histBin) > pixelNumThres):
            out_img[y:topRange:] = 0
            noiseCount +=1
    # special case for the remainder in y
    if y < img_size[1]:
        topRange = img_size[1]
        remainder = topRange-y
        noiseColumnThresh = noiseColumnThresh * (remainder/stepSize)
        pixelNumThres = pixelNumThres * (remainder/stepSize)
        histBinImg = np.sum(binImg[y:topRange:], axis=0)
        nonzeroX_histBin = histBinImg.nonzero()[0]
        if (len(set(np.unique(nonzeroX_histBin))) > noiseColumnThresh) & (len(nonzeroX_histBin) > pixelNumThres):
            out_img[y:topRange:] = 0
            noiseCount +=1
    # if more than 1/2 of image is noisy, throw it out
    if noiseCount > noiseCountTresh:
        return empty_img
    else:
        return out_img

def writeImg(img, outFile, binary=False):
    if binary:
        # scale to 8-bit (0 - 255)
        img = np.uint8(255*img)
    cv2.imwrite(outFile, img)
    

def findLaneBases(binary_warped, xm_per_pix, laneWidthPx):
    '''
    input: binary warped img
    returns: x coordinates of estimated left/right lane bases
    from: Udacity Project 4 lesson, unit 33
    '''
    histogram = np.sum(binary_warped[int(binary_warped.shape[0]/1.75):,:], axis=0)
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0]/2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint
    # VERY simple sanity check
    detectedLaneWidth = rightx_base - leftx_base
    if (detectedLaneWidth > (laneWidthPx*0.8)) & (detectedLaneWidth < (laneWidthPx*1.2)):
        return leftx_base, rightx_base
    else:
        return None, None

def lineConfidence(lanePxY):
    confidence = np.amax(lanePxY) - np.amin(lanePxY)
    return confidence

def slidingWindowFit(binary_warped, x_base, nwindows=9, margin=100, minpix=75, lanePxColor=(0,0,220), visCircleRad=5, fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=0.5, thickness=1):
    '''
    input: binary warped img
           origin of lane in x
           [nwindows: number of sliding windows]
           [margin: width of the windows +/- margin]
           [minpix: minimum number of pixels found to recenter window]
           [lane plot color]
           [font]
           [font scale]
    returns: return status (True/False), polyFit function for lane, image with visualizations
    based on: Udacity Project 4 lesson, unit 33
    TODO: keep track of rate of changes of last n-windows. Make outlier decision more based on that as we get more good windows
    '''
    nWindowsForSuccess = np.floor(nwindows/3)
    filterPrevWindowThresFlipSign = 20 # make threshold large to disable filtering
    filterPrevWindowThres = 40 # make threshold large to disable filtering
    
    # Create an output image to draw on and  visualize the result
    size = binary_warped.shape[0], binary_warped.shape[1], 3
    data_img = np.zeros(size, dtype=np.uint8)

    # Set height of windows
    window_height = np.int(binary_warped.shape[0]/nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated for each window
    x_current = x_base
    # Create empty lists to receive left and right lane pixel indices
    lane_inds = []
    
    # Step through the windows one by one
    x_prev = 0
    x_prevChange = 0
    valid_window_count = 0
    for window in range(nwindows):
        '''
        keep track of previous search window position/rate of change in the curve to prevent outliers
        '''
        valid_window = False
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window+1)*window_height
        win_y_high = binary_warped.shape[0] - window*window_height
        win_y = int(win_y_low+(window_height/2))
        win_x_low = x_current - margin
        win_x_high = x_current + margin
        # Identify the nonzero pixels in x and y within the window
        # take .nonzero()[0] since all we really need is a simple count. Don't care where they actually are
        good_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_x_low) & (nonzerox < win_x_high)).nonzero()[0]
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_inds) > minpix:
            x_current = np.int(np.mean(nonzerox[good_inds]))            
            x_currentChange = x_current - x_prev
            # FILTER: If sign flips and value change is above threshold, ignore window
            # But have to ignore influence of first two windows (always has very large change in x)
            if (window in [0,1]) | (((x_prevChange<0)==(x_currentChange<0)) & (abs(x_prevChange-x_currentChange) < filterPrevWindowThres)) | (((x_prevChange<0)!=(x_currentChange<0)) & (abs(x_prevChange-x_currentChange) < filterPrevWindowThresFlipSign)):
            #if (((x_prevChange<0)==(x_currentChange<0)) or (abs(x_prevChange-x_currentChange) < filterPrevWindowThresFlipSign) or window==1):
                # Append indices to the lists
                lane_inds.append(good_inds)
                valid_window = True
                valid_window_count += 1
            else:
                # if it turns out bad, reset location and keep moving ROI for next window in the previous direction
                x_current = int(x_prev+(x_prevChange/2))
                # if it turns out bad, reset location
        else:
            #x_current = int(x_prev+(x_prevChange/2))
            #x_currentChange = x_current - x_prev
            x_currentChange = 0
            
        # Draw the windows on the visualization image
        visCircleRad = 5
        if valid_window:
            color = (0,255,0)
            # Color in line pixels
            data_img[nonzeroy[good_inds], nonzerox[good_inds]] = lanePxColor
        else:
            color = (100,100,100)
        cv2.rectangle(data_img,(win_x_low,win_y_low),(win_x_high,win_y_high),color, 2)
        cv2.circle(data_img, (x_current, win_y), visCircleRad, color, thickness=-1)
        cv2.putText(data_img, '%s'%x_currentChange, (x_current-70, win_y), fontFace, fontScale,color, thickness)
        x_prev = x_current
        x_prevChange = x_currentChange
        
    # check if enough good points were detected for lane to be valid
    if valid_window_count < nWindowsForSuccess:
        return None, data_img, 0.

    # Concatenate the arrays of indices
    lane_inds = np.concatenate(lane_inds)

    # Extract left and right line pixel positions
    x = nonzerox[lane_inds]
    y = nonzeroy[lane_inds] 
 
    # Fit a second order polynomial to each
    fit = np.polyfit(y, x, 2)
    fit_f = np.poly1d(fit)
    return fit, data_img, lineConfidence(y)

def marginSearch(binary_warped, fit_f, margin=50, lanePxColor=(0,0,220), laneColor=(0,150,0)):
    '''
    input: binary warped img
           curve fit function
           margin: search radius +/- margin
           [lane plot color]
           [lane color]
    returns: return status (True/False), polyFit function for lane, image with visualizations
    based on: Udacity Project 4 lesson, unit 33
    '''
    # Create an output image to draw on and  visualize the result
    size = binary_warped.shape[0], binary_warped.shape[1], 3
    data_img = np.zeros(size, dtype=np.uint8)
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    lane_inds = ((nonzerox > (fit_f(nonzeroy) - margin)) & (nonzerox < (fit_f(nonzeroy) + margin))) 
    # Again, extract line pixel positions
    x = nonzerox[lane_inds]
    y = nonzeroy[lane_inds] 
    # early exit in edge cases
    if len(x) == 0:
        fit = None
        return fit, data_img, 0.
    # Fit a second order polynomial
    fit = np.polyfit(y, x, 2)
    fit_f = np.poly1d(fit)
    # Generate x and y values for plotting
    fity = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
    fit_x = fit_f(fity)
    # Generate a polygon to illustrate the search window area
    # And recast the x and y points into usable format for cv2.fillPoly()
    line_window1 = np.array([np.transpose(np.vstack([fit_x-margin, fity]))])
    line_window2 = np.array([np.flipud(np.transpose(np.vstack([fit_x+margin, fity])))])
    line_pts = np.hstack((line_window1, line_window2))
    # Draw the lane onto the warped blank image
    cv2.fillPoly(data_img, np.int_([line_pts]), laneColor)    
    # Color in line pixels
    #data_img[nonzeroy[lane_inds], nonzerox[lane_inds]] = lanePxColor
    data_img[y, x] = lanePxColor
    confidence = lineConfidence(y)
    if confidence < 0.25:
        fit = None
    return fit, data_img, confidence

def getCurveRadius(fit_f, imgSizeY, xm_per_pix, ym_per_pix):
    '''
    input: curve fit function
           size of image in y
           scale of image in meters/pixel in x and y
    returns: radius of curvature in meters
    based on: Udacity Project 4 lesson, unit 34
    '''   
    ploty = np.linspace(0, imgSizeY-1, imgSizeY)
    # Define y-value where we want radius of curvature. Choose the maximum y-value, corresponding to the bottom of the image
    x = fit_f(ploty)
    # Fit new polynomials to x,y in world space
    fit_cr = np.polyfit(ploty*ym_per_pix, x*xm_per_pix, 2)   
    # Calculate the new radii of curvature
    y_eval1 = np.max(ploty)-20
    curverad1 = ((1 + (2*fit_cr[0]*y_eval1*ym_per_pix + fit_cr[1])**2)**1.5) / np.absolute(2*fit_cr[0])
    y_eval2 = np.max(ploty)-60
    curverad2 = ((1 + (2*fit_cr[0]*y_eval2*ym_per_pix + fit_cr[1])**2)**1.5) / np.absolute(2*fit_cr[0])
    y_eval3 = np.max(ploty)-100
    curverad3 = ((1 + (2*fit_cr[0]*y_eval3*ym_per_pix + fit_cr[1])**2)**1.5) / np.absolute(2*fit_cr[0])
    curverad = (curverad1 + curverad2 + curverad3) / 3
    return curverad

def getCarPositionOffCenter(fit_f_left, fit_f_right, imgSizeX, imgSizeY, xm_per_pix):
    '''
    inputs: function for left and right lanes
            size of image in x
            scale of image in meters/pixel in x
    returns: car position off center in meters
    '''
    base_left = fit_f_left(imgSizeY)
    base_right = fit_f_right(imgSizeY)
    centerOfLanes = base_left+((base_right-base_left)/2)
    offset = ((imgSizeX/2)-centerOfLanes)*xm_per_pix
    return offset

def getLaneWidth(fit_f_left, fit_f_right, imgSizeX, imgSizeY, xm_per_pix):
    '''
    inputs: function for left and right lanes
            size of image in x
            scale of image in meters/pixel in x
    returns: lane width in pixels and meters
    '''
    base_left = fit_f_left(imgSizeY)
    base_right = fit_f_right(imgSizeY)
    #centerOfLanes = base_left+((base_right-base_left)/2)
    laneWidthPx = int(base_right - base_left)
    laneWidthMeters = laneWidthPx * xm_per_pix
    return laneWidthPx, laneWidthMeters

def makeDiagnosticsImage(warped_bin, lineLeft, lineRight, xm_per_pix, data_imgLeft, data_imgRight, fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=0.5, thickness=1):
    warpedImgSize = (warped_bin.shape[1], warped_bin.shape[0])
    # prep data overlay
    warped_bin_data = np.dstack((warped_bin, warped_bin, warped_bin))*255
    if data_imgLeft != None:
        # Want data from lane searches to be fully opaque. Mask out area to fill with data with black first, then overlay
        img2gray = cv2.cvtColor(data_imgLeft,cv2.COLOR_RGB2GRAY)
        warped_bin_data[(img2gray != 0)] = (0,0,0)
        warped_bin_data = cv2.add(warped_bin_data, data_imgLeft)
    if data_imgRight != None:
        img2gray = cv2.cvtColor(data_imgRight,cv2.COLOR_RGB2GRAY)
        warped_bin_data[(img2gray != 0)] = (0,0,0)
        warped_bin_data = cv2.add(warped_bin_data, data_imgRight)

    if lineLeft.best_fit != None:
        fity = lineLeft.fity
        fitx = lineLeft.best_fit_f(fity)
        for y in range(len(fity)):
            visCircleRad = 3
            cv2.circle(warped_bin_data, (int(fitx[y]), int(fity[y])), visCircleRad, (0,220,220), thickness=-1)
        cv2.putText(warped_bin_data, 'left crv rad: {:.0f}m'.format(lineLeft.radius_of_curvature), (int(warpedImgSize[0]/2)-100, warpedImgSize[1]-60), fontFace, fontScale,(0,255,0), thickness)
        cv2.putText(warped_bin_data, 'l coeff: {:.7f}, {:.2f}, {:.2f}'.format(lineLeft.best_fit[0], lineLeft.best_fit[1], lineLeft.best_fit[2]), (20, 100), fontFace, fontScale,(255,255,255), thickness)
        cv2.putText(warped_bin_data, 'l conf: {:.2f}'.format(lineLeft.confidence), (20, 160), fontFace, fontScale,(255,255,255), thickness)
    if lineRight.best_fit != None:
        fity = lineRight.fity
        fitx = lineRight.best_fit_f(fity)
        for y in range(len(fity)):
            visCircleRad = 3
            cv2.circle(warped_bin_data, (int(fitx[y]), int(fity[y])), visCircleRad, (0,220,220), thickness=-1)
        cv2.putText(warped_bin_data, 'right crv rad: {:.0f}m'.format(lineRight.radius_of_curvature), (int(warpedImgSize[0]/2)-100, warpedImgSize[1]-30), fontFace, fontScale,(0,255,0), thickness)
        cv2.putText(warped_bin_data, 'r coeff: {:.7f}, {:.2f}, {:.2f}'.format(lineRight.best_fit[0], lineRight.best_fit[1], lineRight.best_fit[2]), (20, 130), fontFace, fontScale,(255,255,255), thickness)
        cv2.putText(warped_bin_data, 'r conf: {:.2f}'.format(lineRight.confidence), (20, 190), fontFace, fontScale,(255,255,255), thickness)
        # General rule: if one lane has a good fit, then so does the other! At worst, it is mirrored.
        # Calculate how far off-center the vehicle is (meters)
        offset = getCarPositionOffCenter(lineLeft.best_fit_f, lineRight.best_fit_f, warpedImgSize[0], warpedImgSize[1], xm_per_pix)
        cv2.putText(warped_bin_data, 'off center: {:.1f}m'.format(offset), (int(warpedImgSize[0]/2)-100, warpedImgSize[1]-90), fontFace, fontScale,(0,255,0), thickness)
        # Calculate lane width
        laneWidthPx, laneWidthMeters = getLaneWidth(lineLeft.best_fit_f, lineRight.best_fit_f, warpedImgSize[0], warpedImgSize[1], xm_per_pix)
        cv2.putText(warped_bin_data, 'lane width: {}px = {:.1f}m'.format(laneWidthPx, laneWidthMeters), (int(warpedImgSize[0]/2)-100, warpedImgSize[1]-120), fontFace, fontScale,(0,255,0), thickness)
    return warped_bin_data
    

def makeTextDataImage(warped_bin, lineLeft, lineRight, xm_per_pix, fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=0.8, thickness=1):
    warpedImgSize = (warped_bin.shape[1], warped_bin.shape[0])
    imgSize = (390, 500, 3)
    fontColor = (255,255,255)
    data_img = np.zeros(imgSize, dtype=np.uint8)
    if lineLeft.best_fit != None:
        cv2.putText(data_img, 'left crv rad: {:.0f}m'.format(lineLeft.radius_of_curvature), (10, 30), fontFace, fontScale, fontColor, thickness, lineType=cv2.LINE_AA)
        cv2.putText(data_img, 'l coeff: {:.7f}, {:.2f}, {:.2f}'.format(lineLeft.best_fit[0], lineLeft.best_fit[1], lineLeft.best_fit[2]), (10, 110), fontFace, fontScale, fontColor, thickness, lineType=cv2.LINE_AA)
        cv2.putText(data_img, 'l confidence: {:.2f}'.format(lineLeft.confidence), (10, 180), fontFace, fontScale, fontColor, thickness, lineType=cv2.LINE_AA)
    if lineRight.best_fit != None:
        cv2.putText(data_img, 'right crv rad: {:.0f}m'.format(lineRight.radius_of_curvature), (10, 60), fontFace, fontScale, fontColor, thickness, lineType=cv2.LINE_AA)
        cv2.putText(data_img, 'r coeff: {:.7f}, {:.2f}, {:.2f}'.format(lineRight.best_fit[0], lineRight.best_fit[1], lineRight.best_fit[2]), (10, 140), fontFace, fontScale, fontColor, thickness, lineType=cv2.LINE_AA)
        cv2.putText(data_img, 'r confidence: {:.2f}'.format(lineRight.confidence), (10, 210), fontFace, fontScale, fontColor, thickness, lineType=cv2.LINE_AA)
        # General rule: if one lane has a good fit, then so does the other! At worst, it is mirrored.
        # Calculate how far off-center the vehicle is (meters)
        offset = getCarPositionOffCenter(lineLeft.best_fit_f, lineRight.best_fit_f, warpedImgSize[0], warpedImgSize[1], xm_per_pix)
        cv2.putText(data_img, 'off center: {:.1f}m'.format(offset), (10, 260), fontFace, fontScale, fontColor, thickness, lineType=cv2.LINE_AA)
        # Calculate lane width
        laneWidthPx, laneWidthMeters = getLaneWidth(lineLeft.best_fit_f, lineRight.best_fit_f, warpedImgSize[0], warpedImgSize[1], xm_per_pix)
        cv2.putText(data_img, 'lane width: {}px = {:.1f}m'.format(laneWidthPx, laneWidthMeters), (10, 310), fontFace, fontScale, fontColor, thickness, lineType=cv2.LINE_AA)
    return data_img

def makeCtrlImg(finalImg, textDataImg, diagImg, warped_bin):
    finalImg = cv2.cvtColor(finalImg, cv2.COLOR_RGB2BGR)
    imgSize = (750, 1280 , 3)
    ctrl_img = np.zeros(imgSize, dtype=np.uint8)
    #ctrl_img = ctrl_img + (30,30,30)
    smallFinal = cv2.resize(finalImg, (0,0), fx=0.5, fy=0.5)
    smallFinalSize = (smallFinal.shape[1], smallFinal.shape[0])
    ctrl_img[0:smallFinalSize[1], 0:smallFinalSize[0]] = smallFinal

    warped_bin = np.dstack((warped_bin, warped_bin, warped_bin))*255
    xOffset = smallFinalSize[0]+20
    yOffset = 35
    smallWarped = cv2.resize(warped_bin, (0,0), fx=0.45, fy=0.45)
    smallWarpedSize = (smallWarped.shape[1], smallWarped.shape[0])
    ctrl_img[yOffset:yOffset+smallWarpedSize[1], xOffset:xOffset+smallWarpedSize[0]] = smallWarped

    xOffset = xOffset + smallWarpedSize[0]+110
    yOffset = 35
    smallDiag = cv2.resize(diagImg, (0,0), fx=0.45, fy=0.45)
    smallDiagSize = (smallDiag.shape[1], smallDiag.shape[0])
    ctrl_img[yOffset:yOffset+smallDiagSize[1], xOffset:xOffset+smallDiagSize[0]] = smallDiag

    yOffset = smallFinalSize[1]
    #smallDiag = cv2.resize(textDataImg, (0,0), fx=0.5, fy=0.5)
    smallTextSize = (textDataImg.shape[1], textDataImg.shape[0])
    ctrl_img[yOffset:yOffset+smallTextSize[1], 0:smallTextSize[0]] = textDataImg

    cv2.putText(ctrl_img, 'binary image', (670,25), cv2.FONT_HERSHEY_DUPLEX, 1, (255,255,255), 1, lineType=cv2.LINE_AA)
    cv2.putText(ctrl_img, 'lane detection', (1000,25), cv2.FONT_HERSHEY_DUPLEX, 1, (255,255,255), 1, lineType=cv2.LINE_AA)
    return ctrl_img

def makeFinalLaneImage(img, lineLeft, lineRight, Minv, imgSizeX, imgSizeY, xm_per_pix, laneColor=(0,255,0), fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=0.9, thickness=1, featherEdge=False):
    global curvature_ar
    '''
    inputs: original undistorted image
            function for left and right lanes
            inverse perspective transform matrix
            curve radius
            position of car off-center
            size of image in x and y
            [lane color]
            [font]
            [font scale]
            [font thickness]
    output: final image
    '''
    size = imgSizeY, imgSizeX, 3
    data_img = np.zeros(size, dtype=np.uint8)
    ploty = np.linspace(0, imgSizeY-1, imgSizeY)
    if (lineLeft.best_fit != None) & (lineRight.best_fit != None):
        fit_xLeft = lineLeft.best_fit_f(ploty)
        fit_xRight = lineRight.best_fit_f(ploty)
        line_window1 = np.array([np.transpose(np.vstack([fit_xLeft, ploty]))])
        line_window2 = np.array([np.flipud(np.transpose(np.vstack([fit_xRight, ploty])))])
        line_pts = np.hstack((line_window1, line_window2))
        cv2.fillPoly(data_img, np.int_([line_pts]), laneColor)
        # draw center of lane
        centerFit = (lineLeft.best_fit + lineRight.best_fit) / 2
        centerFit_f = np.poly1d(centerFit)
        fit_xLeft = centerFit_f(ploty)-3
        fit_xRight = centerFit_f(ploty)+3
        line_window1 = np.array([np.transpose(np.vstack([fit_xLeft, ploty]))])
        line_window2 = np.array([np.flipud(np.transpose(np.vstack([fit_xRight, ploty])))])
        line_pts = np.hstack((line_window1, line_window2))
        cv2.fillPoly(data_img, np.int_([line_pts]), (0,0,255))
    
    if featherEdge:
        # feather the far edge... BUT THIS IS CURRENTLY RIDICULOUSLY SLOW. Need to somehow do this in numpy
        featherRange = 600
        # set it to 1 on bottom, 0 on top (done), interpolate in between (y=475 is top of ROI)
        for y in range(0,featherRange):
            if y==0:
                scale = 0.0
            else:
                scale = y/featherRange
            for x in range(0,imgSizeX):
                data_img[y][x][0] = int(data_img[y][x][0] * scale)
                data_img[y][x][1] = int(data_img[y][x][1] * scale)
                data_img[y][x][2] = int(data_img[y][x][2] * scale)
        # Warp the blank back to original image space using inverse perspective matrix (Minv)
    data_img = cv2.warpPerspective(data_img, Minv, (img.shape[1], img.shape[0]))

    img = cv2.addWeighted(img, 1., data_img, 0.3, 0.)
    cv2.ellipse(img, (int(img.shape[1]/2),105), (220,65), 0, 0, 360, (50,50,50), -1, lineType=cv2.LINE_AA)
        
    if (lineLeft.best_fit != None) & (lineRight.best_fit != None):
        curvature_ar.append(min(int((lineLeft.radius_of_curvature + lineRight.radius_of_curvature) / 2), 25000))
        if len(curvature_ar) > 30:
            curvature_ar = curvature_ar[1:]
        curvature = int(np.mean(curvature_ar))
        if curvature > 9000:
            curvature = 'straight'
        else:
            curvature = '{}m'.format(curvature)
        offCenter = getCarPositionOffCenter(lineLeft.best_fit_f, lineRight.best_fit_f, imgSizeX, imgSizeY, xm_per_pix)
        if offCenter > 0.:
            offCenter = '{:.1f}m right'.format(abs(offCenter))
        else:
            offCenter = '{:.1f}m left'.format(abs(offCenter))
        
        cv2.putText(img, 'curve radius: {}'.format(curvature), (int(img.shape[1]/2)-165, 100), fontFace, fontScale,(255,255,255), thickness, lineType=cv2.LINE_AA)
        cv2.putText(img, 'off center: {}'.format(offCenter), (int(img.shape[1]/2)-130, 130), fontFace, fontScale,(255,255,255), thickness, lineType=cv2.LINE_AA)
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    '''
    outFile = 'vid_debug/final.jpg'
    writeImg(img, outFile, binary=False)
    '''
    return img