import numpy as np
import cv2

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
        img = cv2.cvtColor(img, cv2.COLOR_RGBGRAY)
    elif colorspace == 'hsv':
        img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    elif colorspace == 'hls':
        img = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    elif colorspace == 'lab':
        img = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    elif colorspace == 'luv':
        img = cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
    elif colorspace == 'yuv':
        img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2RGB)
    
    # isolate channel
    if colorspace != 'gray':
        img = img[:,:,useChannel]

    # apply image mask
    if mask is not None:
        imgMask = np.zeros_like(img)    
        ignore_mask_color = 255
        # filling pixels inside the polygon defined by "vertices" with the fill color    
        cv2.fillPoly(imgMask, mask, ignore_mask_color)
        # returning the image only where mask pixels are nonzero
        img = cv2.bitwise_and(img, imgMask)
    return img
                
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
        sobelX_abs = np.absolute(cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=sobel_kernel))
        scaled_sobelX = np.uint8(255*sobelX_abs/np.max(sobelX_abs))
        sobelY_abs = np.absolute(cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=sobel_kernel))
        scaled_sobelY = np.uint8(255*sobelY_abs/np.max(sobelY_abs))
        # use np.arctan2(abs_sobely, abs_sobelx) to calculate the direction of the gradient 
        gradients = np.arctan2(scaled_sobelY, scaled_sobelX)
        binary[(gradients >= threshold[0]) & (gradients <= threshold[1])] = 1
    else:
        binary[(img >= threshold[0]) & (img <= threshold[1])] = 1  
    return binary


def denoiseBinary(binImg, binImgReplace, stepSize=50, noiseColumnThresh = 120, pixelNumThres = 150):
    '''
    Detects areas of noise and replaces them with the same chunk from the second image OR zero if second image is also noisy
    Noise is defined as lots of positive values on the x-axis
    inputs: image to remove noise from
            image to use for replacement
            [chunks of the image in y that get processed]
            [number of columns with a positive value to use as noise threshold]
    '''
    out_img = binImg.copy()
    img_size = (binImg.shape[1], binImg.shape[0])
    for y in np.arange(0, img_size[1], stepSize):
        topRange = y+stepSize
        histBinImg = np.sum(binImg[y:topRange:], axis=0)
        histBinImgReplace = np.sum(binImgReplace[y:topRange:], axis=0)
        nonzeroX_histBin = histBinImg.nonzero()[0]
        nonzeroX_histBinRepl = histBinImgReplace.nonzero()[0]
        if (len(set(np.unique(nonzeroX_histBin))) > noiseColumnThresh) & (len(nonzeroX_histBin) > pixelNumThres):
            if (len(nonzeroX_histBinRepl) <= noiseColumnThresh):
                out_img[y:topRange:] = binImgReplace[y:topRange:]
            else:
                out_img[y:topRange:] = 0
    # special case for the remainder in y
    if y < img_size[1]:
        topRange = img_size[1]
        remainder = topRange-y
        noiseColumnThresh = noiseColumnThresh * (remainder/stepSize)
        histBinImg = np.sum(binImg[y:topRange:], axis=0)
        histBinImgReplace = np.sum(binImgReplace[y:topRange:], axis=0)
        if (len(histBinImg.nonzero()[0]) > noiseColumnThresh):
            if (len(histBinImgReplace.nonzero()[0]) <= noiseColumnThresh):
                out_img[y:topRange:] = binImgReplace[y:topRange:]
            else:
                out_img[y:topRange:] = 0
    return out_img


def writeImg(img, outFile, binary=False):
    if binary:
        # scale to 8-bit (0 - 255)
        img = np.uint8(255*img)
    cv2.imwrite(outFile, img)
    

def findLaneBases(binary_warped):
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
    return leftx_base, rightx_base

def slidingWindowFit(binary_warped, x_base, nwindows=9, margin=100, minpix=75, lanePxColor=(0,0,220), visCircleRad=5, fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, thickness=2):
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
    '''
    nWindowsForSuccess = np.floor(nwindows/3)
    filterPrevWindowThresFlipSign = 20 # make threshold large to disable filtering
    filterPrevWindowThres = 70 # make threshold large to disable filtering
    
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
            #         But have to ignore influence of first window (always has very large change in x)
            if (((x_prevChange<0)==(x_currentChange<0)) or (abs(x_prevChange-x_currentChange) < filterPrevWindowThresFlipSign) or window==1):
                # Append indices to the lists
                lane_inds.append(good_inds)
                valid_window = True
                valid_window_count += 1
            else:
                # if it turns out bad, reset location and keep moving ROI for next window in the previous direction
                x_current = int(x_prev+(x_prevChange/2))
                 # if it turns out bad, reset location
                #x_current = x_prev
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
        return False, None, data_img

    # Concatenate the arrays of indices
    lane_inds = np.concatenate(lane_inds)

    # Extract left and right line pixel positions
    x = nonzerox[lane_inds]
    y = nonzeroy[lane_inds] 
 
    # Fit a second order polynomial to each
    fit = np.polyfit(y, x, 2)
    fit_f = np.poly1d(fit)
    return True, fit, data_img


def marginSearch(binary_warped, fit_f, margin=100, lanePxColor=(0,0,220), laneColor=(0,150,0)):
    '''
    input: binary warped img
           curve fit function
           margin: search radius +/- margin
           [lane plot color]
           [lane color]
    returns: return status (True/False), polyFit function for lane, image with visualizations
    based on: Udacity Project 4 lesson, unit 33
    '''
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    lane_inds = ((nonzerox > (fit_f(nonzeroy) - margin)) & (nonzerox < (fit_f(nonzeroy) + margin))) 
    # Again, extract line pixel positions
    x = nonzerox[lane_inds]
    y = nonzeroy[lane_inds] 
    # Fit a second order polynomial to each
    fit = np.polyfit(y, x, 2)
    fit_f = np.poly1d(fit)
    # Visualize
    # Create an output image to draw on and  visualize the result
    size = binary_warped.shape[0], binary_warped.shape[1], 3
    data_img = np.zeros(size, dtype=np.uint8)
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
    data_img[nonzeroy[lane_inds], nonzerox[lane_inds]] = lanePxColor
    return True, fit, data_img

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
    y_eval = np.max(ploty)
    x = fit_f(ploty)

    # Fit new polynomials to x,y in world space
    fit_cr = np.polyfit(ploty*ym_per_pix, x*xm_per_pix, 2)   
    # Calculate the new radii of curvature
    curverad = ((1 + (2*fit_cr[0]*y_eval*ym_per_pix + fit_cr[1])**2)**1.5) / np.absolute(2*fit_cr[0])
    # Now our radius of curvature is in meters
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

def makeFinalImage(img, fit_f_left, fit_f_right, Minv, curvature, offCenter, imgSizeX, imgSizeY, laneColor=(0,255,0), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, thickness=2):
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
    fit_xLeft = fit_f_left(ploty)
    fit_xRight = fit_f_right(ploty)
    line_window1 = np.array([np.transpose(np.vstack([fit_xLeft, ploty]))])
    line_window2 = np.array([np.flipud(np.transpose(np.vstack([fit_xRight, ploty])))])
    line_pts = np.hstack((line_window1, line_window2))
    cv2.fillPoly(data_img, np.int_([line_pts]), laneColor)
    
    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    data_img = cv2.warpPerspective(data_img, Minv, (img.shape[1], img.shape[0]))
    
    img = cv2.addWeighted(img, 1., data_img, 1., 0.)
    cv2.putText(img, 'curve radius: {:.0f}m'.format(curvature), (600, 20), fontFace, fontScale,(0,255,0), thickness)
    cv2.putText(img, 'off center: {:.1f}m'.format(offCenter), (600, 100), fontFace, fontScale,(0,255,0), thickness)
    return img