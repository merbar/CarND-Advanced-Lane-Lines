import advLaneDetectUtil as laneUtil
import numpy as np

# Define a class to receive the characteristics of each line detection
class Line:
    def __init__(self, warpedImgSize, xm_per_pix, ym_per_pix):
        self.warpedImgSize = warpedImgSize
        self.xm_per_pix = xm_per_pix
        self.ym_per_pix = ym_per_pix
        # was the line detected in the last iteration?
        self.detected = False 
        self.detectionFailedCount = 0
        self.detectionFailedCountThresh = 5
        self.filterSize = 1
        self.poly0ChangeThresh = 0.00005
        self.poly1ChangeThresh = 0.2
        self.confidence = 0.
        self.confidenceThresh = warpedImgSize[1]*0.8
        self.mirrored = False
        # x values of the last n fits of the line
        #self.recent_xfitted = [] 
        #average x values of the fitted line over the last n iterations
        #self.bestx = None     
        #polynomial coefficients averaged over the last n iterations
        self.best_fit = None
        self.best_fit_f = None
        #polynomial coefficients for the most recent fits
        self.recent_fits = []  
        #radius of curvature of the line in some units
        self.radius_of_curvature = None 
        #distance in meters of vehicle center from the line
        #self.line_base_pos = None 
        #difference in fit coefficients between last and new fits
        self.diffs = np.array([0,0,0], dtype='float') 
        #x values for detected line pixels
        #self.allx = None
        #y values for detected line pixels
        #self.ally = None
        # prep y values for warped image space
        self.fity = np.linspace(0, self.warpedImgSize[1]-1, self.warpedImgSize[1])

    def addOtherLine(self, otherLine):
        self.otherLine = otherLine

    def fitLine(self, fit, detected=True):
        if detected:
            self.detected = True
            self.detectionFailedCount = 0
        self.recent_fits.append(fit)
        if len(self.recent_fits) > self.filterSize:
            self.recent_fits = self.recent_fits[1:]
        self.best_fit = np.average(self.recent_fits, axis=0)
        self.best_fit_f = np.poly1d(self.best_fit)
        self.radius_of_curvature = laneUtil.getCurveRadius(self.best_fit_f, self.warpedImgSize[1], self.xm_per_pix, self.ym_per_pix)

    def reuseCurrentLine(self):
        self.recent_fits.append(self.recent_fits[-1])
        if len(self.recent_fits) > self.filterSize:
            self.recent_fits = self.recent_fits[1:]
        self.best_fit = np.average(self.recent_fits, axis=0)
        self.best_fit_f = np.poly1d(self.best_fit)
        self.radius_of_curvature = laneUtil.getCurveRadius(self.best_fit_f, self.warpedImgSize[1], self.xm_per_pix, self.ym_per_pix)
        self.detectionFailedCount += 1
        if (self.detectionFailedCount < self.detectionFailedCountThresh):
            self.detected = True
        else:
            self.detected = False

    def mirrorOtherLine(self):
        # plot points from other line over to this with lane width = 3.7m
        laneWidthPx = 3.7 / self.xm_per_pix
        derivative_f = np.polyder(self.otherLine.best_fit)
        #x = self.otherLine.best_fit_f(self.fity)+(derivative_f(self.fity)*laneWidthPx)
        # very simplified projection. should use normal, etc
        x = self.otherLine.best_fit_f(self.fity)+laneWidthPx
        
        # Fit a second order polynomial to each
        fit = np.polyfit(self.fity, x, 2)
        self.mirrored = True
        self.fitLine(fit, detected=False)

    def updateData(self, fit, confidence):
        '''
        MAIN FUNCTION
        '''
        self.confidence = confidence
        self.mirrored = False
        # confidence = 0.0 also means a fit is not available

        # ySize*0.625 is a good threshold to start blending in the other curve. BUT, need to also take confidence in the other curve into account

        # good confidence in line?
        if self.confidence > self.confidenceThresh:
            if len(self.recent_fits):
                compFit = self.recent_fits[-1]
            else:
                compFit = fit
            # health check: monitor changes of polynomial coefficients
            if (abs(fit[1]-compFit[1]) < self.poly1ChangeThresh) & (abs(fit[0]-compFit[0]) < self.poly0ChangeThresh):
                # found a good line
                self.fitLine(fit)
            else:
                # found a peaky line
                self.reuseCurrentLine()
        else:
            # poor confidence in line
            self.mirrorOtherLine()
        return self.best_fit, self.radius_of_curvature, self.detectionFailedCount, self.mirrored