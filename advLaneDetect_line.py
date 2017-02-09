import advLaneDetectUtil as laneUtil
import numpy as np

# Define a class to receive the characteristics of each line detection
class Line:
    def __init__(self, warpedImgSize, xm_per_pix, ym_per_pix, side):
        self.warpedImgSize = warpedImgSize
        self.xm_per_pix = xm_per_pix
        self.ym_per_pix = ym_per_pix
        # was the line detected in the last iteration?
        self.detected = False 
        self.detectionFailedCount = 0
        self.detectionFailedCountThresh = 5
        self.filterSize = 6
        self.poly0ChangeThresh = 0.00005
        self.poly1ChangeThresh = 0.2
        self.confidence = 0.
        self.confidenceThresh = 0.6
        self.mirrorFac = 0.
        self.side = side
        self.laneWidth = 3.7 / self.xm_per_pix
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

    def setLaneWidth(self, laneWidth):
        self.laneWidth = laneWidth

    def fitLine(self, fit):
        if fit != None:
            # START BLEND IN MIRRORED/PREVIOUS FIT ACCORDING TO CONFIDENCE
            # blend in mix of other lane and previous fit depending on confidence            
            if (self.confidence < self.confidenceThresh):
                mirrorFit = None
                prevFit = None
                prevFit = self.getPreviousLine()
                if (self.confidence < self.otherLine.confidence) & (self.otherLine.confidence > self.confidenceThresh):
                    mirrorFit = self.getMirroredOtherLine()
                if ((prevFit != None) & (self.confidence < 0.3)): 
                    fit = prevFit
                if (mirrorFit != None):
                    # exponential to amplify blending on lower confidence fits
                    #confidenceFac = (self.confidence**2) / (self.otherLine.confidence**2)
                    if self.confidence > 0.:
                        confidenceFac = self.confidence / self.otherLine.confidence
                    else:
                        confidenceFac = 0.
                    fit = (confidenceFac*fit) + ((1-confidenceFac)*mirrorFit)
            # END BLEND IN MIRRORED/PREVIOUS FIT ACCORDING TO CONFIDENCE            
            self.recent_fits.append(fit)
            if len(self.recent_fits) > self.filterSize:
                self.recent_fits = self.recent_fits[1:]
            self.best_fit = np.average(self.recent_fits, axis=0)
            self.best_fit_f = np.poly1d(self.best_fit)
            self.radius_of_curvature = laneUtil.getCurveRadius(self.best_fit_f, self.warpedImgSize[1], self.xm_per_pix, self.ym_per_pix)
        else:
            # all broken down
            # if fit is None, it means:
            # - it went all the way back to slidingWindows detection and that failed
            # - mirroring the other lane failed as well because that one didn't have a valid detection
            self.recent_fits = []
            self.best_fit = None

    def getPreviousLine(self):
        if len(self.recent_fits):
            return self.recent_fits[-1]
        else:
            return None

    def reusePreviousLine(self):
        # NOT USED
        self.detectionFailedCount += 1
        if (self.detectionFailedCount < self.detectionFailedCountThresh):
            self.detected = True
        else:
            self.detected = False
        return self.recent_fits[-1]

    def getMirroredOtherLine(self):
        if self.otherLine.detected:
            # VERY simplified projection
            if self.side == 'right':
                x = self.otherLine.best_fit_f(self.fity)+self.laneWidth
            else:
                x = self.otherLine.best_fit_f(self.fity)-self.laneWidth            
            # Fit a second order polynomial to each
            fit = np.polyfit(self.fity, x, 2)
            return fit
        else:
            return None

    def updateData(self, fit, fitConfidence):
        '''
        MAIN FUNCTION
        '''
        #confidence == 0.0 means a fit is not available
        if fitConfidence > 0.:
            self.confidence = fitConfidence / self.warpedImgSize[1]# scale confidence from fit function to 0..1
            self.detected = True
            self.detectionFailedCount = 0
            if len(self.recent_fits):
                compFit = self.recent_fits[-1]
            else:
                compFit = fit
            # health check: monitor changes of polynomial coefficients
            if (abs(fit[1]-compFit[1]) > self.poly1ChangeThresh) | (abs(fit[0]-compFit[0]) > self.poly0ChangeThresh):
                # found a peaky line
                self.confidence = 0.
            self.fitLine(fit)
        else:
            self.confidence = 0.
            self.detected = False
            self.detectionFailedCount += 1
            # poor confidence in line
            self.fitLine(self.getMirroredOtherLine())
        return self.best_fit, self.radius_of_curvature