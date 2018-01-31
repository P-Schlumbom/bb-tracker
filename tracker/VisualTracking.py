import cv2
import numpy as np
import datetime
from os import rename, listdir, makedirs, path
import json

class VisTracker:
    def __init__(self, preload=None):

        if preload == None: # if no initialising values are provided, ask for them manually
            if not self.query_fuction(self.setup_new_values, "Set up new values manually? y/n\n"):
                self.setup_values()
        else:
            self.setup_values(preload)

        self.set_background()

        if not self.query_fuction(self.count_bbs, "Let computer count bbs? y/n\n"):
            self.nBBs = (int)(input("Enter number of BBs to look for:\n"))

        self.m = 0
        self.t = 0
        self.cc = 0
        self.e3 = cv2.getTickCount()

        self.imgo = np.ones((self.OUTPUT_DIM, self.OUTPUT_DIM, 3), np.uint8)
        self.imgo[:, :, :] *= 254
        self.thr = np.ones((self.OUTPUT_DIM, self.OUTPUT_DIM, 3), np.uint8)
        self.thr[:, :, :] *= 254
        self.img3 = np.ones((self.OUTPUT_DIM, self.OUTPUT_DIM, 3), np.uint8)
        self.img3[:, :, :] *= 254
        self.cimg = np.ones((self.OUTPUT_DIM, self.OUTPUT_DIM, 3), np.uint8)
        self.cimg[:, :, :] *= 254

        self.run = True

    def setup_values(self, v={}):
        with open('base_files/default_vals', 'r') as f:
            d = json.loads(f.read())

        # VIDEO ANALYSIS SETTINGS
        self.REFERENCE_WIDTH = self.find_match('referenceWidth', v, d)
        self.OUTPUT_DIM = self.find_match('outputDim', v, d)
        self.pixelsPerCentimeter = 1
        self.nOrig = (0, 0)

        self.circset = (15, 324, 18, 5, 30)  # 2mm BBs
        self.circleMinDist = self.find_match('circleMinDist', v, d)
        self.circleEdgeStrength = self.find_match('circleEdgeStrength', v, d)
        self.circleCircularity = self.find_match('circleCircularity', v, d)
        self.circleMinRadius = self.find_match('circleMinRadius', v, d)
        self.circleMaxRadius = self.find_match('circleMaxRadius', v, d)

        self.radius = self.find_match('radius', v, d)

        self.radiusOfTheUniverse = self.find_match('radiusOfTheUniverse', v, d)

        self.markerThreshold = self.find_match('markerThreshold', v, d)

        # CLUSTERING ANALYSIS
        self.samplePeriod = self.find_match('samplePeriod', v, d)
        self.sampleSize = self.find_match('sampleSize', v, d)
        self.publicCircles = list()  # public variable to store circles outside of loop during sampling period
        self.recordCircles = list()  # the calculated centroids to be actually displayed
        self.clusterThreshold = int(self.sampleSize * 0.5)  # how many circles must be in a cluster for it to be considered a cluster
        self.proximityThreshold = self.find_match('proximityThreshold', v, d)
        self.sampleCounter = 0  # how many samples have been taken
        self.snapshotCounter = 0  # how many times a sample has been taken (and stored)

        # RECORDING VARIABLES
        self.animationList = []
        self.animationDict = {}

        # VIDEO CONTROL
        self.msPerFrame = self.find_match('msPerFrame', v, d)
        self.recordingInterval = self.find_match('recordingInterval', v, d)
        self.framesPerRecord = int(self.recordingInterval / self.msPerFrame)  # number of frames between each recorded frame

        self.refsamplesize = 50
        self.quickTargetPts = [(199, 0), (1095, 30), (166, 902), (1062, 937)]

        self.record = self.find_match('record', v, d)
        self.manipulate = self.find_match('manipulate', v, d)
        self.sampling = self.find_match('sampling', v, d)
        self.quickTarget = self.find_match('quickTarget', v, d)
        self.lockMarkers = self.find_match('lockMarkers', v, d)

        # MEMORY OF REFERENCE POINTS
        self.nBBs = self.find_match('nBBs', v, d)

        self.refpts = []
        self.tlref = (0, 0)  # stores last known location of top left reference point

        self.objPermanence = False  # whether tracking has started or not
        self.positionList = []
        self.selected = False  # whether or not a BB is currently selected by the user
        self.BBid = None  # ID# of the currently selected BB

        # VIDEO SETTINGS
        self.cap = cv2.VideoCapture(0)
        self.focus = self.find_match('focus', v, d)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1600)  # 9000 pro
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1200)

        # FILE MANAGEMENT
        self.fileCount = 0

        if path.exists('experiment files/'):
            for f in listdir(r'experiment files'):
                self.fileCount += 1
        else:
            makedirs('experiment files/')

        self.filename = ""

        # NAMING & CREATING VIDEO FILE TO WRITE TO
        self.trashName = 'digital farts'
        self.timestamp = datetime.datetime.now().strftime("%I-%M%p_%d-%m-%Y")
        self.fourcc = cv2.VideoWriter_fourcc(*'DIVX')

        self.out = cv2.VideoWriter('test.avi', self.fourcc, 20.0,(self.OUTPUT_DIM, self.OUTPUT_DIM))
        self.e1, self.e2, self.e3, self.e4, self.t = 0, 0, 0, 0, 0  # used to measure the time

        self.font = cv2.FONT_HERSHEY_SIMPLEX

    def setup_new_values(self):
        #VIDEO ANALYSIS SETTINGS
        self.REFERENCE_WIDTH = self.get_value('float', 'REFERENCE_WIDTH', "actual side length in cm of the reference square")
        self.OUTPUT_DIM = self.get_value('int', "OUTPUT_DIM", "no. of pixels on a side of the output video")
        self.pixelsPerCentimeter = 1
        self.nOrig = (0, 0)

        self.circset = (15, 324, 18, 5, 30)  # 2mm BBs
        self.circleMinDist = self.get_value('float', "circleMinDist", "minimum acceptable distance between detected circles")
        self.circleEdgeStrength = self.get_value('float', "circleEdgeStrength", "how clear and edge must be to be considered part of a circle")
        self.circleCircularity = self.get_value('float', "circleCircularity", "how 'circular' the shape must be to be identified as a circle")
        self.circleMinRadius = self.get_value('float', "circleMinRadius", "minimum acceptable radius (in pixels) of an identified circle")
        self.circleMaxRadius = self.get_value('float', "circleMaxRadius", "maximum acceptable radius (in pixels) of an identified circle")

        self.radius = self.get_value('float', "radius", "approximate radius (in pixels) of balls to be detected in image")

        self.radiusOfTheUniverse = self.get_value('float', "radiusOfTheUniverse", "maximum no. of centimeters a BB may be from the center and still be picked up")

        self.markerThreshold = self.get_value('float', "markerThreshold", "threshold pixel value for the marker detection image")

        #CLUSTERING ANALYSIS
        self.samplePeriod = self.get_value('float', "samplePeriod", "how often to perform sampling analysis, e.g. 600 = every 10 minutes")
        self.sampleSize = self.get_value('int', "sampleSize", "number of images sampled before clustering analysis begins")
        self.publicCircles = list()  # public variable to store circles outside of loop during sampling period
        self.recordCircles = list()  # the calculated centroids to be actually displayed
        self.clusterThreshold = int(self.sampleSize * 0.5)  # how many circles must be in a cluster for it to be considered a cluster
        self.proximityThreshold = self.get_value('float', "proximityThreshold", "how close (in pixels) another circle must be to be considered part of the same cluster")
        self.sampleCounter = 0  # how many samples have been taken
        self.snapshotCounter = 0  # how many times a sample has been taken (and stored)

        #RECORDING VARIABLES
        self.animationList = []
        self.animationDict = {}

        #VIDEO CONTROL
        self.msPerFrame = self.get_value('int', "msPerFrame", "Number of milliseconds passing between each frame (i.e. determines framerate)")
        self.recordingInterval = self.get_value('int', "recordingInterval", "how many ms should pass between each recorded frame")
        self.framesPerRecord = int(self.recordingInterval / self.msPerFrame)  # number of frames between each recorded frame

        self.refsamplesize = 50
        self.quickTargetPts = [(199, 0), (1095, 30), (166, 902), (1062, 937)]

        self.record = self.get_value('bool', "record", "frames are recorded when this is set to true")
        self.manipulate = self.get_value('bool', "manipulate", "manipulate the perspective when this is true")
        self.sampling = self.get_value('bool', "sampling", "while this is true, collect samples for future smoothing")
        self.quickTarget = self.get_value('bool', "quickTarget", "set to true to use manually coded reference points")
        self.lockMarkers = self.get_value('bool', "lockMarkers", "when true, currently detected markers will be used as markers until set false again")

        #MEMORY OF REFERENCE POINTS
        self.nBBs = self.get_value('int', "nBBs", "how many BBs the program should be looking for")

        self.refpts = []
        self.tlref = (0, 0)  # stores last known location of top left reference point

        self.objPermanence = False  # whether tracking has started or not
        self.positionList = []
        self.selected = False  # whether or not a BB is currently selected by the user
        self.BBid = None  # ID# of the currently selected BB

        #VIDEO SETTINGS
        self.cap = cv2.VideoCapture(1)
        self.focus = 110
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1600)  # 9000 pro
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1200)

        #FILE MANAGEMENT
        self.fileCount = 0

        if path.exists('experiment files/'):
            for f in listdir(r'experiment files'):
                self.fileCount += 1
        else:
            makedirs('experiment files/')

        self.filename = ""

        #NAMING & CREATING VIDEO FILE TO WRITE TO
        self.trashName = 'digital farts'
        self.timestamp = datetime.datetime.now().strftime("%I-%M%p_%d-%m-%Y")
        self.fourcc = cv2.VideoWriter_fourcc(*'DIVX')

        self.out = cv2.VideoWriter('test.avi', self.fourcc, 20.0, (self.OUTPUT_DIM, self.OUTPUT_DIM))
        self.e1, self.e2, self.e3, self.e4, self.t = 0, 0, 0, 0, 0  # used to measure the time

        self.font = cv2.FONT_HERSHEY_SIMPLEX

    def set_background(self):
        backgroundSet = False
        takeNew = False
        while not backgroundSet:
            if not takeNew:
                resp = input("Take new background image? y/n\n")
                if resp == 'n':
                    try:
                        self.back = cv2.imread('base_files/background.jpg')
                    except:
                        print("Couldn't load file. A new image must be taken.\n")
                    finally:
                        backgroundSet = True
                elif resp == 'y':
                    takeNew = True
                    backgroundSet = True
                else:
                    print("Please enter exactly 'y' or 'n'.\n")

        if takeNew:
            print("Press 'a' to store current image as background.\n")
            while (self.cap.isOpened()):
                ret, imgo = self.cap.read()
                if ret:
                    k = cv2.waitKey(self.msPerFrame)
                    if k & 0xFF == ord('a'):
                        self.back = imgo.copy()
                        cv2.imwrite('base_files/background.jpg', imgo)
                        break
                    cv2.namedWindow('background', cv2.WINDOW_NORMAL)
                    cv2.imshow('background', imgo)
                else:
                    break
            cv2.destroyAllWindows()

    def count_bbs(self):
        accepted = False
        print("Press 'a' to store current number.\n")
        while not accepted:
            nCs = 0
            while (self.cap.isOpened()):
                ret, imgo = self.cap.read()
                if ret:
                    #DETECT CIRCLES
                    imgrey = cv2.cvtColor(imgo, cv2.COLOR_BGR2GRAY)
                    circles = None
                    nCs = 0
                    try:
                        circles = cv2.HoughCircles(imgrey, cv2.HOUGH_GRADIENT, 1, self.circleMinDist, param1=self.circleEdgeStrength, param2=self.circleCircularity, minRadius=self.circleMinRadius, maxRadius=self.circleMaxRadius)
                        nCs = len(circles[0,:])
                    except:
                        nCs = 0
                    outstr = "\r" + (str)(nCs) + " "
                    print(outstr, end='')


                    #DRAW CIRCLES
                    if circles is not None:
                        for circle in circles[0,:]:
                            cv2.circle(imgo, (circle[0], circle[1]), 20, (0,0,255), 3)


                    k = cv2.waitKey(self.msPerFrame)
                    if k & 0xFF == ord('a'):
                        self.nBBs = nCs
                        break
                    cv2.namedWindow('bb counter', cv2.WINDOW_NORMAL)
                    cv2.imshow('bb counter', imgo)
                else:
                    break
            cv2.destroyAllWindows()
            if input('\n{0} BBs counted. Try again? y/n\n'.format(nCs)) == 'n':
                accepted = True

    def run_full(self):
        while self.cap.isOpened():
            self.process_loop()
            if not self.run:
                break
        self.end_process()

    def process_loop(self):
        ret, self.imgo = self.cap.read()
        if ret and self.run:
            self.cap.set(cv2.CAP_PROP_FOCUS, self.focus)
            self.t += 1
            self.modOrig = [220, 0]
            self.e4 = cv2.getTickCount()

            self.marker_detection(self.imgo)

            currentPoints, distances = self.bb_tracking()

            self.image_sampling(currentPoints)

            self.display_output(distances)

            k = cv2.waitKey(self.msPerFrame)
            if k & 0xFF == ord('q'):  # quit
                self.e2 = cv2.getTickCount()
                if self.record:
                    self.t = (self.e2 - self.e1) / cv2.getTickFrequency()
                self.run = False
            elif k & 0xFF == ord('p'):
                self.lockMarkers = not self.lockMarkers
                print("LOCKMARKERS: ", self.lockMarkers)
            elif k & 0xFF == ord('m'):
                self.manipulate = not self.manipulate
                print("MANIPULATE: ", self.manipulate)
            elif k & 0xFF == ord('r'):  # start recording
                self.record = True
                self.timestamp = datetime.datetime.now().strftime("%I-%M%p_%d-%m-%Y")
                self.e1 = cv2.getTickCount()
                self.filename = "{}_{}mm_BBs_run-{}".format(self.nBBs, 3, str(self.fileCount + -1))  # -1 since the 'waste' folder shouldn't be counted.
                directory = r'experiment files\%s' % self.filename
                if not path.exists(directory):
                    makedirs(directory)
                print("RECORD: ", self.record)
            elif k == ord('s'):  # wait for 's' key to save and exit
                cv2.imwrite('VT2_output.jpg', self.imgo)
            elif k == ord('l'):  # increase focus
                self.focus = np.clip(self.focus+5, 0, 300)
                self.cap.set(cv2.CAP_PROP_FOCUS, self.focus)
                print(self.focus)
            elif k == ord('j'):  # decrease focus
                self.focus = np.clip(self.focus - 5, 0, 300)
                self.cap.set(cv2.CAP_PROP_FOCUS, self.focus)
                print(self.focus)
        return self.positionList

    def marker_detection(self, im):
        # ----------------------------------------MARKER DETECTION-----------------------------------------------------#
        markers = [cv2.MARKER_DIAMOND, cv2.MARKER_CROSS, cv2.MARKER_SQUARE, cv2.MARKER_STAR]

        self.cimg = im.copy()

        s = 13
        kernel = np.ones((s, s), np.uint8)
        to_detect = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        ret, self.thr = cv2.threshold(to_detect, self.markerThreshold, 255, cv2.THRESH_BINARY)
        self.thr = cv2.morphologyEx(self.thr, cv2.MORPH_CLOSE, kernel)

        if self.lockMarkers:
            for pt in self.refpts:
                cv2.drawMarker(self.thr, (pt[0], pt[1]), (255, 255, 255), markerType=cv2.MARKER_CROSS,
                               thickness=2, markerSize=50)
            contlength = len(self.refpts)
        else:
            self.refpts.clear()

            h, contours, hierarchy = cv2.findContours(self.thr, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            rects = list()
            for i in range(1, len(contours)):
                epsilon = 0.04 * cv2.arcLength(contours[i], True)
                approx = cv2.approxPolyDP(contours[i], epsilon, True)
                cv2.drawContours(self.thr, contours[i], -1, (255, 0, 0), 2)
                if len(approx) == 4:
                    rect = cv2.minAreaRect(contours[i])
                    box = cv2.boxPoints(rect)
                    box = np.int0(box)
                    rects.append(approx)
                    cv2.drawContours(self.thr, [box], -1, (255, 255, 255), 2)

            for i in range(len(rects)):
                try:
                    M = cv2.moments(rects[i])
                    cx = int(M['m10'] / M['m00'])
                    cy = int(M['m01'] / M['m00'])
                    self.refpts.append((cx, cy))
                except:
                    self.refpts.append((0, 0))
            sqpts = self.create_square(self.refpts, thr=self.thr)
            self.refpts.clear()
            for i in range(4):
                cv2.drawMarker(self.cimg, (sqpts[i][0], sqpts[i][1]), (255, 255, 255), markerType=cv2.MARKER_CROSS, thickness=2, markerSize=50)
                self.refpts.append((sqpts[i][0] - sqpts[0][1], sqpts[i][1] - sqpts[0][1]))

            contlength = len(contours)

        if self.quickTarget:
            self.refpts = self.quickTargetPts
            contlength = 4

        self.img3 = cv2.absdiff(self.back, im)

        self.nOrig = (int(self.OUTPUT_DIM / 2), int(self.OUTPUT_DIM / 2))  # center of image
        if contlength >= 4 and self.manipulate:
            self.img3, self.pixelsPerCentimeter, self.nOrig = self.compensate_perspective(self.img3,self.refpts, self.img3.shape[0], self.img3.shape[1])  # transforms perspective
            self.cimg, self.pixelsPerCentimeter, self.nOrig = self.compensate_perspective(self.cimg,self.refpts, self.cimg.shape[0], self.cimg.shape[1])  # transforms perspective
        elif self.manipulate:  # in the event that all 4 marker points couldn't be found, use last known reference points
            self.img3 = self.img3[self.tlref[1]:self.OUTPUT_DIM + self.tlref[1], self.tlref[0]:self.OUTPUT_DIM + self.tlref[0]]
            self.cimg = self.img3[self.tlref[1]:self.OUTPUT_DIM + self.tlref[1], self.tlref[0]:self.OUTPUT_DIM + self.tlref[0]]

    def bb_tracking(self):
        s = 5
        kernel = np.ones((s, s), np.uint8)
        self.img3 = cv2.morphologyEx(self.img3, cv2.MORPH_OPEN, kernel)
        self.img3 = cv2.cvtColor(self.img3, cv2.COLOR_BGR2GRAY)

        distances = list()
        currentPoints = []
        tempCirc = []
        circles = None
        try:
            circles = cv2.HoughCircles(self.img3, cv2.HOUGH_GRADIENT, 1, self.circleMinDist,
                                       param1=self.circleEdgeStrength, param2=self.circleCircularity,
                                       minRadius=self.circleMinRadius, maxRadius=self.circleMaxRadius)
            tempCirc = [(c[0],c[1]) for c in circles[0, :]]
        except:
            pass

        # --------------------------------------------BB identity tracking-----------------------------------------------------------------------#
        if self.manipulate:
            if not self.objPermanence and len(tempCirc) == self.nBBs:
                self.objPermanence = True
                self.positionList = tempCirc
            else:
                for i in range(len(self.positionList)):
                    mindist = np.inf
                    minpt = None
                    for j in tempCirc:
                        cdist = self.get_euclidean_dist(self.positionList[i], j)
                        if cdist < mindist:
                            mindist = cdist
                            minpt = j
                    distLim = 1.4
                    if mindist < (distLim * self.radius):
                        proportion = (mindist / (distLim * self.radius))
                        motionVector = ((minpt[0] - self.positionList[i][0]) * proportion,
                                        (minpt[1] - self.positionList[i][1]) * proportion)
                        self.positionList[i] = (int(self.positionList[i][0] + motionVector[0]), int(self.positionList[i][1] + motionVector[1]))
                        tempCirc.remove(minpt)
        else:
            self.objPermanence = False

        if self.positionList is not None:
            for i, circle in enumerate(self.positionList):
                cv2.circle(self.cimg, (circle[0], circle[1]), 20, (0, 0, 255), 3)
                cv2.putText(self.cimg, str(i), (int(circle[0]+self.radius), int(circle[1]+self.radius)), self.font, 1, (255, 255, 255), 2, cv2.LINE_AA)
                if self.selected and self.BBid == i:
                    cv2.drawMarker(self.cimg, self.positionList[i], (50, 255, 50), markerType=cv2.MARKER_CROSS,thickness=2, markerSize=80)
                else:
                    cv2.drawMarker(self.cimg, self.positionList[i], (255, 255, 255), markerType=cv2.MARKER_CROSS,thickness=2, markerSize=(80))

        return currentPoints, distances

    def image_sampling(self, currentPoints):
        if self.record:
            if ((self.e4 - self.e1) / cv2.getTickFrequency()) % self.samplePeriod < 0.5:  # sample the images every [sample_period] seconds
                self.sampling = True
                self.recordCircles.clear()
            if self.sampling and self.sampleCounter < self.sampleSize:  # while sampling, store locations of all circles detected in this round
                cv2.rectangle(self.cimg, (0, 0), (50, 50), (0, 0, 255), -1)
                self.publicCircles = self.publicCircles + currentPoints
                self.sampleCounter += 1
            elif self.sampleCounter >= self.sampleSize and self.sampling:  # once sampling is complete, perform clustering analysis to determine the location of current circles
                self.sampling = False

                while len(self.publicCircles) > 0:  # keep removing elements from the list until none are left
                    c_pt = self.publicCircles[
                        0]  # each iteration, the first vector of the list will be compared to all others to generate a cluster
                    cluster_list = [c_pt]
                    print(len(self.publicCircles))
                    for i in range(1, len(self.publicCircles)):  # find and remove all other points in the list that are close enough to the current point to be considered part of the cluster
                        if self.get_euclidean_dist(c_pt, self.publicCircles[i]) < self.proximityThreshold:
                            cluster_list.append(self.publicCircles[i])
                    for pt in cluster_list:  # remove those points accepted
                        self.publicCircles.remove(pt)
                    # public_circles.pop(0)  # finally remove the current point too
                    if len(cluster_list) > self.clusterThreshold:  # if the cluster has enough points, we'll accept it and treat the average position as a point
                        toadd = self.find_centroid(cluster_list)
                        cv2.circle(self.cimg, tuple(toadd), 20, (255, 0, 255), 2)
                        toadd[0] = toadd[0] / self.pixelsPerCentimeter
                        toadd[1] = toadd[1] / self.pixelsPerCentimeter
                        self.recordCircles.append(toadd)

                self.snapshotCounter += 1
                with open('experiment files\{}\sample_{}.txt'.format(self.filename, str(self.snapshotCounter)),"w+") as f:  # save all BB locations, given as x-y coordinates in cm, as json in a text file for future reference
                    str_out = json.dumps(self.recordCircles)
                    f.write(str_out)

                cv2.imwrite('experiment files\{}\sample_{}.jpg'.format(self.filename, str(self.snapshotCounter)), self.cimg)

                self.sampleCounter = 0
                self.publicCircles.clear()

            cv2.putText(self.cimg, self.format_time(int((self.e4 - self.e1) / cv2.getTickFrequency())), (10, 50), self.font, 1,(255, 255, 255), 2, cv2.LINE_AA)

    def display_output(self, distances):
        rows, cols, h = self.cimg.shape

        try:
            pass
        except ZeroDivisionError as err:
            print("0 division error: ", err)
        except:
            print(
                "I dunno what's wrong, but it ain't a zero division error. \nSo. You know. \nPANIC! FOR ALL IS LOST!!!")
        if len(distances) == self.nBBs:
            self.m += 1

        cv2.drawMarker(self.cimg, self.nOrig, (255, 255, 255), markerType=cv2.MARKER_CROSS, markerSize=100)

        cv2.namedWindow('image', cv2.WINDOW_NORMAL)
        cv2.setMouseCallback('image', self.relocate)
        cv2.imshow('image', self.cimg)
        cv2.namedWindow('markers', cv2.WINDOW_NORMAL)
        cv2.imshow('markers', self.thr)

        if self.record and self.t % self.framesPerRecord == 0:  # if 'r' was pressed, start recording an image every [frames_per_record] frames
            self.out.write(self.cimg)

            cm_pos_list = []
            time = (self.e4 - self.e1) / cv2.getTickFrequency()
            for loc in self.positionList:
                cm_pos_list.append(
                    [(loc[0] - self.nOrig[0]) / self.pixelsPerCentimeter, (loc[1] - self.nOrig[1]) / self.pixelsPerCentimeter])
            self.animationList.append([cm_pos_list, time])
            self.animationDict[time] = cm_pos_list

    def end_process(self):
        cv2.imwrite('background.jpg', self.back)
        self.cap.release()
        self.out.release()
        cv2.destroyAllWindows()
        if self.record:
            with open('experiment files\{}\{}_animation_file.txt'.format(self.filename, self.timestamp),
                      "w+") as f:  # save all BB locations, given as x-y coordinates in cm, as json in a text file for future reference
                str_out = json.dumps(self.animationList)
                f.write(str_out)

            self.t = self.format_time(int(self.t), "_real_runtime_")
            print(self.t)
            print("RENAMING")
            self.timestamp = self.timestamp + "_%s" % self.t
            rename('test.avi','experiment files\{}\{}.avi'.format(self.filename, self.timestamp))

    # ------------------------------------------------FUNCTIONS FOR USE-------------------------------------------------------------------#

    def compensate_perspective(self, to_detect, refpts, cols, rows):
        """
        Warps the perspective of a given image to a new set of points
        :param to_detect: image to warp
        :param refpts: set of 4 points that will be the corners of the new image
        :param cols: number of columns of input image (i.e. width)
        :param rows: number of rows of input image (i.e. height)
        :return: the modified image, the pixels per centimeter conversion rate and the vector of the center of the new image
        """
        # for perspective transform
        pts1 = []
        for point in refpts:
            pts1.append(point)

        pts1 = np.float32(self.create_square(refpts, self.thr))

        n_width = (abs(refpts[3][0] - refpts[2][0]) + abs(
            refpts[1][0] - refpts[0][0])) / 2  # average width of the square formed by the two markers
        n_height = (abs(refpts[3][1] - refpts[1][1]) + abs(
            refpts[2][1] - refpts[0][1])) / 2  # average height of the square formed by the two markers
        cal_width = (n_width + n_height) / 2  # should be square, so average width and height. Used to measure everything
        pixels_per_centimeter = cal_width / self.REFERENCE_WIDTH  # conversion rate
        n_orig = (int(self.OUTPUT_DIM / 2), int(self.OUTPUT_DIM / 2))  # centre of warped image
        pts2 = np.float32([[0, 0], [cal_width, 0], [0, cal_width], [cal_width, cal_width]])
        MP = cv2.getPerspectiveTransform(pts1, pts2)

        cv2.warpPerspective(to_detect, MP, (cols, rows), to_detect)
        to_detect = to_detect[int(pts1[0][1]):int(self.OUTPUT_DIM) + int(pts1[0][1]),
                    int(pts1[0][0]):int(self.OUTPUT_DIM) + int(pts1[0][0])]
        return to_detect, pixels_per_centimeter, n_orig

    def relocate(self, event, x, y, flags, param):
        """
        handles mouse button events (e.g. clicking)
        :param event:
        :return:
        """
        if event == cv2.EVENT_LBUTTONDOWN:
            if self.selected:
                self.positionList[self.BBid] = (x, y)
                self.selected = False
            else:
                for i in range(len(self.positionList)):
                    if self.get_euclidean_dist((x, y), self.positionList[i]) <= 1.5 * self.radius:
                        self.BBid = i
                        self.selected = True

    @staticmethod
    def query_fuction(function, queryTitle="Run function? y/n\n", errorMessage="\nERROR\n"):
        ans = False
        while not ans:
            resp = input(queryTitle)
            if resp == 'n':
                return False
            elif resp == 'y':
                try:
                    function()
                    return True
                except:
                    print(errorMessage)
                    return False
            else:
                print("Please enter exactly 'y' or 'n'.\n")

    @staticmethod
    def find_match(key, a, b):
        """
        returns a[key] if 'a' has 'key', otherwise returns b[key]
        :param key: key to look for in dictionary
        :param a: dictionary to check first
        :param b: backup dictionary to check if 'a' doesn't have 'key'
        :return: value stored at key 'key' in 'a' or 'b'
        """
        return a.get(key, b[key])

    @staticmethod
    def get_value(type, name="UNNAMED", description=""):
        """
        Prompts the user to enter value of certain type and returns it
        :param type: type of value to get
        :param name: name of variable
        :param description: optional description of what that variable represents
        :return: value of type 'type'
        """
        retVal = None
        while not isinstance(retVal, eval(type)):
            try:
                retVal = (eval(type))(input("{}\n{}\nPlease enter {} value:\n".format(name, description, type)))
            except:
                retVal = None
                print("ERROR WRONG TYPE\n")
        return retVal

    @staticmethod
    def get_euclidean_dist(a, b):
        dx = a[0] - b[0]
        dy = a[1] - b[1]
        dist = np.hypot(dx, dy)
        return dist

    @staticmethod
    def create_square(points, thr):
        """
        Given a list of points, picks the 4 most extreme ones (top left, top right, bottom left, bottom right),
        organises them in the format usable for perspective transformations and returns them
        :param points: list of points as vectors
        :return: array of 4 points
        """
        xs = []
        ys = []
        n = 0
        for point in points:
            xs.append(point[0])
            ys.append(point[1])
            n += 1  # counts how many points there are in total
        xs.sort()

        left = xs[:2]  # left 2 points of square
        right = xs[n - 2:]  # right 2 points of square
        ys.sort()
        top = ys[:2]  # top 2 points of square
        bot = ys[n - 2:]  # bottom 2 points of square
        a, b, c, d = [0, 0], [0, 0], [0, 0], [0, 0]
        for point in points:  # note that, if there are more than 4 points, the square will be generated from the 4 most extreme points
            if point[0] in left:  # left points
                if point[1] in top:  # top left point
                    a = [point[0], point[1]]
                elif point[1] in bot:  # bottom left point
                    c = [point[0], point[1]]
            elif point[0] in right:  # right points
                if point[1] in top:  # top right point
                    b = [point[0], point[1]]
                elif point[1] in bot:  # bottom right point
                    d = [point[0], point[1]]
        cv2.drawMarker(thr, (a[0], a[1]), (0, 0, 255), markerType=cv2.MARKER_CROSS, markerSize=200)
        cv2.drawMarker(thr, (b[0], b[1]), (0, 0, 255), markerType=cv2.MARKER_CROSS, markerSize=200)
        cv2.drawMarker(thr, (c[0], c[1]), (0, 0, 255), markerType=cv2.MARKER_CROSS, markerSize=200)
        cv2.drawMarker(thr, (d[0], d[1]), (0, 0, 255), markerType=cv2.MARKER_CROSS, markerSize=200)
        square = [a, b, c, d]
        return square

    @staticmethod
    def find_median_vector(vectors):
        """
        given a list of vectors, returns the median vector
        :param vectors: list of vectors
        :return: the median vector of that list
        """
        xlist = list()
        ylist = list()
        for vector in vectors:
            xlist.append(vector[0])
            ylist.append(vector[1])
        o = (int(np.median(xlist)), int(np.median(ylist)))
        return o

    @staticmethod
    def find_centroid(vectors):
        '''
        finds the mean vector from a list of multidimensional vectors
        :param vectors: list of vectors
        :return: a list of the same dimension as the input vectors
        '''
        n = len(vectors[0])
        outv = list()
        for i in range(n):
            vlist = list()
            for vector in vectors:
                vlist.append(vector[i])
            outv.append(int(np.mean(vlist)))
        return outv

    @staticmethod
    def format_time(seconds, prefix=""):
        return prefix + "{}h{}m{}s".format(int(seconds / (60 * 60)), int(seconds % (60 * 60) / 60), (seconds % 60))