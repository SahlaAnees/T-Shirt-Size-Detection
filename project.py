import cv2
import numpy as np
import matplotlib.pyplot as plt
import imutils
from imutils import contours
from imutils import perspective
import scipy.spatial.distance as dist

def tshirtSizeDetection(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)

##    plt.imshow(img)
##    plt.show()

    #Convert image into grayscale
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #Blur the image 
    img_gray = cv2.GaussianBlur(img_gray, (3,3), 0)

    #Detect edge & Do morphological operations for get a proper edge
    edged = cv2.Canny(img_gray, 80, 200)
    kernal = np.ones((5,5),np.uint8)
    edged = cv2.dilate(edged, kernal, iterations=1)
    edged = cv2.erode(edged, kernal, iterations=1)
   
##    plt.imshow(edged)
##    plt.show()

    #Find the contours of objects in the image
    cnts= cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts= imutils.grab_contours(cnts)

    #Sort the contours, Maximum contour would be the t-shirt object
    (cnts, _) = contours.sort_contours(cnts)
    cmax = max(cnts, key=cv2.contourArea)

    #Calculate moments of binary image
    M = cv2.moments(cmax)

    #Calculate X,Y coordinate of center
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])

    #Create a blank image area of the original image size
    height, width, color = img.shape
    final_img_blank = np.zeros((height, width, 3), np.uint8)
    final_img_blank =cv2.cvtColor(final_img_blank, cv2.COLOR_RGB2BGR)
    final_img_blank[:] = (211, 211, 211)

    # Display the result
##    plt.imshow(final_img_blank)
##    plt.show()

    pixels_per_metric = None
    known_width = 3.9370 # Width of a reference object (in inches)

    for c in cnts:
        if cv2.contourArea(c) < 300:
            continue
        img_ = img.copy()
        box = cv2.minAreaRect(c)
        box = cv2.cv.BoxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box)
        box = np.array(box, dtype="int")
        
        # Draw outline of rotated bounding box
        # after ordering contours in order of top-left, top-right, bottom-right, and bottom-left.
        box = perspective.order_points(box)
        cv2.drawContours(img_, [box.astype("int")], -1, (0, 255, 0), 2)

        #Funtion to calculate the midpoint of 2 pairs of coordinates
        def midpoint(X, Y):
            return ((X[0] + Y[0]) * 0.5, (X[1] + Y[1]) * 0.5)
        
        # Calculate the pixels_per_metric variable
        if pixels_per_metric is None:
            (top_left, top_right, bottom_right, bottom_left) = box
            (tlbl_X, tlbl_Y) = midpoint(top_left, bottom_left)
            (trbr_X, trbr_Y) = midpoint(top_right, bottom_right)
            dB = dist.euclidean((tlbl_X, tlbl_Y), (trbr_X, trbr_Y))
            pixels_per_metric = dB / known_width

        # If T-shirt area and contour area is equal, then ;
        if cv2.contourArea(c) == cv2.contourArea(cmax):
            cv2.drawContours(final_img_blank, [cmax], -1, (255, 255, 255), -1)
            
            # calculate the top for the left sleeve
            leftSleeve_top = tuple(c[c[:, :, 0].argmin()][0])

            # intialization
            extBotleft = (0, 0)
            extTop = (0, 0)
            centre = (cX, cY)

            # Calculate the extreme top point of the contour
            for yp in range(cY, 0, -1):
                isOnContour = cv2.pointPolygonTest(c, (cX, yp), False)
                if isOnContour == 0:
                    extTop = (cX, yp)
                    break
            # calculate the bottom left point of the contour
            for xp in range(cX, 0, -1):
                isOnContour = cv2.pointPolygonTest(c, (xp, cY), False)
                if isOnContour == 0:
                    extBotleft = (xp, cY)
                    break

            widx = dist.euclidean(centre, extBotleft)
            widy = dist.euclidean(centre, extTop)

            extBotRight = (int(cX + widx), cY)
            extDown = (cX, int(cY + widy))

            # calculate the bottom point of the left sleeve
            leftSleeve_bottom = (0, 0)
            yold = leftSleeve_top[1]
            found = False
            for x in cmax:
                for y in x:
                    if leftSleeve_bottom != (0, 0):
                        break
                    if tuple(y) == leftSleeve_top:
                        found = True
                    if found and yold >= y[1]:
                        leftSleeve_bottom = tuple(y)
                    else:
                        yold = y[1]

            # Draw points
            cv2.circle(img_, leftSleeve_top, 8, (0, 0, 255), -1)
            cv2.circle(img_, leftSleeve_bottom, 7, (255, 0, 0), -1)
            cv2.circle(img_, extBotleft, 8, (255, 255, 0), -1)
            cv2.circle(img_, extBotRight, 7, (255, 0, 0), -1)
            cv2.circle(img_, extTop, 8, (255, 255, 0), -1)
            cv2.circle(img_, extDown, 7, (255, 0, 0), -1)
            cv2.circle(img_, centre, 8, (255, 255, 0), -1)

##            plt.imshow(img_)
##            plt.show()

            # Calculate T-shirt width, height, and the dimensions of sleeve)
            outfit_width = widx / pixels_per_metric * 2
            outfit_height = widy / pixels_per_metric * 2
            
            sleevedist = dist.euclidean(leftSleeve_top, leftSleeve_bottom)
            outfit_sleeve = sleevedist / pixels_per_metric
            
            # print the measurements
            print(F"T-shirt width: {outfit_width} inches")
            print(F"T-shirt height: {outfit_height} inches")
            print(F"T-shirt sleeve: {outfit_sleeve} inches")
            
            # Label the measurements on the T-shirt image using arrow-line
            displaypoint = midpoint(leftSleeve_top, leftSleeve_bottom)
            
            cv2.arrowedLine(final_img_blank,
                            (int(centre[0] - widx), int(centre[1]+widy+60)),
                            (int(centre[0] + widx), int(centre[1]+widy+60)),
                            (0, 0, 0), 2, tipLength=0.05)
            
            cv2.arrowedLine(final_img_blank,
                            (int(centre[0] + widx), int(centre[1] + widy + 60)),
                            (int(centre[0] - widx), int(centre[1] + widy + 60)),
                            (0, 0, 0), 2, tipLength=0.05)

            cv2.arrowedLine(final_img_blank,
                            (int(centre[0] + widx+150), int(centre[1] - widy)),
                            (int(centre[0]+widx+150), int(centre[1] + widy)),
                            (0, 0, 0), 2, tipLength=0.05)
            
            cv2.arrowedLine(final_img_blank,
                            (int(centre[0] + widx + 150), int(centre[1] + widy)),
                            (int(centre[0] + widx + 150), int(centre[1] - widy)),
                            (0, 0, 0), 2, tipLength=0.05)

            font = cv2.FONT_HERSHEY_SIMPLEX
            font_size = 0.7
            font_color = (0, 0, 0)
            font_thickness = 2
            
            cv2.putText(final_img_blank,
                        "{:.1f}in (sleeve)".format(outfit_sleeve),
                        (int(displaypoint[0]-180), int(displaypoint[1])),
                        font, font_size, font_color, font_thickness)
            
            cv2.putText(final_img_blank,
                        "{:.1f}in (width)".format(outfit_width),
                        (int(extDown[0]), int(extDown[1] + 25)),
                        font, font_size, font_color, font_thickness)

            cv2.putText(final_img_blank,
                        "{:.1f}in (height)".format(outfit_height),
                        (int(extBotRight[0] + 10), int(extBotRight[1])),
                        font, font_size, font_color, font_thickness)
            

            # Label the t-shirt size as S/M/L/XL on the T-shirt image
            if round(outfit_width) in range(17,19) and round(outfit_height) in range(25,27):
                cv2.putText(final_img_blank, 'S', (cX,cY) , cv2.FONT_HERSHEY_SIMPLEX, 2.0, (255, 0, 0), 3)
                print("small\n")
                
            elif round(outfit_width) in range(18,20) and round(outfit_height) in range(26,28):
                cv2.putText(final_img_blank, 'M', (cX,cY) , cv2.FONT_HERSHEY_SIMPLEX, 2.0, (255, 0, 0), 3)
                print("medium\n")

            elif round(outfit_width) in range(19,21) and round(outfit_height) in range(27,29):
                cv2.putText(final_img_blank, 'L', (cX,cY) , cv2.FONT_HERSHEY_SIMPLEX, 2.00, (255, 0, 0), 3)
                print("large\n")

            elif round(outfit_width) in range(20,22) and round(outfit_height) in range(28,30):
                cv2.putText(final_img_blank, 'XL', (cX,cY) , cv2.FONT_HERSHEY_SIMPLEX, 2.00, (255, 0, 0), 3)
                print("extra large\n")

            else :
                print("size dismatch\n")
            
        # Display the result
##        plt.imshow(final_img_blank)
##        plt.show()

        #Display the output
        COLOR = 'maroon'
        FONT_SIZE = 8

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        plt.subplot(1,2,1)
        plt.title('Original image', c=COLOR, fontsize = FONT_SIZE)
        plt.imshow(img_rgb)
        plt.axis('off')

        plt.subplot(1,2,2)
        plt.title('Final Output', c=COLOR, fontsize = FONT_SIZE)
        plt.imshow(final_img_blank, 'gray')
        plt.axis('off')

        plt.show()

#Plain background
#Small size T-Shirt
tshirtSizeDetection('S_White_1.jpeg')

#Medium size T-Shirt 
tshirtSizeDetection('M_White_1.jpg')

#Medium size T-Shirt 
tshirtSizeDetection('M_Gray_1F.jpeg')

#Large size T-Shirt
tshirtSizeDetection('L_Orange_1.jpeg')

#Extra Large size T-Shirt
tshirtSizeDetection('XL_Blue_1.jpeg')

#Kids T-Shirt
tshirtSizeDetection('Kid_Size.jpeg')

#Camera angle error
tshirtSizeDetection('L_Brown_1F.jpeg')

#Designed Background
#Small size T-Shirt
tshirtSizeDetection('S_White_2.jpeg')

#Medium size T-Shirt 
tshirtSizeDetection('M_White_2.jpeg')

#Medium size T-Shirt 
tshirtSizeDetection('M_Gray2F.jpeg')

#Large size T-Shirt
tshirtSizeDetection('L_Orange_2.jpeg')

#Extra Large size T-Shirt
tshirtSizeDetection('XL_Blue_2.jpeg')
