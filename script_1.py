import cv2 as cv 
import numpy as np
import matplotlib.pyplot as plt

x = []
x2 = []
y = []
y_pred=[]
Co_eff = []
center = []

def least_squares(x, y):
  ones= np.ones(len(x))
  X = np.matrix([x2, x, ones]).T
  Y  = np.matrix([y]).T
  XTX = X.T @ X
  XTY = X.T @ Y
  Co_eff = (np.linalg.inv(XTX) @ XTY)
  a = float(Co_eff[0])
  b = float(Co_eff[1])
  c = float(Co_eff[2])
  for i in x:
    y_pred.append(a*(i**2) + b*i + c)
  return a, b, c

def pixel_mean():
  height, width = np.where(mask!=0)
  W = list(width)
  H = list(height)
  if len(W) != 0 and len(H) != 0:
    mean_x = sum(W)/len(W)
    mean_y = sum(H)/len(H)
    x.append(mean_x)
    x2.append((mean_x)**2)
    y.append(mean_y)
    coords_x = (int(mean_x), int(mean_y))
    center.append(coords_x)
  return 0

def plot_curves():
  plt.scatter(x, y, c = 'r', s = 1, label="Raw Data Curve")
  plt.plot(x, y_pred, label="Best Fit Curve")
  plt.title('Trajectory of the Ball')
  plt.legend(loc = 'upper left')
  plt.show()
  return 0

vid_capture = cv.VideoCapture('ball.mov')
if (vid_capture.isOpened() == False):
  print("Error opening the video file")
else:
  while(True):
    cap, frame = vid_capture.read()
    if not cap:
      break
    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

    lower_red = np.array([0,180,117])
    upper_red = np.array([255,255,255])

    mask = cv.inRange(hsv, lower_red, upper_red)
    result = cv.bitwise_and(frame, frame, mask = mask)
    kernel = np.ones((15,15), np.float32)/225
    smooth = cv.filter2D(result, -1, kernel)
    pixel_mean()
    if len(x) > 0:
      for i in range(0,len(x) - 1):
          cv.circle(frame, (center[i]), 5,  (255, 0, 0), 3)
          cv.line(frame, (center[i]), (center[i+1]), (0, 0, 0), 5)
    cv.imshow('Video',frame)
    cv.imshow('mask',mask)
    # cv.imshow('result',result)
    if cv.waitKey(1) & 0xFF == ord("q"):
      break

a, b, c = least_squares(x, y)
print(" Equation of the Parabola : ")
print('y = {} x\u00b2 {} x + {}'.format(a,b,c))
print(y_pred[0])
y_first = y[0]+300
x_pred = np.roots([a,b,(c-y_first)])
for i in x_pred:
  if i > 0 :
    print("X-coordinate of the ball landing spot in pixels = ",int(x_pred[0]))
  else: 
    continue
plot_curves()
vid_capture.release()
cv.destroyAllWindows()