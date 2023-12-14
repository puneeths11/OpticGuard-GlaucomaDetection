import cv2                                                            #importing necessary library
import numpy as np
from PIL import Image, ImageEnhance

image = cv2.imread('C:\\Users\\Puneeths\\Desktop\\Final project\\datasets\\test\\glau\\1 (12).png')          #Reading a image
image = cv2.resize(image, (256, 256), interpolation=cv2.INTER_AREA)

orig = image.copy()                                                   #copy of the image
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)                        #Gray conversion
gray = cv2.GaussianBlur(gray, (3, 3), 0)                              #applying gaussian blur
(minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(gray)

cv2.circle(image, maxLoc, 80, (0, 0, 0), 2)
disc = 3.14 * 80 * 80
print('Area of Disc:'+str(disc))

r,g,b = cv2.split(orig)

reverse_img = 255 - g

kernel = np.ones((5,5), np.uint8) 
img_dilation = cv2.dilate(g, kernel, iterations=1) 

#stretching
minmax_img = np.zeros((img_dilation.shape[0],img_dilation.shape[1]),dtype = 'uint8')

# Loop over the image and apply Min-Max formulae
for i in range(img_dilation.shape[0]):
    for j in range(img_dilation.shape[1]):
        minmax_img[i,j] = 255*(img_dilation[i,j]-np.min(img_dilation))/(np.max(img_dilation)-np.min(img_dilation))
    
merge = cv2.merge((r,minmax_img,b))

HSV_img = cv2.cvtColor(merge,cv2.COLOR_RGB2HSV)
h,s,v = cv2.split(HSV_img)

median = cv2.medianBlur(s,5)
merge1 = cv2.merge((h,s,median))

cv2.imwrite('C:\\Users\\Puneeths\\Desktop\\Final project\\working1\\merge_oc.jpg',merge1)
image_merge = Image.open('C:\\Users\\Puneeths\\Desktop\\Final project\\working1\\merge_oc.jpg')

enh_col = ImageEnhance.Color(image_merge)
image_colored_oc = enh_col.enhance(7)

cv2.imwrite('C:\\Users\\Puneeths\\Desktop\\Final project\\working1\\image_colored_oc.jpg', np.float32(image_colored_oc))
image_c_oc = cv2.imread('C:\\Users\\Puneeths\\Desktop\\Final project\\working1\\image_colored_oc.jpg')


lab = cv2.cvtColor(image_c_oc, cv2.COLOR_BGR2LAB)

Z = lab.reshape((-1,3))
Z = np.float32(Z)
    
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    
K=2
ret, label1, center1 = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
center1 = np.uint8(center1)
res1 = center1[label1.flatten()]
output1 = res1.reshape((lab.shape))

bilateral_filtered_image = cv2.bilateralFilter(output1, 5, 175, 175)
edge_detected_image = cv2.Canny(bilateral_filtered_image, 75, 200)

contours, _= cv2.findContours(edge_detected_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
contour_list = []
for contour in contours:
      approx = cv2.approxPolyDP(contour,0.01*cv2.arcLength(contour,True),True)
      area = cv2.contourArea(contour)
      if ((len(approx) > 8) & (area > 30) ):
            contour_list.append(contour)
cv2.drawContours(image, contour_list,  -1, (255,0,0), 1)

ellipse = cv2.fitEllipse(contour)
cv2.ellipse(image,ellipse,(0,0,0),1,cv2.LINE_AA)
(x, y), (MA, ma), angle = cv2.fitEllipse(contour)
#print(MA)
#print(ma)
cuparea = (3.14/3) * MA * ma
print('Area of cup:'+str(cuparea))
cdr = cuparea / disc
print('Cup to Disc Ratio:'+str(cdr))

##cv2.imwrite('E:\\working1\\cup\\2.jpg',img_new)
#
if cdr>0.5:
      IP = 1            #glaucoma
else:
      IP = 0            #normal


#cv2.imshow("g", g)
#cv2.imshow("reverse_img", reverse_img)
#cv2.imshow("img_dilation", img_dilation)
#cv2.imshow("merge", merge)
#cv2.imshow("HSV_img", HSV_img)
#cv2.imshow("s", s)
#cv2.imshow("median", median)
#cv2.imshow("merge1", merge1)
#cv2.imshow("image_c_oc", image_c_oc)
#cv2.imshow("lab", lab)
#cv2.imshow("output1", output1)
#cv2.imshow("bilateral_filtered_image", bilateral_filtered_image)
#cv2.imshow("edge_detected_image", edge_detected_image)
cv2.imshow('original', orig)
cv2.imshow('cup to disc', image)


from keras.models import load_model
from PIL import Image
from keras.preprocessing import image
import numpy as np
import cv2
#from keras.preprocessing.image import ImageDataGenerator



#train_datagen = ImageDataGenerator(rescale = 1./255,
#shear_range = 0.2,
#zoom_range = 0.2,
#horizontal_flip = True)
#training_set = train_datagen.flow_from_directory('C:\\Users\\Navyashree HC\\Documents\\navya\\glaucoma project\\Final project\\datasets\\train\\',
#target_size = (240,240),
#batch_size = 32,
#class_mode = 'binary')

#target_size = (240,240)
model=load_model('C:\\Users\\Puneeths\\Desktop\\Final project\\Final project\\glaucoma.h5')
print("model loaded")


test_image = image.load_img('C:\\Users\\Puneeths\\Desktop\\Final project\\datasets\\test\\glau\\1 (12).png', target_size = (240,240))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = model.predict(test_image)
#training_set.class_indices
if result[0][0] != 1:
   DL = 1       #glaucoma
else:
   DL = 0       #normal
        

if (DL and IP) == 1:
   print("{} and {}".format(str(DL),str(IP)))
   print("glaucoma")
   
#        account_sid = 'AC681c3e2f099c76d20de81d3d64cefd78'
#    auth_token = '94c14aafe0c8f4b6b42734607372cc88'
#    client = Client(account_sid, auth_token)
#
#    message = client.messages \
#                          .create(
#                             body="you are affected by glaucoma",
#                             from_='+13203005724',
#                             to='+918296986769'
#                           )
#
#    print(message.sid)
#        
elif (DL and IP) == 0:
     print("{} and {}".format(str(DL),str(IP)))
     print("normal")

#    account_sid = 'AC681c3e2f099c76d20de81d3d64cefd78'
#    auth_token = '94c14aafe0c8f4b6b42734607372cc88'
#    client = Client(account_sid, auth_token)
#
#    message = client.messages \
#                          .create(
#                             body="normal",
#                             from_='+13203005724',
#                             to='+918296986769'
#                           )
#
#    print(message.sid)
#        
else:
    print("{} and {}".format(str(DL),str(IP)))
    print("suspect")

cv2.waitKey(0)
cv2.destroyAllWindows()    

        
