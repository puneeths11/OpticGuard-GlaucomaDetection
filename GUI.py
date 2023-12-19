# import the necessary packages
from tkinter import *
#from PIL import Image
from PIL import ImageTk
from tkinter import ttk
from tkinter import filedialog
from tkinter.ttk import *
import cv2                                                        
import numpy as np
from keras.models import load_model
from PIL import Image
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image, ImageEnhance
from twilio.rest import Client

def gray():
    global panelD
    gray = cv2.cvtColor(prashu, cv2.COLOR_BGR2GRAY)
#    gray = Image.fromarray(gray)
#    gray = ImageTk.PhotoImage(gray)
    display(gray,panelD)
def display(image,panel):
    image =Image.fromarray(image)
    image=ImageTk.PhotoImage(image)
    if panel is None:
        panel=Label(image=image)
        panel.image=image
        panel.pack(side="left",padx=10,pady=10)
    else:
        panel.configure(image=image)
        panel.image=image
    
def select_image():
	# grab a reference to the image panels
	global panelA, panelB,prashu
	# open a file chooser dialog and allow the user to select an input
	# image
	path = filedialog.askopenfilename()
    # ensure a file path was selected
	if len(path) > 0:
		# load the image from disk, convert it to grayscale, and detect
		# edges in it
		image = cv2.imread(path) 

		prashu=image.copy() # OpenCV represents images in BGR order; however PIL represents
		# images in RGB order, so we need to swap the channels
		image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
		#prashu=image.copy()  convert the images to PIL format...
		image = Image.fromarray(image)

		# ...and then to ImageTk format
		image = ImageTk.PhotoImage(image)
        # if the panels are None, initialize them
		if panelA is None or panelB is None:
			# the first panel will store our original image
			panelA = Label(image=image)
			panelA.image = image
			panelA.pack(side="left", padx=10, pady=10)
		else:
			# update the pannels
			panelA.configure(image=image)
			panelA.image = image

            
        # initialize the window toolkit along with the two image panels
def ip():
    global panelC,Ip
    image = prashu.copy()
    image = cv2.resize(image, (256, 256), interpolation=cv2.INTER_AREA)
    orig=image.copy()                                                   #copy of the image
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)                        #Gray conversion
    gray = cv2.GaussianBlur(gray, (3, 3), 0)                              #applying gaussian blur
    (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(gray)

    cv2.circle(image, maxLoc, 80, (0, 0, 0), 2)
    disc = 3.14 * 80 * 80
    print('Area of Disc:'+str(disc))

    r,g,b = cv2.split(orig)

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

    cv2.imwrite('C:\\Users\Puneeths\\Desktop\\Final project\\working1\\merge_oc.jpg',merge1)
    image_merge = Image.open('C:\\Users\Puneeths\\Desktop\\Final project\\working1\\merge_oc.jpg')

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
    image =cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    display(image,panelC)
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
        Ip=1
        print('glaucoma')#glaucoma
        text=Text(root,height=6,font=('calibri',13),foreground='black')
        text.insert(INSERT,"IMAGE PROCESSING RESULTS\n")
        text.insert(INSERT,"Area of Disc : " + str(disc) + "\n")
        text.insert(INSERT,"Area of Cup : " + str(cuparea) + "\n")
        text.insert(INSERT,"Cup to Disc Ratio : " + str(cdr) + "\n")
        text.insert(INSERT,"GLAUCOMA")
        text.pack()  
    else:
        Ip=0
        print('Normal')#normal
        text=Text(root,height=6,font=('calibri',13),foreground='black')
        text.insert(INSERT,"IMAGE PROCESSING RESULTS\n")        
        text.insert(INSERT,"Area of Disc : " + str(disc) + "\n")
        text.insert(INSERT,"Area of Cup : " + str(cuparea) + "\n")
        text.insert(INSERT,"Cup to Disc Ratio : " + str(cdr) + "\n")
        text.insert(INSERT,"NORMAL")
        text.pack()          

def ml():
    print(Ip)
    model=load_model('C:\\Users\\Puneeths\\Desktop\\Final project\\glaucoma.h5')
    print("model loaded")
    cv2.imwrite('C:\\Users\\Puneeths\\Desktop\\Final project\\working1\\prashu.jpg',prashu)
    test_image = tf.keras.utils.load_img('C:\\Users\\Puneeths\\Desktop\\Final project\\working1\\prashu.jpg', target_size = (240,240))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis = 0)
    result = model.predict(test_image)
    if result[0][0] != 1:
        DL = 1;       #glaucoma
    else:
        DL = 0;       #normal
    if (DL and Ip) == 1:
        print("{} and {}".format(str(DL),str(Ip)))
        print("glaucoma")
        text=Text(root,height=9,font=('calibri',13),foreground='black')
        text.insert(INSERT,"MACHINE LEARNING AND IMAGE PROCESSING\n")
        text.insert(INSERT,"Model Loaded\n")
        text.insert(INSERT,"GLAUCOMA")
        text.pack()
        account_sid = '--------------------------------'
        auth_token = '-------------------------------------'
        client = Client(account_sid, auth_token)
        message = client.messages \
        .create(
                body="You are affected by glaucoma",
                from_='------------',
                to='---------------'
                )
        print(message.sid)        
   
root = Tk()
panelA = None
panelB = None
panelC=None
panelD=None
prashu= None
Ip=None
# create a button, then when pressed, will trigger a file chooser
# dialog and allow the user to select an input image; then add the
# button the GUI
root.title("GLAUCOMA GUI /Code Of Duty/2023/PSRR")
root.configure(background='#4798FF')
# kick off the GUI
root.geometry("1200x500")
s=ttk.Style()
s.configure('TButton', font=('calibri',15),foreground='black',anchor=CENTER,borderwidth=2,background='white')
text=Text(root,height=1, width='34',font=('Serif',15,'bold'),foreground='black',borderwidth=2,background='white')
text.insert(INSERT,"\tGlaucoma Detection\n")
text.pack()
btnml = ttk.Button(text = "Machine Learning + IP", command = ml )
btnml.pack(side="bottom", expand="no", padx="10", pady="10")
btnip = ttk.Button(text = "Image Processing", command = ip )
btnip.pack(side="bottom", expand="no", padx="10", pady="10")
#btngray = ttk.Button(text = "         Gray         ", command = gray )
#btngray.pack(side="bottom", expand="no", padx="10", pady="10")
btn = Button(root, text="Select an image", command=select_image)
btn.pack(side="bottom",expand="no", padx="10", pady="10")
root.mainloop()

# ╔════════════════════════════════════════╗
# ║          FOR THE FULL CODE/PROJECT     ║
# ║           CONTACT:                     ║
# ║           PUNEETHSPUNII@GMAIL.COM      ║
# ║           +918296986769                ║
# ╚════════════════════════════════════════╝
