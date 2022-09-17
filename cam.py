# Import fast ai and other required libraries
import fastai
fastai.__version__
from fastai import *
from fastai.vision import *
import os
import cv2
import sys
from PIL import Image, ImageTk
import numpy
import tkinter as tk

# Camara index
camIndex = 0    #CHANGE THIS TO 1,2 etc and re-run code

# Flag to control frame display
cancel = False
 
# Detect Drug and Try Again buttons
def prompt_ok(event = 0):
    global cancel, button, button1, button2
    cancel = True
 
    button.place_forget()
    button1 = tk.Button(mainWindow, text="Detect Drug..", command=saveAndExit)
    button2 = tk.Button(mainWindow, text="Try Again", command=resume)
    button1.place(anchor=tk.CENTER, relx=0.2, rely=0.9, width=150, height=50)
    button2.place(anchor=tk.CENTER, relx=0.8, rely=0.9, width=150, height=50)
    button1.focus()
 
# Save the image and exit to continuous frame capture. -- Call pill detection here
def saveAndExit(event = 0):
    global prevImg
 
    if (len(sys.argv) < 2):
        filepath = "image.png"
    else:
        filepath = sys.argv[1]
 
    print ("Output file to: " + filepath)
    prevImg.save(filepath)
    detectImage()

# Pill detection 
# Create class of pills same as used in Train
# Get the model
# Use the  model to predict it
# Display Name and Accuracy in textbox
def detectImage():
    classes = ['cofsils', 'dolo', 'marcks covid','penicillium']
    data1 = ImageDataBunch.single_from_classes("content", classes, ds_tfms=get_transforms(),
                                               size=224).normalize(imagenet_stats)
    learn1 = cnn_learner(data1, models.resnet34)
    learn1.load('stage-2')

    img = open_image('image.png')
    pred_class, pred_idx, outputs = learn1.predict(img)
    print(outputs)
    val = list(map(float,str(outputs)[8:-2].split(",")))[int(float(str(pred_idx)[7:-1]))]*100
    if val > 99:    
        text_box.delete(1.0, "end-1c")
        text_box.insert("end-1c", str(pred_class) + "  --  " + str(val) + " % Accuracy")
    else:
        text_box.delete(1.0, "end-1c")
        text_box.insert("end-1c", "Unable to Predict, No Accuracy..")
 
 
# Show frame again, if try again pressed
def resume(event = 0):
    global button1, button2, button, lmain, cancel
 
    cancel = False
 
    button1.place_forget()
    button2.place_forget()
 
    mainWindow.bind('<Return>', prompt_ok)
    button.place(bordermode=tk.INSIDE, relx=0.5, rely=0.9, anchor=tk.CENTER, width=300, height=50)
    lmain.after(10, show_frame)
    text_box.delete(1.0, "end-1c")

 
# Capture continuous image frame
cap = cv2.VideoCapture(camIndex)
capWidth = cap.get(3)
capHeight = cap.get(4)
 
success, frame = cap.read()


# Create TK window to display image
mainWindow = tk.Tk()
mainWindow.resizable(width=False, height=False)
mainWindow.bind('<Escape>', lambda e: mainWindow.quit())
lmain = tk.Label(mainWindow, compound=tk.CENTER, anchor=tk.CENTER, relief=tk.RAISED)

text_box = tk.Text(mainWindow, width = 40, height = 2)
text_box.pack()

button = tk.Button(mainWindow, text="Capture", command=prompt_ok)
lmain.pack()
button.place(bordermode=tk.INSIDE, relx=0.5, rely=0.9, anchor=tk.CENTER, width=300, height=50)
button.focus()
 

# Show image frame
def show_frame():
    global cancel, prevImg, button
 
    _, frame = cap.read()
    cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
 
    prevImg = Image.fromarray(cv2image)
    imgtk = ImageTk.PhotoImage(image=prevImg)
    lmain.imgtk = imgtk
    lmain.configure(image=imgtk)
    if not cancel:
        lmain.after(10, show_frame)
 
# Run in infinite loop
show_frame()
mainWindow.mainloop()
