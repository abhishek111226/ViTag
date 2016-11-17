from read import start_extraction
from revsr import start_search
import logging
import os
import shutil
import Tkinter as tk
import tkMessageBox
import tkFileDialog as filedialog
import sys
import cv2
from Tkinter import *
import Image, ImageTk
import ttk

root = tk.Tk()
root.withdraw()
path = filedialog.askopenfilename()
print path
video = path.split('/')[-1]
dir = './extract'+video+'_/'
print "\n\n ---------------- ViTag --------------"
logging.basicConfig(filename='ViTagLog_'+video+'.log',level=logging.DEBUG)
logging.info("\n\n ---------------- ViTag --------------")
logging.info("\n\n ---------------- Logging information--------------")
logging.info("\n\n -----Video: "+video+" path: "+path)
if os.path.exists(dir):
	shutil.rmtree(dir);
os.makedirs(dir)

files = start_extraction(path,dir)
results = start_search(files)
results = filter(None, results)
output = list(set(results)) 
print output
print "\n\n \t\t Developed At IIT Hyderabad, India"
print "\n\n \t\t Abhishek Patwardhan, Santanu Das, Sakshi Varshney, Maundendra Desarkar"
print "\n\n \t\t For Bugs contact : cs15mtech11015@iith.ac.in"
logging.info("Results are ")
logging.info(results)
logging.info("Output shown")
logging.info(output)
logging.info("Program executed successfully")
logging.info("\n\n ----------------END--------------")
logging.info("\n\n ----------------ViTag--------------")
logging.info("\n\n \t\t Developed At IIT Hyderabad, India")
logging.info("\n\n \t\t Abhishek Patwardhan, Santanu Das, Sakshi Varshney, Maundendra Desarkar")
logging.info("\n\n \t\t For Bugs contact : cs15mtech11015@iith.ac.in")

'''
top = tk.Tk()
def playVideo():
   tkMessageBox.showinfo( "Hello Python", "Hello World")

top.minsize(500,400);

top.mainloop()
'''
window = tk.Toplevel()  #Makes main window
window.wm_title("ViTag")
window.config(background="#FFFFFF")
var = StringVar()
label = Label(window, textvariable=var, relief=RAISED, font=("Times", 26))

var.set("ViTag: Automatic Video Tagger")
label.pack()

ttk.Separator(window,orient=HORIZONTAL).pack(fill="x")
#Graphics window
imageFrame = tk.Frame(window, width=600, height=500)
imageFrame.pack(fill=None,expand=False)
#imageFrame.grid(row=0, column=0, padx=10, pady=2, expand=False)
#Capture video frames
lmain = tk.Label(imageFrame, width=550, height=300)
lmain.grid(row=0, column=0)
cap = cv2.VideoCapture(path)
def showframe():
    _, frame = cap.read()
    frame = cv2.flip(frame, 1)
    cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
    cv2image = cv2.resize(cv2image, (500, 250)) 
    img = Image.fromarray(cv2image)
    imgtk = ImageTk.PhotoImage(image=img)
    lmain.imgtk = imgtk
    lmain.configure(image=imgtk)
    lmain.after(10, showframe) 

ttk.Separator(window,orient=HORIZONTAL).pack(fill="x")

tags=''
cnt=0
for x in output:
    cnt=cnt+1;
    if(cnt %3==0):
	tags= tags+"("+str(cnt)+") "+x + '\n'
    else:
	tags=tags+"("+str(cnt)+")"+x + '\t'
print tags
t = Text(window, font=("Times", 16), width=50, height=5)
t.insert(END, tags)
t.config(state=DISABLED)
t.pack()


footer_1 = Label(window, text=" *Developed at IIT Hyderabad", relief=RAISED, font=("Times", 8))
footer_1.pack()
footer_2 = Label(window, text=" ** ViTag internally uses Google Reverse Image Search", relief=RAISED, font=("Times", 8))
footer_2.pack()
B = tk.Button(window, text ="View Tags", command = showframe())
#B.pack()


#Slider window (slider controls stage position)
#sliderFrame = tk.Frame(window, width=600, height=100)
#sliderFrame.grid(row = 600, column=0, padx=10, pady=2)
  #Display 2
window.mainloop()  #Starts GUI


