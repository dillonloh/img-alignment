from tkinter import *
from tkinter.filedialog import askopenfilename

from PIL import Image, ImageTk
from subprocess import run



NO_FLOORS = 3
NO_OF_POINTS = 3
SET_NO = 1
GLOBAL_POINTS = [] # each entry is a floor's dict of points
GLOBAL_IMAGES = []
GLOBAL_IMAGE_PATHS = []
CURRENT_IMG_INDEX = 0

root = Tk()

def create_window():

    global GLOBAL_POINTS, GLOBAL_IMAGES, CURRENT_IMG_INDEX
    global img


    #initialise list of floor dicts
    for i in range(NO_FLOORS):
        f = {}
        GLOBAL_POINTS.append(f)


    window = Toplevel(root)
    frame_btn = Frame(window, bd=2,)

    frame_btn.grid_rowconfigure(0, weight=1)
    frame_btn.grid_columnconfigure(0, weight=1)

    label_btn = Label(master=frame_btn, text='Set coordinates of:')
    label_btn.grid(row=0, column=0, rowspan=2)

    def calibrate():
        global GLOBAL_POINTS, GLOBAL_IMAGES, GLOBAL_IMAGE_PATHS, NO_FLOORS, NO_OF_POINTS
        
        import pickle

        for i in range(NO_FLOORS):
            GLOBAL_POINTS[i]['img'] = GLOBAL_IMAGE_PATHS[i]

        GLOBAL_POINTS.append(NO_FLOORS) # metainfo index -2
        GLOBAL_POINTS.append(NO_OF_POINTS) # metainfo index -1

        shared = GLOBAL_POINTS

        with open('shared.pkl', 'wb') as fp:
            pickle.dump(shared, fp)
    
        run(['python', "mapcalibration.py"])


    end_label = Label(master=frame_btn, text='Click here when done:')
    end_label.grid(row=2, column=0)
    end_btn = Button(master=frame_btn, text='Calibrate and plot now', command=calibrate)
    end_btn.grid(row=2, column=1, columnspan=3)


    def set_floor(floor_no):
        global CURRENT_IMG_INDEX
        CURRENT_IMG_INDEX = (floor_no - 1)
        update_map()


    def set_number(button_no):
        global SET_NO
        SET_NO = button_no


    button_dict = {}
    button_no = {}

    for i in range(1, NO_FLOORS+1):
        button_dict['button{}'.format(i)] = Button(master=frame_btn, text='Floor {}'.format(i), command=lambda j=i: set_floor(j))
        button_dict['button{}'.format(i)].grid(row=0, column=i)

    for i in range(1, NO_OF_POINTS+1):
        button_dict['button{}'.format(i)] = Button(master=frame_btn, text='Point {}'.format(i), command=lambda j=i: set_number(j))
        button_dict['button{}'.format(i)].grid(row=1, column=i)

    frame_btn.pack()

    frame = Frame(window, bd=2, relief=SUNKEN)

    frame.grid_rowconfigure(0, weight=1)
    frame.grid_columnconfigure(0, weight=1)

    xscroll = Scrollbar(frame, orient=HORIZONTAL)
    xscroll.grid(row=1, column=0, sticky=E+W)
    yscroll = Scrollbar(frame)
    yscroll.grid(row=0, column=1, sticky=N+S)

    canvas = Canvas(frame, bd=0, xscrollcommand=xscroll.set, yscrollcommand=yscroll.set)
    canvas.grid(row=0, column=0, sticky=N+S+E+W)

    xscroll.config(command=canvas.xview)
    yscroll.config(command=canvas.yview)

    frame.pack(fill=BOTH,expand=1)

    # adding images
    
    for i in range(NO_FLOORS):

        File = askopenfilename(parent=window, initialdir="../Images",title='Please upload the map image of Floor {}.'.format(i+1))
        GLOBAL_IMAGE_PATHS.append(File)
        GLOBAL_IMAGES.append(ImageTk.PhotoImage(Image.open(File)))
    
    canvas.create_image(0,0,image=GLOBAL_IMAGES[CURRENT_IMG_INDEX],anchor="nw") # always display floor 1 image first
    canvas.config(scrollregion=canvas.bbox(ALL))


    #function to be called when mouse is clicked
    def printcoords(event):
        #outputting x and y coords to console

        global GLOBAL_POINTS

        canvas = event.widget
        y = canvas.canvasx(event.x)
        x = canvas.canvasy(event.y)

        print('[FLOOR {}] Coordinates of point {} is ({},{})'.format(CURRENT_IMG_INDEX+1, SET_NO, x, y))

        GLOBAL_POINTS[CURRENT_IMG_INDEX]['p{}'.format(SET_NO)] = [x, y] # assign to floor's p{} the clicked coordinates

 
    #mouseclick event
    canvas.bind("<Button 1>", printcoords)
    

    def update_map():
        canvas.create_image(0,0,image=GLOBAL_IMAGES[CURRENT_IMG_INDEX],anchor="nw")
        canvas.config(scrollregion=canvas.bbox(ALL))



def submit():
    global NO_FLOORS
    global NO_OF_POINTS
    NO_FLOORS = int(field_floors.get())
    NO_OF_POINTS = int(field_points.get())
    update_params()


frame_root = Frame()
label_floors = Label(master=root, text='Please indicate the number of floors in your building.')
label_f = Label(master=root, text='{} floors registered'.format(NO_FLOORS))
field_floors = Entry(master=root, )

label_floors.grid(row=0,column=0,columnspan=2)
field_floors.grid(row=1,column=0,columnspan=2)
label_f.grid(row=2,column=0,columnspan=2)


label_points = Label(master=root, text='Please enter the number of points you will use for calibration.')
label_p = Label(master=root, text='{} points registered'.format(NO_OF_POINTS))
field_points = Entry(master=root,)
label_points.grid(row=3,column=0,columnspan=2)
field_points.grid(row=4,column=0,columnspan=2)
label_p.grid(row=5,column=0,columnspan=2)

btn_submit = Button(master=root, text='Submit Parameters', command=submit)
btn_submit.grid(row=6, column=0)

def update_params():   
    label_f.configure(text='{} floors registered'.format(NO_FLOORS))
    label_p.configure(text='{} points registered'.format(NO_OF_POINTS))
    root.after(1, update_params)


btn_upload = Button(master=root, text='Begin Calibration', command=create_window)
btn_upload.grid(row=6, column=1)



root.title('3D Alignment of Floor Plans')
root.mainloop()
