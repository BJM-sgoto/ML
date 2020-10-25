import os
import tkinter as tk
from tkinter import messagebox
import math
from PIL import Image, ImageTk, ImageDraw

class Example(tk.Frame):
  
  def __init__(self, root):
    super().__init__()
    self.root = root
    self.root.protocol("WM_DELETE_WINDOW", self.onWindowClose)
    self.marks = []  
    self.dynamic = []
    self.initUI()
    
  def onWindowClose(self):
    self.root.destroy()
    
  def loadNext(self):
    self.canvas_input.delete("all")
    self.file_id = self.file_id + 1
    if self.file_id >= len(self.files):
      messagebox.showinfo("Error", "No more image")
    else:
      print("load file", self.files[self.file_id])
      self.img = Image.open("./train/" + self.files[self.file_id])
      if (self.img.width > 800) or (self.img.height>800):
        rate = 800 / max(self.img.width, self.img.height)
        self.img = self.img.resize(((int)(self.img.width*rate), (int)(self.img.height*rate)))
      
      self.cut_size = min(self.img.width, self.img.height)
      self.cut_size = min(self.cut_size, 256)
      self.cut_image = self.img.crop((0,0,self.cut_size, self.cut_size))
      if self.cut_size!=256:
        self.cut_image = self.cut_image.resize((256,256))
      self.minus_offset = math.floor(self.cut_size/2)
      self.posit_offset = self.cut_size - 1 - self.minus_offset
      self.photo = ImageTk.PhotoImage(self.img)
      self.canvas_input.create_image(0, 0, anchor=tk.NW, image=self.photo)
      self.canvas_input.config(width=self.img.width, height=self.img.height)
      if self.select_mode_option.get() == 3:
        self.cut_image = Image.new("RGB", (self.img.width, self.img.height))
        self.cut_photo = ImageTk.PhotoImage(self.cut_image)
        self.canvas_output.create_image(0, 0, anchor=tk.NW, image=self.cut_photo)
        
        self.canvas_output.config(width=self.img.width, height=self.img.height)
    if self.select_mode_option.get()==2:
      self.dynamic.clear()
        
  def drawFillPolygonOnCanvas(self, polygon):
    p = []
    for i in range(len(polygon)//2):
      p.append([polygon[i*2], polygon[i*2+1]])
    self.canvas_output.create_polygon(p, fill="white")
      
  def saveImage(self):
    if self.size_option.get() == 1:
      self.cut_image = self.cut_image.resize((256,256))
    if self.select_mode_option.get() != 2: 
      print("Save ", self.files[self.file_id]," size", self.cut_image.width, " ", self.cut_image.height)
      self.cut_image.save("./data/" + self.files[self.file_id])    
    else:
      for i in range(len(self.dynamic)):
        square = self.dynamic[i]
        name = self.files[self.file_id]
        name = name[:name.rfind(".")] + "_" + str(i) + ".jpg"
        newFileName = "./data/" + name
        self.img.crop((min(square[0], square[2]), min(square[1], square[3]), max(square[0], square[2]), max(square[1], square[3]))).resize((256,256)).save(newFileName)
        print("Save", newFileName)
  
  def onLeftMousePress(self, e):
    x = e.x
    y = e.y
    if self.select_mode_option.get() == 2:
      self.dynamic.append([x,y,0,0])
    elif self.select_mode_option.get() == 3:
      n = len(self.marks)
      if n == 0:
        self.marks.append([x,y])
        self.canvas_input.create_image(0, 0, anchor=tk.NW, image=self.photo)
      else: #  
        last_polygon = self.marks[len(self.marks) - 1]
        last_polygon.append(x)
        last_polygon.append(y)
        m = len(last_polygon)
        self.canvas_input.create_line(last_polygon[m-4], last_polygon[m-3], last_polygon[m-2], last_polygon[m-1],fill="red")
        
  def onRightMousePress(self, e):
    if self.select_mode_option.get() == 3:
      n = len(self.marks)
      if n > 0:
        last_polygon = self.marks[n-1]
        if len(last_polygon)>=2:
          self.canvas_input.create_image(0, 0, anchor=tk.NW, image=self.photo)
          last_polygon.pop() # pop x
          last_polygon.pop() # pop y
          n = n - 1
          # draw other polygons but the last one
          for i in range(n-2):
            polygon = self.marks[i]
            for j in range(len(polygon)//2):
              m = (j+2)%(len(polygon)//2)
              self.canvas_input.create_line(polygon[j*2], polygon[j*2+1], polygon[m*2], polygon[m*2+1], fill="red")
          # draw last polygon  
          for j in range(len(last_polygon)//2):
            self.canvas_input.create_line(last_polygon[j*2], last_polygon[j*2+1], last_polygon[j*2+2], last_polygon[j*2+3], fill="red")
      else:
        self.marks.clear()
    elif self.select_mode_option.get()==2:
      if len(self.dynamic)!=0:
        self.dynamic.pop()
      self.canvas_input.create_image(0, 0, anchor=tk.NW, image=self.photo)
      for i in range(len(self.dynamic)):
        square = self.dynamic[i]
        self.canvas_input.create_line(square[0], square[1], square[0], square[3],fill="red")
        self.canvas_input.create_line(square[0], square[1], square[2], square[1],fill="red")
        self.canvas_input.create_line(square[2], square[3], square[2], square[1],fill="red")
        self.canvas_input.create_line(square[2], square[3], square[0], square[3],fill="red")
      print("Draw rightr")
        
  def onLeftMouseDoubleClick(self, e):
    if self.select_mode_option.get() == 3:
      last_polygon = self.marks[len(self.marks)-1]
      last_polygon.append(e.x)
      last_polygon.append(e.y)
      drawer = ImageDraw.Draw(self.cut_image)
      self.canvas_input.create_line(e.x, e.y, last_polygon[0], last_polygon[1], fill="red")
      for i in range(len(self.marks)):
        polygon = self.marks[i]
        drawer.polygon(polygon, fill="white")
        self.drawFillPolygonOnCanvas(polygon)
      self.marks.append([])
  
  def onLeftMouseMove(self, e):
    x = e.x
    y = e.y
    if self.select_mode_option.get() == 1 : 
      self.canvas_input.create_image(0, 0, anchor=tk.NW, image=self.photo)
      
      if x - self.minus_offset < 0:
        x = self.minus_offset
      if y - self.minus_offset < 0: 
        y = self.minus_offset
        
      if x + self.posit_offset >= self.img.width:
        x = self.img.width - 1 - self.posit_offset
      if y + self.posit_offset >= self.img.height: 
        y = self.img.height - 1 - self.posit_offset
      self.canvas_input.create_line(x-self.minus_offset, y-self.minus_offset, x-self.minus_offset, y+self.posit_offset,fill="red")
      self.canvas_input.create_line(x-self.minus_offset, y+self.posit_offset, x+self.posit_offset, y+self.posit_offset,fill="red")
      self.canvas_input.create_line(x+self.posit_offset, y+self.posit_offset, x+self.posit_offset, y-self.minus_offset,fill="red")
      self.canvas_input.create_line(x+self.posit_offset, y-self.minus_offset, x-self.minus_offset, y-self.minus_offset,fill="red")
      newPos = (x-self.minus_offset, y-self.minus_offset,x+self.posit_offset, y+self.posit_offset)
      self.cut_image = self.img.crop(newPos)
      if self.cut_size!=256:
        self.cut_image = self.cut_image.resize((256,256))
      self.cut_photo = ImageTk.PhotoImage(self.cut_image)
      self.canvas_output.create_image(0, 0, anchor=tk.NW, image=self.cut_photo)
    elif self.select_mode_option.get() == 2:
    
      if x >= self.img.width:
        x = self.img.width - 1
      elif x < 0:
        x = 0
      
      if y >= self.img.height:
        y = self.img.height - 1
      elif y < 0:
        y = 0
      self.canvas_input.create_image(0, 0, anchor=tk.NW, image=self.photo)
      for i in range(len(self.dynamic)-1):
        square = self.dynamic[i]
        self.canvas_input.create_line(square[0], square[1], square[0], square[3],fill="red")
        self.canvas_input.create_line(square[0], square[1], square[2], square[1],fill="red")
        self.canvas_input.create_line(square[2], square[3], square[2], square[1],fill="red")
        self.canvas_input.create_line(square[2], square[3], square[0], square[3],fill="red")
      x = e.x
      y = e.y
      last_square = self.dynamic[len(self.dynamic)-1]
      side = min(abs(x - last_square[0]), abs(y - last_square[1]))
      if x < last_square[0]:
        x = last_square[0] - side
      else:
        x = last_square[0] + side
        
      if y < last_square[1]:
        y = last_square[1] - side
      else:
        y = last_square[1] + side
      
      self.canvas_input.create_line(last_square[0], last_square[1], last_square[0], y,fill="red")
      self.canvas_input.create_line(last_square[0], last_square[1], x, last_square[1],fill="red")
      self.canvas_input.create_line(x, y, x, last_square[1],fill="red")
      self.canvas_input.create_line(x, y, last_square[0], y,fill="red")
      
    elif self.select_mode_option.get() == 3:
      self.canvas_input.create_image(0, 0, anchor=tk.NW, image=self.photo)
      
  def onLeftMouseRelease(self, e):
    x = e.x
    y = e.y
    
    if self.select_mode_option.get()==2:
      if len(self.dynamic) == 0:
        return
      
      if x >= self.img.width:
        x = self.img.width - 1
      elif x < 0:
        x = 0
      
      if y >= self.img.height:
        y = self.img.height - 1
      elif y < 0:
        y = 0
      
      last_square = self.dynamic[len(self.dynamic)-1]
      last_square[2] = x
      last_square[3] = y
      
      side = min(abs(last_square[2] - last_square[0]), abs(last_square[3] - last_square[1]))
      if side < 10:
        self.dynamic.pop()
        return
      else:
        if last_square[2] < last_square[0]:
          last_square[2] = last_square[0] - side
        else:
          last_square[2] = last_square[0] + side
          
        if last_square[3] < last_square[1]:
          last_square[3] = last_square[1] - side
        else:
          last_square[3] = last_square[1] + side
      self.canvas_input.create_image(0, 0, anchor=tk.NW, image=self.photo)
      for i in range(len(self.dynamic)-1):
        square = self.dynamic[i]
        self.canvas_input.create_line(square[0], square[1], square[0], square[3],fill="red")
        self.canvas_input.create_line(square[0], square[1], square[2], square[1],fill="red")
        self.canvas_input.create_line(square[2], square[3], square[2], square[1],fill="red")
        self.canvas_input.create_line(square[2], square[3], square[0], square[3],fill="red")
      self.canvas_input.create_line(last_square[0], last_square[1], last_square[2], last_square[1],fill="red")
      self.canvas_input.create_line(last_square[0], last_square[1], last_square[0], last_square[3],fill="red")
      self.canvas_input.create_line(last_square[2], last_square[3], last_square[2], last_square[1],fill="red")
      self.canvas_input.create_line(last_square[2], last_square[3], last_square[0], last_square[3],fill="red")
      
      self.cut_image = self.img.crop((min(last_square[0], last_square[2]), min(last_square[1], last_square[3]), max(last_square[0], last_square[2]), max(last_square[1], last_square[3])))
      self.cut_image = self.cut_image.resize((256, 256))
      self.cut_photo = ImageTk.PhotoImage(self.cut_image)
      self.canvas_output.create_image(0, 0, anchor=tk.NW, image=self.cut_photo)
      
  def onKeyPress(self, key):
    if key.char == "s":
      self.saveImage()
  
  def onModeStaticSquare(self):
    self.select_mode_option.set(1)
    self.size_option.set(1)
    self.outputMenu.entryconfig("Original Size", state="disabled")
    self.outputMenu.entryconfig("Size 256X256", state="normal")
    self.canvas_output.config(width=256, height=256)
    
  def onModeDynamicSquare(self):
    self.select_mode_option.set(2)
    self.size_option.set(1)
    self.dynamic = []
    self.outputMenu.entryconfig("Original Size", state="disabled")
    self.outputMenu.entryconfig("Size 256X256", state="normal")
    self.canvas_output.config(width=256, height=256)
    
  def onModePolygon(self):
    self.select_mode_option.set(3)
    self.size_option.set(2)
    self.marks = []
    self.outputMenu.entryconfig("Original Size", state="normal")
    self.outputMenu.entryconfig("Size 256X256", state="disabled")
    
    self.cut_image = Image.new("RGB", (self.img.width, self.img.height))
    self.cut_photo = ImageTk.PhotoImage(self.cut_image)
    self.canvas_output.create_image(0, 0, anchor=tk.NW, image=self.cut_photo)
    
    self.canvas_output.config(width=self.img.width, height=self.img.height)
    
  def onFixedSizeSelect(self):
    self.size_option.set(1)
    self.canvas_output.config(width=256, height=256)
    
  def onOriginalSizeSelect(self):
    self.size_option.set(2)
    
    self.cut_image = self.img.crop((0, 0, self.img.width, self.img.height))
    self.cut_photo = ImageTk.PhotoImage(self.cut_image)
    self.canvas_output.create_image(0, 0, anchor=tk.NW, image=self.cut_photo)
    
    self.canvas_output.config(width=self.img.width, height=self.img.height)
  
  def initUI(self): 
    current_files = os.listdir("./data/")
    if len(current_files)!=0:
      current_files.sort()
    
    self.files = os.listdir("./train")
    if len(self.files)!=0:
      self.files.sort()
      
    n_file = len(self.files)
    i = 0
    
    self.menubar = tk.Menu(self.root)
    self.root.config(menu=self.menubar)
    
    self.select_mode_option = tk.IntVar()
    self.select_mode_option.set(1)
    
    self.modeMenu = tk.Menu(self.menubar, tearoff=0)
    self.modeMenu.add_checkbutton(label="Static Square", onvalue = 1, variable = self.select_mode_option, command=self.onModeStaticSquare)
    self.modeMenu.add_checkbutton(label="Dynamic Square",onvalue = 2, variable = self.select_mode_option, command=self.onModeDynamicSquare)
    self.modeMenu.add_checkbutton(label="Polygon", onvalue = 3, variable = self.select_mode_option , command=self.onModePolygon)
    
    self.outputMenu = tk.Menu(self.menubar, tearoff = 0)
    #output size
    self.size_option = tk.IntVar()
    self.size_option.set(1)
    self.outputMenu.add_checkbutton(label = "Size 256X256", onvalue = 1, variable = self.size_option, command=self.onFixedSizeSelect)
    self.outputMenu.add_checkbutton(label = "Original Size", onvalue = 2, variable = self.size_option, command=self.onOriginalSizeSelect)
    self.outputMenu.entryconfig("Original Size", state = "disabled")
    
    self.menubar.add_cascade(label="Mode", menu=self.modeMenu)
    self.menubar.add_cascade(label="Output", menu=self.outputMenu)
    
    while i<n_file:
      if not self.files[i].endswith(".jpg") and not self.files[i].endswith(".png") and not self.files[i].endswith(".jpeg"):
        self.files.pop(i)
        i = i-1  
        n_file = n_file - 1
      i = i+1
    
    self.file_id = 0
    
    if len(current_files)!=0:
      current_img = current_files[len(current_files)-1]
      if "_" in current_img:
        current_img = current_img[:current_img.rfind("_")] + ".jpg"
      
      for i in range(len(self.files)):
        if self.files[i] == current_img:
          self.file_id = i
          break

    self.master.title("Cut image")
    self.pack(fill=tk.BOTH, expand=1)
    
    self.upper_frame = tk.Frame(self)
    self.upper_frame.pack(side=tk.TOP)
    print("load file", self.files[self.file_id])
    self.img = Image.open("./train/" + self.files[self.file_id])
    if (self.img.width > 800) or (self.img.height>800):
      rate = 800 / max(self.img.width, self.img.height)
      self.img = self.img.resize(((int)(self.img.width*rate), (int)(self.img.height*rate)))
    
    self.cut_size = min(self.img.width, self.img.height)
    self.cut_size = min(self.cut_size, 256)
    self.cut_image = self.img.crop((0,0,self.cut_size, self.cut_size))
    if self.cut_size!=256:
      self.cut_image = self.cut_image.resize((256,256))
    
    self.minus_offset = math.floor(self.cut_size/2)
    self.posit_offset = self.cut_size - 1 - self.minus_offset
    
    self.photo = ImageTk.PhotoImage(self.img)
    self.cut_photo = ImageTk.PhotoImage(self.cut_image)
    
    self.canvas_input = tk.Canvas(
            self.upper_frame, 
            width=self.img.size[0], 
            height=self.img.size[1])
    self.canvas_input.create_image(0, 0, anchor=tk.NW, image=self.photo)
    self.canvas_input.pack(side=tk.LEFT)
    self.canvas_input.bind(sequence="<B1-Motion>", func=self.onLeftMouseMove)
    self.canvas_input.bind(sequence="<Button-1>", func=self.onLeftMousePress)
    self.canvas_input.bind(sequence="<ButtonRelease-1>", func=self.onLeftMouseRelease)
    self.canvas_input.bind(sequence="<Button-3>", func=self.onRightMousePress)
    self.canvas_input.bind(sequence="<Double-Button-1>", func=self.onLeftMouseDoubleClick)
    self.root.bind("<Key>",self.onKeyPress)
    
    self.canvas_output = tk.Canvas(
            self.upper_frame, 
            width=256, 
            height=256)
    self.canvas_output.create_image(0, 0, anchor=tk.NW, image=self.cut_photo)
    self.canvas_output.pack(side=tk.LEFT)
    
    self.lower_frame = tk.Frame(self)
    self.lower_frame.pack(side=tk.BOTTOM, fill = tk.X)
    
    self.button_next = tk.Button(
            self.lower_frame,
            text="Next",
            command=self.loadNext)
    self.button_next.pack(expand = True,side = tk.LEFT, fill = tk.X)
    
    self.button_save = tk.Button(
            self.lower_frame,
            text="Save",
            command=self.saveImage)
    self.button_save.pack(expand = True, side = tk.LEFT, fill = tk.X)
    
def main():
  root = tk.Tk()
  ex = Example(root)
  root.resizable(False, False)
  root.mainloop()  


if __name__ == '__main__':
    main()    