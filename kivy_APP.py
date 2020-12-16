__author__ = 'bunkus'
from kivy.app import App
from kivy.uix.widget import Widget
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.relativelayout import RelativeLayout
from kivy.uix.camera import Camera
from kivy.clock import Clock
from kivy.graphics.texture import Texture
import matplotlib.pyplot as plt
from kivy.uix.button import Button
from kivy.uix.textinput import TextInput
import numpy as np
import os
import cv2 

class CamApp(App):
    def build(self):
        layout = RelativeLayout(size=(500,500))
        self.cam = Camera(play=True,resolution=(1500,1500))
        self.button = Button(text='Detect a face!',
                        size_hint=(.5, .07),
                        pos=(450,100))
        self.button.bind(on_press=self.on_press_button)
        self.button_sketch = Button(text='Create a sketch!',
                        size_hint=(.5, .07),
                        pos=(450,10))
        self.button_paint = Button(text='Create a painting',
                        size_hint=(.5,0.07),
                        pos=(450,190))
        self.button_sketch.bind(on_press=self.on_press_sketch_button)
        self.button_paint.bind(on_press=self.on_press_paint_button)
        layout.add_widget(self.cam)
        layout.add_widget(self.button)
        layout.add_widget(self.button_sketch)
        layout.add_widget(self.button_paint)
        return layout

    def on_press_button(self,instance):
 #       try: 
        self.imagePath = os.path.join('Selfies','Selfie.png')
        self.cam.export_to_png(self.imagePath)
        haarCascade= cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
# Read the image
        image = cv2.imread(self.imagePath)

# Detect faces in the image
        faces = haarCascade.detectMultiScale(
    cv2.cvtColor(image, cv2.COLOR_BGR2GRAY),
    scaleFactor=1.1,
    minNeighbors=5,
    minSize=(30, 30),
    flags = cv2.CASCADE_SCALE_IMAGE
)
# Draw a rectangle around the faces
        for (x, y, w, h) in faces:
            cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

        cv2.imwrite(os.path.join('Selfies','Selfie_detected.jpg'),image)    

        return 0


    def on_press_sketch_button(self,instance):
        self.imagePath = os.path.join('Sketch','Selfie_sketch.png')
        self.cam.export_to_png(self.imagePath)
        image = cv2.imread(self.imagePath)
        img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        img_gray_inv = 255 - img_gray
        img_blur = cv2.GaussianBlur(img_gray_inv, ksize=(21, 21),
                            sigmaX=0, sigmaY=0)
        img_blend = cv2.divide(img_gray, 255-img_blur, scale=256)
        cv2.imwrite(os.path.join('Sketch','transformed.png'),img_blend) 

        return 0

    def on_press_paint_button(self,instance):
        self.imagePath = os.path.join('Paint','Selfie.png')
        self.cam.export_to_png(self.imagePath)
        img = cv2.imread(self.imagePath)
        res = cv2.stylization(img, sigma_s=30, sigma_r=0.3)
        cv2.imwrite(os.path.join('Paint','painting.png'),res)

        return 0
if __name__ == '__main__':
    CamApp().run()