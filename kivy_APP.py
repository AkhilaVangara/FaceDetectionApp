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
        self.cam = Camera(play=True,resolution=(2000,2000))
        self.button = Button(text='Click a photo',
                        size_hint=(.5, .07),
                        pos=(450,100))
        self.button.bind(on_press=self.on_press_button)
 #       self.textinput = TextInput(text='Enter Image Name')
 #       self.textinput.bind(on_text_validate=self.on_enter)
        layout.add_widget(self.cam)
        layout.add_widget(self.button)
        return layout

#    def on_enter(self,instance):
#        self.imagePath = TextInput.text()
#        return 0

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

if __name__ == '__main__':
    CamApp().run()