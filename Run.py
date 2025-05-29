import tkinter as tk
import os
from tkinter import Message, Text
from PIL import Image, ImageTk
import PIL.Image  # Simplified PIL import
import pandas as pd
from tkinter import Tk, Label, Entry, Radiobutton, Button, IntVar, PhotoImage # Specific tkinter imports
import tkinter.ttk as ttk
import tkinter.font as font
import tkinter.messagebox as tm
import matplotlib.pyplot as plt
import csv
import torch
from torchvision import transforms
# Import MyModel from cnn_pytorch.py
from cnn_pytorch import MyModel
import numpy as np
# import main as signtotext # This top-level import is removed to avoid circular dependency
import generate_data as gd
import cnn_pytorch as train # Assuming cnn_pytorch.py is your training script
from itertools import count, cycle
import string
import time
import cv2
from gtts import gTTS
from playsound import playsound
import speech_recognition as sr
import random

# --- PyTorch Model Setup ---
# Load the PyTorch model (MyModel should be defined in cnn_pytorch.py)
num_classes = 35  # Ensure this matches the model saved in trained_model.pth
model = MyModel(num_classes)
# Load the trained weights
# Ensure 'trained_model.pth' is in the same directory or provide the full path
try:
    model.load_state_dict(torch.load('trained_model.pth', map_location=torch.device('cpu')))
except FileNotFoundError:
    tm.showerror("Error", "Trained model file 'trained_model.pth' not found.")
    exit()
except Exception as e:
    tm.showerror("Error", f"Error loading model: {e}")
    exit()
model.eval() # Set the model to evaluation mode

# --- PyTorch based give_char function ---
def give_char():
    try:
        img = PIL.Image.open('tmp1.png').convert('RGB')
    except FileNotFoundError:
        print("Error: tmp1.png not found for prediction.")
        return "Error"
    
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        # Add normalization if your model was trained with it
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    input_tensor = transform(img).unsqueeze(0) # Add batch dimension
    
    with torch.no_grad(): # No need to track gradients for inference
        output = model(input_tensor)
        _, predicted = torch.max(output, 1)
        
    # Ensure this character map matches your model's output classes
    # It should have `num_classes` characters
    chars = "ABCDEFGHIJKMNOPQRSTUVWXYZ0123456789" # 35 chars: A-Z (26) + 0-9 (10) - 1 = 35. Check if this is correct.
                                                # If you have 35 classes, this string should have 35 unique characters.
                                                # Example: "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789" (This is 36 chars, adjust as needed)
                                                # If your classes are 0-34, then this is fine.
    
    indx = predicted.item()
    # print(f"Predicted index: {indx}, Character: {chars[indx] if 0 <= indx < len(chars) else 'Unknown'}")
    if 0 <= indx < len(chars):
        return chars[indx]
    else:
        print(f"Predicted index {indx} is out of bounds for chars string (length {len(chars)})")
        return "?" # Return a placeholder for unknown prediction

# --- Tkinter UI Setup ---
bgcolor="#FFFFF0"
bgcolor1="#F9CEEE"
fgcolor="black"

class ImageLabel(tk.Label):
    def load(self, im):
        if isinstance(im, str):
            try:
                im = PIL.Image.open(im)
            except FileNotFoundError:
                print(f"Error: Image file {im} not found.")
                self.config(text=f"Error: {im} not found")
                return
        frames = []
        try:
            for i in count(1):
                frames.append(ImageTk.PhotoImage(im.copy()))
                im.seek(i)
        except EOFError:
            pass
        self.frames = cycle(frames)
        try:
            self.delay = im.info['duration']
        except:
            self.delay = 100
        if len(frames) == 1:
            self.config(image=next(self.frames))
        else:
            self.next_frame()

    def unload(self):
        self.config(image=None)
        self.frames = None

    def next_frame(self):
        if self.frames:
            try:
                self.config(image=next(self.frames))
                self.after(self.delay, self.next_frame)
            except StopIteration:
                pass # Gif has finished playing (if not looping)

# Corrected paths based on your project structure
# Option 1: Using absolute paths (ensure trailing separator for consistency)
op_dest = r"E:\project\new\Two-Way-Sign-Language-Translator\filtered_data"
alpha_dest =R"E:\project\new\Two-Way-Sign-Language-Translator\alphabet"

# Option 2: Constructing paths relative to the script's location (more portable)
# import os # Make sure this is at the top of your file
# script_dir = os.path.dirname(os.path.abspath(__file__))
# op_dest = os.path.join(script_dir, "filtered_data") + os.sep
# alpha_dest = os.path.join(script_dir, "alphabet") + os.sep

def check_sim(i, file_map):
    for item, words_in_item in file_map.items(): # Use .items() for dictionaries
        for word in words_in_item:
            if i.lower() == word.lower(): # Case-insensitive comparison
                return 1, item
    return -1, ""

def load_file_map():
    file_map = {}
    if not os.path.exists(op_dest):
        print(f"Error: op_dest directory not found: {op_dest}")
        return file_map
    dirListing = os.listdir(op_dest)
    editFiles = [item for item in dirListing if ".webp" in item.lower()]
    for i in editFiles:
        tmp = i.replace(".webp", "")
        tmp = tmp.split() # Splits by whitespace
        file_map[i] = tmp
    return file_map

file_map = load_file_map()

def func(a):
    all_frames = []
    final_gif_image = PIL.Image.new('RGB', (380, 260)) # This will be the first frame
    words = a.split()

    if not os.path.exists(alpha_dest):
        print(f"Error: Alphabet directory not found: {alpha_dest}")
        tm.showerror("Error", f"Alphabet directory not found: {alpha_dest}")
        return []

    for i in words:
        flag, sim_file = check_sim(i, file_map)
        if flag == -1: # Word not found in filtered_data, use alphabet gifs
            for char_in_word in i:
                char_gif_path = os.path.join(alpha_dest, str(char_in_word).lower() + "_small.gif")
                if not os.path.exists(char_gif_path):
                    print(f"Alphabet GIF not found: {char_gif_path}")
                    continue # Skip this character if its GIF is missing
                try:
                    im = PIL.Image.open(char_gif_path)
                    for frame_cnt in range(im.n_frames):
                        im.seek(frame_cnt)
                        # Convert to RGB for consistency, then resize
                        frame_img = im.convert("RGB").resize((380, 260))
                        for _ in range(15): # Repeat frame for slower animation
                            all_frames.append(frame_img)
                except Exception as e:
                    print(f"Error processing alphabet GIF {char_gif_path}: {e}")
        else: # Word found, use the .webp file (converted to GIF)
            webp_path = os.path.join(op_dest, sim_file)
            if not os.path.exists(webp_path):
                print(f"WebP file not found: {webp_path}")
                continue
            try:
                im = PIL.Image.open(webp_path) # PIL can often handle webp if webp plugin is there
                                            # If not, you might need to convert webp to gif/png first externally
                                            # For simplicity, assuming PIL can read it or it's pre-converted.
                # If it's a static webp, it will have 1 frame.
                # If animated, you'd iterate n_frames.
                # For now, let's assume we convert it to a sequence of frames for the GIF.
                # This part might need adjustment based on how webp files are handled.
                # A simple approach for static images:
                frame_img = im.convert("RGB").resize((380, 260))
                all_frames.append(frame_img)
                
                # If it's an animated WebP and PIL supports it directly:
                # for frame_cnt in range(im.n_frames):
                #     im.seek(frame_cnt)
                #     frame_img = im.convert("RGB").resize((380, 260))
                #     all_frames.append(frame_img)

            except Exception as e:
                print(f"Error processing file {webp_path}: {e}")
    
    if all_frames:
        try:
            all_frames[0].save("out.gif", save_all=True, append_images=all_frames[1:], duration=100, loop=0, disposal=2)
        except Exception as e:
            print(f"Error saving GIF: {e}")
            tm.showerror("GIF Error", f"Could not save out.gif: {e}")
            return []
    else:
        print("No frames generated for GIF.")
        tm.showinfo("GIF Info", "No frames were generated to create a GIF.")
    return all_frames

img_counter = 0
img_text = ''

def Home():
    global window, txt, txt1, label, var, lbl_img # Make UI elements accessible if needed

    window = Tk()
    var = IntVar()
    window.title("Sign Language Translation")
    window.geometry('1280x720')
    window.configure(background=bgcolor)
    window.grid_rowconfigure(0, weight=1)
    window.grid_columnconfigure(0, weight=1)

    message1 = Label(window, text="Sign Language Translation", bg=bgcolor, fg=fgcolor, width=70, height=2, font=('times', 25, 'italic bold underline'))
    message1.place(x=50, y=10)

    lbl_text_prompt = Label(window, text="Enter Your Text", width=20, height=2, fg=fgcolor, bg=bgcolor, font=('times', 15, ' bold '))
    lbl_text_prompt.place(x=100, y=100)
    txt = Entry(window, width=30, bg="white", fg="black", font=('times', 15, ' bold '))
    txt.place(x=400, y=115)

    lbl_caption_prompt = Label(window, text="Enter Your Caption", width=20, height=2, fg=fgcolor, bg=bgcolor, font=('times', 15, ' bold '))
    lbl_caption_prompt.place(x=100, y=250)
    txt1 = Entry(window, width=30, bg="white", fg="black", font=('times', 15, ' bold '))
    txt1.place(x=400, y=265)

    def sel():
        selection = str(var.get())
        # label.config(text=selection) # This label might be for displaying the mode
        if selection == "1":
            label.config(text="Mode: Sign to Text")
        elif selection == "2":
            label.config(text="Mode: Text to Sign")
        else:
            label.config(text="Mode: Not Selected")

    R1 = Radiobutton(window, text="Sign2Text", variable=var, value=1, command=sel, bg=bgcolor, font=('times', 12))
    R1.place(x=350, y=160)
    R2 = Radiobutton(window, text="Text2Sign", variable=var, value=2, command=sel, bg=bgcolor, font=('times', 12))
    R2.place(x=450, y=160)

    label = Label(window, text="Mode: Not Selected", width=25, height=2, fg=fgcolor, bg=bgcolor, font=('times', 15, ' bold '))
    label.place(x=350, y=200) # Adjusted y for clarity

    lbl_img = ImageLabel(window) # Changed name from lbl to lbl_img for clarity
    lbl_img.place(x=800, y=350) # Positioned GIF display area

    def clear_text_fields():
        txt.delete(0, 'end')
        txt1.delete(0, 'end')
        if lbl_img:
            lbl_img.unload() # Clear the GIF display
        print("Cleared text fields and GIF display")

    # Function to import main module when needed (avoids circular import)
    def import_signtotext_module():
        import main as signtotext_module
        return signtotext_module

    def sign2text_process():
        mode = var.get()
        if mode == 1: # Sign2Text
            try:
                signtotext_module = import_signtotext_module()
                signtotext_module.process() # Assuming main.py has a process() function
            except AttributeError:
                tm.showerror("Error", "main.py does not have a 'process' function or module not loaded.")
            except Exception as e:
                tm.showerror("Error", f"Failed to run Sign2Text: {e}")
        elif mode == 2: # Text2Sign
            text_to_convert = txt.get()
            if text_to_convert:
                gif_frames_generated = func(text_to_convert)
                if gif_frames_generated:
                    lbl_img.load('out.gif')
                else:
                    tm.showinfo("Text2Sign", "Could not generate GIF. Check console for errors.")
            else:
                tm.showinfo("Input", "Please enter text to convert to sign language GIF.")
        else:
            tm.showinfo("Mode Selection", "Please select a mode (Sign2Text or Text2Sign).")

    def datacreation_process():
        caption_text = txt1.get()
        if caption_text:
            try:
                gd.process(caption_text) # Assuming generate_data.py has a process() function
                tm.showinfo("Data Creation", f"Data creation process initiated for: {caption_text}")
            except AttributeError:
                tm.showerror("Error", "generate_data.py does not have a 'process' function.")
            except Exception as e:
                tm.showerror("Error", f"Data creation failed: {e}")
        else:
            tm.showinfo("Error", "Enter the Caption Letter for data creation.")

    def train_model_process():
        try:
            # Assuming cnn_pytorch.py can be run or has a main training function
            # If cnn_pytorch.py is a script to be run, you might use subprocess
            # For now, assuming it has a callable 'process' or 'train_model' function
            if hasattr(train, 'process'):
                train.process()
                tm.showinfo("Training", "Model training process initiated.")
            elif hasattr(train, 'main'): # Common pattern for scripts
                train.main()
                tm.showinfo("Training", "Model training process initiated.")
            else:
                tm.showerror("Error", "cnn_pytorch.py does not have a 'process' or 'main' function to start training.")
        except Exception as e:
            tm.showerror("Training Error", f"Failed to start training: {e}")

    def record_audio():
        r = sr.Recognizer()
        # m = sr.Microphone() # Not directly used if 'with sr.Microphone() as source:' is used
        text_es = ""
        tm.showinfo("Recording", "Listening... Please speak clearly.")
        try:
            with sr.Microphone() as source:
                r.adjust_for_ambient_noise(source) # Adjust for noise
                print("Listening for audio...")
                audio = r.listen(source, timeout=5, phrase_time_limit=10) # Added timeout
                print("Audio captured, recognizing...")
            text_es = r.recognize_google(audio)
            print("You Said: " + text_es)
            txt.delete(0, 'end') # Clear previous text
            txt.insert('end', text_es)
            tm.showinfo("Speech Recognized", f"You said: {text_es}")
        except sr.WaitTimeoutError:
            print("No speech detected within the time limit.")
            tm.showwarning("Recording", "No speech detected.")
        except sr.UnknownValueError:
            print("Google Speech Recognition could not understand audio.")
            tm.showerror("Recognition Error", "Could not understand audio.")
        except sr.RequestError as e:
            print(f"Could not request results from Google Speech Recognition service; {e}")
            tm.showerror("Service Error", f"Speech service error: {e}")
        except Exception as e:
            print(f"An unexpected error occurred during recording: {e}")
            tm.showerror("Error", f"An unexpected error: {e}")

    browse_btn = Button(window, text="Start Process", command=sign2text_process, fg=fgcolor, bg=bgcolor1, width=15, height=1, activebackground="Red", font=('times', 15, ' bold '))
    browse_btn.place(x=990, y=150) # Adjusted position

    record_btn = Button(window, text="Record Audio", command=record_audio, fg=fgcolor, bg=bgcolor1, width=15, height=1, activebackground="Red", font=('times', 15, ' bold '))
    record_btn.place(x=650, y=110)

    clear_btn = Button(window, text="Clear Fields", command=clear_text_fields, fg=fgcolor, bg=bgcolor1, width=15, height=1, activebackground="Red", font=('times', 15, ' bold '))
    clear_btn.place(x=990, y=200) # Adjusted position

    gen_data_btn = Button(window, text="Data Creation", command=datacreation_process, fg=fgcolor, bg=bgcolor1, width=15, height=1, activebackground="Red", font=('times', 15, ' bold '))
    gen_data_btn.place(x=650, y=265)

    train_btn = Button(window, text="Train Model", command=train_model_process, fg=fgcolor, bg=bgcolor1, width=15, height=1, activebackground="Red", font=('times', 15, ' bold '))
    train_btn.place(x=650, y=350)

    quit_btn = Button(window, text="Quit", command=window.destroy, fg=fgcolor, bg=bgcolor1, width=15, height=1, activebackground="Red", font=('times', 15, ' bold '))
    quit_btn.place(x=990, y=250) # Adjusted position

    window.mainloop()

if __name__ == '__main__':
    Home()
