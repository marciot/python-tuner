#!/usr/bin/python3

#####################################################################
# All User Interface, StringTuner, Graphing and Serial code
#####################################################################
# Author:  Marcio Teixeira
# Date:    August 2018
# License: Creative Commons Attribution-ShareAlike 3.0
#          https://creativecommons.org/licenses/by-sa/3.0/us/
######################################################################


#####################################################################
# FFT processing and mathematical routines in SoundProcessor
#####################################################################
# Author:  Matt Zucker
# Date:    July 2016
# License: Creative Commons Attribution-ShareAlike 3.0
#          https://creativecommons.org/licenses/by-sa/3.0/us/
######################################################################

import numpy as np
import pyaudio
import peakutils
from peakutils.plot import plot as pplot

import glob
import serial
import tkinter as tk
import time
import matplotlib
import matplotlib.pyplot as pyplot
#import winsound

from tkinter import *
from sys     import platform
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2TkAgg
from matplotlib.figure import Figure

matplotlib.use('TkAgg')

#string_axis = ["X", "Y", "Z", "E0", "E1"]
string_axis = ["X", "Y", "Z"]

######################################################################
# Feel free to play with these numbers. Might want to change NOTE_MIN
# and NOTE_MAX especially for guitar/bass. Probably want to keep
# FRAME_SIZE and FRAMES_PER_FFT to be powers of two.

NOTE_MIN = 57       # A3
NOTE_MAX = 69       # A4
FSAMP = 22050       # Sampling frequency in Hz
FRAME_SIZE = 2048   # How many samples per frame?
FRAMES_PER_FFT = 16 # FFT takes average across how many frames?

######################################################################
# Derived quantities from constants above. Note that as
# SAMPLES_PER_FFT goes up, the frequency step size decreases (so
# resolution increases); however, it will incur more delay to process
# new sounds.

SAMPLES_PER_FFT = FRAME_SIZE*FRAMES_PER_FFT
FREQ_STEP = float(FSAMP)/SAMPLES_PER_FFT

######################################################################
# For printing out notes

NOTE_NAMES = 'C C# D D# E F F# G G# A A# B'.split()

######################################################################
# These three functions are based upon this very useful webpage:
# https://newt.phys.unsw.edu.au/jw/notes.html

def freq_to_number(f): return 69 + 12*np.log2(f/440.0)
def number_to_freq(n): return 440 * 2.0**((n-69)/12.0)
def note_name(n): return NOTE_NAMES[n % 12] + str(int(n/12) - 1)

######################################################################
# Ok, ready to go now.

# Get min/max index within FFT of notes we care about.
# See docs for numpy.rfftfreq()
def note_to_fftbin(n): return number_to_freq(n)/FREQ_STEP

class SoundProcessor:
    def __init__(self):
        self.freqs     = []
        self.vals      = []
        self.peaks     = []
        self.maximum   = 0
        self.threshold = 0.20
        
        self.peak_increasing = True
        self.peak_previous   = 0
        
        self.imin = max(0, int(np.floor(note_to_fftbin(NOTE_MIN-1))))
        self.imax = min(SAMPLES_PER_FFT, int(np.ceil(note_to_fftbin(NOTE_MAX+1))))

        # Allocate space to run an FFT. 
        self.buf = np.zeros(SAMPLES_PER_FFT, dtype=np.float32)
        self.num_frames = 0

        # Initialize audio
        self.stream = pyaudio.PyAudio().open(format=pyaudio.paInt16,
                                channels=1,
                                rate=FSAMP,
                                input=True,
                                frames_per_buffer=FRAME_SIZE)

        self.stream.start_stream()

        # Create Hanning window function
        self.window = 0.5 * (1 - np.cos(np.linspace(0, 2*np.pi, SAMPLES_PER_FFT, False)))

        # Print initial text
        print('sampling at', FSAMP, 'Hz with max resolution of', FREQ_STEP, 'Hz')
        print()
        
    def detect_maximum(self, peak_max):
        is_maximum = False
        if peak_max > self.peak_previous:
            self.peak_increasing = True
            self.peak_max        = peak_max
        else:
            if self.peak_increasing and peak_max < 0.95 * self.peak_max:
                self.peak_increasing = False
                is_maximum = True 
        self.peak_previous = peak_max
        return is_maximum
    
    def reset_peak_detector(self):
        self.peak_increasing = False
        self.peak_max        = 0
        self.is_maximum      = False   
        
    def set_threshold(self, y):
        self.threshold = y / self.maximum

    def process_audio_data(self):
        # As long as we are getting data:
        while self.stream.is_active():
        
            # Shift the buffer down and new data in
            self.buf[:-FRAME_SIZE] = self.buf[FRAME_SIZE:]
            self.buf[-FRAME_SIZE:] = np.frombuffer(self.stream.read(FRAME_SIZE), np.int16)
        
            # Run the FFT on the windowed buffer
            fft = np.fft.rfft(self.buf * self.window)
                
            # Begin extracting peaks once we have a full buffer
            if self.num_frames >= FRAMES_PER_FFT:
                #print(freqs,ampls)
                #print('freq: {:7.2f} Hz     note: {:>3s} {:+.2f}'.format(
                    #freq, note_name(n0), n-n0))
                self.vals = np.abs(fft[self.imin:self.imax])
                self.freqs = (np.arange(len(self.vals)) + self.imin) * FREQ_STEP
                self.peaks = peakutils.indexes(self.vals, thres=0.75, min_dist=30)
                if len(self.peaks):
                    peak_max = np.amax(self.vals[self.peaks])
                    self.maximum = max(self.maximum, peak_max)
                    self.peaks = np.extract(self.vals[self.peaks] > self.maximum * self.threshold, self.peaks)
                    self.tones = self.freqs[self.peaks]
                    self.is_maximum = self.detect_maximum(peak_max) and len(self.peaks)
                else:
                    self.reset_peak_detector()
                break
            else:
                self.num_frames += 1

class StringTuner:
    _learning       = None
    _strings        = []
    _note_num       = np.arange(NOTE_MAX - NOTE_MIN + 1) + NOTE_MIN
    _note_frq       = number_to_freq(_note_num)
    _note_str       = [note_name(note) for note in _note_num]
    
    # How much to overshoot and correct to reach a note
    _note_overshoot = 0.25
    _note_wiggle    = 2
    
    def __init__(self, parent, axis):
        self._parent   = parent
        self._axis     = axis
        self._position = 0
        self._fit_m    = 0
        self._fit_b    = 0
        
        self._last_obs = None
        self._note = None
        self._note_err = np.full(NOTE_MAX - NOTE_MIN + 1, 100)
        self._note_pos = np.full(NOTE_MAX - NOTE_MIN + 1, 0)
        
        StringTuner._strings.append(self)
                
    def send_position(self, position):
        self._position = position
        self._parent.send_position_and_wait(self._axis, self._position)
        
    def learn(self):
        if not StringTuner._learning:
            StringTuner._learning   = self
            StringTuner._learn_frq = []
            StringTuner._learn_pos = []
        else:
            StringTuner._learning   = None
            self.compute_slope()
        
    def compute_slope(self):
        if len(StringTuner._learn_frq) > 1:
            x = StringTuner._learn_frq
            y = StringTuner._learn_pos
            m,b = np.polyfit(x, y, 1)
            self._fit_m = m
            self._fit_b = b
            self._note_pos = b + self._note_frq * m
                
    def reset(self):
        self._parent.send("G92 %s%f" % (self._axis, 0))
        self._position  = 0
        
    def motor_off(self):
        self._parent.send("M18 %s" % (self._axis))
        
    def goto_note(self, n):        
        n0 = int(round(n))
        print("Goto note %s" % note_name(n0))
        new_pos = self._note_pos[n0 - NOTE_MIN]
        self.position(new_pos)
        self._note = n0

    def tension(self, amount):
        self.position(self._position + amount)
    
    def position(self, pos):
        if self._fit_m:
            for i in range(self._note_wiggle):
                self.send_position(pos + self._fit_m * self._note_overshoot)
                self.send_position(pos - self._fit_m * self._note_overshoot)
        self.send_position(pos)
        
    def settle_note(self, n0):
        new_pos = self._note_pos[n0 - NOTE_MIN]
        for i in range(self._note_wiggle):
            self.position(new_pos + self._fit_m * self._note_overshoot)
            self.position(new_pos - self._fit_m * self._note_overshoot)
            
    def settle_note1(self, n0):
        new_pos = self._note_pos[n0 - NOTE_MIN]
        if self._note < n0:
            self.position(new_pos + self._fit_m * self._note_overshoot)
        elif self._note > n0:
            self.position(new_pos - self._fit_m * self._note_overshoot)

    def observed_frequency(self, freq):
        # Find nearest note
        n = freq_to_number(freq)
        n0 = int(round(n))
        d  = n-n0
        print('freq: {:7.2f} Hz     note: {:>3s} {:+.2f}'.format(
                freq, note_name(n0), d))
        
        self._last_obs = n

        # Learn from this observation
            
        if not StringTuner._learning:
            self.tune_to_note(freq)
        else:
            self._learn_pos.append(self._position)
            self._learn_frq.append(freq)
            
    def set_note_pos_and_error(self, n0, pos, err):
        i = n0 - NOTE_MIN
        if i > 0 and i < len(self._note_pos):
            self._note_pos[i] = pos
            self._note_err[i] = err
    
    def tune_to_note(self, freq):
        err = number_to_freq(self._note) - freq
        self.tension(err * self._fit_m)
        self.set_note_pos_and_error(self._note, self._position, err)
    
    @classmethod
    def assign_observations(cls, freqs):
        strg_to_freq_assignment = {}
        unassigned_freq        = freqs.tolist()
        unassigned_strg        = StringTuner._strings.copy()
        
        # Map each frequency to the nearest string that has no target
        
        for freq in unassigned_freq:
            # Find nearest note
            n = freq_to_number(freq)
            
            # Assign this tone to the nearest string
            strg   = None
            dist   = 9999
            for s in unassigned_strg:
                if s._note != None and abs(n - s._note) < dist:
                    strg = s

            if strg:
                strg_to_freq_assignment[strg] = freq
                unassigned_freq.remove(freq)
                unassigned_strg.remove(strg)
                
        # Report the observations to the strings
        for stng, freq in strg_to_freq_assignment.items():
            stng.observed_frequency(freq)
        
    @classmethod
    def print_string_list(cls):
        # Print out the updated string list
        for i, stng in enumerate(StringTuner._strings):
            if stng._last_obs:
                print("%d -> %s" % (i,note_name(stng._last_obs)))
        
    @classmethod
    def update(cls, sp):
        freqs = sp.tones 
        if sp.is_maximum and len(freqs):
            if StringTuner._learning:
                StringTuner._learning.observed_frequency(freqs[0])
                StringTuner._learning.tension(-5)
            else:
                cls.assign_observations(freqs)
 
class GraphPage(tk.Frame):
    def __init__(self, parent):
        tk.Frame.__init__(self, parent)
        self.pack()

    def add_mpl_figure(self, fig):
        self.fig = fig
        self.mpl_canvas = FigureCanvasTkAgg(fig, self)
        self.mpl_canvas.draw()
        self.mpl_canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.toolbar = NavigationToolbar2TkAgg(self.mpl_canvas, self)
        self.toolbar.update()
        self.mpl_canvas._tkcanvas.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)
        
        self.mpl_canvas.mpl_connect('button_press_event', self.onclick)
        
    def refresh(self, arg):
        self.fig.refresh(arg)
        self.mpl_canvas.draw()
        
    def onclick(self, event):
        #print('%s click: button=%d, x=%d, y=%d, xdata=%f, ydata=%f' %
        #  ('double' if event.dblclick else 'single', event.button,
        #   event.x, event.y, event.xdata, event.ydata))
        self.fig.onclick(event)

class FFTGraph(Figure):

    def __init__(self):
        Figure.__init__(self, figsize=(5, 5), dpi=100)
        self._subplot = self.add_subplot(111)
                
    def adjustRange(self, y_max):
        if not hasattr(self,"_range"):
            self._range = list(self._subplot.axis())
        if y_max != self._range[3]:
            self._range[3] = y_max
            self._subplot.axis(self._range)            
        
    def refresh(self, fft):
        self._fft = fft
        t = fft.maximum * fft.threshold
        if not hasattr(self,"_fft_plot") or not hasattr(self,"_fft_dots"):
            self._subplot.clear()
            self._fft_plot, = self._subplot.plot(fft.freqs, fft.vals)
            self._fft_dots, = self._subplot.plot(fft.freqs[fft.peaks], fft.vals[fft.peaks], 'ro')
            self._range = list(self._subplot.axis())
            self._fft_thrs = self._subplot.axhline(y=t,color='r',linestyle='--')
            self._subplot.set_xlabel('Frequency (Hz)')
            self._subplot.set_ylabel('Amplitude')
            self._subplot.set_title('Audio Spectrum')
            for n in range(NOTE_MAX - NOTE_MIN + 1):
                freq = number_to_freq(n + NOTE_MIN)
                self._subplot.axvline(x=freq,color='g',linestyle=':')
                
        else:
            self._fft_plot.set_ydata(fft.vals)
            self._fft_thrs.set_ydata([t,t])
            self._fft_dots.set_xdata(fft.freqs[fft.peaks])
            self._fft_dots.set_ydata(fft.vals[fft.peaks])
            self.adjustRange(fft.maximum)
            
    def onclick(self, event):
        if event.xdata:
            StringTuner._strings[0].goto_note(freq_to_number(event.xdata))
        if event.ydata:
            self._fft.set_threshold(event.ydata)
            
class PitchGraph(Figure):

    def __init__(self):
        Figure.__init__(self, figsize=(5, 5), dpi=100)
        self._subplot = self.add_subplot(111)
        self._highlight = []

    def refresh(self, sp):
        if not hasattr(self,"_plots"):
            self._subplot.clear()
            self._subplot.set_xlabel('Pitch')
            self._subplot.set_xticks(StringTuner._note_frq);
            self._subplot.set_xticklabels(StringTuner._note_str, rotation=90)
            self._subplot.set_ylabel('Position')
            self._subplot.set_title('Pitch vs. Position')
            self._plots = []
            self._lines = []
            self._slope = []
            for i,s in enumerate(StringTuner._strings):
                x = s._note_frq
                y = s._note_pos
                p, = self._subplot.plot(x, y,'.')
                self._plots.append(p)
                p = self._subplot.axhline(y=s._position,color='g',linestyle=':')
                self._lines.append(p)
                p, = self._subplot.plot([x[0], x[-1]], [0,0],':')
                self._slope.append(p)
            #self._subplot.autoscale(False)
        else:
            if len(sp.peaks) == 0:
                self.remove_highlights()
            for i,s in enumerate(StringTuner._strings):
                self._plots[i].set_ydata(s._note_pos)
                self._lines[i].set_ydata(s._position)
                if hasattr(s, '_fit_m'):
                    m = s._fit_m
                    b = s._fit_b
                    self._slope[i].set_ydata([s._note_frq[0]*m+b,s._note_frq[-1]*m+b])
            # Highlight the pitches in the data
            if sp.is_maximum:
                for i,s in enumerate(StringTuner._strings):
                    if s._last_obs and s._note:
                        self.highlight_pitch(s._last_obs,s._note)

    def remove_highlights(self):
        for h in self._highlight:
            h.remove()
        self._highlight = []
    
    def highlight_pitch(self, n, n0):
        f  = number_to_freq(n)
        f0 = number_to_freq(n0)
        if abs(n-n0) < 0.1:
            c = 'g'
        elif f0 < f:
            c = 'b'
        else:
            c = 'r'
        p = self._subplot.axvspan(min(f,f0), max(f,f0),facecolor=c,alpha=0.5)
        self._highlight.append(p)
    
    def onclick(self, event):
        if event.xdata:
            StringTuner._strings[0].goto_note(freq_to_number(event.xdata))

class PortSelector:
    def __init__(self, parent):

        self.parent   = parent
        self.portname = None

        top = self.top = Toplevel(parent)

        Label(top, text="Select a serial port:").pack()

        l = self.l = Listbox(top)
        for p in self.serial_ports():
          l.insert(END, p)
        l.pack(padx=5)

        b = Button(top, text="OK", command=self.ok)
        b.pack(pady=5)

    def serial_ports(self):
        """ Lists serial port names

            :raises EnvironmentError:
                On unsupported or unknown platforms
            :returns:
                A list of the serial ports available on the system
        """
        # Reference: https://stackoverflow.com/questions/12090503/listing-available-com-ports-with-python

        if sys.platform.startswith('win'):
            ports = ['COM%s' % (i + 1) for i in range(256)]
        elif sys.platform.startswith('linux') or sys.platform.startswith('cygwin'):
            # this excludes your current terminal "/dev/tty"
            ports = glob.glob('/dev/tty[A-Za-z]*')
        elif sys.platform.startswith('darwin'):
            ports = glob.glob('/dev/tty.*')
        else:
            raise EnvironmentError('Unsupported platform')

        result = []
        for port in ports:
            try:
                s = serial.Serial(port)
                s.close()
                result.append(port)
            except (OSError, serial.SerialException):
                pass
        return result

    def ok(self):
        items = list(map(int, self.l.curselection()))
        if items:
            self.portname = self.l.get(items[0])
        self.top.destroy()

class AxisControl(Frame):
    def __init__(self, parent, character, tuner):
        self._character = character.strip()
        self._parent    = parent
        self._tuner     = tuner
        Frame.__init__(self, parent)
        self.pack()
        self.createWidgets()

    def increment(self):
        self._tuner.tension(5)
        self.updateLabel()

    def decrement(self):
        self._tuner.tension(-5)
        self.updateLabel()

    def set_origin(self):
        self._tuner.reset()        
        self.updateLabel()
        
    def motor_off(self):
        self._tuner.motor_off()
        
    def learn(self):
        self._tuner.learn()
        
    def updateLabel(self):
        self.l.configure(text="%3.1f" % self._tuner._position)

    def createWidgets(self):
        Label(self, text=self._character + ":").pack({"side": "left"})
        Button(self, text="+",         command=self.increment ).pack({"side": "left"})
        Button(self, text="-",         command=self.decrement ).pack({"side": "left"})
        Button(self, text="Reset",     command=self.set_origin).pack({"side": "left"})
        Button(self, text="Off",       command=self.motor_off).pack({"side": "left"})
        Button(self, text="Learn",     command=self.learn    ).pack({"side": "left"})
        self.l = Label(self)
        self.l.pack({"side": "left"})
        self.updateLabel()
        
class PianoRoll(Frame):

    # Sample song, expressed in semitones

    # Edvard Grieg - In The Hall Of The Mountain King
    # https://www.youtube.com/watch?v=K0e3IABZt2s
    song = [-1,1,2,4,6,2,6,5,1,5,4,0,4,-1,1,2,4,6,2,6,11 ,9,6,2,6, 9,
            -1,1,2,4,6,2,6,5,1,5,4,0,4,-1,1,2,4,6,2,6,11 ,9,6,2,6, 9]
    
    left_margin = 70
    note_width  = 20
    
    _playing    = None
    
    def __init__(self, parent):        
        labels = StringTuner._note_str
        line_height = 20
        note_width  = 20
        canvas_width  = len(self.song) * self.note_width + self.left_margin
        canvas_height = line_height*len(labels)
        font=("Helvetica", -line_height+4)
        
        Frame.__init__(self, parent, width=100, bd=2, relief=SUNKEN)
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)

        xscrollbar = Scrollbar(self, orient=HORIZONTAL)
        xscrollbar.grid(row=1, column=0, sticky=E+W)

        yscrollbar = Scrollbar(self)
        yscrollbar.grid(row=0, column=1, sticky=N+S)
        
        c = Canvas(self, bd=0, 
                       scrollregion=(0, 0, canvas_width, canvas_height),
                        xscrollcommand=xscrollbar.set,
                        yscrollcommand=yscrollbar.set,
                        width=300, height=canvas_height)

        c.grid(row=0, column=0, sticky=N+S+E+W)

        xscrollbar.config(command=c.xview)
        yscrollbar.config(command=c.yview)

        for i in range(0,len(labels)):
          c.create_line(0,   line_height*(i+1  ), canvas_width, line_height*(i+1  ))
          c.create_text(self.left_margin  - 15, line_height*(i+0.5), text=labels[i], anchor=E, font=font)

        min_note = min(self.song)          
        for i,n in enumerate(self.song):
            x0 = self.left_margin + i     * self.note_width
            x1 = self.left_margin + (i+1) * self.note_width
            y0 = line_height * (n-min_note)
            y1 = line_height * (n-min_note+1)
            c.create_rectangle(x0, y0, x1, y1, fill="gray")

        self.t = 1
        self.t_line = c.create_line(self.left_margin, 0, self.left_margin, canvas_height, fill="red", dash=(4, 4))
        
        self.canvas = c
        
    def update_time(self):
        x = self.left_margin + self.t
        self.canvas.coords(self.t_line, x, 0, x, self.canvas.cget('height'))
        
    def update(self, sp):
        if sp.is_maximum:
            self.next_note()

    def play(self):
        if not self._playing:
            self._playing = []
            self.t = 0
            m = min(self.song)
            for n in self.song:
                self._playing.append(int(n-m+NOTE_MIN))
                print("Begin playing song")
        else:
            self._playing = False

    def next_note(self):
        if self._playing:
            StringTuner._strings[0].goto_note(self._playing.pop(0))
            self.t += self.note_width
            self.update_time()


class Application(Frame):
    def connect(self):
        if not self._serial:
            d = PortSelector(root)
            root.wait_window(d.top)
            if d.portname:
                self._serial = serial.Serial(d.portname, baudrate=250000)
                self.connect_btn["text"]    = "Disconnect"
                self.connect_btn["command"] = self.disconnect
                self.after(5000, self.initMarlin)

    def disconnect(self, exiting = False):
        if self._serial:
            self._serial.close()
            self._serial = None
        if not exiting:
            self.connect_btn["text"]    = "Connect"
            self.connect_btn["command"] = self.connect
            
    def initMarlin(self):
        self.send("M211 S0") # Turn off endstops
        self.send("M203 X90000 Y90000 Z90000") # Max feedrate

    def send(self, cmd):
        if self._serial:
            self._serial.write((cmd+'\n').encode())
            print(cmd)
            
    def send_position_and_wait(self, axis, position):
        if self._serial:
            self.send('G0 %s%f F90000\nM400\nM114\n' % (axis, position))
            while(1):
                if self._serial.inWaiting():
                    line = self._serial.readline()
                    if line.startswith(b"X:"):
                        break
                
    def receive(self):
        if self._serial:
            if self._serial.inWaiting():
                line = self._serial.readline()
                print("Serial: ",line)
            
    def showSpectrum(self):
        t = tk.Toplevel(self)
        self.fig = FFTGraph()
        self.spec_page = GraphPage(t)
        self.spec_page.add_mpl_figure(self.fig)
        
    def showPitches(self):
        t = tk.Toplevel(self)
        self.fig = PitchGraph()
        self.map_page = GraphPage(t)
        self.map_page.add_mpl_figure(self.fig)
        
    def playSong(self):
        self._pianoroll.play()

    def createWidgets(self):
        # Buttons
        f = Frame(self)
        f.pack({"side": "top", "pady" : 20})
        self.quit_btn    = Button(f, text = "QUIT",     command = self.quit,   fg = "red")
        self.connect_btn = Button(f, text = "Connect",  command = self.connect           )
        self.spec_btn    = Button(f, text = "Spectrum", command = self.showSpectrum      )
        self.pitch_btn   = Button(f, text = "Pitches",  command = self.showPitches       )
        self.play_btn    = Button(f, text = "Song",     command = self.playSong          )
        self.quit_btn.pack(   {"side": "left"})
        self.connect_btn.pack({"side": "left"})
        self.spec_btn.pack({"side": "left"})
        self.pitch_btn.pack({"side": "left"})
        self.play_btn.pack({"side": "left"})

        # Scrolling Canvas
        self._pianoroll = PianoRoll(self)
        self._pianoroll.pack({"side": "top", "pady" : 20, "padx" : 10})
        
        # Axis Control Buttons
        f = Frame(self)
        for i in range(len(string_axis)):
            AxisControl(f, string_axis[i], self._strings[i]).pack({"side": "top"})
        f.pack({"side": "top", "pady" : 20})
        
    def __init__(self, master=None):
        self._serial = None
        self._sp = SoundProcessor()
        
        self._strings = []
        for a in string_axis:
            self._strings.append(StringTuner(self, a)) 
        
        Frame.__init__(self, master)
        self.pack()
        self.createWidgets()
        self.idle()

    def idle(self):
      self.after(100, self.idle)
      
      self._sp.process_audio_data()
      
      StringTuner.update(self._sp)
      
      if hasattr(self,"spec_page"):
          self.spec_page.refresh(self._sp)
          
      if hasattr(self,"map_page"):
          self.map_page.refresh(self._sp)
          
      self._pianoroll.update(self._sp)
      
root = Tk()
app = Application(master=root)
app.mainloop()
app.disconnect(exiting = True)
#root.destroy()