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

NOTE_RNG = 13
NOTE_MIN = 55
NOTE_MAX = NOTE_MIN + NOTE_RNG - 1
FSAMP = 22050       # Sampling frequency in Hz
FRAME_SIZE   = 2048 # How many samples per frame?
FRAMES_PER_FFT =  8 # FFT takes average across how many frames?

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
        
        self.is_maximum  = False
        self.is_falling  = False
        self.is_finished = False
        self.is_starting = False
        self.is_silent   = True
        
        self.peak_increasing = True
        self.peak_previous   = 0
        
        self.imin = max(0, int(np.floor(note_to_fftbin(NOTE_MIN-1))))
        self.imax = min(SAMPLES_PER_FFT, int(np.ceil(note_to_fftbin(NOTE_MAX+1))))

        # Allocate space to run an FFT. 
        self.buf = np.zeros(SAMPLES_PER_FFT, dtype=np.float32)
        self.num_frames = 0
        
        self.freqs = (np.arange(self.imax - self.imin) + self.imin) * FREQ_STEP

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
        self.is_maximum = False
        if self.is_silent:
            self.is_silent = False
            self.is_starting = True
        else:
            self.is_starting = False
        if peak_max > self.peak_previous:
            if not self.peak_increasing:
                self.peak_increasing = True
            self.peak_max        = peak_max
            self.is_falling      = False
        else:
            if self.peak_increasing and peak_max < 0.95 * self.peak_max:
                self.peak_increasing = False
                self.is_maximum = True
                self.is_falling = True
        self.peak_previous = peak_max
        return self.is_maximum
    
    def reset_peak_detector(self):
        if self.is_falling:
            self.is_finished = True
        else:
            self.is_finished = False
        self.peak_increasing = False
        self.peak_max        = 0
        self.is_maximum      = False
        self.is_falling      = False
        self.is_silent       = True
        
    def set_threshold(self, y):
        self.threshold = y / self.maximum
        
    def clear_buffer(self):
        self.num_frames = 0

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
                
                thres = min(1, self.maximum/max(self.vals) * self.threshold)
                self.peaks = peakutils.indexes(self.vals, thres = thres, min_dist=30)
                self.tones = self.freqs[self.peaks]
                if len(self.peaks):
                    peak_max = np.amax(self.vals[self.peaks])
                    self.maximum = max(self.maximum, peak_max)
                    self.detect_maximum(peak_max)
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
    _note_overshoot = 0.20
    _note_wiggle    = 3
    
    def __init__(self, parent, axis):
        self._parent   = parent
        self._axis     = axis
        self._position = 0
        self._fit_m    = 0
        self._fit_b    = 0
        
        self._last_obs = None
        self._last_tune_time = 0
        self._tune_increment = 0.1
        self._tune_step = 1
        self._tune_time = 0
        
        self._motion_t = []
        self._motion_f = []
        
        self._learn_frq = []
        self._learn_pos = []
        
        self._note = None
        
        self._active = False
        
        StringTuner._strings.append(self)
                
    def send_position(self, position, when_done):
        self._position = position
        self._parent.send_position(self._axis, self._position, when_done)
        
    def learn(self):
        if not StringTuner._learning:
            StringTuner._learning   = self
            self._learn_frq = []
            self._learn_pos = []
            self._active = False
        else:
            StringTuner._learning   = None
            self._note = int(round(self._last_obs))
            self._active = True
        
    def compute_slope(self):
        if len(self._learn_frq) > 4:
            x = self._learn_frq
            y = self._learn_pos
            m,b = np.polyfit(x, y, 1)
            self._fit_m = m
            self._fit_b = b
            print("Compute slope!", m, b)
                
    def reset(self):
        self._parent.send("G92 %s%f" % (self._axis, 0))
        self._position  = 0
        
    def motor_off(self):
        self._parent.send("M18 %s" % (self._axis))
        
    def goto_note(self, n):
        if self._active:
            n0 = int(round(n))
            print("Goto note %s" % note_name(n0))
            self.settle_note(n0)
            self._note = n0
            self._last_obs = None
            self._tune_step = 0.75
            self._tune_time = 0

    def tension(self, amount, when_done = None):
        self.position(self._position + amount, when_done)
    
    def position(self, pos, when_done=None):
        self.send_position(pos, when_done)

    def interpolate_note_pos(self, n):
        return self._fit_b + self._fit_m * number_to_freq(n)
    
    def adjust_intercept(self, freq):
        self._fit_b = self._position - self._fit_m * freq
    
    def settle_note(self, n0):
        for i in range(self._note_wiggle, 0, -1):
            self.position(self.interpolate_note_pos(n0 + self._note_overshoot*i))
            self.position(self.interpolate_note_pos(n0 - self._note_overshoot*i))
        self.position(self.interpolate_note_pos(n0))
    
    def observed_frequency(self, freq, sp):
        # Find nearest note
        n = freq_to_number(freq)
        n0 = int(round(n))
        d  = n-n0
        print('freq: {:7.2f} Hz     note: {:>3s} {:+.2f}'.format(
                freq, note_name(n0), d))
        
        self._last_obs = n
        self.record_history(freq)
        
        if sp.is_starting:
            self.clear_history()

        # Learn from this observation
            
        if PianoRoll._playing and self._active:
            self.tune_to_note(freq)   
        
        if sp.is_maximum and not app._marlin._in_motion and StringTuner._learning == self:
            self._learn_pos.append(self._position)
            self._learn_frq.append(freq)
            self.compute_slope()
            
    def clear_history(self):
        self._motion_t = []
        self._motion_f = []
            
    def record_history(self, freq):
        self._motion_t.append(time.time())
        self._motion_f.append(freq)        

    def tune_to_note(self, freq):
        if self._note:
            self.adjust_intercept(freq)
            if not app._marlin._in_motion:
                err = number_to_freq(self._note) - freq
                self.tension(err * self._fit_m * self._tune_step, self.motion_done)
            return
        
            if time.time() - self._last_tune_time > self._tune_time:
                # Figure out when the next tune will take place
                self._last_tune_time = time.time()
                self._tune_time = (FRAME_SIZE * FRAMES_PER_FFT)/FSAMP * self._tune_step * 0.25
                self._tune_step = max(self._tune_step*0.75, 0.1)
            
    def motion_done(self):
        print("Motion complete")
        
    @classmethod
    def play_chord(cls, notes):
        print("Chord:",notes)
        strings = StringTuner.assign_strings_to_notes(notes)
        for stng, note in zip(strings, notes):
            if stng:
                stng.goto_note(note)
            
    @classmethod
    def assign_observations(cls, freqs, sp):
        freqs   = freqs.tolist()
        notes   = [freq_to_number(freq) for freq in freqs]
        strings = StringTuner.assign_strings_to_notes(notes)
        for stng, freq in zip(strings, freqs):
            if stng:
                stng.observed_frequency(freq, sp)
    
    @classmethod
    def assign_strings_to_notes(cls, notes):
        strgs = [s if s._note else None for s in StringTuner._strings]
        
        while len(strgs) < len(notes):
            strgs.append(None)
            
        assignment = None
        distance   = 999999
        for perm in itertools.permutations(strgs, len(notes)):
            d = sum([abs(n - s._note) if s and s._note else 99999 for s,n in zip(perm, notes)])
            if d < distance:
                assignment = perm
                distance   = d
                
        return assignment
        
    @classmethod
    def print_string_list(cls):
        # Print out the updated string list
        for i, stng in enumerate(StringTuner._strings):
            if stng._last_obs:
                print("%d -> %s" % (i,note_name(stng._last_obs)))
        
    @classmethod
    def update(cls, sp):
        freqs = sp.tones
        
        if StringTuner._learning and sp.is_maximum:
            StringTuner._learning.observed_frequency(freqs[0], sp)
            StringTuner._learning.tension(-5)
            
        cls.assign_observations(freqs, sp)
 
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
            self._subplot.axvline(x=FSAMP/2,color='b',linestyle='-')
            self._notes = []
            for i,s in enumerate(StringTuner._strings):
                p = self._subplot.axvline(x=number_to_freq(0),color='g',linestyle='-')
                self._notes.append(p)
        else:
            self._fft_plot.set_ydata(fft.vals)
            self._fft_thrs.set_ydata([t,t])
            self._fft_dots.set_xdata(fft.freqs[fft.peaks])
            self._fft_dots.set_ydata(fft.vals[fft.peaks])
            self.adjustRange(fft.maximum)
            for i,s in enumerate(StringTuner._strings):
                if s._note:
                    self._notes[i].set_xdata(number_to_freq(s._note))
            
    def onclick(self, event):
        if event.ydata:
            self._fft.set_threshold(event.ydata)
            
class PitchGraph(Figure):
    def __init__(self):
        Figure.__init__(self, figsize=(5, 5), dpi=100)
        self._subplot = self.add_subplot(111)
        self._highlight = []

    def refresh(self, sp):
        if not hasattr(self,"_slope"):
            self._subplot.clear()
            self._subplot.set_xlabel('Pitch')
            self._subplot.set_xticks(StringTuner._note_frq);
            self._subplot.set_xticklabels(StringTuner._note_str, rotation=90)
            self._subplot.set_ylabel('Position')
            self._subplot.set_title('Pitch vs. Position')
            self._lines = []
            self._notes = []
            self._slope = []
            for i,s in enumerate(StringTuner._strings):
                if len(s._learn_frq):
                    p, = self._subplot.plot(s._learn_frq, s._learn_pos,'+')
                                
                p = self._subplot.axhline(y=s._position,color='g',linestyle=':')
                self._lines.append(p)
                
                p = self._subplot.axvline(x=s._note_frq[0],color='g',linestyle=':')
                self._notes.append(p)
                
                p, = self._subplot.plot([s._note_frq[0], s._note_frq[-1]], [0,0],':',color='r')
                self._slope.append(p)
            #self._subplot.autoscale(False)
        else:
            if len(sp.peaks) == 0:
                self.remove_highlights()
            for i,s in enumerate(StringTuner._strings):
                self._lines[i].set_ydata(s._position)
                if s._note:
                    self._notes[i].set_xdata(number_to_freq(s._note))
                if hasattr(s, '_fit_m'):
                    m = s._fit_m
                    b = s._fit_b
                    y0 = s._note_frq[ 0]*m+b
                    y1 = s._note_frq[-1]*m+b
                    self._slope[i].set_ydata([y0, y1])
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
            
class MotionGraph(Figure):
    def __init__(self):
        Figure.__init__(self, figsize=(5, 5), dpi=100)
        self._subplot = self.add_subplot(111)
        
    def refresh(self, sp):
        if sp.is_finished:
            self._subplot.clear()
            self._subplot.set_xlabel('Time')
            for i,s in enumerate(StringTuner._strings):
                t = np.array(s._motion_t)
                if len(t):
                    n = freq_to_number(np.array(s._motion_f)) - s._note
                    p, = self._subplot.plot(t - t[0], n, linestyle='-',marker='.')
            p  = self._subplot.axhline(y=0,color='b',linestyle=':')
            p  = self._subplot.axhline(y=0.5,color='r',linestyle=':')
            p  = self._subplot.axhline(y=-0.5,color='r',linestyle=':')
            self._subplot.set_ylim(-1,1)

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
    
    # Row Row Row Your Boat
    # https://www.youtube.com/watch?v=ROqgdTRa0bE
    song = [0,0,0,2,4,4,2,4,5,7,12,12,12,7,7,7,4,4,4,0,0,0,7,5,4,2,0]
    
    song = [0,2,4,5,7,9]

    line_height = 20
    note_width  = 50
    left_margin = 70
    
    _playing    = None
    _notes      = []
    _label      = []
    
    def __init__(self, parent):
        labels = StringTuner._note_str
        canvas_width  = len(self.song) * self.note_width + self.left_margin
        canvas_height = self.line_height*(len(labels)+1)
        font=("Helvetica", -self.line_height+4)
        
        Frame.__init__(self, parent, bd=2, relief=SUNKEN)
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
        self.canvas = c

        c.grid(row=0, column=0, sticky=N+S+E+W)

        xscrollbar.config(command=c.xview)
        yscrollbar.config(command=c.yview)

        for i in range(0,len(labels)):
          c.create_line(0, self.line_height*(i+1  ), 90000, self.line_height*(i+1  ))
          c.create_text(self.left_margin  - 15, self.line_height*(i+1.5), text=labels[i], anchor=E, font=font)

        #min_note = min(self.song)          
        #for i,n in enumerate(self.song):
        #    self.add_note_to_canvas(i, n-min_note+NOTE_MIN):

        self.t = 0
        self.t_line = c.create_line(self.left_margin, 0, self.left_margin, canvas_height, fill="red", dash=(4, 4))
        self.update_time()
        
    def add_notes(self, notes, label=""):
        pos = len(self._notes)
        fill = "red"
        for n in notes:
            self.add_note_to_canvas(pos, n, fill)
            fill = "gray"
        if label:
            self.add_label_to_canvas(pos, label)
        self._notes.append(notes)
        self._label.append(label)

    def add_label_to_canvas(self, pos, label):
        x = self.left_margin + (pos+0.5) * self.note_width
        y = self.line_height * 0.5
        self.canvas.create_text(x, y, text=label)
    
    def add_note_to_canvas(self, pos, note, fill="gray"):
        x0 = self.left_margin + pos     * self.note_width
        x1 = self.left_margin + (pos+1) * self.note_width
        y0 = self.line_height * (note-NOTE_MIN+1)
        y1 = self.line_height * (note-NOTE_MIN+2)
        self.canvas.create_rectangle(x0, y0, x1, y1, fill=fill)
        
    def update_time(self):
        x = self.left_margin + (self.t + 0.5) * self.note_width
        self.canvas.coords(self.t_line, x, 0, x, self.canvas.cget('height'))
        
    def update(self, sp):
        if sp.is_maximum:
            self.next_note()

    def play(self):
        if not self._playing:
            self._playing = True
            self.t = 0
            
    def stop(self):
        self._playing = False

    def next_note(self):
        if self._playing:
            StringTuner.play_chord(self._notes[self.t])
            self.t = (self.t+1) % len(self._notes)
            self.update_time()

class ChordPicker(Frame):
    def __init__(self, parent):
        self.parent = parent
        
        Frame.__init__(self, parent, bd=2, relief=SUNKEN)
        Label(self, text="Chords").grid(row=0,columnspan=12)

        def makeFunc(note, root, chord):
            return lambda: self.selectChord(note,root,chord)

        for i, root in enumerate(["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"]):
            b = Button(self, text=root+"maj", command=makeFunc(i+60, root, "maj")).grid(row=1,column=i)
            b = Button(self, text=root+"min", command=makeFunc(i+60, root, "min")).grid(row=2,column=i)
            
    def selectChord(self, note, root, chord):
        if chord == "maj":
            notes = [note+0, note+4, note+7]
        elif chord == "min":
            notes = [note+0, note+3, note+7]
        
        # If the notes do fall in the range of the instrument,
        # find a chord inversion that does.
        notes = [self.findNoteInRange(note) for note in notes]
        app._pianoroll.add_notes(notes, root+chord)
        StringTuner.play_chord(notes)
        
    def findNoteInRange(self, note):
        """Shift a note up or down an octave until it fits the note range of the instrument"""
        while note < NOTE_MIN:
            note += 12
        while note > NOTE_MAX:
            note -= 12
        return note
    
class Marlin:
    def __init__(self):
        self._serial = None
        self._in_motion = False
        self._when_done = None

    def connect(self, port):
        self._serial = serial.Serial(port, baudrate=250000)
                
    def disconnect(self):
        if self._serial:
            self._serial.close()
            self._serial = None

    def send(self, cmd):
        if self._serial:
            self._serial.write((cmd+'\n').encode())
            print(cmd)

    def send_position(self, axis, position, when_done = None):
        if self._serial:
            self.send('G0 %s%f F90000' % (axis, position))
            if when_done:
                self._in_motion = True
                self.send('M400\nM114')
                self._when_done = when_done
                
    def update(self):
        if not self._serial:
            return
        
        if self._serial.inWaiting():
            line = self._serial.readline()
            print(line.decode().strip())
            
            if self._in_motion and line.startswith(b"X:"):
                self._in_motion = False
                if self._when_done:
                    self._when_done()
                    
            if line.startswith(b"start"):
                self.initMarlin()

    def initMarlin(self):
        self.send("M211 S0") # Turn off endstops
        self.send("M203 X90000 Y90000 Z90000") # Max feedrate
        self.send("M201 X18000 Y18000 Z18000") # Max acceleration
        self.send("M204 T18000")               # Max starting accelration
        self.send("M907 S870")                 # Set motor current
 
class Application(Frame):
    def connect(self):
        d = PortSelector(root)
        root.wait_window(d.top)
        if d.portname:
            self._marlin.connect(d.portname)
            self.connect_btn["text"]    = "Disconnect"
            self.connect_btn["command"] = self.disconnect

    def disconnect(self, exiting = False):
        self._marlin.disconnect()
        if not exiting:
            self.connect_btn["text"]    = "Connect"
            self.connect_btn["command"] = self.connect
            
    def send_position(self, axis, position, when_done = None):
        self._marlin.send_position(axis, position, when_done)
    
    def send(self, cmd):
        self._marlin.send(cmd)
        
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
        
    def showMotion(self):
        t = tk.Toplevel(self)
        self.fig = MotionGraph()
        self.motion_page = GraphPage(t)
        self.motion_page.add_mpl_figure(self.fig)
    
    def startPlaying(self):
        self._pianoroll.play()
        self.play_btn["text"]    = "Stop"
        self.play_btn["command"] = self.stopPlaying
        
    def stopPlaying(self):
        self._pianoroll.stop()
        self.play_btn["text"]    = "Play"
        self.play_btn["command"] = self.startPlaying

    def createWidgets(self):
        # Buttons
        f = Frame(self)
        f.pack({"side": "top", "pady" : 20})
        self.quit_btn    = Button(f, text = "QUIT",     command = self.quit,   fg = "red")
        self.connect_btn = Button(f, text = "Connect",  command = self.connect           )
        self.spec_btn    = Button(f, text = "Spectrum", command = self.showSpectrum      )
        self.pitch_btn   = Button(f, text = "Pitches",  command = self.showPitches       )
        self.motion_btn  = Button(f, text = "Motion",   command = self.showMotion        )
        self.quit_btn.pack(   {"side": "left"})
        self.connect_btn.pack({"side": "left"})
        self.spec_btn.pack({"side": "left"})
        self.pitch_btn.pack({"side": "left"})
        self.motion_btn.pack({"side": "left"})
        
        f = Frame(self)
        # Chord Picker
        self._chord_picker = ChordPicker(f)
        self._chord_picker.pack({"side": "left"})
        
        self.play_btn    = Button(f, text = "Play", command = self.startPlaying, width=10, font=("Helvetica",16,"bold"))
        self.play_btn.pack({"side": "right", "fill":"y", "padx" : 10,  "pady" : 10})
        f.pack({"side": "top", "padx" : 10, "fill":"x", "expand":1})
        
        # Scrolling Canvas
        self._pianoroll = PianoRoll(self)
        self._pianoroll.pack({"side": "top", "padx" : 10, "fill":"x"})
        
        # Axis Control Buttons
        f = Frame(self)
        for i in range(len(string_axis)):
            AxisControl(f, string_axis[i], self._strings[i]).pack({"side": "top"})
        f.pack({"side": "top", "pady" : 20})
        
    def __init__(self, master=None):
        self._marlin = Marlin()
        self._sp = SoundProcessor()
        self._in_motion = False
        
        self._strings = []
        for a in string_axis:
            self._strings.append(StringTuner(self, a)) 
        
        Frame.__init__(self, master)
        self.pack({"fill":"both"})
        self.createWidgets()
        self.idle()

    def idle(self):
      self.after(100, self.idle)
      
      self._marlin.update()
      self._sp.process_audio_data()
      
      StringTuner.update(self._sp)
      
      if hasattr(self,"spec_page"):
          self.spec_page.refresh(self._sp)
          
      if hasattr(self,"map_page"):
          self.map_page.refresh(self._sp)
          
      if hasattr(self,"motion_page"):
          self.motion_page.refresh(self._sp)
          
      self._pianoroll.update(self._sp)
      
root = Tk()
app = Application(master=root)
app.mainloop()
app.disconnect(exiting = True)
#root.destroy()