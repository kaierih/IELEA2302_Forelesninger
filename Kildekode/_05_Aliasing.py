from numpy import sin, cos, pi, exp
from scipy.signal import welch
import numpy as np
import matplotlib.pyplot as plt
from ipywidgets import interact, fixed, FloatSlider, IntSlider, HBox, VBox, interactive_output, Layout
import ipywidgets as widget
from scipy.io import wavfile    

def make_stem_segments(n, xn):
    """
    Create list of line segments from x and y coordinates, in the correct format
    for LineCollection: an array of the form numlines x (points per line) x 2 (x
    and y) array
    """

    points = np.array([n, xn]).T.reshape(-1, 1, 2)
    start_points = np.array([n, np.zeros(len(n))]).T.reshape(-1, 1, 2)

    segments = np.concatenate([start_points, points], axis=1)
    return segments

class interactiveStem:
    def __init__(self, ax, N = 2, A_max=1):
        self.N = N
        self.ax = ax
        self.n = np.arange(self.N)
        self.samples = self.ax.stem(self.n, # 
                                    np.zeros(self.N), # Nullsampler
                                    linefmt='C3', # Linjestil stolper
                                    markerfmt='oC3', # Punktstil for stem-markere. Default er 'o' (stor prikk)
                                    basefmt='black', # Farge på y=0 aksen
                                    use_line_collection=True # Hvordan "stem" skal håndtere dataene. Bruk alltid True.
                                    )
        self.samples.baseline.set_linewidth(0.5)
        # avgrensning av akser, rutenett, merkede punkt på aksene, tittel, aksenavn
        self.ax.axis([0, self.N, -A_max, A_max])
        self.ax.set_xticks(np.arange(N+1))
        self.ax.set_xlabel("Samplenummer 'n'")
        
    def update(self, n, xn):
        self.N = len(n)
        
        # Adjust stemlines, markerline, baseline in stemcontainer
        segments = make_stem_segments(n, xn)
        self.samples.stemlines.set_segments(segments)
        self.samples.markerline.set_xdata(n)
        self.samples.markerline.set_ydata(xn)
        self.samples.baseline.set_xdata([0, self.N])
        self.samples.baseline.set_ydata([0, 0])
        
        # Adjust sample markers
        self.ax.set_xticks(np.arange(self.N+1))
        
        # Adjust axis limits
        self.ax.set_xlim([-0.0, self.N])
        
class timeSeriesPlot:
    def __init__(self, ax, t, A_max, N=1, t_unit='s'):
        res  = len(t)
        self.N = N
        t_nd = np.outer(t, np.ones(self.N))
        x_t = np.zeros((res, self.N))          

        self.ax = ax
        self.lines = self.ax.plot(t_nd, x_t, zorder=10)
        
        # avgrensning av akser, rutenett, merkede punkt på aksene, tittel, aksenavn
        self.ax.axis([t[0], t[-1], -A_max, A_max])
        self.ax.grid(True)
        self.ax.set_xticks(np.linspace(t[0],t[-1],11))
        self.ax.set_xlabel("Tid (" + t_unit + ")")
        
    def update(self, new_lines):
        assert self.N == len(new_lines), "Error: Parameter lenght different from number of sines."
        for i in range(self.N):
            self.lines[i].set_ydata(new_lines[i])
            
    def setLabels(self, names):
        #self.ax.legend(self.lines, names, loc='upper right')
        for i in range(len(names)):
            self.lines[i].set_label(names[i])
        
    def setStyles(self, styles):
        for i in range(min(len(styles), len(self.lines))):
            try:
                self.lines[i].set_color(styles[i]['color'])
            except:
                pass
            
            try:
                self.lines[i].set_linestyle(styles[i]['linestyle'])
            except:
                pass
            
# Samplet sinussignal
class aliasingDemo():
    def __init__(self, fig_num=1):
        # Set up canvas
        plt.close(fig_num)
        self.fig = plt.figure(fig_num, figsize=(12, 6))
        
        # Set up sine waves on canvas
        ax1 = plt.subplot()
        self.t = np.linspace(0, 1, 501)
        self.SineWaves = timeSeriesPlot(ax1, self.t, A_max = 1.1, N=2)
        self.SineWaves.setLabels([r'Opprinnelig signal $x(t)$',
                                  r'Rekonstruert singal $\hat{x}(t)$'])

        # Stem samples on overlay canvas
        ax2 = ax1.twiny()
        self.discreteSignal = interactiveStem(ax2, A_max = 1.1)
        self.discreteSignal.samples.set_label(r"Samplet signal $x[n]$")
        self.fig.legend(bbox_to_anchor=(0.99, 0.92), loc=1)
        
        # Tilpass figur-layout

        self.fig.tight_layout(pad=0.1, w_pad=1.0, h_pad=1.0)

        # Set up slider panel
        # Set up UI panel
        signal_freq = widget.FloatSlider(
                                        value = 1,
                                        min=1,
                                        max=10,
                                        step = 0.25,
                                        description='Sine wave frequency (Hz):',
                                        disabled=False,
                                        style = {'description_width': 'initial'},
                                        layout=Layout(width='95%'),
                                        continuous_update=True
        )
        sampling_freq = widget.IntSlider(
                                        value = 10,
                                        min=2,
                                        max=20,
                                        step = 1,
                                        description='Sampling frequency (samples/s):',
                                        disabled=False,
                                        style = {'description_width': 'initial'},
                                        layout=Layout(width='95%'),
                                        continuous_update=True
        )
        self.layout = VBox([signal_freq, sampling_freq])
        self.userInput = {
            'F': signal_freq,
            'Fs': sampling_freq
        }
        
        # Run demo
        out = interactive_output(self.update, self.userInput)
        display(self.layout, out)
        

        
    def update(self, **kwargs):
        F = kwargs['F']
        Fs = kwargs['Fs']
        F_rec = (F-Fs/2)%Fs-Fs/2
        A_rec = 1*(F != Fs/2)
        
        # Update waveforms
        x1_t = sin(2*pi*F*self.t)
        x2_t = A_rec*sin(2*pi*F_rec*self.t)
        
        self.SineWaves.update([x1_t, x2_t])
        
        # Update discrete samples
        n = np.arange(0, Fs)
        w_d = 2*pi*F/Fs
        xn = sin(w_d*n)
        self.discreteSignal.update(n, xn)