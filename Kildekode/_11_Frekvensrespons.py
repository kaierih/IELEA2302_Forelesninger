from numpy import sin, cos, pi, exp, real, imag
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig
from ipywidgets import interact, fixed, FloatSlider, IntSlider, HBox, VBox, interactive_output, Layout
import ipywidgets as widget

# Interactive stem plot with matlab-ish default config
class interactiveStem:
    def __init__(self, ax, n, xn, color='tab:blue', marker='o', label=None, filled=False):
        self.ax = ax
        self.samples = self.ax.stem(n, # 
                                    xn, # Nullsampler
                                    basefmt='black', # Farge på y=0 aksen
                                    label=label,
                                    use_line_collection=True # Hvordan "stem" skal håndtere dataene. Bruk alltid True.
                                    )
        self.samples.baseline.set_linewidth(0.5)
        self.samples.baseline.set_xdata([0, len(n)])
        self.samples.markerline.set_color(color)
        if not filled:
            self.samples.markerline.set_markerfacecolor('none')
        self.samples.stemlines.set_color(color)
        self.ax.grid(True)
        
    def update(self, n, xn):
        self.N = len(n)
        
        # Make new line collection
        points = np.array([n, xn]).T.reshape(-1, 1, 2)
        start_points = np.array([n, np.zeros(len(n))]).T.reshape(-1, 1, 2)
        segments = np.concatenate([start_points, points], axis=1)
        
        # Adjust markers and lines
        self.samples.stemlines.set_segments(segments)
        self.samples.markerline.set_xdata(n)
        self.samples.markerline.set_ydata(xn)

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
        #self.ax.set_xticks(np.linspace(t[0],t[-1],11))
        
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

class FreqRespDemo:
    def __init__(self, hn, fig_num=1, figsize=(8,6)):
        
        plt.close(fig_num)
        self.fig = plt.figure(fig_num, figsize=figsize)
        
        self.hn = hn
        self.M = len(hn)
        self.w, Hw = sig.freqz(hn, [1], worN=512)
        self.Hw_amp = np.abs(Hw)
        self.Hw_phase = np.unwrap(np.angle(Hw))
        
        self.t_n = np.linspace(0, 16, 501)
        self.n = np.arange(16)
        
        
        # Amplituderespons
        ax11 = plt.subplot(2,2,1)
        ax11.plot(self.w, self.Hw_amp)
        ax11.set_xlim([0, pi])
        ax11.set_ylim(ymin=0)
        ax11.set_xticks(np.linspace(0, 1, 5)*pi)
        ax11.set_xticklabels([r'$'+str(round(i,2))+'\pi$' for i in np.linspace(0, 1, 5)])
        ax11.set_xlabel(r'Digital Frekvens $\hat{\omega}$')
        ax11.set_ylabel(r'Amplituderespons $\left| H\left(e^{j\hat{\omega}} \right)\right|$')
        ax11.grid(True)
        ax11.set_title('placeholder')
        self.ax11 = ax11
        
        # Markør for valgt frekvens:
        self.ampMarker, = ax11.plot([0], [1], 'oC3')
        

        # Frekvensrespons
        ax12 = plt.subplot(2,2,2)
        ax12.plot(self.w, self.Hw_phase/pi)
        phaseLabels = ax12.get_yticks()
        phaseLim = ax12.get_ylim()
        ax12.set_yticks(phaseLabels)
        ax12.set_ylim(phaseLim)
        ax12.set_yticklabels([r'$'+str(round(i,2))+'\pi$' for i in phaseLabels])
        ax12.set_xlim([0, pi])
        ax12.set_xticks(np.linspace(0, 1, 5)*pi)
        ax12.set_xticklabels([r'$'+str(round(i,2))+'\pi$' for i in np.linspace(0, 1, 5)])
        ax12.set_xlabel(r'Digital Frekvens $\hat{\omega}$')
        ax12.set_ylabel(r'Faserespons $\angle H\left(e^{j\hat{\omega}} \right)$')
        ax12.grid(True)
        ax12.set_title('placeholder')
        self.ax12 = ax12
        
        # Markør for valgt frekvens:
        self.phaseMarker, = ax12.plot([0], [0], 'oC3')

        # Sinusfigurer
        ax2 = plt.subplot(2,2,(3,4))
        ax2.set_title('placeholder')
        self.waveforms = timeSeriesPlot(ax2, self.t_n, A_max=1.1, N=2)
        self.waveforms.setStyles([{'color':'tab:blue', 'linestyle':'-.'}, 
                                  {'color':'tab:red', 'linestyle':'-.'}])
        self.waveforms.lines[0].set_linewidth(0.5)
        self.waveforms.lines[1].set_linewidth(0.5)
        
        self.xn_stem = interactiveStem(ax2, self.n, sin(0*self.n), color='tab:blue', label=r'x[n]')
        self.yn_stem = interactiveStem(ax2, self.n, sin(0*self.n), color='tab:red', label=r'y[n]')
        ax2.legend(loc='upper right')
        ax2.set_xlabel(r'Samplenummer $n$')
        self.ax2 = ax2
        
        # Confiugre Layout
        self.fig.tight_layout(pad=0.5, w_pad=1.0, h_pad=1.0)
        
        #Set up slider panel
        normFreq = widget.FloatSlider(
                                    value = 1/8,
                                    min=0,
                                    max=127/128,
                                    step = 1/128,
                                    description=r'Digital Frekvens $\hat{\omega}\ (\times \pi)$',
                                    disabled=False,
                                    style = {'description_width': 'initial'},
                                    layout=Layout(width='95%'),
                                    continuous_update=False
                                    )
        self.layout = VBox([normFreq])
        self.userInput = {'w': normFreq}
        
        # Run demo
        out = interactive_output(self.update, self.userInput)
        display(self.layout, out)

    
    def update(self, w):
        index = int(w*128)*4
        
        self.ampMarker.set_xdata(self.w[index])
        self.ampMarker.set_ydata(self.Hw_amp[index])
        self.phaseMarker.set_xdata(self.w[index])
        self.phaseMarker.set_ydata(self.Hw_phase[index]/pi)
        self.ax11.set_title(r"$\left| H \left( e^{j"+str(round(w,2))+r"\pi} \right) \right| = "+str(round(self.Hw_amp[index],2))+"$")
        self.ax12.set_title(r"$\angle H \left( e^{j"+str(round(w,2))+r"\pi} \right) = "+str(round(self.Hw_phase[index]/pi,2))+"\pi$")
        titlestr = (r"$x[n] = \sin("+str(round(w,2))+r"\pi \cdot n), \ \ \ \ y[n]="+
                          str(round(self.Hw_amp[index],2))+r"\cdot\sin("+str(round(w,2))+r"\pi \cdot n +"+str(round(self.Hw_phase[index]/pi,2))+r"\pi)$")
        titlestr=titlestr.replace("+-", "-")
        self.ax2.set_title(titlestr)        
        
        xt = sin(pi*w*self.t_n)
        yt = self.Hw_amp[index]*sin(pi*w*self.t_n+self.Hw_phase[index])
        self.waveforms.update([xt, yt])
        
        xn = sin(pi*w*self.n)
        yn = self.Hw_amp[index]*sin(pi*w*self.n+self.Hw_phase[index])
        self.xn_stem.update(self.n, xn)
        self.yn_stem.update(self.n, yn)

##########################################
# Funksjoner og klasser for for DTFT demo#
##########################################
def getArrow(x, y, dx, dy, arrowhead_scale=1):
    r = np.hypot(dx, dy)
    theta = np.arctan2(dy,dx)
    len_arrowhead = min(arrowhead_scale/16, r/2)
    x_arrow = np.array([x, x+dx, x+dx+len_arrowhead*cos(theta-4*pi/5), x+dx, x+dx+len_arrowhead*cos(theta+4*pi/5)])
    y_arrow = np.array([y, y+dy, y+dy+len_arrowhead*sin(theta-4*pi/5), y+dy, y+dy+len_arrowhead*sin(theta+4*pi/5)])
    return x_arrow, y_arrow

class vectorPlot:
    def __init__(self, ax, A_max, N=1, arrowhead_scale=1):
        self.ax = ax
        self.N = N
        self.arrowhead_scale=arrowhead_scale
        init_values = np.zeros((2, N))
        self.lines = self.ax.plot(init_values, init_values)
        self.ax.grid(True)
        self.ax.set_xlabel("Reell akse")
        self.ax.set_ylabel("Imaginær akse")
        self.ax.axis([-A_max, A_max, -A_max, A_max])
        
    def update(self, x_new_lines, y_new_lines):
        assert len(x_new_lines)==len(y_new_lines)==self.N, 'Error: mismatch between x and y dimensions.'
        for i in range(self.N):
            x_line = x_new_lines[i]
            y_line = y_new_lines[i]
            L = len(x_line)
            assert len(y_line)==L, 'Error: mismatch between x and y dimensions.'
            x_arrows = np.zeros((L-1)*5)
            y_arrows = np.zeros((L-1)*5)
            for j in range(1, L):
                b = j*5
                a = b-5
                x_arrows[a:b], y_arrows[a:b] = getArrow(x_line[j-1], y_line[j-1], x_line[j]-x_line[j-1], y_line[j]-y_line[j-1], arrowhead_scale = self.arrowhead_scale)
            self.lines[i].set_xdata(x_arrows)
            self.lines[i].set_ydata(y_arrows)
            
    def setLabels(self, names):
        self.ax.legend(self.lines, names, loc='upper right')
        
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

# Vektorrepresentasjon av komplekst tall på polarform        
class DTFT_demo:
    def __init__(self, xn, fig_num=5, figsize=(8,6)):
        self.xn = xn
        self.N = len(xn)
        self.f = np.linspace(-1, 1, 501)
        self.H_f = np.fft.fftshift(np.fft.fft(xn, 501))
        
        
        # Set up canvas
        plt.close(fig_num)
        self.fig = plt.figure(fig_num, figsize=figsize)
        
        ax = plt.subplot(2,3, (1, 5))
    
     
        ax.set_title(r"$\sum_{n=0}^{M} h[n]\cdot e^{j\hat{\omega}\cdot n}$")
        ax.set_aspect(1)
        self.VectorFig = vectorPlot(ax, A_max = sum(np.absolute(xn)), N = 2, arrowhead_scale = sum(abs(xn))/2)
        
        self.VectorFig.setStyles([{'color': 'tab:blue'}])
        self.VectorFig.lines[1].set_color('C3')
        self.VectorFig.ax.plot(np.real(self.H_f), np.imag(self.H_f), ':C0')
        
        ax2 = plt.subplot(2,3,3)
        ax2.plot(self.f, np.abs(self.H_f))
        ax2.set_xlim([-1, 1])
        ax2.set_ylim(ymin=0)
        ax2.grid(True)
        ax2.set_ylabel(r'$\left|H\left(e^{j\hat{\omega}} \right) \right|$')
        ax2.set_xticks(np.linspace(-1, 1, 5))
        ax2.set_xticklabels([str(round(f,1))+'$\pi$' for f in np.linspace(-1, 1, 5)])
        
        self.MagHighlight, = ax2.plot([0,0], [0,0], 'C3')
        self.MagHighlight.set_ydata([0, 100])
        
      
        ax3 = plt.subplot(2,3,6)
        ax3.plot(self.f, np.angle(self.H_f)*(np.abs(self.H_f)>1e-10))
        ax3.set_xlim([-1, 1])
        ax3.set_ylim([-pi, pi])
        ax3.set_yticks(np.linspace(-1, 1, 9)*pi)
        ax3.set_yticklabels([str(round(f,1))+'$\pi$' for f in np.linspace(-1, 1, 9)])
        ax3.set_ylabel(r'$\angle H\left(e^{j\hat{\omega}} \right)$')
        ax3.set_xticks(np.linspace(-1, 1, 5))
        ax3.set_xticklabels([str(round(f,1))+'$\pi$' for f in np.linspace(-1, 1, 5)])
        ax3.set_xlabel(r'Digital Frekvens $\hat{\omega}$')
        
        self.AngleHighlight, = ax3.plot([0,0], [0,0], 'C3')
        self.AngleHighlight.set_ydata([-100, 100])
        
        ax3.grid(True)
        
        # Tilpass figur-layout
        self.fig.tight_layout(pad=0.1, w_pad=1.0, h_pad=1.0)

        signal_freq = widget.FloatSlider(
                                        value = 0.0,
                                        min=-1,
                                        max=1,
                                        step = 1/60,
                                        description=r'Digital Frekvens $\hat{\omega}\ (\times \pi)$:',
                                        disabled=False,
                                        style = {'description_width': '30%'},
                                        layout=Layout(width='95%'),
                                        continuous_update=True
        )
        layout = HBox([signal_freq])
        # Run Demo:
        out = interactive_output(self.update, {'omega': signal_freq})
        display(layout, out)
        
    def update(self, omega):
        vectors = self.xn*np.exp(1j*omega*pi*np.arange(self.N))
        vectorSums = np.array([np.sum(vectors[0:i]) for i in range(self.N+1)])
        x = np.append(np.array([0]), np.real(vectorSums))
        y = np.append(np.array([0]), np.imag(vectorSums))
        self.VectorFig.update([x, np.array([0, x[-1]])], [y, np.array([0, y[-1]])])
        
        self.MagHighlight.set_xdata([omega, omega])
        self.AngleHighlight.set_xdata([omega, omega])