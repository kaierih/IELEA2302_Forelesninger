from numpy import sin, cos, pi, exp
from scipy.signal import welch
import numpy as np
import matplotlib.pyplot as plt
from ipywidgets import interact, fixed, FloatSlider, IntSlider, HBox, VBox, interactive_output, Layout
import ipywidgets as widget

# Funksjon brukt i eksempler om frekvensforskyvning
def displayDualSpectrum(x, fs):
    N = len(x)
    Xk = np.fft.fft(x)/N
    Xk = np.fft.fftshift(Xk)
    f = np.array(np.arange(-fs/2, fs/2, fs/N))
    plt.xlim([-fs/2, fs/2])
    plt.plot(f, np.abs(Xk))
    plt.xlabel("Frekvens (Hz)")
    plt.grid(True)
    plt.ylim(ymin=0)





# UI tool to inspect segments of a larger signal in time and frequency domain   
class signalAnalyzer:
    def __init__(self, x_n, f_s, fig_num=1):
        self.x_n = x_n
        self.f_s = f_s
        self.t = np.linspace(0, len(self.x_n)/self.f_s, len(self.x_n), endpoint=False)
        
        
        # Set up canvas
        plt.close(fig_num)
        self.fig = plt.figure(fig_num, figsize=(12, 8))
        
        
        # Set up plot showing signal selection
        self.ax1 = plt.subplot(4, 1, 1)
        self.ax1.set_title(r"Full Signal Plot")
        self.ax1.plot(self.t, self.x_n,  color='tab:blue')
        self.ax1.grid(True)
        self.ax1.set_xlabel('Time t (seconds)')
        self.highlight, = self.ax1.plot(self.t, self.x_n, color='tab:orange')
        self.ax1.axis(xmin=self.t[0], xmax=self.t[-1])
        
        
        
        # Set up signal segment inspection plot
        self.ax2 = plt.subplot(4, 1, (2,4))
        
        self.selectionCurve, = self.ax2.plot(self.t, self.x_n, color='tab:blue')
        self.ax2.grid(True)
        
        
        # Adjust figure layout
        self.fig.tight_layout(pad=0.1, w_pad=1.0, h_pad=2.0)
        
        # Set up UI panel
        domainSelection = widget.RadioButtons(
            options=['Time Trace', 'Frequency Spectrum'],
            value='Time Trace',
            description='Display: ',
            disabled=False,
            continuous_update=False
        )
        win_start = widget.FloatSlider(
            value = 0.0,
            min=0.0,
            max=(len(self.x_n)-1)/self.f_s,
            step = 0.01,
            description='Signal segment start (seconds):',
            disabled=False,
            style = {'description_width': 'initial'},
            layout=Layout(width='95%'),
            continuous_update=False
        )
        win_length = widget.BoundedFloatText(    
            value=0.1,
            min=0.0,
            max=(len(self.x_n)-1)/self.f_s,
            step=0.01,
            description='Signal segment lenght (seconds):',
            disabled=False,
            style = {'description_width': 'initial'},
            #layout=Layout(width='95%'),
            continuous_update=False
        )
        self.layout = VBox(
            [win_start, 
             HBox([win_length, domainSelection])]
        )
        self.userInput = {
            't_start': win_start,
            't_length': win_length,
            'domain': domainSelection
        }
        out = interactive_output(self.update, self.userInput)
        display(self.layout, out)
        # Run demo
        #out = interactive_output(self.update, self.sliders)
        #display(self.layout, out)
        
    def update(self, t_start, t_length, domain):
        n_start = int(t_start*self.f_s)
        n_stop = int((t_start+t_length)*self.f_s)
        self.highlight.set_ydata(self.x_n[n_start:n_stop])
        self.highlight.set_xdata(self.t[n_start:n_stop])
        if domain=='Time Trace':
            self.ax2.set_xlabel('Time t (seconds)')
            self.ax2.set_ylabel('Value x(t)')
            self.selectionCurve.set_ydata(self.x_n[n_start:n_stop])
            self.selectionCurve.set_xdata(self.t[n_start:n_stop])
            self.ax2.set_xlabel("Time t (seconds)")
            self.ax2.set_ylabel("Value x(t)")
            self.ax2.set_title("Time plot of selected signal segment")
            self.ax2.axis(xmin=self.t[n_start], xmax=self.t[n_stop], ymin=min(self.x_n)*1.1, ymax=max(self.x_n)*1.1)
            
        elif domain=='Frequency Spectrum':
            M = n_stop-n_start
            f, Sxx_sub = welch(self.x_n[n_start:n_stop], self.f_s, 'hamming', int(M/4), int(M/8), int(M/2))
            Sxx_sub_dB = 10*np.log10(Sxx_sub)
            
            self.ax2.set_xlabel("Frequency f (Hz)")
            self.ax2.set_ylabel("Power Pxx(f) (dB)")
            self.ax2.set_title("Frequency content of selected signal segment")
            self.selectionCurve.set_ydata(Sxx_sub_dB)
            self.selectionCurve.set_xdata(f)
            self.ax2.axis(xmin=0, xmax=self.f_s/2, ymin=min(Sxx_sub_dB), ymax=max(Sxx_sub_dB)+5)
            
        else:
            pass
        self.fig.tight_layout(pad=0.1, w_pad=1.0, h_pad=2.0)

        
# Funksjoner og klassedefinisjon for fourier demonstrasjon.

def getDiamond(x, y, x_size=0.1, y_size=0.1):
    x_cross = np.array([1, 0, -1, 0, 1])*x_size+x
    y_cross = np.array([0, 1, 0, -1, 0])*y_size+y
    return [x_cross, y_cross]

def getCross(x, y, x_size=0.1, y_size=0.1):
    x_cross = np.array([1, 0, -1, 0, -1, 0, 1])*x_size+x
    y_cross = np.array([1, 0, 1, 0, -1, 0, -1])*y_size+y
    return [x_cross, y_cross]

# UI tool to inspect segments of a larger signal in time and frequency domain   
class FourierDemo:
    def __init__(self, x_t, f_s, max_k=7, fig_num=1):
        self.L = len(x_t)
        self.r_t = np.absolute(x_t)
        self.phi_t = np.angle(x_t)
        self.f_s = f_s
        self.t = np.linspace(0, 1, self.L)
        self.mixer = 2*pi*self.t
        self.k = 0
        self.animationSteps = np.linspace(1/20, 1, 20)
        self.a_k = []
        self.max_k = max_k
        
        for i in range(-max_k, max_k+1):
            self.a_k.append(np.mean(x_t*np.exp(-2j*pi*self.t*i)))
        
        # Set up canvas
        plt.close(fig_num)
        self.fig = plt.figure(fig_num, figsize=(9, 6))
        
        
        # Set up plot showing signal
        self.ax1 = plt.subplot(2, 3, (1,5))
        self.ax1.set_title(r"$x(t)\cdot e^{-j\frac{2\pi\cdot 0}{T_0}\cdot t}$", fontsize=18)
        self.signalCurve, = self.ax1.plot(np.cos(self.phi_t)*self.r_t, np.sin(self.phi_t)*self.r_t,  color='tab:blue')
        self.ax1.grid(True)
        self.ax1.set_aspect(1)
        self.ax1.set_xlabel('Real axis')
        self.ax1.set_ylabel('Imaginary axis')
        axisLimit = max(self.r_t)
        self.ax1.axis([-axisLimit, axisLimit, -axisLimit, axisLimit])
        
        x, y = getCross(np.real(self.a_k[max_k]), np.imag(self.a_k[max_k]), 0.2, 0.2)
        self.coeffSquare, = self.ax1.plot(x, y, color='tab:red')
        
        # Set up plot showing magnitude spectrum
        self.ax2 = plt.subplot(2,3, 3)
        self.ax2.stem(np.arange(-max_k, max_k+1), # Samplenummer
                      np.absolute(self.a_k), # Signalverdier gitt samplenummer (x[n])
                      linefmt='-.', # Linjestil stolper
                      markerfmt='o', # Punktstil for stem-markere. Default er 'o' (stor prikk)
                      basefmt='grey', # Farge på y=0 aksen
                      use_line_collection=True # Hvordan "stem" skal håndtere dataene. Bruk alltid True.
                      )
        self.ax2.set_ylim(ymin=0)
        self.ax2.set_xlim([-max_k, max_k])
        self.ax2.set_xticks(np.arange(-max_k, max_k+1))
        self.ax2.grid(True)
        self.ax2.set_ylabel(r'$A_k$')
        x, y = getDiamond(0, np.absolute(self.a_k[max_k]), 0.5, 0.05)
        self.ampCoeff, = self.ax2.plot(x, y, color='tab:red')


        
        # Set up plot showing phase spectrum
        self.ax3 = plt.subplot(2,3, 6)
        self.ax3.stem(np.arange(-max_k, max_k+1), # Samplenummer
                      np.angle(self.a_k), # Signalverdier gitt samplenummer (x[n])
                      linefmt='-.', # Linjestil stolper
                      markerfmt='o', # Punktstil for stem-markere. Default er 'o' (stor prikk)
                      basefmt='grey', # Farge på y=0 aksen
                      use_line_collection=True # Hvordan "stem" skal håndtere dataene. Bruk alltid True.
                      )        
        self.ax3.set_ylim([-np.pi, np.pi])
        self.ax3.set_xticks(np.arange(-max_k, max_k+1))
        self.ax3.set_xlim([-max_k, max_k])
        self.ax3.set_yticks(np.linspace(-np.pi, np.pi, 5))
        self.ax3.set_yticklabels([r'$-\pi$', r'$-\frac{\pi}{2}$', r'$0$', r'$\frac{\pi}{2}$', r'$\pi$'])
        self.ax3.grid(True)
        self.ax3.set_xlabel('k')
        self.ax3.set_ylabel(r'$\phi_k$')
        x, y = getDiamond(0, np.angle(self.a_k[max_k]), 0.5, 0.4)
        self.angleCoeff, = self.ax3.plot(x, y, color='tab:red')
        
        # Adjust figure layout
        self.fig.tight_layout(pad=0.5, w_pad=1.0, h_pad=1.0)
        
        # Set up UI panel
        coeff_k = widget.BoundedIntText(    
            value=0,
            min=-max_k,
            max=max_k,
            step=1,
            description='Number of rotations "k":',
            disabled=False,
            style = {'description_width': 'initial'},
            #layout=Layout(width='95%'),
            continuous_update=False
        )
        self.layout = HBox(
            [coeff_k]
        )
        self.userInput = {
            'k': coeff_k
        }
        out = interactive_output(self.update, self.userInput)
        display(self.layout, out)
        # Run demo
        #out = interactive_output(self.update, self.sliders)
        #display(self.layout, out)
        
    def update(self, k):
        
        for i in (self.animationSteps*(k-self.k)+self.k):
            phi_t = self.phi_t-self.mixer*i
            self.signalCurve.set_xdata(np.cos(phi_t)*self.r_t)
            self.signalCurve.set_ydata(np.sin(phi_t)*self.r_t)
            #self.fig.tight_layout(pad=0.1, w_pad=1.0, h_pad=2.0)
            self.fig.canvas.draw()
        titleStr = r"$x(t)\cdot e^{-j\frac{2\pi \cdot" +str(k)+"}{T_0}\cdot t}$"
        self.ax1.set_title(titleStr, fontsize=18)
        self.k = k
        
        x, y = getDiamond(k, np.angle(self.a_k[self.max_k+k]), 0.5, 0.4)
        self.angleCoeff.set_xdata(x)
        self.angleCoeff.set_ydata(y)
        
        x, y = getDiamond(k, np.absolute(self.a_k[self.max_k+k]), 0.5, 0.05)
        self.ampCoeff.set_xdata(x)
        self.ampCoeff.set_ydata(y)
        
        x, y = getCross(np.real(self.a_k[self.max_k+self.k]), np.imag(self.a_k[self.max_k+self.k]), 0.2, 0.2)
        self.coeffSquare.set_xdata(x)
        self.coeffSquare.set_ydata(y)