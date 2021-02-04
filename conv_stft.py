


import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from scipy.signal import get_window
import torchaudio as audio_nn

def istft(stft_matrix, hop_length=None, win_length=None, window='hann',
          center=True, normalized=False, onesided=True, length=None):
    # keunwoochoi's implementation
    # https://gist.github.com/keunwoochoi/2f349e72cc941f6f10d4adf9b0d3f37e

    """stft_matrix = (batch, freq, time, complex)

    All based on librosa
        - http://librosa.github.io/librosa/_modules/librosa/core/spectrum.html#istft
    What's missing?
        - normalize by sum of squared window --> do we need it here?
        Actually the result is ok by simply dividing y by 2.
    """
    assert normalized == False
    assert onesided == True
    assert window == "hann"
    assert center == True

    device = stft_matrix.device
    n_fft = 2 * (stft_matrix.shape[-3] - 1)

    batch = stft_matrix.shape[0]

    # By default, use the entire frame
    if win_length is None:
        win_length = n_fft

    if hop_length is None:
        hop_length = int(win_length // 4)

    istft_window = torch.hann_window(n_fft).to(device).view(1, -1)  # (batch, freq)
    # istft_window = torch.hann_window(win_length).to(device).view(1, -1)  # (batch, freq)

    n_frames = stft_matrix.shape[-2]
    expected_signal_len = n_fft + hop_length * (n_frames - 1)
    # expected_signal_len = win_length + hop_length * (n_frames - 1)
    # print(expected_signal_len)

    y = torch.zeros(batch, expected_signal_len, device=device)
    for i in range(n_frames):
        sample = i * hop_length
        spec = stft_matrix[:, :, i]
        iffted = torch.irfft(spec, signal_ndim=1, signal_sizes=(win_length,))

        ytmp = istft_window * iffted
        y[:, sample:(sample + n_fft)] += ytmp

    y = y[:, n_fft // 2:]

    if length is not None:
        if y.shape[1] > length:
            y = y[:, :length]
        elif y.shape[1] < length:
            y = F.pad(y, (0, length - y.shape[1]))
            # y = torch.cat((y[:, :length], torch.zeros(y.shape[0], length - y.shape[1])))

    coeff = n_fft / float(
        hop_length) / 2.0  # -> this might go wrong if curretnly asserted values (especially, `normalized`) changes.
    return y / coeff


def init_kernels(win_len, win_inc, fft_len, win_type=None, invers=False):
    if win_type == 'None' or win_type is None:
        window = np.ones(win_len)
    else:
        window = get_window(win_type, win_len, fftbins=True)#**0.5
    
    N = fft_len
    fourier_basis = np.fft.rfft(np.eye(N))[:win_len]
    real_kernel = np.real(fourier_basis)
    imag_kernel = np.imag(fourier_basis)
    kernel = np.concatenate([real_kernel, imag_kernel], 1).T
    
    if invers :
        kernel = np.linalg.pinv(kernel).T 

    kernel = kernel*window
    kernel = kernel[:, None, :]
    return torch.from_numpy(kernel.astype(np.float32)), torch.from_numpy(window[None,:,None].astype(np.float32))


class ConvSTFT1(nn.Module):

    def __init__(self, win_len, win_inc, fft_len=None, win_type='hamming', feature_type='real', fix=True):
        super(ConvSTFT, self).__init__() 
        
        if fft_len == None:
            self.fft_len = np.int(2**np.ceil(np.log2(win_len)))
        else:
            self.fft_len = fft_len
        
        kernel, _ = init_kernels(win_len, win_inc, self.fft_len, win_type)
        #self.weight = nn.Parameter(kernel, requires_grad=(not fix))
        self.register_buffer('weight', kernel)
        self.feature_type = feature_type
        self.stride = win_inc
        self.win_len = win_len
        self.dim = self.fft_len

    def forward(self, inputs):
        if inputs.dim() == 2:
            inputs = torch.unsqueeze(inputs, 1)
        with torch.no_grad():
            inputs = F.pad(inputs,[self.win_len-self.stride, self.win_len-self.stride])
            outputs = F.conv1d(inputs, self.weight, stride=self.stride)
            
            if self.feature_type == 'complex':
                return outputs
            else:
                dim = self.dim//2+1
                real = outputs[:, :dim, :]
                imag = outputs[:, dim:, :]
                mags = torch.sqrt(real**2+imag**2)
                phase = torch.atan2(imag, real)
                return mags, phase

class ConviSTFT1(nn.Module):

    def __init__(self, win_len, win_inc, fft_len=None, win_type='hamming', feature_type='real', fix=True):
        super(ConviSTFT, self).__init__() 
        if fft_len == None:
            self.fft_len = np.int(2**np.ceil(np.log2(win_len)))
        else:
            self.fft_len = fft_len
        kernel, window = init_kernels(win_len, win_inc, self.fft_len, win_type, invers=True)
        #self.weight = nn.Parameter(kernel, requires_grad=(not fix))
        self.register_buffer('weight', kernel)
        self.feature_type = feature_type
        self.win_type = win_type
        self.win_len = win_len
        self.stride = win_inc
        self.stride = win_inc
        self.dim = self.fft_len
        self.register_buffer('window', window)
        self.register_buffer('enframe', torch.eye(win_len)[:,None,:])

    def forward(self, inputs, phase=None):
        """
        inputs : [B, N+2, T] (complex spec) or [B, N//2+1, T] (mags)
        phase: [B, N//2+1, T] (if not none)
        """ 

        if phase is not None:
            real = inputs*torch.cos(phase)
            imag = inputs*torch.sin(phase)
            inputs = torch.cat([real, imag], 1)
        outputs = F.conv_transpose1d(inputs, self.weight, stride=self.stride) 

        # this is from torch-stft: https://github.com/pseeth/torch-stft
        t = self.window.repeat(1,1,inputs.size(-1))**2
        coff = F.conv_transpose1d(t, self.enframe, stride=self.stride)
        outputs = outputs/(coff+1e-8)
        #outputs = torch.where(coff == 0, outputs, outputs/coff)
        outputs = outputs[...,self.win_len-self.stride:-(self.win_len-self.stride)]
        
        return outputs

class ConvSTFT(nn.Module):
    def __init__(self, n_fft, hop_length, win_length):
        super().__init__()
        self.n_fft, self.hop_length = n_fft, hop_length
        self.window = torch.hamming_window(self.n_fft).to("cuda")
        #self.stft = audio_nn.STFT(fft_length=self.n_fft, hop_length=self.hop_length, win_length=win_length)

    def forward(self, signal):
        with torch.no_grad():
            x = torch.stft( signal, n_fft=self.n_fft, hop_length=self.hop_length, win_length=self.n_fft, window=self.window )
            real = x[...,0]
            imag = x[...,1]
            mag = torch.sqrt(real ** 2 + imag ** 2 )
            phase = torch.atan2( imag, real )

        mix = torch.stack((mag, phase), dim=-1)
        return mix


class ConviSTFT(nn.Module):
    def __init__(self, n_fft, hop_length, win_length):
        super().__init__()
        self.n_fft, self.hop_length, self.win_length = n_fft, hop_length, win_length
        self.window = torch.hamming_window(self.n_fft).to("cuda")

    def forward(self, x):
        #B, C, F, T, D = x.shape
        #x = x.view(B, F, T, D)
        x_istft = torch.istft(x, n_fft=self.n_fft, hop_length=self.hop_length, win_length=self.n_fft, window=self.window)
        return x_istft
        
def test_fft():
    torch.manual_seed(20)
    win_len = 320
    win_inc = 160 
    fft_len = 512
    inputs = torch.randn([1, 1, 16000*4])
    fft = ConvSTFT(win_len, win_inc, fft_len, win_type='hanning', feature_type='real')
    import librosa

    outputs1 = fft(inputs)[0]
    outputs1 = outputs1.numpy()[0]
    np_inputs = inputs.numpy().reshape([-1])
    librosa_stft = librosa.stft(np_inputs, win_length=win_len, n_fft=fft_len, hop_length=win_inc, center=False)
    print(np.mean((outputs1 - np.abs(librosa_stft))**2))


def test_ifft1():
    import soundfile as sf
    N = 400
    inc = 100
    fft_len=512
    torch.manual_seed(N)
    data = np.random.randn(16000*8)[None,None,:]
#    data = sf.read('../ori.wav')[0]
    inputs = data.reshape([1,1,-1])
    fft = ConvSTFT(N, inc, fft_len=fft_len, win_type='hanning', feature_type='complex')
    ifft = ConviSTFT(N, inc, fft_len=fft_len, win_type='hanning', feature_type='complex')
    inputs = torch.from_numpy(inputs.astype(np.float32))
    outputs1 = fft(inputs)
    print(outputs1.shape) 
    outputs2 = ifft(outputs1)
    sf.write('conv_stft.wav', outputs2.numpy()[0,0,:],16000)
    print('wav MSE', torch.mean(torch.abs(inputs[...,:outputs2.size(2)]-outputs2)**2))


def test_ifft2():
    N = 400
    inc = 100
    fft_len=512
    np.random.seed(20)
    torch.manual_seed(20)
    t = np.random.randn(16000*4)*0.001
    t = np.clip(t, -1, 1)
    #input = torch.randn([1,16000*4]) 
    input = torch.from_numpy(t[None,None,:].astype(np.float32))
    
    fft = ConvSTFT(N, inc, fft_len=fft_len, win_type='hanning', feature_type='complex')
    ifft = ConviSTFT(N, inc, fft_len=fft_len, win_type='hanning', feature_type='complex')
    
    out1 = fft(input)
    output = ifft(out1)
    print('random MSE', torch.mean(torch.abs(input-output)**2))
    import soundfile as sf
    sf.write('zero.wav', output[0,0].numpy(),16000)


if __name__ == '__main__':
    #test_fft()
    test_ifft1()
    #test_ifft2()
