function [noisePowMat,snrPost1,SPP] = noisepowproposed(noisy, fs, noise_true)
% noisy = clean + noise_true;





noise_true_stft = stft(noise_true,16000,Window=hann(256,"periodic"),OverlapLength=128,FFTLength=256);
noise_stft_half = noise_true_stft(128:256,:);
noise_stft_abs = abs(noise_stft_half);
real_noise = noise_stft_abs.^2;
% 
% [real_noise, F, T] = spectrogram(noise_true, hann(256,"periodic"), 128, 256, 16000, 'yaxis');
% % real_noise = abs(real_noise).^2;
% real_noise = real_noise.*conj(real_noise);

%% some constants
frLen   = 256;  % frame size
% frLen   = 32e-3*fs*2;  % frame size
fShift  = frLen/2;   % fShift size
nFrames = floor(length(noisy)/fShift)-1; % number of frames

anWin  = hann(frLen,'periodic'); %analysis window

%% allocate some memory
noisePowMat = zeros(frLen/2+1,nFrames);

%% initialize
noisePow = init_noise_tracker_ideal_vad(noisy,frLen,frLen,fShift, anWin); % This function computes the initial noise PSD estimate. It is assumed that the first 5 time-frames are noise-only.
% noisePowMat(:,1)=noisePow;
noisePowMat(:,1)=real_noise(:, 1);


%
PH1mean  = 0.5;
alphaPH1mean = 0.9;
alphaPSD = 0.8;

%constants for a posteriori SPP
q          = 0.5; % a priori probability of speech presence:
priorFact  = q./(1-q);
xiOptDb    = 15; % optimal fixed a priori SNR for SPP estimation
%calculated in the 2.D by Charlie.Tang
xiOpt      = 10.^(xiOptDb./10);
logGLRFact = log(1./(1+xiOpt));
GLRexp     = xiOpt./(1+xiOpt);

SPP = zeros(129, nFrames);

for indFr = 1:nFrames
    indices       = (indFr-1)*fShift+1:(indFr-1)*fShift+frLen;
    noisy_frame   = anWin.*noisy(indices);
    noisyDftFrame = fft(noisy_frame,frLen);
    noisyDftFrame = noisyDftFrame(1:frLen/2+1);
	
    noisyPer = noisyDftFrame.*conj(noisyDftFrame);
    snrPost1 =  noisyPer./(real_noise(:, indFr));% a posteriori SNR based on old noise power estimate

    %% noise power estimation
	GLR     = priorFact .* exp(min(logGLRFact + GLRexp.*snrPost1,200));
	PH1     = GLR./(1+GLR); % a posteriori speech presence probability

	PH1mean  = alphaPH1mean * PH1mean + (1-alphaPH1mean) * PH1;
	stuckInd = PH1mean > 0.99;
	PH1(stuckInd) = min(PH1(stuckInd),0.99);
	estimate =  PH1 .* real_noise(:, indFr) + (1-PH1) .* noisyPer ;
	% noisePow = alphaPSD *noisePow+(1-alphaPSD)*estimate;
        
	noisePowMat(:,indFr) =real_noise(:, indFr);
    
    %% SPP
    SPP(:, indFr) = PH1;
    
end

return

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function   noise_psd_init =init_noise_tracker_ideal_vad(noisy,fr_size,fft_size,hop,sq_hann_window)
 
for I=1:5
    noisy_frame=sq_hann_window.*noisy((I-1)*hop+1:(I-1)*hop+fr_size,:);
    noisy_dft_frame_matrix(:,I)=fft(noisy_frame,fft_size);
end
noise_psd_init=mean(abs(noisy_dft_frame_matrix(1:fr_size/2+1,1:end)).^2,2);%%%compute the initialisation of the noise tracking algorithms.
return
