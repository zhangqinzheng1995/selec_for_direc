function [specGlobal,specInst] = doa_srp2(x,method, Param)
%% 
if(~any(strcmp(method, {'SRP-PHAT' 'SRP-NON'})))
    error('ERROR[doa_srp]: method参数错误');   
end
%% STFT
X = ssl_stft2(x.',Param.window, Param.noverlap, Param.nfft, Param.fs);
X = X(2:end,:,:);
%% 
if strcmp(method,'SRP-PHAT')
    [specGlobal, specInst] = ssl_srpPhat2(X,Param);
else
    specGlobal = ssl_srp_nonlin2(X,Param);
end

end

function X=ssl_stft2(x,window,noverlap,nfft,fs)

% Inputs:x: nchan x nsampl  window = blackman(wlen);
% Output:X: nbin x nfram x nchan matrix 

[nchan,~]=size(x);
[Xtemp,F,T,~] = spectrogram(x(1,:),window,noverlap,nfft,fs); % S nbin x nframe
nbin = length(F);
nframe = length(T);
X = zeros(nbin,nframe,nchan);
X(:,:,1) = Xtemp;
for ichan = 2:nchan
    X(:,:,ichan) = spectrogram(x(ichan,:),window,noverlap,nfft,fs); 
end

end

function [specGlobal,specInst] = ssl_srpPhat2(X,Param)
[~,nFrames,~] = size(X);
specInst = zeros(Param.nGrid, nFrames);

for i = 1:Param.nPairs
%     XX=[];
%     for kk = 1:length(Param.freqBins)
%         if mod(kk,2) == 1
%         X_new = X(kk:kk+1,:,Param.pairId(i,:));
%         XX = cat(1,XX,X_new);
%         kk
%         size(XX)
%         size(X_new)
%         end
%     end
%     spec = srpPhat_spec(XX, Param.f(Param.freqBins), Param.tauGrid{i}); % NV % [freq x fram x local angle for each pair]
    spec = srpPhat_spec(X(Param.freqBins,:,Param.pairId(i,:)), Param.f(Param.freqBins), Param.tauGrid{i}); % NV % [freq x fram x local angle for each pair]
    switch Param.pooling
        case 'max'
            specSampledgrid = (shiftdim((sum(spec,1)/size(spec,1))))';
        case 'sum'
            specSampledgrid = (shiftdim(sum(spec,1)))';
    end
    specCurrentPair = interp1q(Param.alphaSampled{i}', specSampledgrid, Param.alpha(i,:)');
    specInst(:,:) = specInst(:,:) + specCurrentPair;
end

switch Param.pooling
    case 'max'
        specGlobal = shiftdim(max(specInst,[],2));
    case 'sum'
        specGlobal = shiftdim(sum(specInst,2));
end
end

function [specGlobal] = ssl_srp_nonlin2(X,Param)

alpha_meth = (10*Param.c)./(Param.d*Param.fs);
[~,nFrames,~] = size(X);
specInst = zeros(Param.nGrid, nFrames);

for i = 1:Param.nPairs
    spec = srpNonlin_spec(X(Param.freqBins,:,Param.pairId(i,:)), Param.f(Param.freqBins), alpha_meth(i), Param.tauGrid{i});
    specSampledgrid = (shiftdim(sum(spec,1)))';
    specCurrentPair = interp1q(Param.alphaSampled{i}', specSampledgrid, Param.alpha(i,:)');
    specInst = specInst + specCurrentPair;
end

switch Param.pooling
    case 'max'
        specGlobal = shiftdim(max(specInst,[],2));
    case 'sum'
        specGlobal = shiftdim(sum(specInst,2));
end
end

