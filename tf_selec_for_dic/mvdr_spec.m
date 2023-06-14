function spec = mvdr_spec(hatRxx, f, tauGrid)

[nbin,nFrames] = size(hatRxx(:,:,1,1));
ngrid = length(tauGrid);
R11 = hatRxx(:,:,1,1);
R12 = hatRxx(:,:,1,2);
R21 = hatRxx(:,:,2,1);
R22 = hatRxx(:,:,2,2);
traceRxx = real(R11 + R22);   % tr(hatRxx)

SNR = zeros(nbin,nFrames,ngrid);
for pkInd=1:ngrid,
    EXP = repmat(exp(-2*1i*pi*tauGrid(pkInd)*f),1,nFrames); % d = [1 EXP],EXP=exp(ix)
    power_y = real(R11.*R22 - R12.*R21)./(traceRxx - 2*real(R12.*EXP)); 
    % power_y = (d'(hatRxx^-1)d)^-1����MVDR�ķ�λ�׺����������ᴿ�źŵĹ���(������two_decades_of_array_signal_processing_research)
    % �������ڶ��׾���M = [a b; c d];�����M^-1 = (1/det(M))*[d -b; -c a]
    % det(hatRxx) = real(R11.*R22 - R12.*R21)  hatRxx^-1 = [R22 -R12; -R21 R11]/det(hatRxx)
    % d'(hatRxx^-1)d = [1 exp(-ix)][R22 -R12; -R21 R11][1;exp(ix)]/det(hatRxx)
    %                =[R22-exp(-ix)R12,-R21+exp(-ix)R11][1;exp(ix)]/det(hatRxx)
    %                = (R22-exp(-ix)R12-R21exp(ix)+R11)/det(hatRxx)
    %                        |  Э�������Ϊʵ�Գ���R12=R21��trace=R11+R22, EXP=exp(ix)
    %                =(trace-2*R12(cos(x)-isin(x)+cos(x)+isin(x)))/det(hatRxx)
    %                = (trace-2*real(EXP)*R12)/det(hatRxx)
    % ��Ϊd'(hatRxx^-1)d�Ǳ���������power_y = (d'(hatRxx^-1)d)^-1 = det(hatRxx)/(trace-2*real(EXP)*R12)
    SNR(:,:,pkInd) = power_y./(.5*traceRxx-power_y); % �������� SNR_out = �ᴿ�źŹ���/�����������ʣ�
                                             % ������������ = �����źŹ���-�ᴿ�źŹ���
                                             % Э�������Խ���Ϊ��ͨ�������źŵķ�����ʣ�,����trace(Rxx)/nmic(����ͨ��������ͨ����ƽ��)�ķ�������һ��ͨ���Ľ����źŹ��ʣ�������ֻȡ����һ��ͨ������ķ���
                                             % �����������pair����MVDR���ף�����nmic=2�������źŹ���=0.5*trace(Rxx)
end
spec = SNR;

end