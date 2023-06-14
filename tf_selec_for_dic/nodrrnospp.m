% restoredefaultpath
% 这个是不分频率的
clc;
clear;
close all;
% load hxkuaipai_nono_new5_noise0.mat
% load y_new5_noise0.mat
% addpath 'D:\DOA'

SNR = -25

hunxiangjishu = 0;
for hunxiang = 20:20:70
    hunxiangjishu = hunxiangjishu+1;
    yongjuzhenbiaoda =0;
for xuanzejiaodu = 2:12
xuanzejiaodu   
yongjuzhenbiaoda =yongjuzhenbiaoda+1;

m = 0;

filename = ['E:\SPP_DRR_rangeSNR_back\data\' 'y_' num2str(SNR) 'db_0' num2str(hunxiang) '_noise92_100_05s_ang' num2str(xuanzejiaodu) '.mat'];

load(filename) ;

error_total=[];
for yshuliang = 1:250
    y = total_withnoise_signal(:,:,yshuliang);
    % load ix_total_new-5.mat
%     addpath 'D:\DOA\sound-source-localization-algorithm_DOA_estimation-master\ssl_tools'
    addpath(genpath('./../'));
    % addpath('./wav files');
    %% 音频文件和传声器位置坐标
    % fileName = 'example.wav';  
    % micPos = ... 
    % ...%  mic1	 mic2   mic3   mic4   mic5   mic6   mic7  mic8
    %     [ 0.037 -0.034 -0.056 -0.056 -0.037  0.034  0.056 0.056;  % x
    %       0.056  0.056  0.037 -0.034 -0.056 -0.056 -0.037 0.034;  % y
    %     -0.038   0.038 -0.038  0.038 -0.038  0.038 -0.038 0.038]; % z
    
    micPos =  [0.5 2.72+1.5 1.2               %2米   0到180°
    0.5 2.8+1.5 1.2  
    0.5 2.88+1.5 1.2 
    0.5 2.96+1.5 1.2 
    0.5 3.04+1.5 1.2 
    0.5 3.12+1.5 1.2 
    0.5 3.20+1.5 1.2 
    0.5 3.28+1.5 1.2]'; 
    azBound = [-90 90]; % 方位角搜索范围
    elBound = 0;   % 俯仰角搜索范围。若只有水平面：则elBound=0;
    gridRes = 0.1;          % 方位角/俯仰角的分辨率
    alphaRes = 0.1;          % 分辨率
    
    method = 'SRP-PHAT';
    % method = 'SNR-MVDR';
    wlen = 256;
    window = hann(wlen);
    noverlap = 0.5*wlen;
    nfft = 256;
    nsrc = 1;               % 声源个数
    c = 340;                % 声速
    freqRange = [ ];         % 计算的频率范围 []为所有频率
    pooling = 'sum';        % 如何聚合各帧的结果：所有帧取最大或求和{'max' 'sum'}
    
    %% 读取音频文件(fix)
    % [x,fs] = audioread(fileName);
    % x_full = hxkuaipai_nono;
    
    x_full = y;
    % x_full = ix_total;
    qiepian = 4480;
    for qiepianshu = 1 : 1
        
    x = x_full((qiepianshu-1)*qiepian+1:qiepianshu*qiepian,:);
    fs=16e3;
        [nSample,nChannel] = size(x);
        if nChannel>nSample, error('ERROR:输入信号为nSample x nChannel'); end
        [~,nMic,~] = size(micPos);
        if nChannel~=nMic, error('ERROR:麦克风数应与信号通道数相等'); end
        %% 保存参数(fix)
        Param = pre_paramInit(c,window, noverlap, nfft,pooling,azBound,elBound,gridRes,alphaRes,fs,freqRange,micPos);
        %% 定位(fix)
        if strfind(method,'SRP')
            specGlobal = doa_srp(x,method, Param);
        elseif strfind(method,'SNR')
            specGlobal = doa_mvdr(x,method,Param);
        elseif strfind(method,'MUSIC')
            specGlobal = doa_music(x,Param,nsrc);
        else 
        end
        
        %% 计算角度
        minAngle                   = 10;         % 搜索时两峰之间最小夹角
        specDisplay                = 0;          % 是否展示角度谱{1,0}
        % pfEstAngles = post_sslResult(specGlobal, nsrc, Param.azimuth, Param.elevation, minAngle);
        % 绘制角谱
        % [pfEstAngles,figHandle] = post_findPeaks(specGlobal, Param.azimuth, Param.elevation, Param.azimuthGrid, Param.elevationGrid, nsrc, minAngle, specDisplay);
        [pfEstAngles,figHandle] = post_findPeaks(specGlobal, Param.azimuth, Param.elevation, Param.azimuthGrid, Param.elevationGrid, nsrc, minAngle, specDisplay);
        
        azEst = pfEstAngles(:,1)';
        elEst = pfEstAngles(:,2)';
%         for i = 1:nsrc
%             fprintf('切片数为：%d \n 第 %d 个声源方位为: \n Azimuth (Theta): %.0f \t Elevation (Phi): %.0f \n\n',qiepianshu,i,azEst(i),elEst(i));
%         end
        close all;
                if xuanzejiaodu == 2
                    trueang=-75;
                elseif xuanzejiaodu ==3
                    trueang=-60;
                elseif xuanzejiaodu ==4
                    trueang=-45;
                elseif xuanzejiaodu ==5
                    trueang=-30;
                elseif xuanzejiaodu ==6
                    trueang=-15;
                elseif xuanzejiaodu ==7
                    trueang=0;
                elseif xuanzejiaodu ==8
                    trueang=15;
                elseif xuanzejiaodu ==9
                    trueang=30;
                elseif xuanzejiaodu ==10
                    trueang=45;
                elseif xuanzejiaodu ==11
                    trueang=60;
                elseif xuanzejiaodu ==12
                    trueang=75;
                end
        result(yshuliang) = azEst;
        error_total(yshuliang) = abs(azEst-trueang);
        if error_total(yshuliang)<=5;
            m = m+1;
        end
    end
end
error_final = mean(error_total)
aaa_juzhenbiaoda(1,yongjuzhenbiaoda)=error_final;
zhunquelv = m/250
aaa_juzhenbiaoda(2,yongjuzhenbiaoda)=zhunquelv;
end
a_sanwei_juzhenbiaoda(:,:,hunxiangjishu)=aaa_juzhenbiaoda;
end
