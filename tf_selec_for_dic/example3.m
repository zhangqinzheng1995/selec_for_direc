%与example3是同样的，方便多跑几个
% restoredefaultpath
clc;clear;close all;
addpath(genpath('./../'));

% load hxkuaipai_nono_new5_noise0.mat
% load y_new5_noise0.mat
load y_-20db_016_noise92_100_verify.mat;
data = h5read('result_fit2_-20db_016_noise92_100.h5','/result_SPP_0db');  %不是0db，7db
data = permute(data,[3,2,1]);
errot_total=[];
jilu = [];
countcount = 1;
for threshold = 0.9:0.9
% for threshold = 0:0
        threshold
        
    for yshuliang = 1:100
        % load ix_total_new-5.mat
        y = total_withnoise_signal(:,:,yshuliang);
        SPP_channel1 =data(2:129,:,yshuliang);
        [row,col]=find(SPP_channel1>threshold);
        SPECINST_ME=ones(181,249);
        for nnnnn = 1:249                        %找end
    %         nnnnn
            panduan = ismember(nnnnn,col);
            if panduan ==0
                SPECINST_ME(:,nnnnn)=0;
            
            elseif panduan ==1
    %             panduan
    %             nnnnn
                num_col = find(col == nnnnn);
                shuliang = length(num_col);
                f = [];
                a = [];
                b = []; 
                f_dian=[];
                 for oo = 1:shuliang
                     a(2) = 8000/128*row(num_col(oo));
                     a(1) = 8000/128*(row(num_col(oo))-1);
                     b = row(num_col(oo));
                     f = [f,a];
                     f_dian = [f_dian,b];
                     a = [];
                     b=[];
                 end
                freqRange = f;         % 计算的频率范围 []为所有频率
                fre_use = f_dian;
               
                % addpath('./wav files');
                %% 音频文件和传声器位置坐标
                % fileName = 'example.wav';  
                % micPos = ... 
                % ...%  mic1	 mic2   mic3   mic4   mic5   mic6   mic7  mic8
                %     [ 0.037 -0.034 -0.056 -0.056 -0.037  0.034  0.056 0.056;  % x
                %       0.056  0.056  0.037 -0.034 -0.056 -0.056 -0.037 0.034;  % y
                %     -0.038   0.038 -0.038  0.038 -0.038  0.038 -0.038 0.038]; % z
                
                micPos = [1 2.72 1.2               %2米   0到180°
                    1 2.8 1.2  
                    1 2.88 1.2 
                    1 2.96 1.2 
                    1 3.04 1.2 
                    1 3.12 1.2 
                    1 3.20 1.2 
                    1 3.28 1.2]'; 
                azBound = [-90 90]; % 方位角搜索范围
                elBound = 0;   % 俯仰角搜索范围。若只有水平面：则elBound=0;
                gridRes = 1;          % 方位角/俯仰角的分辨率
                alphaRes = 1;          % 分辨率
                
                method = 'SRP-PHAT';
                % method = 'SNR-MVDR';
                wlen = 256;
                window = hann(wlen);
                noverlap = 0.5*wlen;
                nfft = 256;
                nsrc = 1;               % 声源个数
                c = 340;                % 声速
                
                pooling = 'sum';        % 如何聚合各帧的结果：所有帧取最大或求和{'max' 'sum'}
                
                %% 读取音频文件(fix)
                % [x,fs] = audioread(fileName);
                % x_full = hxkuaipai_nono;
                
                x_full = y;
                % x_full = ix_total;
                qiepian = 32000;
                for qiepianshu = 1 : 32000/qiepian
                    
                x = x_full((qiepianshu-1)*qiepian+1:qiepianshu*qiepian,:);
                fs=16e3;
                    [nSample,nChannel] = size(x);
                    if nChannel>nSample, error('ERROR:输入信号为nSample x nChannel'); end
                    [~,nMic,~] = size(micPos);
                    if nChannel~=nMic, error('ERROR:麦克风数应与信号通道数相等'); end
                    %% 保存参数(fix)
                    Param = pre_paramInit2(c,window, noverlap, nfft,pooling,azBound,elBound,gridRes,alphaRes,fs,freqRange,micPos,fre_use);
                    %% 定位(fix)
                    if strfind(method,'SRP')
                        [specGlobal,specInst] = doa_srp2(x,method, Param);
                    elseif strfind(method,'SNR')
                        specGlobal = doa_mvdr(x,method,Param);
                    elseif strfind(method,'MUSIC')
                        specGlobal = doa_music(x,Param,nsrc);
                    else 
                    end
                    SPECINST_ME(:,nnnnn) = specInst(:,nnnnn);
                end
            end
        end
        switch Param.pooling
            case 'max'
                specGlobal_ME = shiftdim(max(SPECINST_ME,[],2));
            case 'sum'
                specGlobal_ME = shiftdim(sum(SPECINST_ME,2));
        end
        
                    %% 计算角度
                    minAngle                   = 10;         % 搜索时两峰之间最小夹角
                    specDisplay                = 1;          % 是否展示角度谱{1,0}
                    % pfEstAngles = post_sslResult(specGlobal, nsrc, Param.azimuth, Param.elevation, minAngle);
                    % 绘制角谱
                    % [pfEstAngles,figHandle] = post_findPeaks(specGlobal, Param.azimuth, Param.elevation, Param.azimuthGrid, Param.elevationGrid, nsrc, minAngle, specDisplay);
                    [pfEstAngles,figHandle] = post_findPeaks(specGlobal_ME, Param.azimuth, Param.elevation, Param.azimuthGrid, Param.elevationGrid, nsrc, minAngle, specDisplay);
                    
                    azEst = pfEstAngles(:,1)';
                    elEst = pfEstAngles(:,2)';
                    for i = 1:nsrc
                        fprintf('切片数为：%d \n 第 %d 个声源方位为: \n Azimuth (Theta): %.0f \t Elevation (Phi): %.0f \n\n',qiepianshu,i,azEst(i),elEst(i));
                    end
                    close all;
         error_total(yshuliang) = abs(azEst-45);
         jilu(countcount,yshuliang) = azEst;
    end      
    error_final(countcount) = mean(error_total);
    countcount = countcount +1;
end
