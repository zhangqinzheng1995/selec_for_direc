% restoredefaultpath
% 这个是不分频率的
clc;clear;close all;
% load hxkuaipai_nono_new5_noise0.mat
% load y_new5_noise0.mat
% addpath 'D:\DOA'
SNR = -15
for xuanzejiaodu = 2:12
    xuanzejiaodu
        filename = ['D:\matlab2\toolbox\RIR-Generator-master\sound-source-localization-algorithm_DOA_estimation-master\ssl_tools\example\11_22\' 'y_' num2str(SNR) 'db_036_noise92_100_05s_ang' num2str(xuanzejiaodu) '.mat'];
        load(filename);
        filename2=['D:\matlab2\toolbox\RIR-Generator-master\sound-source-localization-algorithm_DOA_estimation-master\ssl_tools\example\11_22\' 'result_fit2_' num2str(SNR) 'db_036_noise92_100_05s_ang' num2str(xuanzejiaodu) '.h5'];
        data = h5read(filename2,'/result_SPP_0db');  %不是0db，7db
        data = permute(data,[3,2,1]);
        errot_total=[];
        jilu = [];
        countcount = 1;
                for yshuliang = 1:100
%                     yshuliang
                    y = total_withnoise_signal(:,:,yshuliang);
                    % load ix_total_new-5.mat
                    addpath(genpath('./../'));
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
                    freqRange = [ ];         % 计算的频率范围 []为所有频率
                    pooling = 'sum';        % 如何聚合各帧的结果：所有帧取最大或求和{'max' 'sum'}
                    
                    %% 读取音频文件(fix)
                    % [x,fs] = audioread(fileName);
                    % x_full = hxkuaipai_nono;
                    
                    x_full = y;
                    % x_full = ix_total;
                    
                %     for qiepianshu = 1 : 32000/qiepian
                        
                    x = x_full;
                    fs=16e3;
                        [nSample,nChannel] = size(x);
                        if nChannel>nSample, error('ERROR:输入信号为nSample x nChannel'); end
                        [~,nMic,~] = size(micPos);
                        if nChannel~=nMic, error('ERROR:麦克风数应与信号通道数相等'); end
                        %% 保存参数(fix)
                        Param = pre_paramInit(c,window, noverlap, nfft,pooling,azBound,elBound,gridRes,alphaRes,fs,freqRange,micPos);
                        %% 定位(fix)
                        if strfind(method,'SRP')
                            [specGlobal,specInst,spec_me] = doa_srp3(x,method, Param);
                        elseif strfind(method,'SNR')
                            specGlobal = doa_mvdr(x,method,Param);
                        elseif strfind(method,'MUSIC')
                            specGlobal = doa_music(x,Param,nsrc);
                        else 
                        end
                   
                         menxianshuliang = 1;
                         for threshold = 0.5:0.03:0.86
        %                  for threshold = 0.89:0.03:0.92
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
                %                      f = [f,a];
                                     f_dian = [f_dian,b];
                                     a = [];
                                     b=[];
                                 end
                %                 freqRange = f;         % 计算的频率范围 []为所有频率
                                fre_use = f_dian;
                                spec_me_thisthreshold =spec_me(fre_use,:,:);    %128(25)*249*181
                                specSampledgrid = (shiftdim(sum(spec_me_thisthreshold,1)))';   %181*249
                                SPECINST_ME(:,nnnnn)=specSampledgrid(:,nnnnn);
                            end
                           end
                        specGlobal_ME = shiftdim(sum(SPECINST_ME,2));
                           %% 计算角度
                        minAngle                   = 10;         % 搜索时两峰之间最小夹角
                        specDisplay                = 1;          % 是否展示角度谱{1,0}
                        % pfEstAngles = post_sslResult(specGlobal, nsrc, Param.azimuth, Param.elevation, minAngle);
                        % 绘制角谱
                        % [pfEstAngles,figHandle] = post_findPeaks(specGlobal, Param.azimuth, Param.elevation, Param.azimuthGrid, Param.elevationGrid, nsrc, minAngle, specDisplay);
                        [pfEstAngles,figHandle] = post_findPeaks(specGlobal_ME, Param.azimuth, Param.elevation, Param.azimuthGrid, Param.elevationGrid, nsrc, minAngle, specDisplay);
                        
                        azEst = pfEstAngles(:,1)';
                        elEst = pfEstAngles(:,2)';
%                         for i = 1:nsrc
%                             fprintf(' 第 %d 个声源方位为: \n Azimuth (Theta): %.0f \t Elevation (Phi): %.0f \n\n',i,azEst(i),elEst(i));
%                         end
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
        
                         error_total(yshuliang,menxianshuliang) = abs(azEst-trueang);
                             if error_total(yshuliang,menxianshuliang)<=5
                                 m(yshuliang,menxianshuliang)=1;
                             elseif error_total(yshuliang,menxianshuliang)>5
                                 m(yshuliang,menxianshuliang)=0;
                             end
                         jilu(menxianshuliang,yshuliang) = azEst;
                         menxianshuliang = menxianshuliang+1;
                         end
                     end
         for i = 1:menxianshuliang-1
             error_final(i)=mean(error_total(:,i)); 
             m_final(i)=mean(m(:,i)); 
         end
         error_final = error_final';
         m_final = m_final';
         le1=length(error_final);
         for ooxx = 1:le1   
          atongji(2*ooxx-1,xuanzejiaodu) = error_final(ooxx);
          atongji(2*ooxx,xuanzejiaodu)=m_final(ooxx);
         end
end