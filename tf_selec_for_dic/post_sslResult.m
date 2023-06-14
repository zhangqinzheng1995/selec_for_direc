function result = post_sslResult(specGlobal, nsrc, azimuth, elevation, minAngle)
gridRes = abs(azimuth(1)-azimuth(2));
result = zeros(nsrc,2);
nAz = length(azimuth);
nEl = length(elevation);

if(nAz == 1)%��ֱ��
    for isrc = 1:nsrc
        [~,index] = sort(specGlobal);
        result(isrc,1) = azimuth; 
        result(isrc,2) = elevation(1)+gridRes*(index(end)-1);
        specGlobal(index-floor(minAngle/2/gridRes):index+floor(minAngle/2/gridRes))=-inf;
    end
elseif(nEl == 1)%ˮƽ��
%     plot(azimuth, specGlobal);
    for isrc = 1:nsrc
        [~,index] = sort(specGlobal);
        result(isrc,1) = azimuth(1)+gridRes*(index(end)-1);
        result(isrc,2) = elevation;
        specGlobal(index-floor(minAngle/2/gridRes):index+floor(minAngle/2/gridRes))=-inf;
    end
else % ��ά
    Spec2D = (reshape(specGlobal,nAz,nEl))';
    for isrc = 1:nsrc
        [x, y] = find(Spec2D==max(max(Spec2D)));%���ֵ������
        result(isrc,1) = azimuth(1)+gridRes*(y-1);
        result(isrc,2) = elevation(1)+gridRes*(x-1);
        %% minAngle��Ϊ��Сֵ����Ѱ��
        Spec2D(x-floor(minAngle/2/gridRes):x+floor(minAngle/2/gridRes),y-floor(minAngle/2/gridRes):y+floor(minAngle/2/gridRes))=-inf;
    end
end
end