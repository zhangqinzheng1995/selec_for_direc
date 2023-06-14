
fs=16e3;
% load SPP_n.mat
s = clean_DRR_duicheng;



jiequshijian = 2;

[flen,tlen] = size(s);
t = jiequshijian/tlen:jiequshijian/tlen:jiequshijian;
t=t';
f = (fs/2)/flen:(fs/2)/flen:(fs/2);
f=f';


% mesh(t,f/1000,s);

mesh(1:129,1:249,s);
view(2);
colorbar();
clim([-20 -10])
%%%%%%%%
