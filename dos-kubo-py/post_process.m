#!/usr/bin/octave  -qf

fname=(argv () {1});
%clear all
%clc
dat=load(fname);
%dat=res_512_3;
wlist=linspace(0.0000001,2,length(dat));
for w=wlist
disp([w,(exp(-w/0.01)-1)/w*sum(exp(-0.0001*dat(:,1).^2).*sin(dat(:,1)*w)*2.*dat(:,3))])
end