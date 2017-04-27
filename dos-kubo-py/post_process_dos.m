#!/usr/bin/octave  -qf

fname=(argv () {1});
%clear all
%clc
dat=load(fname);
dat=[-dat(end:-1:2,1),dat(end:-1:2,2),-dat(end:-1:2,3);dat]
%dat=[-dat(end:-1:2,1),dat(end:-1:2,2);dat];
res=dat;
d=res(:,2)+1.0i*res(:,3);
%d=res(:,1)+1.0i*res(:,2);
t=dat(:,1);
%t=[1:length(dat)];
d=d.*hanning(length(d)).^1;

t_max=max(t);
dt=abs(t(1)-t(2));
nf=1.0;
fft_d=fftshift(fft(ifftshift(d)))/nf*dt/pi
%fft_d=fft(d)/nf*dt/pi;
%w=[-pi/dt:(pi/t_max):(pi/dt)]*4;
w=linspace(-nf*pi/dt,nf*pi/dt,length(fft_d))';

%plot(w,abs(fft_d), w,imag(fft_d),'+', w,real(fft_d),'o')
%semilogy(w,abs(fft_d), w,imag(fft_d),'+', w,real(fft_d),'o')
%plot(w,abs(fft_d))
%ylim([0,1])
%xlim([-4.2,4.2])
%grid on
format long e
disp([w,real(fft_d),imag(fft_d)])
