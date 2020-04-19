clc; clear all; close all
fs=1200;
addpath('C:\Users\USER\Downloads\Thesis data\g.tec')
addpath('C:\Users\USER\Downloads\Thesis data\polhemus')
addpath('C:\Users\USER\Downloads\Thesis data\ArmAble')
%arm_position=readmatrix('
polhemus_position=readmatrix('AB_14_G1_0 - Report1.txt');
A= polhemus_position(:,3)*100;
B= polhemus_position(:,4)*100;
C= polhemus_position(:,5)*100;
X = smoothdata(polhemus_position(:,3)*100,'gaussian',200);
Y = smoothdata(polhemus_position(:,4)*100,'gaussian',200);
Z = smoothdata(polhemus_position(:,5)*100,'gaussian',200);
k = smoothdata(polhemus_position(:,3)*100,'gaussian',1200);

% A_std = std(A);
% 
%  ANorm=normalize(A);
% ANorm1=(A+A_std);
% Amean=mean(A);
% Avg=(A-mean(A));



t = 0:1/fs:length(X)/fs -1/fs;
feq= 1./t;
%t= polhemus_position(:,1);
% pos = rand(100,1) ;
% t = 1:100 ;
p = abs(fft(A(1:end-1)));


v1 = zeros(length(t)-1,1) ;
for i = 1:length(t)-1
    v1(i) = (X(i+1)-X(i))/(t(i+1)-t(i)) ;
end
v2 = zeros(length(t)-1,1) ;
for i = 1:length(t)-1
    v2(i) = (Y(i+1)-Y(i))/(t(i+1)-t(i)) ;
end
v3 = zeros(length(t)-1,1) ;
for i = 1:length(t)-1
    v3(i) = (Z(i+1)-Z(i))/(t(i+1)-t(i)) ;
end
figure,plot(t(1:end-1),v1')
title('velocity')
xlabel ('(TIME(s))');
ylabel ('(m/s)');
figure,plot(t(1:end-1),v2')
title('velocity')
xlabel ('(TIME(s))');
ylabel ('(m/s)');
figure,plot(t(1:end-1),v3')
title('velocity')
xlabel ('(TIME(s))');
ylabel ('(cm/s)');
figure,plot3(v1,v2,v3)
title('velocity 3-D')
xlabel('X-axis (cm/s)')
ylabel('Y-axis (cm/s)')
zlabel('Z-axis (cm/s)')
figure, plot(feq(1:end-1),p')
v1mean=mean(abs(v1));
v2mean=mean(abs(v2));
v3mean=mean(abs(v3));
% KX=zeros(length(X),1);
% for i=1:length(X)
%     KX(i) = (X(i,:));
%      if ((KX(i)>= -1) && (KX(i)<= 1))
%         disp(KX(i));
%     else
%         disp(median(KX(i++) ,KX( i+));
%      end     
% end

% z=diff(v1);
% figure,plot(z)
% k= gradient(v1);
% figure, plot(k)


