filename = 'data_Tokyo2.csv';

M = readmatrix(filename);
[sz,~]=size(M)
temprature = M(:,2);%2列目のみ取り出す気温のデータ
delimiterIn = ' ';
headerlinesIn = sz;%読みとる行
A = importdata(filename,delimiterIn,headerlinesIn);


detrend_sdata =detrend(temprature);%トレンドのける
detrend_sdata=detrend_sdata - mean(detrend_sdata)
x=detrend_sdata
wname='amor'%morlet = ガボール


[wt,f] = cwt(x,wname,1/(60*60)); %サンプリング周波数が1hour = 1/3600 Hz
%cwt(x,wname,1/(60*60));
%{
% hour単位で行うもの
D = duration(1,0,0)% D = duration(H,MI,S)
[wt,f] = cwt(x,wname,D);%hour単位
%cwt(x,wname,D);
%}
t = 0:sz-1



plot(t,temprature)
xlabel('Time (year)')
xlim([0,sz])
xticks([0 2256 4440 6660 sz])
xticklabels({ '2021/4/1', '2021/7/4','2021/10/1','2022/1/1','2022/4/1'})
ylabel('Temprature[℃]')
title('Raw Temprature Data')
saveas(gcf,'raw_data.png')
hold off


plt1=plot(t,temprature,'DisplayName','raw data')
hold on
%plt2=plot(t,x)
plt2=plot(t,x,'DisplayName','corrected data')
%legend([plt1,plt2],["Raw data","corrected_data"])
%legend(["Raw data","corrected_data"])
xlabel('Time (year)')
xlim([0,sz])
xticks([0 2256 4440 6660 sz])
xticklabels({ '2021/4/1', '2021/7/4','2021/10/1','2022/1/1','2022/4/1'})
ylabel('Temprature[℃]')
title('Temprature Data')
legend
saveas(gcf,'raw_and_corrected_data.png')
hold off



figure,surface(t,f,abs(wt))
axis tight, shading flat
xlabel('Time (hour)')
ylabel('Frequency (Hz)')
%xticks([0 2256 4440 6660 sz])
%xticklabels({ '2021/4/1', '2021/7/4','2021/10/1','2022/1/1','2022/4/1'})
set(gca,'yscale','log')
colorbar
saveas(gcf,'temprature_tokyo2')
saveas(gcf,'wavelet_temprature_tokyo_one_year2.png')



