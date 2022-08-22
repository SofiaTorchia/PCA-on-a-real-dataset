
clc;
close all;
clear;

file_name = 'day.csv';
num_col = 15;
data = readtable(file_name);
display(data)

data = removevars(data,{'dteday'}); % Discard column "dteday"
data = table2array(data);
stand_data = zscore(data); % Standardize date

corr_matrix = corrcoef(stand_data); %Correlation matrix
figure(9)
    heatmap(corr_matrix)


figure(1) % Details on features correlated with d15
    tiledlayout(2,2)
    nexttile
    plot(stand_data(:,15),stand_data(:,14),'o')
    xlabel('Variabile 14','FontSize',14);
    ylabel('Variabile 15','FontSize',14);
    nexttile
    plot(stand_data(:,15),stand_data(:,9),'o')
    xlabel('Variabile 9','FontSize',14);
    ylabel('Variabile 15','FontSize',14);
    nexttile
    plot(stand_data(:,15),stand_data(:,12),'o')
    xlabel('Variabile 12','FontSize',14);
    ylabel('Variabile 15','FontSize',14);
    nexttile
    plot(stand_data(:,15),stand_data(:,11),'o')
    xlabel('Variabile 11','FontSize',14);
    ylabel('Variabile 15','FontSize',14);
   

[V,D] = eig(corr_matrix);
eig_values = flip(diag(D));
fprintf('Eigenvalues for the correlation matrix: \n')
display(eig_values)



cum_sum = cumsum(eig_values);
proportions = [];
for i=1:num_col
    proportions(i) = cum_sum(i)/cum_sum(num_col);
end

figure(2) 
    plot((0:num_col-1),eig_values,'-o','LineWidth',2)
    hold on 
    plot((0:num_col-1),proportions,'-o','LineWidth',2,'Color','r')
    hold on
    yline(1,'k')
    hold on
    yline(0.9,'m')
    hold on 
    yline(0.8,'g')
    hold on
    xline(5)
    hold on 
    xline(6)

    ylim([0 5])
    txt = 'Total variance';
    text(0.5,1.06,txt)

    txt = '90% of the total variance';
    text(2.5,0.96,txt)

    txt = '80% of the total variance';
    text(10.5,0.86,txt)
    legend('Eigenvalues distribution', 'Normalized cumulated sums', ...
        'Total variance',...
    '90% of the total variance','80% of the total variance','Fontsize',12);
    title('Eigenvalues truncation','FontSize',15)
    %xlabel('Number of eigenvalues')
    ylabel('Eigenvalues','FontSize',15)

fprintf('\nFist eigenvector: \n')
display(V(:,num_col))
fprintf('\nSecond eigenvector: \n')
display(V(:,num_col-1))

figure(3) %feature relevance for y1 and y2
    plot(V(:,num_col),V(:,num_col-1),'x','LineWidth',2.5,'Color','r')
    hold on 
    xline(0)
    hold on
    yline(0)
    hold on

    ang=0:0.01:2*pi; 
    xp=cos(ang);
    yp=sin(ang);
    plot(xp,yp,'Color','k');

    for i=1:num_col
        txt = ['d', int2str(i)];
        text(V(i,num_col)-0.05,V(i,num_col-1)+0.03,txt)
    end

    xlim([-1.0 1.0])
    ylim([-1.0 1.0])
    xlabel('Fist eigenvector','FontSize',15)
    ylabel('Second eigenvector','FontSize',15)
    axis equal


% We analyse now the correlations between columns d1,...,d15 and y1 = X*v1
% y1 = v1,1*d1 + v1,2*d2 .. + v1,15*d15 where the weights v1,j are the
% elements of the first eigenvector of R.
r1 = zeros(1,num_col);
y1 = stand_data*V(:,num_col);  % y1 = X*v1
for i=1:num_col
    r1(i) = corr(y1,stand_data(:,i)); %corr between y1 and the features
end

% Same thing for y2
r2 = zeros(1,num_col);
y2 = stand_data*V(:,num_col-1);  % y2 = X*v2
for i=1:15
    r2(i) = corr(y2,stand_data(:,i));
end

figure(4) 
    tiledlayout(1,2)
    nexttile
        plot((1:num_col),r1','-*','LineWidth',2)
        hold on
        plot((1:num_col),V(:,num_col),'-*','LineWidth',2)
        hold on 
        yline(0)
        hold on 
        yline(-0.5)
        hold on 
        yline(0.5)
        axis([1 15 -1 1])
        legend('Correlation between y_1 and d_1,...,d_{15}', ...
            'First eigenvector v_1','Fontsize',13)
        xlabel('Features','FontSize',13)
        ylabel('Correlation','FontSize',13)
     nexttile %grafico di r2 e v2
        plot((1:num_col),r2','-*','LineWidth',2)
        hold on
        plot((1:num_col),V(:,num_col-1),'-*','LineWidth',2)
        hold on
        yline(0)
        hold on 
        yline(-0.5)
        hold on 
        yline(0.5)
        axis([1 15 -1 1])
        legend('Correlation between y_2 and d_1,...,d_{15}', ...
            'Second eigenvector v_2','Fontsize',13)
        xlabel('Features','FontSize',13)
        ylabel('Correlation','FontSize',13)



% We analyse feature relevance for y1...y6
s = [];
s1 = [];
for i=0:5
    lambda = D(15-i,15-i);
    s = [s,abs(V(:,15-i))];
    s1 = [s1,lambda*abs(V(:,15-i))];
end
figure(5)
    nexttile
    bar(s,'stacked')
    xlabel('Dataset columns','FontSize',14)
    ylabel('Relevance','FontSize',14)
    legend('v1','v2','v3','v4','v5','v6')

    nexttile
    bar(s1,'stacked')
    xlabel('Dataset columns','FontSize',14)
    ylabel('Weighted relevance','FontSize',14)
    legend('v1','v2','v3','v4','v5','v6')
 


% Scatter plots for y1 and the 15 features
figure(6) 
tiledlayout(3,5)
for i=1:num_col
    nexttile
    plot(stand_data(:,i),y1,'o');
    title(['Pict.',int2str(i)],'FontSize',10)
end


% Anomaly detection and examination of day 26.
figure(7)
    plot(y1,y2,'o','MarkerSize',3,'LineWidth',3)
    hold on
    xline(0)
    hold on 
    yline(0)

    j = 26;
    hold on
    plot(y1(j),y2(j),'x','LineWidth',4,'Color','r')
    txt = 'day26';
    text(y1(j)-0.4,y2(j)+0.3,txt,'FontSize',12)

    j = 90;
    hold on
    plot(y1(j),y2(j),'x','LineWidth',4,'Color','r')
    j = 463;
    hold on
    plot(y1(j),y2(j),'x','LineWidth',4,'Color','r')
    j = 302;
    hold on
    plot(y1(j),y2(j),'x','LineWidth',4,'Color','r')
    xlabel('First principal component y1','FontSize',13)
    ylabel('Second principal component y2','FontSize',13)
    title('Anomaly detection','FontSize',13)


figure(8)
    plot(stand_data(26,:),'LineWidth',3)
    hold on
    plot(mean(stand_data(1:50,:)),'LineWidth',3)
    legend('Day 26','Seasonal trend','Fontsize',13)
    xlim([1 15])
    xlabel('Columns','FontSize',15) 














