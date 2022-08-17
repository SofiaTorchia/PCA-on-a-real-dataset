
clc;
close all;
clear;

file_name = 'Bike-Sharing-Dataset/day.csv';
num_col = 15;
data = readtable(file_name);
display(data)

% 1) Indice giornaliero;
% 2) Data;
% 3) Stagione (1 Inverno, 2 Primavera, 3 Estate, 4 Autunno);
% 4) Anno (0 per il 2011, 1 per il 2012);
% 5) Mese dell’anno (da 1 a 12);
% 6) Giorni festivi americani (0 No, 1 Si);
% 7) Giorno della settimana (da 0 a 6);
% 8) Tipo di giorno (0 Feriale, 1 Festivo);
% 9) Condizioni atmosferiche
% 10) Temperatura effettiva;
% 11) Temperatura percepita;
% 12) Umidit`a dell’aria;
% 13) Velocit`a del vento;
% 14) Noleggio da parte di utenti non registrati;
% 15) Noleggio da parte di utenti registrati;
% 16) Noleggi totali 

data = removevars(data,{'dteday'}); % Eliminiamo dall'analisi la colonna "dteday"
data = table2array(data);
stand_data = zscore(data); % Normalizzo i dati

corr_matrix = corrcoef(stand_data); % Matrice di correlazione 
figure(9)
    heatmap(corr_matrix)


figure(1) % Correlazioni più significative con la colonna d_15
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
fprintf('Autovalori della matrice di correlazione: \n')
display(eig_values)



% Cerco un numero di autovalori K sufficiente a spiegare l'80-90% della varianza
cum_sum = cumsum(eig_values);
proportions = [];
for i=1:num_col
    proportions(i) = cum_sum(i)/cum_sum(num_col);
end

figure(2) %distribuzione della varianza
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
    txt = 'Varianza totale';
    text(0.5,1.06,txt)

    txt = '90% varianza totale';
    text(2.5,0.96,txt)

    txt = '80% varianza totale';
    text(10.5,0.86,txt)
    legend('Distribuzione autovalori', 'Somme cumulate normalizzate','Varianza Totale',...
    '90% varianza totale','80% varianza totale');
    title('Troncamento autovalori')
    xlabel('Numero di autovalori')
    ylabel('Autovalori')

% Se X è la matrice dei dati, quello che voglio fare è trovare v tale che Xv
% abbia varianza massima. v è dato dal primo autovalore della matrice di correlazione
fprintf('\nPrimo autovettore: \n')
display(V(:,num_col))
fprintf('\nSecondo autovettore: \n')
display(V(:,num_col-1))

figure(3) %rilevanza feature per le prime due componenti principali
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
    xlabel('Primo autovettore')
    ylabel('Secondo autovettore')
    axis equal


% Studiamo le correlazioni tra le colonne d1,...,d15 e y1 = X*v1 (prima CP)
% y1 = v1,1*d1 + v1,2*d2 .. + v1,15*d15 dove i pesi v1,j sono gli elementi
% del primo autovettore di R (matrice di correlazione).
r1 = zeros(1,num_col);
y1 = stand_data*V(:,num_col);  % y1 = X*v1
for i=1:num_col
    r1(i) = corr(y1,stand_data(:,i)); %corr tra y1 e le feature
end

% Stessa cosa per la seconda componente principale
r2 = zeros(1,num_col);
y2 = stand_data*V(:,num_col-1);  % y2 = X*v2
for i=1:15
    r2(i) = corr(y2,stand_data(:,i));
end

figure(4) %grafico di r1 e v1
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
        legend('Correlazione tra y_1 e d_1,...,d_{15}','Primo autovettore v_1')
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
        legend('Correlazione tra y_2 e d_1,...,d_{15}','Secondo autovettore v_2')



% Voglio capire quali colonne influiscono maggiormente sui primi sei
% autovettori contemporaneamente (e quindi sulle prime sei componenti
% principali
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
    xlabel('Colonne del dataset','FontSize',14)
    ylabel('Importanza','FontSize',14)
    legend('v1','v2','v3','v4','v5','v6')

    nexttile
    bar(s1,'stacked')
    xlabel('Colonne del dataset','FontSize',14)
    ylabel('Importanza pesata','FontSize',14)
    legend('v1','v2','v3','v4','v5','v6')
 


% Studio del diagramma di dispersione 
figure(6) %diagramma di dispersione tra y1 e le feature
tiledlayout(3,5)
for i=1:num_col
    nexttile
    plot(stand_data(:,i),y1,'o');
end


% Grafico per lo studio di eventi anomali
% Sono stati individuati nel grafico vari punti classificabili come
% "anomali". Studiamo uno di questi (giorno 26).
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
    xlabel('Prima componente principale y1')
    ylabel('Seconda componente principale y2')



% Ora guardiamo cosa succede al giorno 26:
% Consideriamo la media delle 15 variabili nei giorni {1..50}.
% Confrontiamo il giorno 26 con l'andamento "stagionale" di quel periodo.
figure(8)
    plot(stand_data(26,:),'LineWidth',3)
    hold on
    plot(mean(stand_data(1:50,:)),'LineWidth',3)
    legend('Giorno 26','Andamento stagionale')
    xlim([1 15])
    xlabel('Colonne di X','FontSize',15) 














