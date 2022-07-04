ProjectFolder='D:\CowBhave\';
DataFolder=[ProjectFolder 'Labeled25\'];
SaveDataFolder=[ProjectFolder ''];
P=10:10:90;
WS=60; Freq=25;%Hz 
MinimalClassN=WS*Freq*10;

% Percentage, continious
FileList=dir(DataFolder);
for File_i=1:1:length(FileList)
    FileName=FileList(File_i).name;
    if contains(FileName,'AccDataLabeled_Tag') && contains(FileName,'.csv')
        disp([num2str(File_i) '/' num2str(length(FileList)) ', ' FileName]);
        T=readtable([DataFolder FileName],'Delimiter',';','ReadRowNames',false);
        DateStr=table2array(T(:,1));
        X=table2array(T(:,2));
        Y=table2array(T(:,3));
        Z=table2array(T(:,4));
        L=table2array(T(:,5));
        for i=P
            a=1; b=round(length(X)*i/100);
            L1=L(a:b);
            N1=sum(L1==1);
            N2=sum(L1==2);
            N3=sum(L1==3);
            while b<length(X)-1000 && (N1<MinimalClassN || N2<MinimalClassN || N3<MinimalClassN)
                a=a+1000; b=b+1000;
                L1=L(a:b);
                N1=sum(L1==1);
                N2=sum(L1==2);
                N3=sum(L1==3);
            end
            DateStr1=DateStr(a:b);
            X1=X(a:b);
            Y1=Y(a:b);
            Z1=Z(a:b);
            if N1>ClassN && N2>ClassN && N3>ClassN
                T1=table(DateStr1,X1,Y1,Z1,L1);
                FileName1=[SaveDataFolder 'Labeled25_' num2str(i) '\' FileName];
                writetable(T1,FileName1,'WriteVariableNames',false,'Delimiter',';');
            else
                disp(['Missing class ' FileName]);
            end
        end
    end
end

load gong.mat;
sound(y);
