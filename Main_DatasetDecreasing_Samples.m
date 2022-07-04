ProjectFolder='D:\CowBhave\';
DataFolder=[ProjectFolder 'Labeled25\'];
SaveDataFolder=[ProjectFolder ''];
WS=60; Freq=25;%Hz 
NS=WS*Freq;
LabeledSamplesN=9;

%% Labeled samples (classification window), separated
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
        n=length(L);
        w=[];
        f=1;
        
        for cl=1:3%classes
            i=1;
            bn=0;
            while bn<LabeledSamplesN && i<n
                while L(i)~=cl && i<n
                    i=i+1;
                end
                sn=0;
                while L(i+sn)==cl && i+sn<n && sn<NS
                    sn=sn+1;
                end
                if sn==NS
                    bn=bn+1;
                    w=[w; i+(0:sn-1)'];
                end
                i=i+sn;
            end
            if bn<LabeledSamplesN
                f=0;
            end
        end
        if f==1
            X1=X(w);
            Y1=Y(w);
            Z1=Z(w);
            L1=L(w);
            DateStr1=DateStr(w);
            T1=table(DateStr1,X1,Y1,Z1,L1);
            FileName1=[SaveDataFolder 'Labeled25_' num2str(WS) 'B' num2str(LabeledSamplesN) '\' FileName];
            writetable(T1,FileName1,'WriteVariableNames',false,'Delimiter',';');
        else
            disp(['Missing class ' FileName]);
        end
    end
end

load gong.mat;
sound(y);
