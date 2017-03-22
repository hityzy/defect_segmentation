file_path = pwd;
files = dir(pwd);
files(1,:) = [];
files(1,:) = [];
files(end,:)=[];
for i = 1:size(files,1)
oldname = files(i).name;
newname = ['class2_def-' , oldname];
movefile(fullfile(pwd , oldname) , fullfile(pwd , newname))
end