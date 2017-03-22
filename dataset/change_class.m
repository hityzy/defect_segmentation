%%change class
file_path = pwd;
% files = dir([pwd , '/*.png']);
files = dir([pwd ]);
files(1,:) = [];
files(1,:) = [];
files(end,:)=[];
for i = 1:size(files,1)
oldname = files(i).name;
if strcmp(oldname , 'change_class.m')
    continue;
end
newname = oldname;
newname(6) = '9';
movefile(fullfile(pwd , oldname) , fullfile(pwd , newname))
end