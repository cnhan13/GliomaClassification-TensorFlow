clc;
workspace;
format longg;
format compact;

% Define a starting folder
start_path = fullfile('/home/nhan/Documents/dl/ChiNhan_2016/mha/BRATS2014_training');
topLevelFolder = uigetdir(start_path);
if topLevelFolder == 0
    return;
end

% Get list of all subfolders
allSubFolders = genpath(topLevelFolder);
% Parse into a cell array
remain = allSubFolders;
listOfFolderNames = {};
while true
    [singleSubFolder, remain] = strtok(remain, ':');
    if isempty(singleSubFolder)
        break;
    end
    listOfFolderNames = [listOfFolderNames singleSubFolder];
end
numberOfFolders = length(listOfFolderNames)

% Process all *.mha files in those folders
for i=1:numberOfFolders
    % Get this folder and print it out
    thisFolder = listOfFolderNames{i};
    fprintf('Processing folder %s\n', thisFolder);
    
    % Get *.mha files
    filePattern = sprintf('%s/*.mha', thisFolder);
    baseFileNames = dir(filePattern);
    
    % regexp(thisFolder, '[/.]', 'split');
    folderDepth = strsplit(thisFolder,'/');
    if length(folderDepth) == 11 % including '' at front
        % input file
        length(baseFileNames)
        infileName = fullfile(thisFolder, baseFileNames(1).name);
        fprintf('\tProcessing file %s\n',infileName);
        [V,~] = ReadData3D(infileName);
        V = normalize(V);
        
        % ouput file
        outfolderName = folderDepth(end-1);
        containingFolder = strsplit(char(folderDepth(end)),'.');
        outfileExt = containingFolder(end-1);
        outfileName = fullfile(thisFolder,'..','..',strcat(outfolderName,'_',outfileExt,'.mat'));
        save(char(outfileName),'V');
    end
end