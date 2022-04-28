function [img,imgList] = loadimg(path)

% path: e.g "orginal/"

imgPath = path;
imgDir = dir([imgPath '*.jpg']);
img{1,length(imgDir)} = {};
for i=1:length(imgDir)
    img(i) = {imread(strcat(imgPath,imgDir(i).name))};
end

imgList = imgDir;

end