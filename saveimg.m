function isComplete = saveimg(img,imgDir,path)

for i=1:length(img)
    imwrite(img{i},strcat(path,imgDir(i).name));
end

isComplete = true;

end