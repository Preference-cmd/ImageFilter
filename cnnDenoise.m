function [output] = cnnDenoise(img,net)


output{1,length(img)} = {};
for i=1:length(img)
     output(i) = {cnnDe(img{i},net)};
end

end