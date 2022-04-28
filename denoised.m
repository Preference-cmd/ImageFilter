function [output] = denoised(img, core, step, mode, weights,isGPU)

h = fspecial('average',core);

if isGPU
    imgs{1,length(img)} = {};
    for i=1:length(img)
        imgs(i) = img(i);
    end
end

output{1,length(img)} = {};
if isGPU
    for i=1:length(img)
         output(i) = {convCoreGPU(imgs{i},mode,h)};
    end
else
    for i=1:length(img)
         output(i) = {convCore(img{i},core,step,mode,weights)};
    end
end



end