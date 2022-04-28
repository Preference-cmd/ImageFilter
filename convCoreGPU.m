function output=convCoreGPU(data,mode,h)


if mode==1
    output = imfilter(data,h);
else
    output = cat(3,medfilt2(data(:,:,1)),medfilt2(data(:,:,2)),medfilt2(data(:,:,3)));
end