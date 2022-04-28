function [output] = scores(img,ref_img)

% img&ref_img: expected cell structure: {{X1},{X2},...}
%                        where X are 3D-array of images

len = length(img);

output = randn(len,3);

for i=1:len
    output(i,:) = [psnr(img{i},ref_img{i}),mean(multissim(img{i},ref_img{i})), niqe(img{i})];
end

output = array2table(output,"VariableNames",{'psnr','ssim','niqe'});


end
