function [output]=addnoise(img)



output{1,length(img)} = {};

for i=1:length(img)
    output(i) = {imnoise(img{i},'gaussian')};
end






end
