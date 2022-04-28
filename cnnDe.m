function output = cnnDe(data,net)

R = denoiseImage(data(:,:,1),net);
G = denoiseImage(data(:,:,2),net);
B = denoiseImage(data(:,:,3),net);

output =cat(3,R,G,B);


end