%[img,~] = loadimg('Original/');




avg_scores = readtable('avg.csv');
med_scores = readtable('med.csv');
cnn_scores = readtable('cnn.csv');
stack_scores = readtable('stack.csv');

%[noised,~] = loadimg('Noised/');
%[avg,~] = loadimg('Denoised/avg/');
%[cnn,~] = loadimg('Denoised/dncnn/');
%[med,~] = loadimg('Denoised/med/');
