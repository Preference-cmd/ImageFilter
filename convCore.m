function output=convCore(data, filter_size, step, mode, weights)

% data：图像数据对应的数组 
% filter_size：卷积核的大小
% step：平移的步长
% weights：卷积核内的权重，通常为一个n维数组或者服从某种概率分布的权重集合，非全1矩阵时可能需要进行归一化

[n,m,d] = size(data); % 获取图像维度信息
edge = floor(filter_size/2); % 边缘到中心点的距离
pole = [edge+1,edge+1, 1]; % 中心点初始位置
height = ((n-2*edge) / step)-1; % 纵向移动次数
width = ((m-2*edge) / step)-1; % 横向移动次数
datax = data(:,:,:); % 复制图像数组

if mode==1

    for h=0:height % 卷积核依次向下滑动
        for w=0:width % 卷积核依次向右滑动
            values = data(pole(1)-edge:pole(1)+edge,pole(2)-edge:pole(2)+edge,:);
            conv_values = values.*weights;
            rep_values = mean(conv_values,"all");
            datax(pole(1),pole(2)) = rep_values;
            pole(2) = pole(2) + step;
        end
        pole(2) = edge+1;
        pole(1) = pole(1) + step;
    end

else

    for h=0:height % 卷积核依次向下滑动
        for w=0:width % 卷积核依次向右滑动
            values = data(pole(1)-edge:pole(1)+edge,pole(2)-edge:pole(2)+edge,:);
            conv_values = values.*weights;
            rep_values = median(conv_values, "all");
            datax(pole(1),pole(2)) = rep_values;
            pole(2) = pole(2) + step;
        end
        pole(2) = edge+1;
        pole(1) = pole(1) + step;
    end
    
end



%end

output = datax;
end