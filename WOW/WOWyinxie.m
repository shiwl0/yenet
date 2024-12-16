payload = single(0.4);% set payload
params.p = -1;  % set params   holder norm parameter
file_path =  '/Users/shiwenlve/Downloads/new_BOSSbase_1.01/';% 图像文件夹路径
img_path_list = dir(strcat(file_path,'*.pgm'));%获取该文件夹中所有.pgm格式的图像
img_num = length(img_path_list);%获取图像总数
if img_num > 0 %有满足条件的图像
        for pn = 1:img_num %逐一读取图像
            image_name = img_path_list(pn).name;% 图像名
            cover =  imread(strcat(file_path,image_name));%读取图像
            fprintf('\n\n%d %s\n',pn,strcat(file_path,image_name));% 显示正在处理的图像名
            MEXstart = tic;
            [stego, distortion] = WOW(cover, payload, params);% Run embedding simulation
            MEXend = toc(MEXstart);
            fprintf('\n\nImage embedded in %.2f seconds, change rate: %.4f, distortion per pixel: %.6f\n', MEXend, sum(cover(:)~=stego(:))/numel(cover), distortion/numel(cover));
            imwrite(uint8(stego),['/Users/shiwenlve/Downloads/W_BOSSbase_1.01/',image_name])%保存图像
        end
end