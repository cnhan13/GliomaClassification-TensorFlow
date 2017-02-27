function demo()
[V,info] = ReadData3D('file.mha');
save('file.mat','V');

imwrite(V(:,:,75),'slice75.jpg');
end


function outimg = normalize(img)
outmin=0;
outmax=255;
img=double(img);
mx=max(img(:));
mn=min(img(:));
outimg = round((outmax - outmin)*((img-mn)/(mx-mn)));
end
