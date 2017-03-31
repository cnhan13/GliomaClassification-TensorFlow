function OutImg = normalize(img, OutputMin, OutputMax)
% This function can be used to normalize the input image of any size and dimension to specific range defined by user

%Inputs:
%       img:  is the image to be normalized
%       OutputMin: is the minimum graylevel required in the output image, default is 0
%       OutputMax: is the maximum graylevel required in the output image, default is 255
%Output:
%       OutImg: is the output image with the range [OutputMin OutputMax]

% Created 21/3/2014 by Ahmed Elazab, PhD candidate, Shenzhen Institutes of Advanced Technology, Chinese Academy of Sciences

if (nargin < 2)
    OutputMin=0;
    OutputMax=255;
    if (nargin < 1)
        error('No enough inputs !');
        return
    end
end

img=double(img);
mx=max(img(:));
mn=min(img(:));
OutImg=round((OutputMax - OutputMin)*((img-mn)/(mx-mn)));
end

