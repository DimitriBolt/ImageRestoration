%----------------------------------------------------------------
% File:     basic_script.m
%----------------------------------------------------------------
%
% Author:   Marek Rychlik (rychlik@arizona.edu)
% Date:     Mon Apr  7 14:19:32 2025
% Copying:  (C) Marek Rychlik, 2020. All rights reserved.
% 
%----------------------------------------------------------------
% Basic operations on raw images
% Source image: https://www.reddit.com/r/EditMyRaw/comments/1jt4ecw/the_official_weekly_raw_editing_challenge/
raw_img_filename=fullfile('.','images','DSC00099.ARW')
%raw_img_filename=fullfile('.','images','credit @signatureeditsco - signatureedits.com _DSC4583.dng')
t=tiledlayout('TileSpacing','compact','Padding','compact');
linked_axes=[]
cfa=rawread(raw_img_filename);
ax=nexttile, imagesc(bitand(cfa,  7)),title('xor with 7');
info = rawinfo(raw_img_filename);
disp(info);
Iplanar=raw2planar(cfa);
planes='rggb';
ang = info.ImageSizeInfo.ImageRotation;
for j=1:size(Iplanar,3)
    ax=nexttile; imagesc(imrotate(Iplanar(:,:,j),ang)), title(['Plane ',num2str(j),': ', planes(j)]), colorbar, colormap gray
    linked_axes=[linked_axes,ax];
end
Idemosaic=demosaic(cfa,planes);
% nexttile, image(Idemosaic), colorbar;
Irgb=raw2rgb(raw_img_filename);
% Resized RGB image, so that size matches the size of the planes
ax=nexttile; image(imresize(Irgb,.5)), colorbar,
title('Demosaiced and scaled RGB image');
linked_axes=[linked_axes,ax];
linkaxes(linked_axes);
