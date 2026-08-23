% I=imread('C:\Users\SUS\Desktop\Final Image DB\Temp\OriginalImage.jpg');
singleimagedir='C:\Users\SUS\Desktop\Final Image DB\Temp';
d=dir(strcat(singleimagedir,'\*.jpeg'));
for z = 1:length(d)
X=imread('C:\Users\SUS\Downloads\I_2(1).JPG');
[rows, columns]=size(I);
J = imnoise(I,'gaussian');
K = imnoise(I(:,:,1),'salt & pepper');
L = imnoise(I(:,:,2),'speckle');
M = imnoise(I,'poisson');

% subplot(2,2,1),imshow(J); title('image with gaussian noise');
% subplot(2,2,2),imshow(K); title('image with salt & pepper noise');
% subplot(2,2,3),imshow(L); title('image with speckle Noise');
% subplot(2,2,4),imshow(M); title('image with poisson noise');

%Kmedian = medfilt2(I);
% K = wiener2(img1,[5 5]);
P=I;
Q=I(:,:,1);
R=I(:,:,2);

Resize=imresize(X,[1024,1024]);
S=Resize;
% subplot(2,2,1),imshow(P); title('image after  Guassian Filter');
% subplot(2,2,2),imshow(Q); title('image after Median Filter');
% subplot(2,2,3),imshow(R); title('image after Weiner filter');
% subplot(2,2,4),imshow(S); title('image after NL filter');
%  P1= psnr(intlut(J,P),J); 
%  P2= psnr(intlut(K,Q),K); 
%   P3= psnr(intlut(L,R),L); 
%    P4= psnr(intlut(M,S),M); 
            for i = 1:1
            [peaksnr1, snr1] = psnr(double(J), double(P));
            [peaksnr2, snr2] = psnr(double(K), double(Q));
            [peaksnr3, snr3] = psnr(double(L), double(R));
            [peaksnr4, snr4] = psnr(double(M), double(S));


% message = sprintf('\nThe SNR1 = %.2f.\nThe PSNR1 = %.2f.\nThe SNR2 = %.2f.\nThe PSNR2 = %.2f.\nThe SNR3 = %.2f.\nThe PSNR3 = %.2f.\nThe SNR4 = %.2f.\nThe PSNR4 = %.2f',snr1,peaksnr1,snr2,peaksnr2,snr3,peaksnr3,snr4,peaksnr4);
% msgbox(message);
            S1=ssim(J,P);
            S2=ssim(K,Q);
            S3=ssim(L,R);
            S4=ssim(M,S);
% message = sprintf('\nThe value of S1 = %.2f.The Value of S2 %.2f.\nThe value of S3 = %.2f.The Value of S4 %.2f.',S1,S2,S3,S4);
% msgbox(message);
            t1=sum(sum(P));
            mse1 = t1 / (rows * columns);

            t2=sum(sum(Q));
            mse2 = t2 / (rows * columns);

            t3=sum(sum(R));
            mse3 = t3 / (rows * columns);

            t4=sum(sum(S));
            mse4 = t4/ (rows * columns);
% message = sprintf('\nThe value of MSE1 = %.2f.The Value of MSE2 %.2f.\nThe value of MSE3 = %.2f.The Value of MSE4 %.2f.',mse1,mse2,mse3,mse4);
% msgbox(message)
            end
            querymat(z,:)=[peaksnr1(1,1:1),peaksnr2(1,1:1),peaksnr3(1,1:1),peaksnr4(1,1:1)];
end
A = {'PSNR1','PSNR2','PSNR3','PSNR4'};
 xlswrite('C:\Users\SUS\Desktop\Final Image DB\Temp\shilalekhfeature.xlsx',A);
 xlRange = 'A2:B2:C2:D2:A10:B10:C10:D10';
xlswrite('C:\Users\SUS\Desktop\Final Image DB\Temp\shilalekhfeature.xlsx',querymat,xlRange);
