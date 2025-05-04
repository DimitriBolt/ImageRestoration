% --- Prerequisites (Run basic_script.m first) ---
run("basic_script")
% Make sure Iplanar, ang, etc. are in your workspace.

% --- Define Parameters ---
lambda_vec = logspace(-1, 2, 15); % Adjust N values as needed
epsilon_vec = logspace(-4, -1, 10); % Adjust M values as needed

% --- Process the Red Plane (Plane 1) ---
fprintf('Processing Red Plane...\n');
f_r_uint16 = Iplanar(:,:,1); % Extract the Red plane (likely uint16)
f_r = im2single(f_r_uint16); % Convert to single precision float (or use im2double)
% Optional: Rotate if needed, although ROF is rotation invariant
% f_r = imrotate(f_r, ang);

% Calculate MSD for the Red plane over the parameter grid
msd_r = calculate_msd(f_r, lambda_vec, epsilon_vec);
% msd_r will be a 15x10 matrix (size KxL)

% Optionally, get the smoothed images for the Red plane
% This might consume a lot of memory if K*L is large!
% u_r_all = smooth_image_rof(f_r, lambda_vec, epsilon_vec);
% u_r_all would be a [H, W, 15, 10] array

% Example: Get smoothed image for specific parameters (e.g., 5th lambda, 3rd epsilon)
lambda_scalar = lambda_vec(5);
epsilon_scalar = epsilon_vec(3);
u_r_specific = smooth_image_rof(f_r, lambda_scalar, epsilon_scalar);
% u_r_specific would be an [H, W] image

% --- Display results ---
figure;
[LL, EE] = meshgrid(lambda_vec, epsilon_vec); % Create grid for plotting
surf(LL, EE, msd_r', 'EdgeColor', 'none', 'FaceAlpha', 0.8); % Note the transpose on msd_r
set(gca, 'XScale', 'log', 'YScale', 'log', 'ZScale', 'log'); % Use log scale for axes
xlabel('\lambda');
ylabel('\epsilon');
zlabel('MSD (Red Plane)');
title('MSD Surface for Red Plane');
colorbar;

figure;
subplot(1,2,1); imagesc(f_r_uint16); axis image; colormap gray; title('Original Red Plane');
subplot(1,2,2); imagesc(u_r_specific); axis image; colormap gray; title(sprintf('Smoothed Red (\\lambda=%.2e, \\epsilon=%.2e)', lambda_scalar, epsilon_scalar));

% --- Repeat for G1, G2, B planes ---
% f_g1 = im2single(Iplanar(:,:,2));
% msd_g1 = calculate_msd(f_g1, lambda_vec, epsilon_vec);
% ... and so on for G2 and B planes ...