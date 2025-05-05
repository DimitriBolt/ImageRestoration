function msd = calculate_msd(f, lambda, epsilon)
%CALCULATE_MSD Returns MSD for a given degraded image and ROF parameters.
%   MSD = CALCULATE_MSD(F, LAMBDA, EPSILON) finds the Mean Square Difference (MSD)
%   between the degraded image F and the ROF-restored image U, obtained using
%   parameters LAMBDA and EPSILON.
%
%   Arguments:
%       F         - The degraded image (2D matrix, HxW).
%       LAMBDA    - The 'smoothing' parameter (scalar or vector).
%       EPSILON   - The 'regularization' parameter (scalar or vector).
%
%   Returns:
%       MSD       - The MSD of the degraded image.
%                   If LAMBDA or EPSILON is a vector, the result is a 2D array
%                   of size K-by-L, where K = length(LAMBDA) and L = length(EPSILON).
%                   MSD(k, l) corresponds to LAMBDA(k) and EPSILON(l).
%
%   Notes:
%   - Calls smooth_image_rof to get the restored image U.
%   - Handles GPU/CPU execution based on smooth_image_rof's behavior.

    [H, W] = size(f);
    lambda = lambda(:);  % Ensure column vector
    epsilon = epsilon(:); % Ensure column vector
    K = length(lambda);
    L = length(epsilon);

    fprintf('Calculating MSD for %d parameter combinations...\n', K*L);

    % Call smooth_image_rof to get the restored image(s) U.
    % U will be HxWxKxL and will reside on the CPU after smooth_image_rof finishes.
    u = smooth_image_rof(f, lambda, epsilon);

    % --- Calculate MSD ---
    % Ensure f is on the CPU and has the same type as u for subtraction
    if isa(u, 'gpuArray')
        % This case should ideally not happen as smooth_image_rof gathers
        warning('U is unexpectedly on GPU in calculate_msd. Gathering F.');
        f_cpu = gather(f);
        u_cpu = u; % U is already on CPU if gathered in smooth_image_rof
    else
        f_cpu = f;
        u_cpu = u;
    end
    
    % Ensure data types match for subtraction (e.g., convert f to single if u is single)
    if ~strcmp(class(f_cpu), class(u_cpu))
       f_cpu = cast(f_cpu, 'like', u_cpu); 
    end

    % Preallocate MSD array (on CPU)
    msd = zeros(K, L, 'like', u_cpu); % Match data type of u_cpu (likely double or single)

    % Calculate MSD for each parameter combination
    % This loop is relatively fast compared to the ROF smoothing
    for k = 1:K
        for l = 1:L
            % Difference between restored and original image for this (k, l)
            diff = u_cpu(:,:,k,l) - f_cpu;

            % Calculate squared difference sum
            sum_sq_diff = sum(diff(:).^2);

            % Calculate MSD
            msd(k,l) = sqrt(sum_sq_diff / (H * W));
        end
    end

    fprintf('MSD calculation complete.\n');

end