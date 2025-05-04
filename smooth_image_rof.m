function u = smooth_image_rof(f, lambda, epsilon)
%SMOOTH_IMAGE_ROF Perform ROF image restoration using fixed-point iteration.
%   U = SMOOTH_IMAGE_ROF(F, LAMBDA, EPSILON) performs ROF image restoration
%   on the degraded image F using the smoothing parameter LAMBDA and the
%   regularization parameter EPSILON.
%
%   Arguments:
%       F         - The degraded image (2D matrix, HxW). Assumed to be double or single.
%       LAMBDA    - The 'smoothing' parameter (scalar or vector).
%       EPSILON   - The 'regularization' parameter (scalar or vector).
%
%   Returns:
%       U         - The restored/smoothed image.
%                   If LAMBDA or EPSILON is a vector, the result is a 4D array
%                   of size H-by-W-by-K-by-L, where K = length(LAMBDA) and
%                   L = length(EPSILON).
%
%   Notes:
%   - Uses Neumann boundary conditions (symmetric padding).
%   - Automatically uses GPU if available, otherwise uses parallel CPU processing
%     for vector LAMBDA/EPSILON inputs.
%   - Fully vectorized implementation for the iterative updates. [cite: 25]

    [H, W] = size(f);
    lambda = lambda(:);  % Ensure column vector
    epsilon = epsilon(:); % Ensure column vector
    K = length(lambda);
    L = length(epsilon);

    % Determine execution environment (GPU or CPU)
    use_gpu = false;
    if gpuDeviceCount > 0
        try
            gpuDevice; % Check if a GPU is actually usable
            use_gpu = true;
            fprintf('GPU detected. Using GPU for computation.\n');
            f_device = gpuArray(f); % Move input image to GPU
        catch ME
            warning('GPU detected but unusable. Error: %s Using CPU instead.', ME.message);
            f_device = f; % Keep image on CPU
        end
    else
        fprintf('No GPU detected. Using CPU.\n');
        f_device = f; % Keep image on CPU
    end

    % Allocate output array on the appropriate device (CPU or GPU)
    % Using 'like' ensures the output has the same data type and device as f_device
    u = zeros(H, W, K, L, 'like', f_device);

    % Parameters for the iterative solver
    max_iter = 100; % Maximum number of iterations
    tol = 1e-4;     % Convergence tolerance

    % --- Main Loop ---
    % If LAMBDA and EPSILON are scalars, run a simple loop.
    % If either is a vector, use parfor for CPU or run sequentially for GPU.
    if K == 1 && L == 1
        % --- Scalar Parameter Case ---
        lam = lambda(1);
        eps2 = epsilon(1)^2;
        uk = f_device; % Initialize u for this parameter set

        for iter = 1:max_iter
            % Pad for Neumann boundary conditions [cite: 106]
            up = padarray(uk, [1 1], 'symmetric');

            % Compute forward differences (gradients) [cite: 87, 113, 114]
            ux = up(2:end-1, 3:end, :, :)   - up(2:end-1, 2:end-1, :, :);
            uy = up(3:end,   2:end-1, :, :) - up(2:end-1, 2:end-1, :, :);

            % Calculate magnitude squared + epsilon^2 [cite: 94, 114]
            mag_sq = eps2 + ux.^2 + uy.^2;
            mag = sqrt(mag_sq); % Avoid division by zero if mag is exactly 0

            % Compute flux terms [cite: 95, 96, 115, 116]
            % Add small value to denominator to prevent NaN/Inf if mag is zero
            px = ux ./ (mag + eps); 
            py = uy ./ (mag + eps);

            % Compute backward divergence [cite: 87, 97, 118]
            % Pad flux terms before differencing (Neumann implies zero flux gradient at boundary)
            px_padded = padarray(px, [0 1], 0, 'pre'); % Pad left column with 0
            py_padded = padarray(py, [1 0], 0, 'pre'); % Pad top row with 0
            
            div_p = (px_padded(:, 2:end, :, :) - px_padded(:, 1:end-1, :, :)) + ...
                    (py_padded(2:end, :, :, :) - py_padded(1:end-1, :, :, :));

            % Update step [cite: 88, 98, 119]
            unew = f_device - lam * div_p;

            % Check convergence [cite: 99, 120, 121]
            rel_change = norm(unew(:) - uk(:), 2) / (norm(uk(:), 2) + eps); % Add eps for stability if norm is zero
            if rel_change < tol
                break;
            end
            uk = unew;
        end
        u(:,:,1,1) = uk; % Store result

    else
        % --- Vector Parameter Case (GPU or Parallel CPU) ---
        if use_gpu
             % --- GPU Execution (Batched - similar to smooth_image_rof_batched.m logic) ---
            fprintf('Processing %d parameter combinations on GPU...\n', K*L);
             % Expand f to match the output dimensions [H, W, K, L]
             f_big = repmat(f_device, 1, 1, K, L);

             % Expand parameters to match dimensions [1, 1, K, L]
             [LL, EE] = ndgrid(lambda, epsilon); % Creates KxL matrices
             Lambda = reshape(LL, 1, 1, K, L);
             Eps2 = reshape(EE.^2, 1, 1, K, L);

             % Initialize u on the GPU [H, W, K, L]
             uk = f_big; % Start iteration with the input image for all parameters

             for iter = 1:max_iter
                 % Pad for Neumann boundary conditions
                  up = padarray(uk, [1 1], 'symmetric');

                 % Forward differences (Vectorized across K, L)
                 ux = up(2:end-1, 3:end, :, :)   - up(2:end-1, 2:end-1, :, :);
                 uy = up(3:end,   2:end-1, :, :) - up(2:end-1, 2:end-1, :, :);

                 % Magnitude (Vectorized) - Use expanded Epsilon^2
                 mag_sq = Eps2 + ux.^2 + uy.^2;
                 mag = sqrt(mag_sq);

                 % Flux (Vectorized)
                 px = ux ./ (mag + eps); % Add eps for numerical stability
                 py = uy ./ (mag + eps);

                 % Divergence (Vectorized)
                 px_padded = padarray(px, [0 1], 0, 'pre');
                 py_padded = padarray(py, [1 0], 0, 'pre');
                 div_p = (px_padded(:, 2:end, :, :) - px_padded(:, 1:end-1, :, :)) + ...
                         (py_padded(2:end, :, :, :) - py_padded(1:end-1, :, :, :));

                 % Update (Vectorized) - Use expanded Lambda
                 unew = f_big - Lambda .* div_p;

                 % Convergence Check (check average change across all images)
                 delta_norm = norm(unew(:) - uk(:), 2);
                 uk_norm = norm(uk(:), 2);
                 rel_change = delta_norm / (uk_norm + eps);

                 fprintf('GPU Iter %d, Relative Change: %e\n', iter, gather(rel_change)); % Gather for display

                 if rel_change < tol
                     break;
                 end
                 uk = unew;
             end
             u = uk; % Assign the final result
             fprintf('GPU processing finished after %d iterations.\n', iter);

        else
            % --- Parallel CPU Execution (using parfor) --- [cite: 31, 33, 111]
            fprintf('Processing %d parameter combinations using parallel CPU (%d threads)...\n', K*L, maxNumCompThreads);
            % Use a temporary cell array to store results from parfor workers [cite: 164, 167]
            uCell = cell(K*L, 1);

            parfor idx = 1:(K*L) % Loop over all parameter combinations linearly
                % Convert linear index back to (k, l) indices [cite: 168, 177]
                [k, l] = ind2sub([K, L], idx);
                lam = lambda(k);
                eps2 = epsilon(l)^2;

                uk_par = f_device; % Initialize u for this worker (f_device is on CPU here)

                for iter = 1:max_iter
                    up = padarray(uk_par, [1 1], 'symmetric');
                    ux = up(2:end-1, 3:end)   - up(2:end-1, 2:end-1);
                    uy = up(3:end,   2:end-1) - up(2:end-1, 2:end-1);
                    mag_sq = eps2 + ux.^2 + uy.^2;
                    mag = sqrt(mag_sq);
                    px = ux ./ (mag + eps);
                    py = uy ./ (mag + eps);
                    px_padded = padarray(px, [0 1], 0, 'pre');
                    py_padded = padarray(py, [1 0], 0, 'pre');
                    div_p = (px_padded(:, 2:end) - px_padded(:, 1:end-1)) + ...
                            (py_padded(2:end, :) - py_padded(1:end-1, :));
                    unew = f_device - lam * div_p;

                    rel_change = norm(unew(:) - uk_par(:), 2) / (norm(uk_par(:), 2) + eps);
                    if rel_change < tol
                        break;
                    end
                    uk_par = unew;
                end
                 % Store the result for this (k,l) in the cell array [cite: 175]
                uCell{idx} = uk_par;
            end

            % Reassemble the 4D array from the cell array [cite: 176, 177, 178]
            % Ensure the final array 'u' is on the CPU, matching f_device type
            u = zeros(H, W, K, L, 'like', f_device); % Create CPU array
            for idx = 1:(K*L)
                [k, l] = ind2sub([K, L], idx);
                u(:,:,k,l) = uCell{idx};
            end
            fprintf('Parallel CPU processing finished.\n');
        end
    end

    % If computation was done on GPU, gather the result back to CPU memory
    if use_gpu
        fprintf('Gathering results from GPU to CPU memory...\n');
        u = gather(u);
        fprintf('Gathering complete.\n');
    end

end