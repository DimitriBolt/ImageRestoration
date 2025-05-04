function [u, msd] = smooth_image_rof_batched(f, lambda, epsilon)
% Batches the ROF smoothing and MSD computation over a grid of (lambda, epsilon)
% using full GPU parallelism. Assumes f is 2D and of type single or double.

    assert(isa(f, 'single') || isa(f, 'double'), 'f must be floating-point.');
    f = gpuArray(f);  % Move to GPU

    [H, W] = size(f);
    lambda = lambda(:); epsilon = epsilon(:);
    K = length(lambda); L = length(epsilon);
    max_iter = 100; tol = 1e-4;

    % Expand input image into 4D grid: [H, W, K, L]
    f_big = repmat(f, 1, 1, K, L);

    % Expand parameters
    [LL, EE] = ndgrid(lambda, epsilon);
    Lambda = reshape(LL, 1, 1, K, L);
    Eps2 = reshape(EE.^2, 1, 1, K, L);

    % Initialize u
    u = f_big;

    for iter = 1:max_iter
        % Pad for Neumann BCs
        up = padarray(u, [1 1], 'symmetric');

        % Forward differences
        ux = up(2:end-1,3:end,:,:) - up(2:end-1,2:end-1,:,:);
        uy = up(3:end,2:end-1,:,:) - up(2:end-1,2:end-1,:,:);
        mag = sqrt(Eps2 + ux.^2 + uy.^2);

        % Flux terms
        px = ux ./ mag;
        py = uy ./ mag;

        % Backward divergence
        pxp = padarray(px, [0 1], 'pre');
        pyp = padarray(py, [1 0], 'pre');
        div = pxp(:,2:end,:,:)-pxp(:,1:end-1,:,:) + pyp(2:end,:,:,:) - pyp(1:end-1,:,:,:);

        % Update
        unew = f_big - Lambda .* div;

        % Convergence check (optional, single sample)
        if iter > 1
            delta = norm(unew(:,:,1,1) - u(:,:,1,1), 'fro') / norm(u(:,:,1,1), 'fro');
            if delta < tol
                break;
            end
        end
        u = unew;
    end

    % Compute MSD
    diff = u - f_big;
    msd = sqrt(sum(diff.^2, [1 2]) / (H * W));
    msd = gather(squeeze(msd));  % K x L matrix on CPU
    u = gather(u);               % optionally return denoised images
end
