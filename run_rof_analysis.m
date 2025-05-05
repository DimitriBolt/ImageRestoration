% run_rof_analysis.m
run("basic_script.m"); % Run basic script to get Iplanar etc.

% --- Define Parameters ---
lambda_vec = logspace(-1, 2, 15); % K=15 values
epsilon_vec = logspace(-4, -1, 10);% L=10 values
K = length(lambda_vec);
L = length(epsilon_vec);

% --- Chunking Parameters ---
% Determine a chunk size that fits your GPU memory.
% ¡¡I must to adjust this value (smaller in case of errors, larger for performance)!!
combinations_per_chunk = 15; % ¡¡ NOT MORE THAN 15!!

num_total_combinations = K * L;
num_chunks = ceil(num_total_combinations / combinations_per_chunk);
fprintf('Total combinations: %d. Processing in %d chunks of size up to %d.\n', num_total_combinations, num_chunks, combinations_per_chunk);

% --- Prepare Storage for Results ---
% Preallocate cell arrays to store MSD results for each plane
msd_results = cell(1, 4); % One cell for R, G1, G2, B
msd_results{1} = NaN(K, L); % Preallocate with NaN
msd_results{2} = NaN(K, L);
msd_results{3} = NaN(K, L);
msd_results{4} = NaN(K, L);

plane_names = {'Red', 'Green1', 'Green2', 'Blue'};

% --- Process Each Plane ---
for j = 1:4 % Loop through R, G1, G2, B
    fprintf('\n--- Processing %s Plane ---\n', plane_names{j});
    f_plane_uint16 = Iplanar(:,:,j);
    f_plane = im2single(f_plane_uint16); % Convert to single

    % --- Loop Through Chunks ---
    start_idx = 1;
    for chunk_num = 1:num_chunks
        % Determine the linear indices for this chunk
        end_idx = min(start_idx + combinations_per_chunk - 1, num_total_combinations);
        current_indices = start_idx:end_idx;
        num_in_chunk = length(current_indices);

        fprintf('Processing chunk %d/%d (combinations %d to %d)...\n', chunk_num, num_chunks, start_idx, end_idx);

        % Convert linear indices back to K, L subscripts
        [k_indices_chunk, l_indices_chunk] = ind2sub([K, L], current_indices);

        % Get the unique lambda and epsilon values needed JUST for this chunk
        lambda_chunk_unique = unique(lambda_vec(k_indices_chunk));
        epsilon_chunk_unique = unique(epsilon_vec(l_indices_chunk));

        % --- Call calculate_msd for the current chunk ---
        % Pass only the subset of parameters needed for this chunk
        msd_chunk = calculate_msd(f_plane, lambda_chunk_unique, epsilon_chunk_unique);

        % --- Map results back to the full KxL grid ---
        % Need to carefully place the results from msd_chunk into the correct
        % positions in the full msd_results matrix for this plane.
        for i = 1:num_in_chunk
            linear_idx = current_indices(i); % Original linear index (1 to K*L)
            [k_orig, l_orig] = ind2sub([K, L], linear_idx); % Original (k,l) position

            % Find where the parameters for this combo ended up in the unique chunk vectors
            k_chunk_idx = find(lambda_chunk_unique == lambda_vec(k_orig), 1);
            l_chunk_idx = find(epsilon_chunk_unique == epsilon_vec(l_orig), 1);

            % Assign the calculated MSD value to the correct overall position
            msd_results{j}(k_orig, l_orig) = msd_chunk(k_chunk_idx, l_chunk_idx);
        end

        fprintf('Chunk %d finished.\n', chunk_num);

        % Optional: Clear GPU memory explicitly between chunks if needed
        clear msd_chunk; % Clear CPU variable
        gd = gpuDevice(); reset(gd); % Resets GPU, clears all GPU vars
        fprintf('GPU memory reset.\n');

        % Update start index for the next chunk
        start_idx = end_idx + 1;
    end
    fprintf('Finished processing %s Plane.\n', plane_names{j});
end

fprintf('\n--- All planes processed ---\n');

% --- Now I plot the results from msd_results ---
figure;
hold on;
colors = {'red', 'green', [0 0.7 0.7], 'blue'}; % Example colors including a distinct second green
markers = {'o', 's', '^', 'd'}; % Optional markers
plot_handles = gobjects(1, 4);

[LL, EE] = meshgrid(lambda_vec, epsilon_vec); % Create grid for plotting

for j = 1:4
    msd_matrix = msd_results{j}; % This is KxL
    % Note: surf needs Z to be LxK if X=lambda_vec (K) and Y=epsilon_vec (L) from meshgrid
    plot_handles(j) = surf(LL, EE, msd_matrix', ... % Transpose MSD matrix
                           'DisplayName', plane_names{j}, ...
                           'FaceColor', colors{j}, ...
                           'EdgeColor', 'none', ...
                           'FaceAlpha', 0.6); % Adjust transparency
end

set(gca, 'XScale', 'log', 'YScale', 'log', 'ZScale', 'log'); % Use log scale
xlabel('\lambda');
ylabel('\epsilon');
zlabel('MSD');
title('Stacked MSD Surfaces for R, G1, G2, B Planes');
legend(plot_handles, 'Location', 'best');
grid on;
view(3); % Standard 3D view
hold off;

disp('Analysis complete. Check the plot.');