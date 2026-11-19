%% Recompute the authors' topology measures on a sample of their networks,
%% and compare with the values they stored.

%% Where things are: change these if your folders are elsewhere
capsule_path = fullfile(getenv('HOME'), 'seRNN-capsule');   % the authors' CodeOcean capsule, unzipped

addpath(genpath(fullfile(capsule_path, 'data', 'prerequisites')));   % Brain Connectivity Toolbox
if ~exist('network_data', 'var')   % load the authors' networks if they aren't loaded yet (~1 min)
    load(fullfile(capsule_path, 'data', 'derivatives', 'network_data.mat'));
end

rng(1);          % fixed random seed, so the check gives the same result every time
nsample = 10;    % networks per group
epoch = 11;      % 11 = end of training (1 = before training)
nperm = 1000;    % random comparison networks for small-worldness, as in the paper
D = network_data.trained_rnns.space.euclidean;

groups = {'l1', 'seRNN'};   % connectivity{1} is L1, connectivity{2} is seRNN
for g = 1:2
    stored = network_data.trained_rnns.(groups{g}).topology.statistics;
    nets = randperm(1001, nsample);   % pick 10 networks at random
    fprintf('\n%s: mine vs theirs\n', groups{g});
    fprintf('%8s %23s %23s %23s %23s\n', 'network', 'total weight', 'corr(w,d)', 'modularity', 'small-worldness');
    for i = nets
        n = abs(squeeze(network_data.trained_rnns.connectivity{g}(i, epoch, :, :)));

        % the four measures, using the authors' own lines of code
        tw = sum(n, 'all');
        r = corr(D(n>0), n(n>0));
        [~, q] = modularity_und(n);
        a = threshold_proportional(n, 0.1); a = double(a > 0);
        m = nnz(a)/2;
        clu_perm = zeros(1, nperm); cpl_perm = zeros(1, nperm);
        for j = 1:nperm
            Arand = makerandCIJ_und(100, m);
            clu_perm(j) = mean(clustering_coef_bu(Arand));
            cpl_perm(j) = charpath(Arand);
        end
        smw = (mean(clustering_coef_bu(a))/mean(clu_perm)) / (charpath(a)/mean(cpl_perm));

        % print mine next to theirs
        mine = [tw r q smw];
        theirs = squeeze(stored(i, epoch, :))';
        fprintf('%8d', i);
        fprintf('   %8.4f vs %8.4f', [mine; theirs]);
        fprintf('\n');
    end
end