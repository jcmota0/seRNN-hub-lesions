%% Compare my trained networks with their "twins" in the authors' data.
% Each of my networks used the same cost strength as one of the authors' networks
% (same network number). For every pair, this computes the paper's measures on my
% network at the end of training, with the authors' own lines of code, and puts them
% next to the values the authors stored.

%% Where things are: change these if your folders are elsewhere
capsule_path = fullfile(getenv('HOME'), 'seRNN-capsule');   % the authors' CodeOcean capsule, unzipped
my_folder = fullfile(getenv('HOME'), 'seRNN-runs-notebook');  % my networks (train_networks.py --strengths notebook)
results_path = fullfile(capsule_path, 'results');               % where the figure is saved

addpath(genpath(fullfile(capsule_path, 'data', 'prerequisites')));   % Brain Connectivity Toolbox
if ~exist('network_data', 'var')   % load the authors' networks if they aren't loaded yet (~1 min)
    load(fullfile(capsule_path, 'data', 'derivatives', 'network_data.mat'));
end

epoch = 11;          % end of training (1 = before training)
nperm = 1000;        % random comparison networks for small-worldness, as in the paper
accthr = 0.9;        % the paper's threshold for "learned the task"
rng(1);

D = network_data.trained_rnns.space.euclidean;
groups = {'l1', 'seRNN'};          % the authors' names
my_names = {'L1', 'seRNN'};        % my file names
labels = {'total weight', 'corr(w,d)', 'modularity', 'small-worldness'};

results = struct();
for g = 1:2
    files = dir(fullfile(my_folder, [my_names{g} '_*.mat']));
    stored = network_data.trained_rnns.(groups{g}).topology.statistics;
    stored_acc = network_data.accuracy.accuracy{g};
    nnet = numel(files);
    mine = zeros(nnet, 4); theirs = zeros(nnet, 4);
    acc_mine = zeros(nnet, 1); acc_theirs = zeros(nnet, 1); nets = zeros(nnet, 1);

    for k = 1:nnet
        S = load(fullfile(my_folder, files(k).name));
        i = double(S.network);
        n = abs(double(squeeze(S.w_rec(epoch, :, :))));

        % the four measures, with the authors' own lines of code (computing_topology.m)
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

        nets(k) = i;
        mine(k, :) = [tw r q smw];
        theirs(k, :) = squeeze(stored(i, epoch, :))';
        acc_mine(k) = S.accuracy(epoch, 2);
        acc_theirs(k) = stored_acc(i, epoch, 2);
    end
    results.(my_names{g}) = struct('nets', nets, 'mine', mine, 'theirs', theirs, ...
        'acc_mine', acc_mine, 'acc_theirs', acc_theirs);
end

%% 1. Pair by pair: how closely do my networks track their twins?
fprintf('\nPair by pair (correlation between my value and my twin''s, across networks)\n');
fprintf('%8s %15s %15s %15s %15s %15s\n', 'group', labels{:}, 'accuracy');
for g = 1:2
    R = results.(my_names{g});
    c = arrayfun(@(k) corr(R.mine(:, k), R.theirs(:, k)), 1:4);
    fprintf('%8s %15.3f %15.3f %15.3f %15.3f %15.3f\n', my_names{g}, c, corr(R.acc_mine, R.acc_theirs));
end

%% 2. The paper's comparison: seRNN vs L1, among networks that learned the task
fprintf('\nseRNN vs L1 among networks with accuracy >= %.0f%% (mean)\n', 100*accthr);
fprintf('%22s %10s %10s %10s %10s\n', '', 'n', labels{2:4});
for who = {'mine', 'theirs'}
    for g = 1:2
        R = results.(my_names{g});
        ok = R.(['acc_' who{1}]) >= accthr;
        x = R.(who{1});
        fprintf('%22s %10d %10.3f %10.3f %10.3f\n', [who{1} ', ' my_names{g}], sum(ok), mean(x(ok, 2:4), 1));
    end
end

%% 3. Figure: my value against my twin's, for each measure
h = figure; h.Position = [100 100 1200 300];
colors = {[0.1 0.6 0.6], [0.85 0.3 0.4]};   % L1 teal, seRNN pink, as in the paper's figures
for k = 1:4
    subplot(1, 4, k); hold on;
    for g = 1:2
        R = results.(my_names{g});
        scatter(R.theirs(:, k), R.mine(:, k), 25, colors{g}, 'filled', 'MarkerFaceAlpha', 0.6);
    end
    lims = [min([xlim ylim]) max([xlim ylim])];
    plot(lims, lims, 'k--');   % on this line, mine = theirs
    xlabel('authors'' network'); ylabel('my network'); title(labels{k});
    axis square; box off;
end
legend(my_names, 'Location', 'best');
if ~exist(results_path, 'dir'), mkdir(results_path); end
saveas(gcf, fullfile(results_path, 'compare_twins.png'));
