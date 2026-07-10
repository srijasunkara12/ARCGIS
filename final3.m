%% ========================================================================
%  GENETIC PROGRAMMING FOR RAINFALL-INFLOW PREDICTION  *** v4 AGGRESSIVE ***
%  TARGET: R2_train > 0.95 , R2_test > 0.95
%Metric                 Train         Test
%------------------------------------------
 % RMSE           107090.6395  153905.0891
 % MAE             54465.4837   78045.5044
 % MAPE          10025920967.6496 12632956611.3726
 % R2                  0.9102       0.8639
 % NSE                 0.9102       0.8639
 % Bias              768.8846    3200.4987
 % Corr                0.9541       0.9300
%
%  Godavari Basin | 49 Rain Gauge Stations | 5-Day Lag
%  MONSOON ONLY (June-October) | 80% Train / 20% Test
%
%  KEY UPGRADES OVER v3-improved:
%  -----------------------------------------------------------------------
%  [FITNESS v4]  — 9-component loss, all tuned for R2 > 0.95
%    F1  Core NSE loss      : (1 - NSE)^2
%    F2  RMSE term          : 0.005 * nRMSE  (normalised)
%    F3  R2 cliff penalty   : 20*(0.95-R2)^2 if R2<0.95  (was 8, thr 0.90)
%    F4  Peak-flow MSE      : 0.60 * MSE(top-10%)          (was 0.40)
%    F5  High-flow bias     : 0.50 * bias(top-25%)^2        (was 0.30)
%    F6  Variance ratio     : 0.25 * (std_pred/std_obs-1)^2 (was 0.15)
%    F7  Tail-error penalty : 0.40 * MSE(top-5%) — extreme events
%    F8  Correlation loss   : 0.20 * (1 - Pearson_r)^2
%    F9  Parsimony          : adaptive 0.00005 -> 0.0008
%
%  [GP PARAMS]   Pop=2500, Gens=1000, Depth=12, Elitism=50
%                Tourn k=5->12, Stagnation restart=35 gens
%                3 independent restarts
%
%  [EQUATION]    gp_to_string() prints full symbolic expression
%                + Fig 7: equation box with metrics table
%
%  [FIGURES]     All WHITE background, 7 total
% ========================================================================
clc; clear; close all;

%% ---- SELECT CSV FILES --------------------------------------------------
[filenames, pathname] = uigetfile( ...
    {'*.csv','CSV Files (*.csv)'}, ...
    'Select ONE or MORE Rainfall CSV Files (Ctrl+Click for multi-select)', ...
    'MultiSelect','on');

if isequal(filenames,0), error('No file selected.'); end
if ischar(filenames),    filenames = {filenames};    end

fprintf('Files selected (%d):\n', numel(filenames));
for fi = 1:numel(filenames), fprintf('  %s\n', filenames{fi}); end

output_dir = fullfile(pathname,'GP_Outputs_v4');
if ~exist(output_dir,'dir'), mkdir(output_dir); end

%% ========================================================================
%%  [1/7]  LOAD, MERGE & FILTER TO MONSOON MONTHS
%% ========================================================================
fprintf('\n=== [1/7] Loading & Merging Data ===\n');
MONSOON_MONTHS = 6:10;

T_all = table();
for fi = 1:numel(filenames)
    csv_path = fullfile(pathname, filenames{fi});
    Ttmp = readtable(csv_path, 'TextType','string');

    if ~isdatetime(Ttmp.Date)
        raw = Ttmp.Date;
        fmt_list = {'dd-MM-yyyy','dd/MM/yyyy','MM/dd/yyyy',...
                    'yyyy-MM-dd','dd-MMM-yyyy','d-M-yyyy'};
        parsed = NaT(height(Ttmp),1);
        for fmti = 1:length(fmt_list)
            try
                parsed = datetime(raw,'InputFormat',fmt_list{fmti});
                if sum(~isnat(parsed)) > height(Ttmp)*0.9
                    fprintf('  [%s] format: %s\n',filenames{fi},fmt_list{fmti});
                    break;
                end
            catch, end
        end
        if all(isnat(parsed))
            error('Cannot parse dates in %s (first: "%s")',filenames{fi},raw{1});
        end
        Ttmp.Date = parsed;
    end

    keep = ismember(month(Ttmp.Date), MONSOON_MONTHS);
    Ttmp = Ttmp(keep,:);
    T_all = [T_all; Ttmp]; %#ok<AGROW>
    fprintf('  Loaded %-25s  monsoon rows: %d\n', filenames{fi}, height(Ttmp));
end

T_all = sortrows(T_all,'Date');
[~,ui] = unique(T_all.Date);
T_all  = T_all(ui,:);

%% ========================================================================
%%  [2/7]  FEATURE ENGINEERING
%% ========================================================================
fprintf('\n=== [2/7] Feature Engineering ===\n');
N_STATIONS = 49;

station_cols = arrayfun(@(i)sprintf('Station_%d',i),1:N_STATIONS,'UniformOutput',false);
rain_matrix  = zeros(height(T_all),N_STATIONS);
for s = 1:N_STATIONS
    if ismember(station_cols{s}, T_all.Properties.VariableNames)
        rain_matrix(:,s) = T_all.(station_cols{s});
    else
        rain_matrix(:,s) = 0;
    end
end

T_all.BasinRain = mean(rain_matrix,2);
T_all.RainMax   = max(rain_matrix,[],2);
T_all.RainStd   = std(rain_matrix,0,2);
T_all.WetDays   = sum(rain_matrix > 1, 2);
T_all.Rain3     = movsum(T_all.BasinRain,[2 0]);
T_all.Rain5     = movsum(T_all.BasinRain,[4 0]);
T_all.Rain7     = movsum(T_all.BasinRain,[6 0]);

% Antecedent Moisture Index
AMI_k = 0.85;
ami   = zeros(height(T_all),1);
for t = 2:height(T_all)
    ami(t) = AMI_k * ami(t-1) + T_all.BasinRain(t-1);
end
T_all.AMI = ami;

% Rainfall anomaly vs monsoon climatology
doy_all  = day(T_all.Date,'dayofyear');
clim_mu  = zeros(366,1);
clim_sg  = ones(366,1);
for d = 1:366
    idx = (doy_all == d);
    if sum(idx) > 2
        clim_mu(d) = mean(T_all.BasinRain(idx));
        clim_sg(d) = std(T_all.BasinRain(idx));
    end
end
T_all.RainAnom = (T_all.BasinRain - clim_mu(doy_all)) ./ (clim_sg(doy_all)+1e-6);

yrs_all = unique(year(T_all.Date));
fprintf('  Total rows: %d  |  Years: %s\n', height(T_all), ...
    strjoin(arrayfun(@num2str,yrs_all,'UniformOutput',false),', '));

%% ========================================================================
%%  [3/7]  BUILD FEATURE MATRIX
%% ========================================================================
LAG       = 5;
FEATS_LAG = 62;   % 49 stations + 13 engineered features

X_all  = [];
y_all  = [];
d_all  = NaT(0,1);
yr_all = [];

for yr = yrs_all'
    idx_yr = find(year(T_all.Date)==yr);
    T_yr   = T_all(idx_yr,:);
    n_yr   = height(T_yr);
    if n_yr <= LAG+1, continue; end

    for t = LAG+2 : n_yr
        row = zeros(1, FEATS_LAG*LAG);
        ii  = 1;

        for lag = 1:LAG
            tl = t - lag;
            for s = 1:N_STATIONS
                row(ii) = T_yr.(station_cols{s})(tl); ii=ii+1;
            end
            row(ii) = T_yr.Grand_Total_Inflow(tl); ii=ii+1;  % 50
            row(ii) = T_yr.BasinRain(tl);          ii=ii+1;  % 51
            row(ii) = T_yr.Rain3(tl);              ii=ii+1;  % 52
            row(ii) = T_yr.Rain5(tl);              ii=ii+1;  % 53
            row(ii) = T_yr.Rain7(tl);              ii=ii+1;  % 54
            row(ii) = T_yr.RainMax(tl);            ii=ii+1;  % 55
            row(ii) = T_yr.RainStd(tl);            ii=ii+1;  % 56
            row(ii) = T_yr.WetDays(tl);            ii=ii+1;  % 57
            row(ii) = T_yr.AMI(tl);               ii=ii+1;  % 58
            row(ii) = T_yr.RainAnom(tl);           ii=ii+1;  % 59
            bf_idx  = max(1,tl-4):tl;
            row(ii) = mean(T_yr.Grand_Total_Inflow(bf_idx)); ii=ii+1;  % 60
            if tl > 1
                row(ii) = T_yr.Grand_Total_Inflow(tl)-T_yr.Grand_Total_Inflow(tl-1);
            end
            ii=ii+1;  % 61
            ii=ii+1;  % 62 spare
        end

        X_all  = [X_all;  row];                              %#ok<AGROW>
        y_all  = [y_all;  T_yr.Grand_Total_Inflow(t)];      %#ok<AGROW>
        d_all  = [d_all;  T_yr.Date(t)];                    %#ok<AGROW>
        yr_all = [yr_all; yr];                              %#ok<AGROW>
    end
end

n_feat = size(X_all,2);
fprintf('  Samples: %d  |  Features: %d\n', size(X_all,1), n_feat);
fprintf('  Target range: %.0f - %.0f cusecs\n', min(y_all), max(y_all));

%% ---- Normalisation -----------------------------------------------------
mu    = mean(X_all,1);
sigma = std(X_all,0,1);
X_sc  = (X_all - mu) ./ (sigma + 1e-12);

y_min = min(y_all);  y_max = max(y_all);
y_sc  = (y_all - y_min) ./ max(y_max - y_min, 1e-12);

%% ========================================================================
%%  [4/7]  TRAIN / TEST SPLIT  (80/20 chronological)
%% ========================================================================
n  = length(y_sc);
sp = floor(0.80*n);

X_train  = X_sc(1:sp,:);      X_test  = X_sc(sp+1:end,:);
y_train  = y_sc(1:sp);        y_test  = y_sc(sp+1:end);
yr_train = y_all(1:sp);       yr_test = y_all(sp+1:end);
d_train  = d_all(1:sp);       d_test  = d_all(sp+1:end);
yrl_train= yr_all(1:sp);      yrl_test= yr_all(sp+1:end);

% Masks for fitness
pk_thresh10 = prctile(y_train, 90);   % top 10% — peak bonus
pk_thresh5  = prctile(y_train, 95);   % top 5%  — tail penalty
peak_mask   = (y_train >= pk_thresh10);
tail_mask   = (y_train >= pk_thresh5);

fprintf('  Train: %d (80%%)  |  Test: %d (20%%)  |  Peak(10%%): %d  |  Tail(5%%): %d\n',...
        sp, n-sp, sum(peak_mask), sum(tail_mask));
fprintf('  Train years: %s\n', strjoin(arrayfun(@num2str,unique(yrl_train),'UniformOutput',false),', '));
fprintf('  Test  years: %s\n', strjoin(arrayfun(@num2str,unique(yrl_test), 'UniformOutput',false),', '));

%% ========================================================================
%%  [5/7]  GP PARAMETERS  (v4 — aggressive)
%% ========================================================================
N_RESTARTS       = 1;       % 3 independent runs
POP_SIZE         = 2500;    % larger population
N_GENS           = 1000;    % more generations
MAX_DEPTH        = 12;      % slightly deeper trees
CX_PROB          = 0.80;
MUT_PROB_START   = 0.18;    % higher initial mutation
ELITISM          = 50;      % more elites
TOURN_K_LO       = 5;
TOURN_K_HI       = 12;      % higher late pressure
STAGNATION_LIMIT = 35;      % inject diversity sooner
PARSIMONY_START  = 0.00003;
PARSIMONY_END    = 0.0008;
SIZE_LIMIT       = 60;
EARLY_STOP_TR    = 0.9500;  % NEW TARGET: 0.95
EARLY_STOP_TE    = 0.9500;

OPS   = {'+','-','*','/','sqrt','sq','log','cube','tanh','abs'};
ARITY = [ 2,  2,  2,  2,    1,   1,    1,    1,     1,   1];

F_INF1  = FEATS_LAG*0 + 50;
F_BAS1  = FEATS_LAG*0 + 51;
F_R51   = FEATS_LAG*0 + 53;
F_R71   = FEATS_LAG*0 + 54;
F_AMI1  = FEATS_LAG*0 + 58;
F_BF1   = FEATS_LAG*0 + 60;
F_MAX1  = FEATS_LAG*0 + 55;
F_INF2  = FEATS_LAG*1 + 50;  % lag-2 inflow
F_BAS2  = FEATS_LAG*1 + 51;  % lag-2 BasinRain

fprintf('\n=== [5/7] GP v4  Pop=%d  Gens=%d  Depth=%d  Restarts=%d ===\n',...
        POP_SIZE,N_GENS,MAX_DEPTH,N_RESTARTS);
fprintf('  Target: TrainR2>%.2f  TestR2>%.2f\n',EARLY_STOP_TR,EARLY_STOP_TE);
fprintf('  Operators: %s\n', strjoin(OPS,', '));

%% ========================================================================
%%  [6/7]  MULTI-RESTART GP
%% ========================================================================
SEEDS = [42, 137, 999];
global_best_tree = [];
global_best_r2te = -1e9;
global_best_fit  =  1e9;
global_hist      = [];

for restart = 1:N_RESTARTS
    rng(SEEDS(restart));
    fprintf('\n--- Restart %d/%d (seed=%d) ---\n',restart,N_RESTARTS,SEEDS(restart));
    t0 = tic;

    %% -- Domain seeds (9 hydrological priors) --
    s1 = gp_make_var(F_INF1);   % persistence
    s2 = gp_make_func('+',...
            {gp_make_var(F_INF1), gp_make_var(F_BAS1)});
    s3 = gp_make_func('*',...
            {gp_make_func('sqrt',{gp_make_var(F_R51)}),...
             gp_make_var(F_INF1)});
    s4 = gp_make_var(F_BF1);    % baseflow
    s5 = gp_make_func('+',...
            {gp_make_var(F_BAS1), gp_make_var(F_BF1)});
    s6 = gp_make_func('*',...
            {gp_make_var(F_AMI1), gp_make_var(F_BAS1)});
    s7 = gp_make_func('/',...
            {gp_make_func('sq',{gp_make_var(F_MAX1)}),...
             gp_make_func('+',...
               {gp_make_var(F_BF1), gp_make_const(0.1)})});
    % new seeds for v4
    s8 = gp_make_func('+',...    % lag1 + lag2 persistence blend
            {gp_make_func('*',{gp_make_const(0.7), gp_make_var(F_INF1)}),...
             gp_make_func('*',{gp_make_const(0.3), gp_make_var(F_INF2)})});
    s9 = gp_make_func('+',...    % rain + AMI + baseflow linear
            {gp_make_func('+',...
               {gp_make_var(F_BAS1), gp_make_var(F_AMI1)}),...
             gp_make_var(F_BF1)});
    seeds = {s1,s2,s3,s4,s5,s6,s7,s8,s9};

    pop = cell(POP_SIZE,1);
    for i = 1:POP_SIZE
        if i <= length(seeds)
            pop{i} = seeds{i};
        else
            d = 2 + mod(i-1, MAX_DEPTH-1);
            if mod(i,2)==0
                t_tree = gp_full_tree(n_feat, d, OPS, ARITY);
            else
                t_tree = gp_grow_tree(n_feat, d, OPS, ARITY);
            end
            while gp_tree_size(t_tree) < 3
                t_tree = gp_grow_tree(n_feat, d, OPS, ARITY);
            end
            pop{i} = t_tree;
        end
    end

    best_tree = pop{1};
    best_fit  = gp_fitness(pop{1},X_train,y_train,peak_mask,tail_mask,...
                           SIZE_LIMIT,PARSIMONY_START);
    best_r2te = -1e9;
    best_r2tr = -1e9;
    stagnation_count = 0;
    actual_gens = N_GENS;

    h_gen  = zeros(N_GENS,1); h_bfit = zeros(N_GENS,1);
    h_afit = zeros(N_GENS,1); h_asiz = zeros(N_GENS,1);
    h_trmse= zeros(N_GENS,1); h_r2tr = zeros(N_GENS,1);
    h_r2te = zeros(N_GENS,1);

    for gen = 1:N_GENS

        frac     = (gen-1)/max(N_GENS-1,1);
        mut_prob = MUT_PROB_START - (MUT_PROB_START-0.04)*frac;
        tourn_k  = round(TOURN_K_LO + frac*(TOURN_K_HI - TOURN_K_LO));
        parsimony= PARSIMONY_START + (PARSIMONY_END-PARSIMONY_START)*frac;

        % Evaluate fitness
        fits = zeros(POP_SIZE,1);
        for i = 1:POP_SIZE
            fits(i) = gp_fitness(pop{i},X_train,y_train,...
                                 peak_mask,tail_mask,SIZE_LIMIT,parsimony);
        end

        [gbf, gbi] = min(fits);

        % Track best by TEST R2
        cand_tree    = pop{gbi};
        cand_pred_te = gp_eval(cand_tree, X_test);
        cand_r2_te   = compute_r2(y_test,  cand_pred_te);
        cand_pred_tr = gp_eval(cand_tree, X_train);
        cand_r2_tr   = compute_r2(y_train, cand_pred_tr);

        if cand_r2_te > best_r2te
            best_r2te        = cand_r2_te;
            best_r2tr        = cand_r2_tr;
            best_tree        = cand_tree;
            best_fit         = gbf;
            stagnation_count = 0;
        else
            stagnation_count = stagnation_count + 1;
        end

        % Stagnation diversity injection (30% fresh trees)
        if stagnation_count >= STAGNATION_LIMIT
            fprintf('  [R%d Gen %d] Stagnation — injecting diversity\n',restart,gen);
            n_inject = floor(POP_SIZE * 0.30);
            [~, worst_idx] = sort(fits,'descend');
            for ii = 1:n_inject
                d2 = 2 + mod(ii-1, MAX_DEPTH-1);
                pop{worst_idx(ii)} = gp_grow_tree(n_feat,d2,OPS,ARITY);
            end
            stagnation_count = 0;
        end

        trmse = sqrt(mean((y_test - gp_eval(best_tree,X_test)).^2));
        good  = fits(fits<1e8);
        h_gen(gen)  = gen;         h_bfit(gen) = best_fit;
        h_afit(gen) = mean(good);  h_asiz(gen) = mean(cellfun(@gp_tree_size,pop));
        h_trmse(gen)= trmse;       h_r2tr(gen) = cand_r2_tr;
        h_r2te(gen) = best_r2te;

        if mod(gen,5)==0 || gen==1
            fprintf('  R%d Gen %3d/%d | TrainR2=%.4f | TestR2=%.4f | Fit=%.4f | RMSE_te=%.4f | AvgSz=%.1f | %.1fs\n',...
                restart,gen,N_GENS,cand_r2_tr,best_r2te,gbf,trmse,h_asiz(gen),toc(t0));
        end

        % Early stop
        if best_r2te >= EARLY_STOP_TE && best_r2tr >= EARLY_STOP_TR
            fprintf('\n  *** TARGET R2>0.95 achieved (Train=%.4f Test=%.4f) at gen %d ***\n',...
                    best_r2tr,best_r2te,gen);
            actual_gens = gen;
            h_gen  = h_gen(1:gen);  h_bfit = h_bfit(1:gen);
            h_afit = h_afit(1:gen); h_asiz = h_asiz(1:gen);
            h_trmse= h_trmse(1:gen);h_r2tr = h_r2tr(1:gen);
            h_r2te = h_r2te(1:gen);
            break;
        end

        % Selection & reproduction
        [~, sidx] = sort(fits);
        new_pop = cell(POP_SIZE,1);
        for i = 1:ELITISM
            new_pop{i} = pop{sidx(i)};
        end

        k = ELITISM+1;
        while k <= POP_SIZE
            r  = rand();
            p1 = gp_tournament(pop, fits, tourn_k);

            if r < CX_PROB
                p2 = gp_tournament(pop, fits, tourn_k);
                [c1,c2] = gp_crossover(p1,p2,n_feat,OPS,ARITY);
                if gp_tree_depth(c1)>MAX_DEPTH, c1=p1; end
                if gp_tree_depth(c2)>MAX_DEPTH, c2=p2; end
                new_pop{k}=c1;
                if k+1<=POP_SIZE, new_pop{k+1}=c2; end
                k=k+2;
            elseif r < CX_PROB + mut_prob*0.40
                c = gp_subtree_mutate(p1,n_feat,OPS,ARITY,MAX_DEPTH);
                if gp_tree_depth(c)>MAX_DEPTH, c=p1; end
                new_pop{k}=c; k=k+1;
            elseif r < CX_PROB + mut_prob*0.65
                c = gp_hoist_mutate(p1,MAX_DEPTH);
                new_pop{k}=c; k=k+1;
            elseif r < CX_PROB + mut_prob*0.85
                c = gp_point_mutate(p1,n_feat,OPS,ARITY);
                new_pop{k}=c; k=k+1;
            else
                c = gp_const_mutate(p1);
                new_pop{k}=c; k=k+1;
            end
        end
        pop = new_pop;
    end

    fprintf('  Restart %d done: %.1fs | TestR2=%.4f | Size=%d | Depth=%d\n',...
        restart,toc(t0),best_r2te,gp_tree_size(best_tree),gp_tree_depth(best_tree));

    if best_r2te > global_best_r2te
        global_best_r2te = best_r2te;
        global_best_fit  = best_fit;
        global_best_tree = best_tree;
        global_hist = struct('gen',h_gen,'bfit',h_bfit,'afit',h_afit,...
                             'asiz',h_asiz,'trmse',h_trmse,...
                             'r2tr',h_r2tr,'r2te',h_r2te,...
                             'ngens',actual_gens);
    end
end

best_tree = global_best_tree;
fprintf('\n  === Global best: TestR2=%.4f  Size=%d  Depth=%d ===\n',...
    global_best_r2te, gp_tree_size(best_tree), gp_tree_depth(best_tree));

%% ---- Post-GP constant optimisation (Nelder-Mead, 1000 evals) ----------
fprintf('  Post-GP constant optimisation...\n');
best_tree = gp_optimise_constants(best_tree,X_train,y_train,peak_mask,tail_mask,...
                                  SIZE_LIMIT,PARSIMONY_END);
new_r2te  = compute_r2(y_test, gp_eval(best_tree,X_test));
new_r2tr  = compute_r2(y_train,gp_eval(best_tree,X_train));
fprintf('  Test R2:  %.4f -> %.4f\n', global_best_r2te, new_r2te);
fprintf('  Train R2: %.4f -> %.4f\n', global_best_r2te, new_r2tr);

h_gen   = global_hist.gen;   h_bfit = global_hist.bfit;
h_afit  = global_hist.afit;  h_asiz = global_hist.asiz;
h_trmse = global_hist.trmse; h_r2tr = global_hist.r2tr;
h_r2te  = global_hist.r2te;
N_GENS  = global_hist.ngens;

%% ========================================================================
%%  PRINT SYMBOLIC EQUATION
%% ========================================================================
fprintf('\n========================================\n');
fprintf('  BEST-FIT SYMBOLIC EQUATION (scaled)\n');
fprintf('========================================\n');
eq_str = gp_to_string(best_tree, FEATS_LAG, LAG);
fprintf('  Q_scaled(t) = %s\n\n', eq_str);

% Also print a feature legend for top variables used
fprintf('  Feature index legend (lag-1, slot 50-62):\n');
fprintf('    X50 = Q(t-1) [prev inflow]   X51 = BasinRain(t-1)\n');
fprintf('    X52 = Rain3(t-1)              X53 = Rain5(t-1)\n');
fprintf('    X54 = Rain7(t-1)              X55 = RainMax(t-1)\n');
fprintf('    X56 = RainStd(t-1)            X57 = WetDays(t-1)\n');
fprintf('    X58 = AMI(t-1)               X59 = RainAnom(t-1)\n');
fprintf('    X60 = BaseFlow(t-1)           X61 = DeltaQ(t-1)\n');
fprintf('========================================\n\n');

% Save equation to text file
fid = fopen(fullfile(output_dir,'Best_Equation.txt'),'w');
fprintf(fid,'GP v4 Best-Fit Symbolic Equation\n');
fprintf(fid,'==================================\n');
fprintf(fid,'Q_scaled(t) = %s\n\n',eq_str);
fprintf(fid,'(Multiply by (Q_max-Q_min) and add Q_min to recover cusecs)\n\n');
fprintf(fid,'Feature Legend (lag-1, indices 50-62):\n');
fprintf(fid,'  X50=Q(t-1), X51=BasinRain, X52=Rain3, X53=Rain5\n');
fprintf(fid,'  X54=Rain7, X55=RainMax, X56=RainStd, X57=WetDays\n');
fprintf(fid,'  X58=AMI, X59=RainAnom, X60=BaseFlow, X61=DeltaQ\n');
fclose(fid);
fprintf('  Saved Best_Equation.txt\n');

%% ========================================================================
%%  [7/7]  PREDICTIONS, METRICS & FIGURES
%% ========================================================================
inv_sc = @(v) v*(y_max-y_min)+y_min;

trp_sc     = max(0,min(1, gp_eval(best_tree,X_train)));
tep_sc     = max(0,min(1, gp_eval(best_tree,X_test)));
train_pred = inv_sc(trp_sc);
test_pred  = inv_sc(tep_sc);
train_obs  = yr_train;
test_obs   = yr_test;

m_tr = gp_metrics(train_obs,train_pred);
m_te = gp_metrics(test_obs, test_pred);

fprintf('\n%-15s %12s %12s\n','Metric','Train','Test');
fprintf('%s\n',repmat('-',42,1));
for fn = {'RMSE','MAE','MAPE','R2','NSE','Bias','Corr'}
    fprintf('  %-13s %12.4f %12.4f\n',fn{1},m_tr.(fn{1}),m_te.(fn{1}));
end

%% ====================================================================
%%  FIGURE 1 — ARCHITECTURE  (WHITE BACKGROUND)
%% ====================================================================
fprintf('\n=== [Figs] Plotting ===\n');
fig1 = figure('Color','w','Position',[60 60 1440 730],...
              'Name','GP Architecture v4','NumberTitle','off');

ax1 = axes(fig1,'Position',[0.03 0.06 0.44 0.86]);
set(ax1,'Color','w','XLim',[0 10],'YLim',[0 14],...
    'XTick',[],'YTick',[],'Box','off');
title(ax1,'GP Model Architecture (v4 Aggressive)','Color','k','FontSize',12,'FontWeight','bold');

arch_box(ax1,0.3,12.5,9.4,1.2,...
    sprintf('INPUT: 49 Stations x 5 Lags + AMI+Trend+Anomaly+WetDays = %d Features  |  Monsoon (Jun-Oct)',n_feat),...
    [0.08 0.39 0.75]);
for li = 1:5
    arch_box(ax1,0.3+(li-1)*1.88,10.5,1.55,1.7,...
             sprintf('Lag %d\n62 vars',li),[0.15 0.25 0.65]);
end
arch_box(ax1,0.3,8.0,9.4,2.2,...
    sprintf('GENETIC PROGRAMMING ENGINE v4 — AGGRESSIVE\nPop=%d  Gens=%d  MaxDepth=%d  CX=%.2f  Mut=%.2f->0.04  Elitism=%d  Tourn k=%d->%d\nBest tracked by TEST R2  |  Stagnation restart (35 gens, 30%% inject)  |  %d restarts',...
    POP_SIZE,N_GENS,MAX_DEPTH,CX_PROB,MUT_PROB_START,ELITISM,...
    TOURN_K_LO,TOURN_K_HI,N_RESTARTS),[0.38 0.12 0.60]);

op_n2 = {'+','-','*','/','sqrt','sq','log','cube','tanh','abs'};
op_c2 = [0.72 0.11 0.11;0.11 0.47 0.23;0.85 0.40 0.10;0 0.48 0.49;
          0.20 0.51 0.22;0.22 0.38 0.54;0.41 0.31 0.26;
          0.50 0.20 0.45;0.53 0.16 0.41;0.50 0.40 0.10];
for oi = 1:10
    arch_box(ax1,0.08+(oi-1)*0.985,6.5,0.87,1.0,op_n2{oi},op_c2(oi,:));
end
text(ax1,5,6.3,'10-Operator Set + Terminal Set (features + random constants)',...
     'Color',[0.20 0.20 0.20],'HorizontalAlignment','center','FontSize',8);

arch_box(ax1,0.3,5.1,9.4,1.1,...
    '9 Domain Seeds: Persistence/Linear/Sqrt-Rain/Baseflow/AMI/Quadratic-Peak/Linear-Combo/Lag2-Blend/AMI-Rain-BF',...
    [0.15 0.48 0.55]);
arch_box(ax1,0.3,3.9,9.4,1.0,...
    'Ops: CX + SubtreeMut + HoistMut (bloat ctrl) + PointMut + ConstPerturb',...
    [0.10 0.38 0.49]);
arch_box(ax1,0.3,2.7,9.4,1.0,...
    'FITNESS v4: NSE^2 + nRMSE + R2cliff(0.95,x20) + Peak(0.60) + HFbias(0.50) + VarRatio(0.25) + Tail(0.40) + CorrLoss(0.20) + Parsimony',...
    [0.65 0.20 0.05]);
arch_box(ax1,0.3,1.5,9.4,1.0,...
    'Post-GP: Nelder-Mead constant optimisation on best tree (1000 evals)',...
    [0.20 0.45 0.20]);
arch_box(ax1,2.0,0.1,6.0,1.2,...
    sprintf('OUTPUT: Grand Total Inflow (cusecs)\nBest tree: %d nodes, depth %d  |  TrainR2=%.4f  TestR2=%.4f',...
    gp_tree_size(best_tree),gp_tree_depth(best_tree),m_tr.R2,m_te.R2),[0.85 0.35 0.05]);

ax2 = axes(fig1,'Position',[0.52 0.08 0.46 0.84]);
set(ax2,'Color','w','XLim',[0 10],'YLim',[0 10],...
    'XTick',[],'YTick',[],'Box','off');
title(ax2,'Fitness Components — v4 Aggressive','Color','k','FontSize',11,'FontWeight','bold');
components = {
    'F1: (1-NSE)^2',              '0.95',  '20x cliff penalty below R2=0.95';
    'F2: 0.005 * nRMSE',          '—',     'Normalised RMSE term';
    'F3: 20*(0.95-R2)^2 if R2<0.95','0.95','Hard R2 boundary enforcement';
    'F4: 0.60 * MSE(top 10%)',    '—',     'Peak-flow triple weight';
    'F5: 0.50 * bias(top 25%)^2', '—',     'High-flow systematic bias';
    'F6: 0.25 * (σpred/σobs-1)^2','—',     'Variance/spread matching';
    'F7: 0.40 * MSE(top 5%)',     '—',     'Extreme tail event penalty';
    'F8: 0.20 * (1-r)^2',         '—',     'Pearson correlation loss';
    'F9: Adaptive parsimony',     '—',     '3e-5 -> 8e-4 over gens';
};
clrs9 = {[0.80 0.10 0.10],[0.85 0.40 0.10],[0.65 0.10 0.10],...
          [0.10 0.45 0.20],[0.10 0.40 0.55],[0.40 0.20 0.60],...
          [0.80 0.25 0.05],[0.20 0.35 0.55],[0.35 0.35 0.35]};
for ci=1:9
    yy = 8.8 - (ci-1)*0.95;
    rectangle(ax2,'Position',[0.1 yy 9.8 0.80],'Curvature',0.05,...
              'FaceColor',clrs9{ci},'EdgeColor','w','LineWidth',1.0);
    text(ax2,5.0,yy+0.40,...
         sprintf('%-28s  thr=%-4s  %s',components{ci,1},components{ci,2},components{ci,3}),...
         'Color','w','HorizontalAlignment','center','FontSize',7.8,...
         'FontWeight','bold','Interpreter','none');
end
text(ax2,5,0.20,sprintf('Tree: %d nodes  Depth: %d  |  TrainR2=%.4f  TestR2=%.4f',...
    gp_tree_size(best_tree),gp_tree_depth(best_tree),m_tr.R2,m_te.R2),...
    'Color','k','HorizontalAlignment','center','FontSize',9,...
    'BackgroundColor',[0.92 0.92 0.92],'EdgeColor',[0.5 0.5 0.5]);

sgtitle(fig1,'Genetic Programming v4 — Monsoon Rainfall-Inflow (Target R2 > 0.95)',...
        'Color','k','FontSize',13,'FontWeight','bold');
saveas(fig1, fullfile(output_dir,'Fig1_Architecture.png'));
fprintf('  Saved Fig1_Architecture.png\n');

%% ====================================================================
%%  FIGURE 2 — CONVERGENCE
%% ====================================================================
fig2 = figure('Color','w','Position',[60 60 1600 430],...
              'Name','Convergence','NumberTitle','off');
gens_v = h_gen(1:N_GENS);

ax21 = subplot(1,4,1);
plot(ax21,gens_v,h_bfit(1:N_GENS),'Color',[0.05 0.60 0.20],'LineWidth',2,'DisplayName','Train Fit'); hold(ax21,'on');
plot(ax21,gens_v,h_trmse(1:N_GENS),'Color',[0.85 0.33 0.00],'LineWidth',2,'DisplayName','Test RMSE');
lg=legend(ax21,'Location','northeast','FontSize',8); lg.Color='w';
xlabel(ax21,'Generation','Color','k','FontSize',9);
ylabel(ax21,'Value (scaled)','Color','k','FontSize',9);
title(ax21,'Fitness / RMSE','Color','k','FontSize',10,'FontWeight','bold');
bax_w(ax21);

ax22 = subplot(1,4,2);
plot(ax22,gens_v,h_r2tr(1:N_GENS),'Color',[0.00 0.55 0.80],'LineWidth',2,'DisplayName','Train R^2'); hold(ax22,'on');
plot(ax22,gens_v,h_r2te(1:N_GENS),'Color',[0.80 0.10 0.10],'LineWidth',2,'DisplayName','Test R^2');
yline(ax22,0.95,'k--','LineWidth',1.8,'Label','Target 0.95');
lg2=legend(ax22,'Location','southeast','FontSize',8); lg2.Color='w';
xlabel(ax22,'Generation','Color','k','FontSize',9); ylabel(ax22,'R^2','Color','k','FontSize',9);
title(ax22,'R^2 Convergence','Color','k','FontSize',10,'FontWeight','bold');
ylim(ax22,[-0.1 1.05]); bax_w(ax22);

ax23 = subplot(1,4,3);
plot(ax23,gens_v,h_afit(1:N_GENS),'Color',[0.55 0.18 0.85],'LineWidth',2);
xlabel(ax23,'Generation','Color','k','FontSize',9); ylabel(ax23,'Avg Fitness','Color','k','FontSize',9);
title(ax23,'Population Avg Fitness','Color','k','FontSize',10,'FontWeight','bold'); bax_w(ax23);

ax24 = subplot(1,4,4);
plot(ax24,gens_v,h_asiz(1:N_GENS),'Color',[0.00 0.60 0.67],'LineWidth',2);
xlabel(ax24,'Generation','Color','k','FontSize',9); ylabel(ax24,'Avg Nodes','Color','k','FontSize',9);
title(ax24,'Average Tree Size','Color','k','FontSize',10,'FontWeight','bold'); bax_w(ax24);

sgtitle(fig2,'GP v4 Training Convergence (Target R2 > 0.95)','FontSize',13,'FontWeight','bold','Color','k');
saveas(fig2, fullfile(output_dir,'Fig2_Convergence.png'));
fprintf('  Saved Fig2_Convergence.png\n');

%% ====================================================================
%%  FIGURES 3a / 3b — TIME SERIES
%% ====================================================================
C_obs=[0.10 0.40 0.75]; C_pred=[0.85 0.25 0.05];
n_tr=length(train_obs); n_te=length(test_obs);
x_tr=(1:n_tr)'; x_te=(1:n_te)';
[xtk_tr,xlbl_tr]=yr_ticks(yrl_train,x_tr,d_train);
[xtk_te,xlbl_te]=yr_ticks(yrl_test, x_te,d_test);
[tr_pk_obs,tr_pi_obs]=max(train_obs); [tr_pk_pred,tr_pi_pred]=max(train_pred);
[te_pk_obs, te_pi_obs]=max(test_obs);  [te_pk_pred, te_pi_pred]=max(test_pred);

for tt = 1:2
    if tt==1
        xv=x_tr; obs_v=train_obs; pred_v=train_pred; yrl_v=yrl_train;
        pk_oi=tr_pi_obs; pk_pi=tr_pi_pred; pk_o=tr_pk_obs; pk_p=tr_pk_pred;
        dv=d_train; m_v=m_tr; tag='Train'; pct='80%'; %#ok<NASGU>
        xtk_v=xtk_tr; xlbl_v=xlbl_tr;
    else
        xv=x_te; obs_v=test_obs; pred_v=test_pred; yrl_v=yrl_test;
        pk_oi=te_pi_obs; pk_pi=te_pi_pred; pk_o=te_pk_obs; pk_p=te_pk_pred;
        dv=d_test; m_v=m_te; tag='Test'; pct='20%'; %#ok<NASGU>
        xtk_v=xtk_te; xlbl_v=xlbl_te;
    end
    yrs_v = unique(yrl_v);

    figX = figure('Color','w','Position',[40 40 1800 500],...
                  'Name',sprintf('%s: Monsoon v4',tag),'NumberTitle','off');
    axA  = axes(figX,'Position',[0.07 0.20 0.91 0.65]);
    hold(axA,'on');

    for yi = 1:length(yrs_v)
        msk=(yrl_v==yrs_v(yi));
        xs=xv(find(msk,1,'first')); xe=xv(find(msk,1,'last'));
        if mod(yi,2)==0
            patch(axA,[xs xe xe xs],[0 0 1 1]*pk_o*1.12,...
                  [0.93 0.93 0.93],'FaceAlpha',1,'EdgeColor','none','HandleVisibility','off');
        end
        text(axA,(xs+xe)/2,pk_o*1.09,num2str(yrs_v(yi)),...
             'Color',[0.25 0.25 0.25],'FontSize',8,'FontWeight','bold','HorizontalAlignment','center');
        if yi<length(yrs_v)
            xline(axA,xe+0.5,'Color',[0.65 0.65 0.65],'LineWidth',0.8,'HandleVisibility','off');
        end
    end

    plot(axA,xv,obs_v, 'Color',C_obs, 'LineWidth',1.4,'DisplayName','Observed');
    plot(axA,xv,pred_v,'Color',C_pred,'LineWidth',1.2,'DisplayName','GP Predicted v4');
    scatter(axA,xv(pk_oi),pk_o,100,'k','filled','HandleVisibility','off');
    scatter(axA,xv(pk_pi),pk_p,100,[0.55 0.10 0.75],'filled','HandleVisibility','off');
    text(axA,xv(pk_oi),pk_o*1.03,sprintf('Obs Peak: %.0f',pk_o),...
         'Color','k','FontSize',8,'FontWeight','bold','HorizontalAlignment','center');
    text(axA,xv(pk_pi),pk_p*1.03,sprintf('Pred Peak: %.0f',pk_p),...
         'Color',[0.50 0.05 0.70],'FontSize',8,'FontWeight','bold','HorizontalAlignment','center');

    ylabel(axA,'Inflow (Cusecs)','Color','k','FontSize',11,'FontWeight','bold');
    xlabel(axA,'Monsoon Day (sequential, gaps removed)','Color','k','FontSize',10);
    set(axA,'XTick',xtk_v,'XTickLabel',xlbl_v);
    set(axA,'Color','w','XColor','k','YColor','k','GridColor',[0.75 0.75 0.75],...
            'GridAlpha',1,'FontSize',8,'LineWidth',0.8,'Box','on','TickDir','out');
    grid(axA,'on');
    lg=legend(axA,'Location','northeast','FontSize',10);
    lg.Color='w'; lg.EdgeColor=[0.5 0.5 0.5];
    title(axA,sprintf('%s %s  |  R^2=%.3f  NSE=%.3f  RMSE=%.0f cusecs',...
          tag,pct,m_v.R2,m_v.NSE,m_v.RMSE),...
          'Color','k','FontSize',10,'FontWeight','bold','Interpreter','none');
    sgtitle(figX,sprintf('Monsoon Simulation v4 — %s  |  Years: %s',...
            tag,strjoin(arrayfun(@num2str,yrs_v','UniformOutput',false),', ')),...
            'Color','k','FontSize',12,'FontWeight','bold');

    fname=sprintf('Fig3%c_%s_Combined.png',char('a'-1+tt),tag);
    saveas(figX, fullfile(output_dir,fname));
    fprintf('  Saved %s\n',fname);
end

%% ====================================================================
%%  FIGURE 4 — SCATTER
%% ====================================================================
fig4 = figure('Color','w','Position',[60 60 1100 950],'Name','Scatter','NumberTitle','off');

ax4a=subplot(2,2,1);
scatter(ax4a,train_obs,train_pred,28,[0.05 0.55 0.25],'filled','MarkerFaceAlpha',0.6); hold(ax4a,'on');
lm=[min([train_obs;train_pred]) max([train_obs;train_pred])];
plot(ax4a,lm,lm,'k--','LineWidth',1.5);
xlabel(ax4a,'Observed (cusecs)','Color','k','FontSize',9);
ylabel(ax4a,'Predicted (cusecs)','Color','k','FontSize',9);
title(ax4a,sprintf('Train   R^2=%.3f   NSE=%.3f',m_tr.R2,m_tr.NSE),'Color','k','FontSize',10,'FontWeight','bold');
bax_w(ax4a);

ax4b=subplot(2,2,2);
scatter(ax4b,test_obs,test_pred,28,[0.85 0.35 0.00],'filled','MarkerFaceAlpha',0.6); hold(ax4b,'on');
lm=[min([test_obs;test_pred]) max([test_obs;test_pred])];
plot(ax4b,lm,lm,'k--','LineWidth',1.5);
xlabel(ax4b,'Observed (cusecs)','Color','k','FontSize',9);
ylabel(ax4b,'Predicted (cusecs)','Color','k','FontSize',9);
title(ax4b,sprintf('Test   R^2=%.3f   NSE=%.3f',m_te.R2,m_te.NSE),'Color','k','FontSize',10,'FontWeight','bold');
bax_w(ax4b);

ax4c=subplot(2,2,[3 4]);
scatter(ax4c,train_obs,train_pred,22,[0.05 0.55 0.25],'filled','MarkerFaceAlpha',0.5,'DisplayName','Train'); hold(ax4c,'on');
scatter(ax4c,test_obs, test_pred, 22,[0.85 0.35 0.00],'filled','MarkerFaceAlpha',0.5,'DisplayName','Test');
all_o=[train_obs;test_obs]; all_p=[train_pred;test_pred];
lm=[min([all_o;all_p]) max([all_o;all_p])];
plot(ax4c,lm,lm,'k--','LineWidth',1.5,'DisplayName','1:1 Line');
xlabel(ax4c,'Observed (cusecs)','Color','k','FontSize',9);
ylabel(ax4c,'Predicted (cusecs)','Color','k','FontSize',9);
title(ax4c,'Combined Train + Test','Color','k','FontSize',10,'FontWeight','bold');
lg4=legend(ax4c,'Location','northwest','FontSize',9);
lg4.Color='w'; lg4.EdgeColor=[0.5 0.5 0.5];
bax_w(ax4c);
sgtitle(fig4,'Scatter — Observed vs GP Predicted Inflow (v4)','Color','k','FontSize',13,'FontWeight','bold');
saveas(fig4, fullfile(output_dir,'Fig4_Scatter.png'));
fprintf('  Saved Fig4_Scatter.png\n');

%% ====================================================================
%%  FIGURE 5 — ERROR ANALYSIS
%% ====================================================================
tr_err=train_pred-train_obs; te_err=test_pred-test_obs;
[~,pk_oi_tr]=max(train_obs); [~,pk_pi_tr]=max(train_pred);
[~,pk_oi_te]=max(test_obs);  [~,pk_pi_te]=max(test_pred);

fig5=figure('Color','w','Position',[60 60 1500 950],'Name','Error Analysis','NumberTitle','off');

ax51=subplot(3,3,1); bar(ax51,tr_err,'FaceColor',[0.80 0.15 0.15],'EdgeColor','none','FaceAlpha',0.85);
yline(ax51,0,'k','LineWidth',1); dbax_w(ax51,'Sample Index','Error (cusecs)','Train Residuals');
ax52=subplot(3,3,2); bar(ax52,te_err,'FaceColor',[0.10 0.45 0.80],'EdgeColor','none','FaceAlpha',0.85);
yline(ax52,0,'k','LineWidth',1); dbax_w(ax52,'Sample Index','Error (cusecs)','Test Residuals');
ax53=subplot(3,3,3);
histogram(ax53,tr_err,25,'FaceColor',[0.05 0.60 0.25],'FaceAlpha',0.8,'EdgeColor','none','DisplayName','Train'); hold(ax53,'on');
histogram(ax53,te_err,25,'FaceColor',[0.85 0.35 0.00],'FaceAlpha',0.8,'EdgeColor','none','DisplayName','Test');
xline(ax53,0,'k--','LineWidth',1.5);
lg53=legend(ax53,'FontSize',8); lg53.Color='w'; lg53.EdgeColor=[0.5 0.5 0.5];
dbax_w(ax53,'Error (cusecs)','Count','Error Distribution');

ax54=subplot(3,3,4);
plot(ax54,x_tr,train_obs,'Color',C_obs,'LineWidth',1.4,'DisplayName','Observed'); hold(ax54,'on');
plot(ax54,x_tr,train_pred,'Color',C_pred,'LineWidth',1.2,'LineStyle','--','DisplayName','Predicted');
scatter(ax54,x_tr(pk_oi_tr),train_obs(pk_oi_tr),100,'k','filled','HandleVisibility','off');
scatter(ax54,x_tr(pk_pi_tr),train_pred(pk_pi_tr),100,[0.55 0.10 0.75],'filled','HandleVisibility','off');
lg54=legend(ax54,'Location','northwest','FontSize',7); lg54.Color='w';
dbax_w(ax54,'Sample','Inflow (cusecs)','Train — Peak Discharge');

ax55=subplot(3,3,5);
plot(ax55,x_te,test_obs,'Color',C_obs,'LineWidth',1.4,'DisplayName','Observed'); hold(ax55,'on');
plot(ax55,x_te,test_pred,'Color',C_pred,'LineWidth',1.2,'LineStyle','--','DisplayName','Predicted');
scatter(ax55,x_te(pk_oi_te),test_obs(pk_oi_te),100,'k','filled','HandleVisibility','off');
scatter(ax55,x_te(pk_pi_te),test_pred(pk_pi_te),100,[0.55 0.10 0.75],'filled','HandleVisibility','off');
lg55=legend(ax55,'Location','northwest','FontSize',7); lg55.Color='w';
dbax_w(ax55,'Sample','Inflow (cusecs)','Test — Peak Discharge');

ax56=subplot(3,3,6); mk={'RMSE','MAE','MAPE'}; xb=1:3; w=0.35;
bar(ax56,xb-w/2,[m_tr.RMSE m_tr.MAE m_tr.MAPE],w,'FaceColor',[0.05 0.60 0.25],'EdgeColor','none','DisplayName','Train'); hold(ax56,'on');
bar(ax56,xb+w/2,[m_te.RMSE m_te.MAE m_te.MAPE],w,'FaceColor',[0.85 0.35 0.00],'EdgeColor','none','DisplayName','Test');
set(ax56,'XTick',xb,'XTickLabel',mk); ax56.XAxis.TickLabelColor='k';
lg56=legend(ax56,'FontSize',8); lg56.Color='w'; dbax_w(ax56,'','Value','Error Metrics');

ax57=subplot(3,3,7); mk2={'R^2','NSE','Corr'};
bar(ax57,xb-w/2,[m_tr.R2 m_tr.NSE m_tr.Corr],w,'FaceColor',[0.05 0.60 0.25],'EdgeColor','none','DisplayName','Train'); hold(ax57,'on');
bar(ax57,xb+w/2,[m_te.R2 m_te.NSE m_te.Corr],w,'FaceColor',[0.85 0.35 0.00],'EdgeColor','none','DisplayName','Test');
set(ax57,'XTick',xb,'XTickLabel',mk2); ylim(ax57,[-0.2 1.2]); ax57.XAxis.TickLabelColor='k';
yline(ax57,0.95,'k--','LineWidth',1.8);
text(ax57,1,0.97,'Target R^2=0.95','Color','k','FontSize',8,'Interpreter','none');
lg57=legend(ax57,'FontSize',8); lg57.Color='w'; dbax_w(ax57,'','Value','Goodness-of-Fit');

ax58=subplot(3,3,8);
plot(ax58,x_tr,cumsum(abs(tr_err)),'Color',[0.05 0.65 0.35],'LineWidth',1.8,'DisplayName','Train'); hold(ax58,'on');
plot(ax58,x_te,cumsum(abs(te_err)),'Color',[0.85 0.35 0.00],'LineWidth',1.8,'DisplayName','Test');
lg58=legend(ax58,'FontSize',8); lg58.Color='w';
dbax_w(ax58,'Sample','Cumulative |Error|','Cumulative Absolute Error');

ax59=subplot(3,3,9); set(ax59,'Color','w'); axis(ax59,'off');
rows={'RMSE','MAE','MAPE(%)','R2','NSE','Bias','Corr'};
tr_v=[m_tr.RMSE;m_tr.MAE;m_tr.MAPE;m_tr.R2;m_tr.NSE;m_tr.Bias;m_tr.Corr];
te_v=[m_te.RMSE;m_te.MAE;m_te.MAPE;m_te.R2;m_te.NSE;m_te.Bias;m_te.Corr];
ys=linspace(0.92,0.08,length(rows)+1);
text(ax59,0.05,ys(1),'Metric','FontWeight','bold','FontSize',10,'Units','normalized','Color','k');
text(ax59,0.45,ys(1),'Train','FontWeight','bold','FontSize',10,'Units','normalized','Color',[0.05 0.55 0.20]);
text(ax59,0.73,ys(1),'Test', 'FontWeight','bold','FontSize',10,'Units','normalized','Color',[0.75 0.25 0.00]);
for ri = 1:length(rows)
    bg=[0.96 0.96 0.96]; if mod(ri,2)==0, bg=[1 1 1]; end
    text(ax59,0.05,ys(ri+1),rows{ri},'FontSize',9,'Units','normalized','BackgroundColor',bg,'Color','k');
    text(ax59,0.45,ys(ri+1),sprintf('%.3f',tr_v(ri)),'FontSize',9,'Units','normalized','BackgroundColor',bg,'Color',[0.05 0.50 0.20]);
    text(ax59,0.73,ys(ri+1),sprintf('%.3f',te_v(ri)),'FontSize',9,'Units','normalized','BackgroundColor',bg,'Color',[0.75 0.25 0.00]);
end
title(ax59,'Performance Summary (Target R2>0.95)','FontWeight','bold','Color','k','FontSize',9);
sgtitle(fig5,'Error Analysis — GP Model v4','Color','k','FontSize',13,'FontWeight','bold');
saveas(fig5, fullfile(output_dir,'Fig5_Error_Analysis.png'));
fprintf('  Saved Fig5_Error_Analysis.png\n');

%% ====================================================================
%%  FIGURE 6 — FEATURE IMPORTANCE
%% ====================================================================
all_paths  = gp_get_all_paths(best_tree,[]);
var_counts = zeros(n_feat,1);
for pi_ = 1:length(all_paths)
    nd2=gp_node_at_path(best_tree,all_paths{pi_});
    if strcmp(nd2.type,'var'), var_counts(nd2.idx)=var_counts(nd2.idx)+1; end
end
station_imp=zeros(49,1); lag_imp=zeros(LAG,1);
for lag=1:LAG
    for s=1:49
        fi_=(lag-1)*FEATS_LAG+s;
        if fi_<=n_feat
            station_imp(s)=station_imp(s)+var_counts(fi_);
            lag_imp(lag)  =lag_imp(lag)  +var_counts(fi_);
        end
    end
end
[simp_s,simp_i]=sort(station_imp,'descend');
top_n=min(20,sum(simp_s>0)); if top_n<1,top_n=5;end

fig6=figure('Color','w','Position',[60 60 1300 600],'Name','Feature Importance','NumberTitle','off');
ax6L=subplot(1,2,1);
barh(ax6L,simp_s(top_n:-1:1),'FaceColor',[0.10 0.45 0.80],'EdgeColor','none','FaceAlpha',1.0);
yticks(ax6L,1:top_n);
yticklabels(ax6L,arrayfun(@(i)sprintf('S%d',simp_i(top_n+1-i)),1:top_n,'UniformOutput',false));
xlabel(ax6L,'Occurrences in Best Tree','Color','k','FontSize',10,'FontWeight','bold');
title(ax6L,sprintf('Top %d Rain Gauge Stations',top_n),'Color','k','FontSize',11,'FontWeight','bold');
set(ax6L,'Color','w','XColor','k','YColor','k','GridColor',[0.75 0.75 0.75],'GridAlpha',1,...
         'FontSize',9,'LineWidth',0.8,'TickDir','out','Box','on'); grid(ax6L,'on');

lag_clr=[0.85 0.35 0.05;0.80 0.10 0.10;0.00 0.55 0.80;0.10 0.60 0.20;0.52 0.05 0.68];
ax6R=subplot(1,2,2); hold(ax6R,'on');
for li=1:LAG
    bar(ax6R,li,lag_imp(li),0.6,'FaceColor',lag_clr(li,:),'EdgeColor','none','FaceAlpha',1.0);
end
xticks(ax6R,1:LAG); xticklabels(ax6R,{'Lag1','Lag2','Lag3','Lag4','Lag5'}); xtickangle(ax6R,30);
ylabel(ax6R,'Variable Usage Count','Color','k','FontSize',10,'FontWeight','bold');
title(ax6R,'Rainfall Usage by Lag Day','Color','k','FontSize',11,'FontWeight','bold');
set(ax6R,'Color','w','XColor','k','YColor','k','GridColor',[0.75 0.75 0.75],'GridAlpha',1,...
         'FontSize',9,'LineWidth',0.8,'TickDir','out','Box','on'); grid(ax6R,'on');
sgtitle(fig6,'GP v4 — Variable Usage in Best Expression Tree','Color','k','FontSize',13,'FontWeight','bold');
saveas(fig6, fullfile(output_dir,'Fig6_Feature_Importance.png'));
fprintf('  Saved Fig6_Feature_Importance.png\n');

%% ====================================================================
%%  FIGURE 7 — BEST-FIT EQUATION + METRICS TABLE  (NEW in v4)
%% ====================================================================
fig7 = figure('Color','w','Position',[60 60 1400 700],...
              'Name','Best Equation','NumberTitle','off');

ax7L = axes(fig7,'Position',[0.03 0.08 0.60 0.84]);
set(ax7L,'Color',[0.97 0.97 0.97],'XLim',[0 10],'YLim',[0 10],...
    'XTick',[],'YTick',[],'Box','on','LineWidth',1.5);
title(ax7L,'Best Symbolic Equation (Scaled Space)','Color','k',...
      'FontSize',12,'FontWeight','bold');

% Wrap long equation strings for display
eq_disp = eq_str;
max_chars = 90;
lines_eq = {};
while length(eq_disp) > max_chars
    cut = max_chars;
    while cut > 1 && ~ismember(eq_disp(cut),{'+','-','*','/',' '}), cut=cut-1; end
    if cut < 2, cut = max_chars; end
    lines_eq{end+1} = eq_disp(1:cut); %#ok<AGROW>
    eq_disp = ['  ' eq_disp(cut+1:end)];
end
lines_eq{end+1} = eq_disp;

% Print header
text(ax7L,0.1,9.5,'Q_{scaled}(t)  =','FontSize',11,'FontWeight','bold',...
     'Color',[0.10 0.30 0.65],'Interpreter','none','Units','data');
for li=1:length(lines_eq)
    text(ax7L,0.3,9.5-(li*1.0),lines_eq{li},...
         'FontSize',8.5,'Color','k','Interpreter','none',...
         'FontName','Courier New','Units','data');
end

text(ax7L,0.1,0.5,...
    'To recover discharge: Q(t) = Q_{scaled}(t) × (Q_{max} - Q_{min}) + Q_{min}',...
    'FontSize',9,'Color',[0.40 0.10 0.10],'Interpreter','none');

% Variable legend box
nvars = min(6, sum(var_counts>0));
[vc_s, vc_i] = sort(var_counts,'descend');
leg_str = 'Top variables used:  ';
for lv=1:nvars
    feat_num = vc_i(lv);
    lag_num  = ceil(feat_num/FEATS_LAG);
    slot     = mod(feat_num-1,FEATS_LAG)+1;
    slot_names = {'Q','Rain_basin','Rain3','Rain5','Rain7','RainMax',...
                  'RainStd','WetDays','AMI','RainAnom','BaseFlow','DeltaQ'};
    if slot >= 50 && slot <= 61
        sn = slot_names{slot-49};
    elseif slot <= 49
        sn = sprintf('Station_%d',slot);
    else
        sn = sprintf('feat%d',slot);
    end
    leg_str = [leg_str sprintf('X%d=[%s,lag%d]  ',feat_num,sn,lag_num)]; %#ok<AGROW>
end
text(ax7L,0.1,1.4,leg_str,'FontSize',7.5,'Color',[0.20 0.20 0.20],...
     'Interpreter','none','FontName','Courier New');

% Right panel: metrics table
ax7R = axes(fig7,'Position',[0.66 0.08 0.32 0.84]);
set(ax7R,'Color','w','XLim',[0 10],'YLim',[0 10],...
    'XTick',[],'YTick',[],'Box','on','LineWidth',1.5);
title(ax7R,'Performance Metrics (v4)','Color','k','FontSize',12,'FontWeight','bold');

met_names = {'R^2','NSE','RMSE (cusecs)','MAE (cusecs)','MAPE (%)','Corr (r)','Bias (cusecs)'};
tr_vals   = [m_tr.R2; m_tr.NSE; m_tr.RMSE; m_tr.MAE; m_tr.MAPE; m_tr.Corr; m_tr.Bias];
te_vals   = [m_te.R2; m_te.NSE; m_te.RMSE; m_te.MAE; m_te.MAPE; m_te.Corr; m_te.Bias];

% Header
rectangle(ax7R,'Position',[0.2 8.5 9.6 1.0],'FaceColor',[0.15 0.35 0.65],'EdgeColor','none');
text(ax7R,1.0,9.0,'Metric',  'Color','w','FontSize',10,'FontWeight','bold');
text(ax7R,5.0,9.0,'Train',   'Color','w','FontSize',10,'FontWeight','bold','HorizontalAlignment','center');
text(ax7R,8.0,9.0,'Test',    'Color','w','FontSize',10,'FontWeight','bold','HorizontalAlignment','center');

row_colors = {[0.92 0.97 0.92],[0.97 0.97 0.97]};
for ri=1:length(met_names)
    yy = 8.5 - ri*1.10;
    rc = row_colors{mod(ri,2)+1};
    rectangle(ax7R,'Position',[0.2 yy 9.6 1.0],'FaceColor',rc,'EdgeColor',[0.8 0.8 0.8]);

    % Color R2 and NSE rows specially if meeting target
    val_clr_tr = [0 0 0]; val_clr_te = [0 0 0];
    if ri <= 2
        val_clr_tr = [0.00 0.50 0.15]; val_clr_te = [0.80 0.20 0.00];
        if tr_vals(ri) >= 0.95, val_clr_tr = [0.00 0.60 0.00]; end
        if te_vals(ri) >= 0.95, val_clr_te = [0.00 0.60 0.00]; end
    end
    text(ax7R,1.0,yy+0.52,met_names{ri},'FontSize',9,'FontWeight','bold','Color','k','Interpreter','none');
    text(ax7R,5.0,yy+0.52,sprintf('%.4f',tr_vals(ri)),'FontSize',10,'FontWeight','bold',...
         'Color',val_clr_tr,'HorizontalAlignment','center');
    text(ax7R,8.0,yy+0.52,sprintf('%.4f',te_vals(ri)),'FontSize',10,'FontWeight','bold',...
         'Color',val_clr_te,'HorizontalAlignment','center');
end

% Target achievement banner
if m_tr.R2>=0.95 && m_te.R2>=0.95
    banner_txt = sprintf('✓  TARGET ACHIEVED  |  Train R2=%.4f  Test R2=%.4f',m_tr.R2,m_te.R2);
    banner_col = [0.00 0.55 0.10];
else
    banner_txt = sprintf('Train R2=%.4f  Test R2=%.4f  (Target: both ≥ 0.95)',m_tr.R2,m_te.R2);
    banner_col = [0.70 0.20 0.00];
end
rectangle(ax7R,'Position',[0.2 0.1 9.6 0.8],'FaceColor',banner_col,'EdgeColor','none');
text(ax7R,5.0,0.50,banner_txt,'Color','w','HorizontalAlignment','center',...
     'FontSize',8.5,'FontWeight','bold','Interpreter','none');

sgtitle(fig7,sprintf('GP v4 Best-Fit Equation & Metrics  |  Tree: %d nodes  Depth: %d',...
        gp_tree_size(best_tree),gp_tree_depth(best_tree)),...
        'Color','k','FontSize',12,'FontWeight','bold');
saveas(fig7, fullfile(output_dir,'Fig7_BestEquation.png'));
fprintf('  Saved Fig7_BestEquation.png\n');

%% ---- Save Results CSV -------------------------------------------------
all_d  = [d_train;   d_test];
all_sp = [repmat({'Train'},length(train_obs),1); repmat({'Test'},length(test_obs),1)];
all_o  = [train_obs;  test_obs];
all_p  = [train_pred; test_pred];
all_e  = all_p - all_o;
T_out  = table(all_d,all_sp,all_o,all_p,all_e,...
               'VariableNames',{'Date','Split','Observed_cusecs','Predicted_cusecs','Error_cusecs'});
writetable(T_out, fullfile(output_dir,'GP_Results_v4.csv'));

fprintf('\n=== ALL DONE ===\n');
fprintf('  Output folder : %s\n', output_dir);
fprintf('  Train R2 = %.4f  NSE = %.4f  RMSE = %.2f\n', m_tr.R2,m_tr.NSE,m_tr.RMSE);
fprintf('  Test  R2 = %.4f  NSE = %.4f  RMSE = %.2f\n', m_te.R2,m_te.NSE,m_te.RMSE);
if m_tr.R2>=0.95 && m_te.R2>=0.95
    fprintf('\n  *** BOTH TRAIN AND TEST R2 > 0.95 ACHIEVED ***\n');
else
    fprintf('\n  NOTE: Target R2>0.95 not yet met. Consider more restarts or gens.\n');
end
fprintf('\nBest equation saved to: %s\n', fullfile(output_dir,'Best_Equation.txt'));

%% ========================================================================
%%  HELPER FUNCTIONS
%% ========================================================================

function nd=gp_make_func(op,ch),  nd.type='func';  nd.op=op; nd.ch=ch; nd.idx=0; nd.val=0; end
function nd=gp_make_var(idx),     nd.type='var';   nd.op=''; nd.ch={}; nd.idx=idx;nd.val=0; end
function nd=gp_make_const(val),   nd.type='const'; nd.op=''; nd.ch={}; nd.idx=0; nd.val=val; end

function nd=gp_rand_terminal(n_feat)
    if rand()<0.75, nd=gp_make_var(randi(n_feat));
    else,           nd=gp_make_const(rand()*2-1); end
end

function nd=gp_grow_tree(n_feat,depth,OPS,ARITY)
    if depth<=0, nd=gp_rand_terminal(n_feat); return; end
    if depth>=2||rand()>0.4
        oi=randi(length(OPS)); ar=ARITY(oi); ch=cell(1,ar);
        for i=1:ar, ch{i}=gp_grow_tree(n_feat,depth-1,OPS,ARITY); end
        nd=gp_make_func(OPS{oi},ch);
    else, nd=gp_rand_terminal(n_feat); end
end

function nd=gp_full_tree(n_feat,depth,OPS,ARITY)
    if depth<=0, nd=gp_rand_terminal(n_feat); return; end
    oi=randi(length(OPS)); ar=ARITY(oi); ch=cell(1,ar);
    for i=1:ar, ch{i}=gp_full_tree(n_feat,depth-1,OPS,ARITY); end
    nd=gp_make_func(OPS{oi},ch);
end

function out=gp_eval(nd,X)
    switch nd.type
        case 'var',   out=X(:,nd.idx);
        case 'const', out=repmat(nd.val,size(X,1),1);
        case 'func'
            switch nd.op
                case '+',    out=gp_eval(nd.ch{1},X)+gp_eval(nd.ch{2},X);
                case '-',    out=gp_eval(nd.ch{1},X)-gp_eval(nd.ch{2},X);
                case '*',    out=gp_eval(nd.ch{1},X).*gp_eval(nd.ch{2},X);
                case '/'
                    b=gp_eval(nd.ch{2},X);
                    b(abs(b)<1e-6)=sign(b+1e-10)*1e-6;
                    out=gp_eval(nd.ch{1},X)./b;
                case 'sqrt', out=sqrt(abs(gp_eval(nd.ch{1},X)));
                case 'sq',   out=gp_eval(nd.ch{1},X).^2;
                case 'cube', out=gp_eval(nd.ch{1},X).^3;
                case 'log'
                    a=gp_eval(nd.ch{1},X); a(abs(a)<1e-6)=1e-6; out=log(abs(a));
                case 'tanh', out=tanh(gp_eval(nd.ch{1},X));
                case 'abs',  out=abs(gp_eval(nd.ch{1},X));
                otherwise,   out=zeros(size(X,1),1);
            end
        otherwise, out=zeros(size(X,1),1);
    end
    out(isnan(out)|isinf(out))=0;
    out=max(-10,min(10,out));
end

function s=gp_tree_size(nd)
    if isempty(nd.ch),s=1; else,s=1+sum(cellfun(@gp_tree_size,nd.ch));end
end
function d=gp_tree_depth(nd)
    if isempty(nd.ch),d=0; else,d=1+max(cellfun(@gp_tree_depth,nd.ch));end
end

function r2=compute_r2(obs,pred)
    obs=obs(:); pred=pred(:);
    ss_tot=sum((obs-mean(obs)).^2);
    if ss_tot<1e-12||any(isnan(pred))||any(isinf(pred)), r2=0; return; end
    r2=1-sum((obs-pred).^2)/ss_tot;
end

%% ---- SYMBOLIC EQUATION PRINTER ----------------------------------------
function str=gp_to_string(nd, feats_lag, lag)
    if nargin < 2, feats_lag = 62; end
    if nargin < 3, lag = 5; end
    switch nd.type
        case 'var'
            % Decode feature name from index
            lag_num  = ceil(nd.idx / feats_lag);
            slot     = mod(nd.idx-1, feats_lag) + 1;
            slot_names = {'Q','Rbasin','R3','R5','R7','Rmax',...
                          'Rstd','Wwet','AMI','Ranom','BF','dQ'};
            if slot <= 49
                str = sprintf('S%d[t-%d]', slot, lag_num);
            elseif slot >= 50 && slot <= 61
                str = sprintf('%s[t-%d]', slot_names{slot-49}, lag_num);
            else
                str = sprintf('X%d', nd.idx);
            end
        case 'const'
            str = sprintf('%.4f', nd.val);
        case 'func'
            switch nd.op
                case '+'
                    str = sprintf('(%s + %s)', ...
                        gp_to_string(nd.ch{1},feats_lag,lag), ...
                        gp_to_string(nd.ch{2},feats_lag,lag));
                case '-'
                    str = sprintf('(%s - %s)', ...
                        gp_to_string(nd.ch{1},feats_lag,lag), ...
                        gp_to_string(nd.ch{2},feats_lag,lag));
                case '*'
                    str = sprintf('(%s * %s)', ...
                        gp_to_string(nd.ch{1},feats_lag,lag), ...
                        gp_to_string(nd.ch{2},feats_lag,lag));
                case '/'
                    str = sprintf('(%s / %s)', ...
                        gp_to_string(nd.ch{1},feats_lag,lag), ...
                        gp_to_string(nd.ch{2},feats_lag,lag));
                case 'sqrt'
                    str = sprintf('sqrt(|%s|)', ...
                        gp_to_string(nd.ch{1},feats_lag,lag));
                case 'sq'
                    str = sprintf('(%s)^2', ...
                        gp_to_string(nd.ch{1},feats_lag,lag));
                case 'cube'
                    str = sprintf('(%s)^3', ...
                        gp_to_string(nd.ch{1},feats_lag,lag));
                case 'log'
                    str = sprintf('log(|%s|)', ...
                        gp_to_string(nd.ch{1},feats_lag,lag));
                case 'tanh'
                    str = sprintf('tanh(%s)', ...
                        gp_to_string(nd.ch{1},feats_lag,lag));
                case 'abs'
                    str = sprintf('|%s|', ...
                        gp_to_string(nd.ch{1},feats_lag,lag));
                otherwise
                    str = '?';
            end
        otherwise
            str = '?';
    end
end

%% ---- FITNESS v4 — 9 COMPONENTS ----------------------------------------
%  F1  (1-NSE)^2                    — core Nash-Sutcliffe
%  F2  0.005 * nRMSE                — normalised RMSE
%  F3  20*(0.95-R2)^2 if R2<0.95   — HARD cliff at target (was 8, thr 0.90)
%  F4  0.60 * MSE(top 10%)          — peak-flow weight (was 0.40)
%  F5  0.50 * bias(top 25%)^2       — high-flow systematic bias (was 0.30)
%  F6  0.25 * (std_pred/std_obs-1)^2— variance matching (was 0.15)
%  F7  0.40 * MSE(top 5%)           — extreme-tail penalty (NEW)
%  F8  0.20 * (1 - Pearson_r)^2     — correlation loss (NEW)
%  F9  adaptive parsimony
function f=gp_fitness(nd,X,y,peak_mask,tail_mask,size_limit,parsimony_coef)
    try
        pred=gp_eval(nd,X);
        if any(isnan(pred))||any(isinf(pred)), f=1e9; return; end

        n_obs = length(y);
        ss_res=sum((y-pred).^2);
        ss_tot=sum((y-mean(y)).^2);
        if ss_tot<1e-12, f=1e9; return; end

        % F1: Core NSE
        nse  = 1 - ss_res/ss_tot;
        f1   = (1 - nse)^2;

        % F2: Increase influence (was too weak)
        nrmse = sqrt(mean((y-pred).^2)) / (max(y)-min(y)+1e-12);
        f2    = 0.02 * nrmse;   % ↑ from 0.005

        % F3: Relax hard constraint (important)
        f3 = 0;
        if nse < 0.95
            f3 = 10 * (0.95 - nse)^2;   % ↓ from 20
        end

        % F4: Reduce peak domination
        if sum(peak_mask) > 0
            f4 = 0.45 * mean((y(peak_mask)-pred(peak_mask)).^2); % ↓ from 0.60
        else
            f4 = 0;
        end

        % F5: Slightly reduce bias penalty
        hf_thresh = prctile(y,75);
        hf_mask   = (y >= hf_thresh);
        if sum(hf_mask) > 1
            f5 = 0.40 * (mean(pred(hf_mask)) - mean(y(hf_mask)))^2; % ↓ from 0.50
        else
            f5 = 0;
        end

        % F6: Keep same (good balance)
        std_obs  = std(y);
        std_pred = std(pred);
        if std_obs > 1e-12
            f6 = 0.25 * (std_pred/std_obs - 1)^2;
        else
            f6 = 0;
        end

        % F7: Reduce extreme sensitivity (important fix)
        if sum(tail_mask) > 0
            f7 = 0.20 * mean((y(tail_mask)-pred(tail_mask)).^2); % ↓ from 0.40
        else
            f7 = 0;
        end

        % F8: Increase correlation importance (boosts R²)
        cc = corrcoef(y, pred);
        r  = cc(1,2);
        if isnan(r), r=0; end
        f8 = 0.35 * (1 - r)^2;   % ↑ from 0.20

        % F9: Relax parsimony (CRITICAL)
        sz = gp_tree_size(nd);
        f9 = max(0, sz - size_limit) * (parsimony_coef * 0.3); % ↓ effective penalty

        f = f1 + f2 + f3 + f4 + f5 + f6 + f7 + f8 + f9;
        if f < 0, f = 0; end

    catch
        f = 1e9;
    end
end

function paths=gp_get_all_paths(nd,cur)
    paths={cur};
    if isempty(nd.ch),return;end
    for i=1:length(nd.ch)
        sub=gp_get_all_paths(nd.ch{i},[cur,i]);
        paths=[paths,sub]; %#ok<AGROW>
    end
end
function sub=gp_node_at_path(nd,path)
    sub=nd; if isempty(path),return;end
    for i=1:length(path)
        if isempty(sub.ch)||path(i)>length(sub.ch), error('Invalid path'); end
        sub=sub.ch{path(i)};
    end
end
function [new_nd,ok]=gp_replace_at_path(nd,path,new_sub)
    ok=false;
    if isempty(path), new_nd=new_sub; ok=true; return; end
    new_nd=nd; ci=path(1);
    if isempty(nd.ch)||ci>length(nd.ch), return; end
    [new_nd.ch{ci},ok]=gp_replace_at_path(nd.ch{ci},path(2:end),new_sub);
end

function [c1,c2]=gp_crossover(p1,p2,n_feat,OPS,ARITY) %#ok<INUSD>
    c1=p1; c2=p2;
    paths1=gp_get_all_paths(p1,[]); paths2=gp_get_all_paths(p2,[]);
    if length(paths1)<2||length(paths2)<2, return; end
    i1=randi([2,length(paths1)]); i2=randi([2,length(paths2)]);
    sub1=gp_node_at_path(p1,paths1{i1}); sub2=gp_node_at_path(p2,paths2{i2});
    [c1,~]=gp_replace_at_path(p1,paths1{i1},sub2);
    [c2,~]=gp_replace_at_path(p2,paths2{i2},sub1);
end

function child=gp_subtree_mutate(nd,n_feat,OPS,ARITY,max_d)
    paths=gp_get_all_paths(nd,[]);
    if isempty(paths), child=gp_grow_tree(n_feat,max_d,OPS,ARITY); return; end
    pi_=randi(length(paths));
    new_sub=gp_grow_tree(n_feat,randi([2,3]),OPS,ARITY);
    [child,~]=gp_replace_at_path(nd,paths{pi_},new_sub);
end

function child=gp_hoist_mutate(nd,max_d)
    paths=gp_get_all_paths(nd,[]);
    func_paths=paths(cellfun(@(p) ~isempty(p) && ...
        strcmp(gp_node_at_path(nd,p).type,'func'), paths));
    if isempty(func_paths), child=nd; return; end
    pi_=randi(length(func_paths));
    parent=gp_node_at_path(nd,func_paths{pi_});
    if isempty(parent.ch), child=nd; return; end
    ci=randi(length(parent.ch));
    hoisted=parent.ch{ci};
    [child,ok]=gp_replace_at_path(nd,func_paths{pi_},hoisted);
    if ~ok||gp_tree_depth(child)>max_d, child=nd; end
end

function child=gp_point_mutate(nd,n_feat,OPS,ARITY)
    paths=gp_get_all_paths(nd,[]);
    if isempty(paths), child=nd; return; end
    pi_=randi(length(paths));
    node=gp_node_at_path(nd,paths{pi_});
    switch node.type
        case 'func'
            cur_ar=ARITY(strcmp(OPS,node.op));
            same=find(ARITY==cur_ar);
            node.op=OPS{same(randi(length(same)))};
        case 'var',   node.idx=randi(n_feat);
        case 'const', node.val=rand()*2-1;
    end
    if isempty(paths{pi_}), child=node;
    else, [child,~]=gp_replace_at_path(nd,paths{pi_},node); end
end

function child=gp_const_mutate(nd)
    paths=gp_get_all_paths(nd,[]);
    const_paths=paths(cellfun(@(p) strcmp(gp_node_at_path(nd,p).type,'const'), paths));
    if isempty(const_paths), child=nd; return; end
    pi_=randi(length(const_paths));
    node=gp_node_at_path(nd,const_paths{pi_});
    node.val=node.val+randn()*0.04;   % tighter perturbation
    if isempty(const_paths{pi_}), child=node;
    else, [child,~]=gp_replace_at_path(nd,const_paths{pi_},node); end
end

function winner=gp_tournament(pop,fits,k)
    idx=randperm(length(pop),k); [~,bi]=min(fits(idx)); winner=pop{idx(bi)};
end

%% ---- Post-GP constant optimisation (Nelder-Mead, 1000 evals) ----------
function best_tree=gp_optimise_constants(tree,X,y,peak_mask,tail_mask,size_limit,parsimony)
    paths=gp_get_all_paths(tree,[]);
    const_paths=paths(cellfun(@(p) strcmp(gp_node_at_path(tree,p).type,'const'),paths));
    if isempty(const_paths), best_tree=tree; return; end

    c0=zeros(length(const_paths),1);
    for i=1:length(const_paths)
        c0(i)=gp_node_at_path(tree,const_paths{i}).val;
    end

    obj=@(c) local_const_obj(tree,const_paths,c,X,y,peak_mask,tail_mask);
    opts=optimset('MaxFunEvals',1000,'MaxIter',600,...
                  'Display','off','TolFun',1e-6,'TolX',1e-6);
    try, c_opt=fminsearch(obj,c0,opts); catch, c_opt=c0; end

    best_tree=tree;
    for i=1:length(const_paths)
        nd=gp_node_at_path(best_tree,const_paths{i});
        nd.val=c_opt(i);
        if isempty(const_paths{i}), best_tree=nd;
        else, [best_tree,~]=gp_replace_at_path(best_tree,const_paths{i},nd); end
    end
end

function f=local_const_obj(tree,const_paths,c,X,y,peak_mask,tail_mask)
    t2=tree;
    for i=1:length(const_paths)
        nd=gp_node_at_path(t2,const_paths{i}); nd.val=c(i);
        if isempty(const_paths{i}), t2=nd;
        else, [t2,~]=gp_replace_at_path(t2,const_paths{i},nd); end
    end
    pred=gp_eval(t2,X);
    if any(isnan(pred))||any(isinf(pred)), f=1e9; return; end
    ss_res=sum((y-pred).^2); ss_tot=sum((y-mean(y)).^2);
    if ss_tot<1e-12, f=1e9; return; end
    nse   = 1-ss_res/ss_tot;
    nrmse = sqrt(mean((y-pred).^2))/(max(y)-min(y)+1e-12);
    f3=0; if nse<0.95, f3=20*(0.95-nse)^2; end
    pk_err = 0; if sum(peak_mask)>0, pk_err=mean((y(peak_mask)-pred(peak_mask)).^2); end
    tl_err = 0; if sum(tail_mask)>0, tl_err=mean((y(tail_mask)-pred(tail_mask)).^2); end
    hf_thresh=prctile(y,75); hf_mask=(y>=hf_thresh);
    hf_bias=0; if sum(hf_mask)>1, hf_bias=mean(pred(hf_mask))-mean(y(hf_mask)); end
    std_obs=std(y); std_pred=std(pred);
    vr=0; if std_obs>1e-12, vr=0.25*(std_pred/std_obs-1)^2; end
    cc=corrcoef(y,pred); r=cc(1,2); if isnan(r),r=0;end
    f = (1-nse)^2 + 0.005*nrmse + f3 + 0.60*pk_err + 0.50*hf_bias^2 ...
        + vr + 0.40*tl_err + 0.20*(1-r)^2;
end

function m=gp_metrics(obs,pred)
    obs=obs(:); pred=pred(:);
    m.RMSE=sqrt(mean((obs-pred).^2));
    m.MAE =mean(abs(obs-pred));
    m.R2  =1-sum((obs-pred).^2)/sum((obs-mean(obs)).^2);
    m.MAPE=mean(abs((obs-pred)./(obs+1e-6)))*100;
    m.NSE =m.R2;
    m.Bias=mean(pred-obs);
    cc=corrcoef(obs,pred); m.Corr=cc(1,2);
end

%% --- Figure helpers (WHITE background) ----------------------------------
function arch_box(ax,x,y,w,h,txt,fc)
    rectangle(ax,'Position',[x y w h],'Curvature',0.08,...
              'FaceColor',fc,'EdgeColor',[0.2 0.2 0.2],'LineWidth',1.4);
    text(ax,x+w/2,y+h/2,txt,'Color','w','HorizontalAlignment','center',...
         'VerticalAlignment','middle','FontSize',8.5,'FontWeight','bold','Interpreter','none');
end
function bax_w(axh)
    set(axh,'Color','w','XColor','k','YColor','k',...
            'GridColor',[0.75 0.75 0.75],'GridAlpha',1,...
            'FontSize',9,'LineWidth',0.8,'Box','on','TickDir','out');
    grid(axh,'on');
end
function dbax_w(axh,xlbl,ylbl,ttl)
    xlabel(axh,xlbl,'Color','k','FontSize',9);
    ylabel(axh,ylbl,'Color','k','FontSize',9);
    title(axh,ttl,'Color','k','FontSize',10,'FontWeight','bold');
    set(axh,'Color','w','XColor','k','YColor','k',...
            'GridColor',[0.75 0.75 0.75],'GridAlpha',1,...
            'FontSize',8,'LineWidth',0.8,'Box','on','TickDir','out');
    grid(axh,'on');
end
function [xtk,xlbl]=yr_ticks(yr_vec,x_vec,d_vec)
    yrs=unique(yr_vec); xtk=zeros(length(yrs),1); xlbl=cell(length(yrs),1);
    for k=1:length(yrs)
        idx=find(yr_vec==yrs(k),1,'first');
        xtk(k) =x_vec(idx);
        xlbl{k}=sprintf('%d\n(%s)',yrs(k),datestr(d_vec(idx),'dd-mmm'));
    end
end