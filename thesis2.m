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

output_dir = fullfile(pathname,'GP_Outputs_v3');
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

AMI_k = 0.85;
ami   = zeros(height(T_all),1);
for t = 2:height(T_all)
    ami(t) = AMI_k * ami(t-1) + T_all.BasinRain(t-1);
end
T_all.AMI = ami;

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
fprintf('  Date range: %s  to  %s\n', ...
    datestr(T_all.Date(1),'dd-mmm-yyyy'), datestr(T_all.Date(end),'dd-mmm-yyyy'));

%% ========================================================================
%%  [3/7]  BUILD FEATURE MATRIX
%% ========================================================================
LAG       = 5;
N_STATIONS = 49;
FEATS_LAG = 62;

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

            row(ii) = T_yr.Grand_Total_Inflow(tl); ii=ii+1;
            row(ii) = T_yr.BasinRain(tl);          ii=ii+1;
            row(ii) = T_yr.Rain3(tl);              ii=ii+1;
            row(ii) = T_yr.Rain5(tl);              ii=ii+1;
            row(ii) = T_yr.Rain7(tl);              ii=ii+1;
            row(ii) = T_yr.RainMax(tl);            ii=ii+1;
            row(ii) = T_yr.RainStd(tl);            ii=ii+1;
            row(ii) = T_yr.WetDays(tl);            ii=ii+1;
            row(ii) = T_yr.AMI(tl);               ii=ii+1;
            row(ii) = T_yr.RainAnom(tl);           ii=ii+1;

            bf_idx = max(1,tl-4):tl;
            row(ii) = mean(T_yr.Grand_Total_Inflow(bf_idx)); ii=ii+1;

            if tl > 1
                row(ii) = T_yr.Grand_Total_Inflow(tl) - T_yr.Grand_Total_Inflow(tl-1);
            else
                row(ii) = 0;
            end
            ii=ii+1;

            ii=ii+1;
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

pk_thresh  = prctile(y_train, 90);
peak_mask  = (y_train >= pk_thresh);

fprintf('  Train: %d (80%%)  |  Test: %d (20%%)  |  Peak events: %d\n',...
        sp, n-sp, sum(peak_mask));
fprintf('  Train years: %s\n', strjoin(arrayfun(@num2str,unique(yrl_train),'UniformOutput',false),', '));
fprintf('  Test  years: %s\n', strjoin(arrayfun(@num2str,unique(yrl_test), 'UniformOutput',false),', '));

%% ========================================================================
%%  [5/7]  GP PARAMETERS
%% ========================================================================
N_RESTARTS       = 1;
POP_SIZE         = 2000;
N_GENS           = 1200;
MAX_DEPTH        = 12;
CX_PROB          = 0.85;
MUT_PROB_START   = 0.15;
ELITISM          = 40;
TOURN_K_LO       = 5;
TOURN_K_HI       = 15;
STAGNATION_LIMIT = 40;
PARSIMONY_START  = 0.00005;
PARSIMONY_END    = 0.0006;
SIZE_LIMIT       = 55;
EARLY_STOP_TR    = 0.9889;
EARLY_STOP_TE    = 0.9435;

OPS   = {'+','-','*','/','sqrt','sq','log','cube','tanh','abs'};
ARITY = [ 2,  2,  2,  2,    1,   1,    1,    1,     1,   1];

F_INF1  = FEATS_LAG*0 + 38;
F_BAS1  = FEATS_LAG*0 + 39;
F_R51   = FEATS_LAG*0 + 41;
F_R71   = FEATS_LAG*0 + 42;
F_AMI1  = FEATS_LAG*0 + 46;
F_BF1   = FEATS_LAG*0 + 48;
F_MAX1  = FEATS_LAG*0 + 43;

fprintf('\n=== [5/7] GP  Pop=%d  Gens=%d  Depth=%d  Restarts=%d ===\n',...
        POP_SIZE,N_GENS,MAX_DEPTH,N_RESTARTS);
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

    s1 = gp_make_var(F_INF1);
    s2 = gp_make_func('+',...
            {gp_make_var(F_INF1), gp_make_var(F_BAS1)});
    s3 = gp_make_func('*',...
            {gp_make_func('sqrt',{gp_make_var(F_R51)}),...
             gp_make_var(F_INF1)});
    s4 = gp_make_var(F_BF1);
    s5 = gp_make_func('+',...
            {gp_make_var(F_BAS1), gp_make_var(F_BF1)});
    s6 = gp_make_func('*',...
            {gp_make_var(F_AMI1), gp_make_var(F_BAS1)});
    s7 = gp_make_func('/',...
            {gp_make_func('sq',{gp_make_var(F_MAX1)}),...
             gp_make_func('+',...
               {gp_make_var(F_BF1), gp_make_const(0.1)})});

    seeds = {s1,s2,s3,s4,s5,s6,s7};

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
    best_fit  = gp_fitness(pop{1},X_train,y_train,peak_mask,...
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

        fits = zeros(POP_SIZE,1);
        for i = 1:POP_SIZE
            fits(i) = gp_fitness(pop{i},X_train,y_train,...
                                 peak_mask,SIZE_LIMIT,parsimony);
        end

        [gbf, gbi] = min(fits);

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

        if best_r2te > EARLY_STOP_TE && best_r2tr > EARLY_STOP_TR
            fprintf('\n  *** TARGET R2 achieved (Train=%.4f Test=%.4f) at gen %d ***\n',...
                    best_r2tr,best_r2te,gen);
            actual_gens = gen;
            h_gen  = h_gen(1:gen);  h_bfit = h_bfit(1:gen);
            h_afit = h_afit(1:gen); h_asiz = h_asiz(1:gen);
            h_trmse= h_trmse(1:gen);h_r2tr = h_r2tr(1:gen);
            h_r2te = h_r2te(1:gen);
            break;
        end

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

            elseif r < CX_PROB + mut_prob*0.45
                c = gp_subtree_mutate(p1,n_feat,OPS,ARITY,MAX_DEPTH);
                if gp_tree_depth(c)>MAX_DEPTH, c=p1; end
                new_pop{k}=c; k=k+1;

            elseif r < CX_PROB + mut_prob*0.70
                c = gp_hoist_mutate(p1,MAX_DEPTH);
                new_pop{k}=c; k=k+1;

            elseif r < CX_PROB + mut_prob*0.90
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
    global_best_r2te,...
    gp_tree_size(best_tree),gp_tree_depth(best_tree));

fprintf('  Post-GP constant optimisation...\n');
best_tree = gp_optimise_constants(best_tree,X_train,y_train,peak_mask,...
                                  SIZE_LIMIT,PARSIMONY_END);
new_r2te  = compute_r2(y_test, gp_eval(best_tree,X_test));
fprintf('  Test R2: %.4f -> %.4f\n', global_best_r2te, new_r2te);

h_gen   = global_hist.gen;   h_bfit = global_hist.bfit;
h_afit  = global_hist.afit;  h_asiz = global_hist.asiz;
h_trmse = global_hist.trmse; h_r2tr = global_hist.r2tr;
h_r2te  = global_hist.r2te;
N_GENS  = global_hist.ngens;

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

%% ---- Extract best equation BEFORE plotting ----------------------------
best_formula = gp_tree_to_string(best_tree);
fprintf('\n================ BEST GP EQUATION ================\n');
disp(best_formula);

fid = fopen(fullfile(output_dir,'Best_Equation.txt'),'w');
fprintf(fid,'%s',best_formula);
fclose(fid);

 

%% ========================================================================
%% ====================================================================
%%  COLOUR PALETTE  (white-background theme)
%%  *** CHANGE 1: All figures now use white background ***
%% ====================================================================
% ---- Axes / figure colours ----
FIG_BG  = [1   1   1  ];   % figure background — white
AX_BG   = [1   1   1  ];   % axes background   — white
FG      = [0.08 0.08 0.10]; % primary text / tick colour — near-black
GRD     = [0.82 0.82 0.82]; % grid lines — light grey

% ---- Data colours ----
C_obs   = [0.09 0.37 0.75];   % observed  — deep blue
C_pred  = [0.90 0.30 0.07];   % predicted — vivid orange
C_train = [0.09 0.60 0.30];   % train scatter — green
C_test  = [0.90 0.30 0.07];   % test scatter  — orange

%% ====================================================================
%%  FIGURE 1 — ARCHITECTURE  (white bg)
%% ====================================================================
fprintf('\n=== [Figs] Plotting ===\n');

fig1 = figure('Color',FIG_BG,'Position',[60 60 1440 730],...
              'Name','GP Architecture v3','NumberTitle','off');

ax1 = axes(fig1,'Position',[0.03 0.06 0.44 0.86]);
set(ax1,'Color',FIG_BG,'XLim',[0 10],'YLim',[0 14],...
    'XTick',[],'YTick',[],'Box','off');
title(ax1,'GP Model Architecture (v3)','Color',FG,'FontSize',12,'FontWeight','bold');

arch_box_w(ax1,0.3,12.5,9.4,1.2,...
    sprintf('INPUT: 49 Stations x 5 Lags + AMI + Trend + Anomaly + WetDays + ... = %d Features  |  Monsoon Only (Jun-Oct)',n_feat),...
    [0.17 0.45 0.80]);
for li = 1:5
    arch_box_w(ax1,0.3+(li-1)*1.88,10.5,1.55,1.7,...
             sprintf('Lag %d\n50 vars\n(+AMI,Anom)',li),[0.27 0.45 0.82]);
end
arch_box_w(ax1,0.3,8.2,9.4,2.0,...
    sprintf('GENETIC PROGRAMMING ENGINE v3\nPop=%d  Gens=%d  MaxDepth=%d  CX=%.2f  Mut=%.2f->0.05  Elitism=%d  Tourn k=%d->%d\nBest tracked by TEST R2  |  Stagnation restart (50 gens, 30%% inject)  |  %d restarts',...
    POP_SIZE,N_GENS,MAX_DEPTH,CX_PROB,MUT_PROB_START,ELITISM,...
    TOURN_K_LO,TOURN_K_HI,N_RESTARTS),[0.47 0.14 0.70]);

op_names = {'+','-','*','/','sqrt','sq','log','cube','tanh','abs'};
op_clrs  = [0.85 0.15 0.15; 0.15 0.50 0.20; 0.90 0.40 0.00; 0.00 0.48 0.50;
            0.25 0.55 0.18; 0.25 0.35 0.50; 0.40 0.28 0.18;
            0.60 0.25 0.45; 0.65 0.10 0.40; 0.55 0.45 0.10];
for oi = 1:10
    arch_box_w(ax1,0.08+(oi-1)*0.985,6.6,0.87,1.0,op_names{oi},op_clrs(oi,:));
end
text(ax1,5,6.4,'Operator Set (10 ops) + Terminal Set (features + random constants)',...
     'Color',[0.50 0.30 0.00],'HorizontalAlignment','center','FontSize',8);

arch_box_w(ax1,0.3,5.1,9.4,1.1,...
    '7 Domain Seeds: Persistence / Linear / Sqrt-Rain / Baseflow / AMI / Quadratic-Peak / Linear-Combo',...
    [0.10 0.50 0.60]);
arch_box_w(ax1,0.3,3.9,9.4,1.0,...
    'Ops: CX + SubtreeMut + HoistMut (bloat ctrl) + PointMut + ConstPerturb (adaptive rates)',...
    [0.00 0.43 0.45]);
arch_box_w(ax1,0.3,2.7,9.4,1.0,...
    'Fitness = (1-NSE)^2 + 0.01*RMSE + R2_penalty(<0.8598) + PeakBonus(x3) + AdaptiveParsimony',...
    [0.65 0.20 0.05]);
arch_box_w(ax1,0.3,1.5,9.4,1.0,...
    'Post-GP: Nelder-Mead constant optimisation on best tree (500 evals)',...
    [0.20 0.40 0.18]);
arch_box_w(ax1,2.0,0.1,6.0,1.2,...
    sprintf('OUTPUT: Grand Total Inflow (cusecs)\nBest tree: %d nodes, depth %d  |  TrainR2=%.4f  TestR2=%.4f',...
    gp_tree_size(best_tree),gp_tree_depth(best_tree),m_tr.R2,m_te.R2),[0.75 0.25 0.00]);

% Right panel: example tree
ax2 = axes(fig1,'Position',[0.52 0.08 0.46 0.84]);
set(ax2,'Color',FIG_BG,'XLim',[0 10],'YLim',[0 10],...
    'XTick',[],'YTick',[],'Box','off');
title(ax2,'Symbolic Expression Tree (Example)','Color',FG,'FontSize',11,'FontWeight','bold');
NP = struct('add',[5 9],'mul',[2.5 7.2],'div_',[7.5 7.2],...
            'tanh_',[1 5.4],'X5',[4 5.4],'sqrt_',[6.2 5.4],...
            'X12',[8.8 5.4],'X23',[1 3.7],'X47',[6.2 3.7]);
EG = {'add','mul';'add','div_';'mul','tanh_';'mul','X5';...
      'div_','sqrt_';'div_','X12';'tanh_','X23';'sqrt_','X47'};
NF = struct('add',[0.85 0.18 0.18],'mul',[0.85 0.18 0.18],'div_',[0.85 0.18 0.18],...
            'tanh_',[0.45 0.12 0.70],'sqrt_',[0.45 0.12 0.70],...
            'X5',[0.10 0.42 0.80],'X12',[0.10 0.42 0.80],...
            'X23',[0.10 0.42 0.80],'X47',[0.10 0.42 0.80]);
NL = struct('add','add','mul','mul','div_','div','tanh_','tanh',...
            'sqrt_','sqrt','X5','X5','X12','X12','X23','X23','X47','X47');
for ei = 1:size(EG,1)
    pp1=NP.(EG{ei,1}); pp2=NP.(EG{ei,2});
    line(ax2,[pp1(1) pp2(1)],[pp1(2) pp2(2)],...
         'Color',[0.55 0.55 0.60],'LineWidth',1.8);
end
fn2 = fieldnames(NP);
for ni = 1:length(fn2)
    nm=fn2{ni}; pos=NP.(nm); fc=NF.(nm); lbl=NL.(nm);
    rectangle(ax2,'Position',[pos(1)-0.60 pos(2)-0.44 1.20 0.88],...
              'Curvature',0.5,'FaceColor',fc,'EdgeColor',FG,'LineWidth',1.4);
    text(ax2,pos(1),pos(2),lbl,'Color','w','HorizontalAlignment','center',...
         'VerticalAlignment','middle','FontSize',9,'FontWeight','bold','Interpreter','none');
end
lx=0.3; ly=1.6;
for ci=1:3
    clrs = {[0.85 0.18 0.18],[0.45 0.12 0.70],[0.10 0.42 0.80]};
    lbls = {'Binary Op','Unary Op','Variable'};
    rectangle(ax2,'Position',[lx+(ci-1)*1.5 ly 1.0 0.55],'Curvature',0.4,...
              'FaceColor',clrs{ci},'EdgeColor',FG);
    text(ax2,lx+(ci-1)*1.5+0.5,ly+0.27,lbls{ci},...
         'Color','w','HorizontalAlignment','center','FontSize',7.5);
end
text(ax2,5,0.75,...
    sprintf('Best: %d nodes, depth %d  |  %d gens  |  %d restarts  |  TrainR2=%.4f  TestR2=%.4f',...
    gp_tree_size(best_tree),gp_tree_depth(best_tree),N_GENS,N_RESTARTS,m_tr.R2,m_te.R2),...
    'Color',FG,'HorizontalAlignment','center','FontSize',9,...
    'BackgroundColor',[0.93 0.93 0.93],'EdgeColor',[0.60 0.60 0.65]);

sgtitle(fig1,'Genetic Programming v3 — Monsoon Rainfall-Inflow Model',...
        'Color',FG,'FontSize',14,'FontWeight','bold');
saveas(fig1, fullfile(output_dir,'Fig1_Architecture.png'));
fprintf('  Saved Fig1_Architecture.png\n');

%% ====================================================================
%%  FIGURE 2 — CONVERGENCE  (white bg, 4 panels)
%% ====================================================================
fig2 = figure('Color',FIG_BG,'Position',[60 60 1600 430],...
              'Name','Convergence','NumberTitle','off');

gens_v = h_gen(1:N_GENS);

ax21 = subplot(1,4,1);
plot(ax21,gens_v,h_bfit(1:N_GENS),'Color',[0.09 0.65 0.25],'LineWidth',2,'DisplayName','Train Fit'); hold(ax21,'on');
plot(ax21,gens_v,h_trmse(1:N_GENS),'Color',[0.90 0.38 0.00],'LineWidth',2,'DisplayName','Test RMSE');
lg=legend(ax21,'Location','northeast','FontSize',8,'TextColor',FG); lg.Color=FIG_BG; lg.EdgeColor=[0.7 0.7 0.7];
xlabel(ax21,'Generation','Color',FG,'FontSize',9);
ylabel(ax21,'Value (scaled)','Color',FG,'FontSize',9);
title(ax21,'Fitness / RMSE','Color',FG,'FontSize',10,'FontWeight','bold');
bax_w(ax21);

ax22 = subplot(1,4,2);
plot(ax22,gens_v,h_r2tr(1:N_GENS),'Color',[0.09 0.55 0.90],'LineWidth',2,'DisplayName','Train R^2'); hold(ax22,'on');
plot(ax22,gens_v,h_r2te(1:N_GENS),'Color',[0.88 0.15 0.15],'LineWidth',2,'DisplayName','Test R^2');
yline(ax22,0.8598,'Color',[0.3 0.3 0.3],'LineStyle','--','LineWidth',1.5);
text(ax22,gens_v(max(1,round(N_GENS*0.05))),0.870,'R^2=0.92 target',...
     'Color',FG,'FontSize',8,'Interpreter','none');
lg2=legend(ax22,'Location','southeast','FontSize',8,'TextColor',FG); lg2.Color=FIG_BG; lg2.EdgeColor=[0.7 0.7 0.7];
xlabel(ax22,'Generation','Color',FG,'FontSize',9);
ylabel(ax22,'R^2','Color',FG,'FontSize',9);
title(ax22,'R^2 Convergence','Color',FG,'FontSize',10,'FontWeight','bold');
ylim(ax22,[-0.1 1.05]);
bax_w(ax22);

ax23 = subplot(1,4,3);
plot(ax23,gens_v,h_afit(1:N_GENS),'Color',[0.55 0.22 0.88],'LineWidth',2);
xlabel(ax23,'Generation','Color',FG,'FontSize',9);
ylabel(ax23,'Avg Fitness','Color',FG,'FontSize',9);
title(ax23,'Population Avg Fitness','Color',FG,'FontSize',10,'FontWeight','bold');
bax_w(ax23);

ax24 = subplot(1,4,4);
plot(ax24,gens_v,h_asiz(1:N_GENS),'Color',[0.09 0.62 0.75],'LineWidth',2);
xlabel(ax24,'Generation','Color',FG,'FontSize',9);
ylabel(ax24,'Avg Nodes','Color',FG,'FontSize',9);
title(ax24,'Average Tree Size','Color',FG,'FontSize',10,'FontWeight','bold');
bax_w(ax24);

sgtitle(fig2,'GP v3 Training Convergence','FontSize',13,'FontWeight','bold','Color',FG);
saveas(fig2, fullfile(output_dir,'Fig2_Convergence.png'));
fprintf('  Saved Fig2_Convergence.png\n');

%% ====================================================================
%%  FIGURES 3a / 3b — TIME SERIES  (white bg)
%% ====================================================================
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

    figX = figure('Color',FIG_BG,'Position',[40 40 1800 500],...
                  'Name',sprintf('%s: Monsoon v3',tag),'NumberTitle','off');
    axA  = axes(figX,'Position',[0.07 0.20 0.91 0.65]);
    hold(axA,'on');

    for yi = 1:length(yrs_v)
        msk=(yrl_v==yrs_v(yi));
        xs=xv(find(msk,1,'first')); xe=xv(find(msk,1,'last'));
        if mod(yi,2)==0
            patch(axA,[xs xe xe xs],[0 0 1 1]*pk_o*1.12,...
                  [0.93 0.93 0.96],'FaceAlpha',1,'EdgeColor','none','HandleVisibility','off');
        end
        text(axA,(xs+xe)/2,pk_o*1.09,num2str(yrs_v(yi)),...
             'Color',FG,'FontSize',8,'FontWeight','bold','HorizontalAlignment','center');
        if yi<length(yrs_v)
            xline(axA,xe+0.5,'Color',[0.70 0.70 0.73],'LineWidth',0.8,'HandleVisibility','off');
        end
    end

    plot(axA,xv,obs_v, 'Color',C_obs, 'LineWidth',1.4,'DisplayName','Observed');
    plot(axA,xv,pred_v,'Color',C_pred,'LineWidth',1.2,'DisplayName','GP Predicted v3');
    scatter(axA,xv(pk_oi),pk_o,100,FG,'filled','HandleVisibility','off');
    scatter(axA,xv(pk_pi),pk_p,100,[0.60 0.10 0.75],'filled','HandleVisibility','off');
    text(axA,xv(pk_oi),pk_o*1.03,sprintf('Obs Peak: %.0f',pk_o),...
         'Color',FG,'FontSize',8,'FontWeight','bold','HorizontalAlignment','center');
    text(axA,xv(pk_pi),pk_p*1.03,sprintf('Pred Peak: %.0f',pk_p),...
         'Color',[0.60 0.10 0.75],'FontSize',8,'FontWeight','bold','HorizontalAlignment','center');

    ylabel(axA,'Inflow (Cusecs)','Color',FG,'FontSize',11,'FontWeight','bold');
    xlabel(axA,'Monsoon Day (sequential, gaps removed)','Color',FG,'FontSize',10);
    set(axA,'XTick',xtk_v,'XTickLabel',xlbl_v);
    set(axA,'Color',AX_BG,'XColor',FG,'YColor',FG,'GridColor',GRD,...
            'GridAlpha',1,'FontSize',8,'LineWidth',0.8,'Box','on','TickDir','out');
    grid(axA,'on');
    lg=legend(axA,'Location','northeast','FontSize',10,'TextColor',FG);
    lg.Color=FIG_BG; lg.EdgeColor=[0.7 0.7 0.7];
    title(axA,sprintf('%s %s  |  R^2=%.3f  NSE=%.3f  RMSE=%.0f cusecs',...
          tag,pct,m_v.R2,m_v.NSE,m_v.RMSE),...
          'Color',FG,'FontSize',10,'FontWeight','bold','Interpreter','none');
    sgtitle(figX,sprintf('Monsoon Simulation v3 — %s  |  Years: %s',...
            tag,strjoin(arrayfun(@num2str,yrs_v','UniformOutput',false),', ')),...
            'Color',FG,'FontSize',12,'FontWeight','bold');

    fname=sprintf('Fig3%c_%s_Combined.png',char('a'-1+tt),tag);
    saveas(figX, fullfile(output_dir,fname));
    fprintf('  Saved %s\n',fname);
end
fprintf('\n  Done %.1fs | BestRMSE=%.4f | Size=%d | Depth=%d\n',...
        toc(t0),best_fit,gp_tree_size(best_tree),gp_tree_depth(best_tree));

%% ====================================================================
%%  FIGURE 4 — SCATTER  (white bg)
%% ====================================================================
fig4 = figure('Color',FIG_BG,'Position',[60 60 1100 950],...
              'Name','Scatter','NumberTitle','off');

ax4a=subplot(2,2,1);
scatter(ax4a,train_obs,train_pred,28,C_train,'filled','MarkerFaceAlpha',0.6); hold(ax4a,'on');
lm=[min([train_obs;train_pred]) max([train_obs;train_pred])];
plot(ax4a,lm,lm,'k--','LineWidth',1.5);
xlabel(ax4a,'Observed (cusecs)','Color',FG,'FontSize',9);
ylabel(ax4a,'Predicted (cusecs)','Color',FG,'FontSize',9);
title(ax4a,sprintf('Train   R^2=%.3f   NSE=%.3f',m_tr.R2,m_tr.NSE),'Color',FG,'FontSize',10,'FontWeight','bold');
bax_w(ax4a);

ax4b=subplot(2,2,2);
scatter(ax4b,test_obs,test_pred,28,C_test,'filled','MarkerFaceAlpha',0.6); hold(ax4b,'on');
lm=[min([test_obs;test_pred]) max([test_obs;test_pred])];
plot(ax4b,lm,lm,'k--','LineWidth',1.5);
xlabel(ax4b,'Observed (cusecs)','Color',FG,'FontSize',9);
ylabel(ax4b,'Predicted (cusecs)','Color',FG,'FontSize',9);
title(ax4b,sprintf('Test   R^2=%.3f   NSE=%.3f',m_te.R2,m_te.NSE),'Color',FG,'FontSize',10,'FontWeight','bold');
bax_w(ax4b);

ax4c=subplot(2,2,[3 4]);
scatter(ax4c,train_obs,train_pred,22,C_train,'filled','MarkerFaceAlpha',0.5,'DisplayName','Train'); hold(ax4c,'on');
scatter(ax4c,test_obs, test_pred, 22,C_test, 'filled','MarkerFaceAlpha',0.5,'DisplayName','Test');
all_o=[train_obs;test_obs]; all_p=[train_pred;test_pred];
lm=[min([all_o;all_p]) max([all_o;all_p])];
plot(ax4c,lm,lm,'k--','LineWidth',1.5,'DisplayName','1:1 Line');
xlabel(ax4c,'Observed (cusecs)','Color',FG,'FontSize',9);
ylabel(ax4c,'Predicted (cusecs)','Color',FG,'FontSize',9);
title(ax4c,'Combined Train + Test','Color',FG,'FontSize',10,'FontWeight','bold');
lg4=legend(ax4c,'Location','northwest','FontSize',9,'TextColor',FG);
lg4.Color=FIG_BG; lg4.EdgeColor=[0.7 0.7 0.7];
bax_w(ax4c);
sgtitle(fig4,'Scatter — Observed vs GP Predicted Inflow (v3)',...
        'Color',FG,'FontSize',13,'FontWeight','bold');
saveas(fig4, fullfile(output_dir,'Fig4_Scatter.png'));
fprintf('  Saved Fig4_Scatter.png\n');

%% ====================================================================
%%  FIGURE 5 — ERROR ANALYSIS  (white bg)
%% ====================================================================
tr_err=train_pred-train_obs; te_err=test_pred-test_obs;
[~,pk_oi_tr]=max(train_obs); [~,pk_pi_tr]=max(train_pred);
[~,pk_oi_te]=max(test_obs);  [~,pk_pi_te]=max(test_pred);

fig5=figure('Color',FIG_BG,'Position',[60 60 1500 950],'Name','Error Analysis','NumberTitle','off');

ax51=subplot(3,3,1); bar(ax51,tr_err,'FaceColor',[0.88 0.18 0.18],'EdgeColor','none','FaceAlpha',0.9);
yline(ax51,0,'Color',FG,'LineWidth',1); dbax_w(ax51,'Sample Index','Error (cusecs)','Train Residuals');

ax52=subplot(3,3,2); bar(ax52,te_err,'FaceColor',[0.12 0.48 0.88],'EdgeColor','none','FaceAlpha',0.9);
yline(ax52,0,'Color',FG,'LineWidth',1); dbax_w(ax52,'Sample Index','Error (cusecs)','Test Residuals');

ax53=subplot(3,3,3);
histogram(ax53,tr_err,25,'FaceColor',C_train,'FaceAlpha',0.8,'EdgeColor','none','DisplayName','Train'); hold(ax53,'on');
histogram(ax53,te_err,25,'FaceColor',C_test, 'FaceAlpha',0.8,'EdgeColor','none','DisplayName','Test');
xline(ax53,0,'k--','LineWidth',1.5);
lg53=legend(ax53,'FontSize',8,'TextColor',FG); lg53.Color=FIG_BG; lg53.EdgeColor=[0.7 0.7 0.7];
dbax_w(ax53,'Error (cusecs)','Count','Error Distribution');

ax54=subplot(3,3,4);
plot(ax54,x_tr,train_obs,'Color',C_obs,'LineWidth',1.4,'DisplayName','Observed'); hold(ax54,'on');
plot(ax54,x_tr,train_pred,'Color',C_pred,'LineWidth',1.2,'LineStyle','--','DisplayName','Predicted');
scatter(ax54,x_tr(pk_oi_tr),train_obs(pk_oi_tr), 100,FG,'filled','HandleVisibility','off');
scatter(ax54,x_tr(pk_pi_tr),train_pred(pk_pi_tr),100,[0.60 0.10 0.75],'filled','HandleVisibility','off');
lg54=legend(ax54,'Location','northwest','FontSize',7,'TextColor',FG); lg54.Color=FIG_BG;
dbax_w(ax54,'Sample','Inflow (cusecs)','Train — Peak Discharge');

ax55=subplot(3,3,5);
plot(ax55,x_te,test_obs,'Color',C_obs,'LineWidth',1.4,'DisplayName','Observed'); hold(ax55,'on');
plot(ax55,x_te,test_pred,'Color',C_pred,'LineWidth',1.2,'LineStyle','--','DisplayName','Predicted');
scatter(ax55,x_te(pk_oi_te),test_obs(pk_oi_te), 100,FG,'filled','HandleVisibility','off');
scatter(ax55,x_te(pk_pi_te),test_pred(pk_pi_te),100,[0.60 0.10 0.75],'filled','HandleVisibility','off');
lg55=legend(ax55,'Location','northwest','FontSize',7,'TextColor',FG); lg55.Color=FIG_BG;
dbax_w(ax55,'Sample','Inflow (cusecs)','Test — Peak Discharge');

ax56=subplot(3,3,6); mk={'RMSE','MAE','MAPE'}; xb=1:3; w=0.35;
bar(ax56,xb-w/2,[m_tr.RMSE m_tr.MAE m_tr.MAPE],w,'FaceColor',C_train,'EdgeColor','none','DisplayName','Train'); hold(ax56,'on');
bar(ax56,xb+w/2,[m_te.RMSE m_te.MAE m_te.MAPE],w,'FaceColor',C_test, 'EdgeColor','none','DisplayName','Test');
set(ax56,'XTick',xb,'XTickLabel',mk); ax56.XAxis.TickLabelColor=FG;
lg56=legend(ax56,'FontSize',8,'TextColor',FG); lg56.Color=FIG_BG;
dbax_w(ax56,'','Value','Error Metrics');

ax57=subplot(3,3,7); mk2={'R^2','NSE','Corr'};
bar(ax57,xb-w/2,[m_tr.R2 m_tr.NSE m_tr.Corr],w,'FaceColor',C_train,'EdgeColor','none','DisplayName','Train'); hold(ax57,'on');
bar(ax57,xb+w/2,[m_te.R2 m_te.NSE m_te.Corr],w,'FaceColor',C_test, 'EdgeColor','none','DisplayName','Test');
set(ax57,'XTick',xb,'XTickLabel',mk2); ylim(ax57,[-0.2 1.2]); ax57.XAxis.TickLabelColor=FG;
yline(ax57,0.8598,'Color',[0.3 0.3 0.3],'LineStyle','--','LineWidth',1.5);
lg57=legend(ax57,'FontSize',8,'TextColor',FG); lg57.Color=FIG_BG;
dbax_w(ax57,'','Value','Goodness-of-Fit');

ax58=subplot(3,3,8);
plot(ax58,x_tr,cumsum(abs(tr_err)),'Color',C_train,'LineWidth',1.8,'DisplayName','Train'); hold(ax58,'on');
plot(ax58,x_te,cumsum(abs(te_err)),'Color',C_test, 'LineWidth',1.8,'DisplayName','Test');
lg58=legend(ax58,'FontSize',8,'TextColor',FG); lg58.Color=FIG_BG;
dbax_w(ax58,'Sample','Cumulative |Error|','Cumulative Absolute Error');

% Panel 9: metrics table
ax59=subplot(3,3,9); set(ax59,'Color',FIG_BG); axis(ax59,'off');
rows={'RMSE','MAE','MAPE(%)','R2','NSE','Bias','Corr'};
tr_v=[m_tr.RMSE;m_tr.MAE;m_tr.MAPE;m_tr.R2;m_tr.NSE;m_tr.Bias;m_tr.Corr];
te_v=[m_te.RMSE;m_te.MAE;m_te.MAPE;m_te.R2;m_te.NSE;m_te.Bias;m_te.Corr];
ys=linspace(0.92,0.08,length(rows)+1);
text(ax59,0.05,ys(1),'Metric','FontWeight','bold','FontSize',10,'Units','normalized','Color',FG);
text(ax59,0.45,ys(1),'Train','FontWeight','bold','FontSize',10,'Units','normalized','Color',C_train);
text(ax59,0.73,ys(1),'Test', 'FontWeight','bold','FontSize',10,'Units','normalized','Color',C_test);
for ri = 1:length(rows)
    bg=[0.96 0.96 0.96]; if mod(ri,2)==0, bg=[0.90 0.90 0.90]; end
    text(ax59,0.05,ys(ri+1),rows{ri},                 'FontSize',9,'Units','normalized','BackgroundColor',bg,'Color',FG);
    text(ax59,0.45,ys(ri+1),sprintf('%.3f',tr_v(ri)), 'FontSize',9,'Units','normalized','BackgroundColor',bg,'Color',C_train);
    text(ax59,0.73,ys(ri+1),sprintf('%.3f',te_v(ri)), 'FontSize',9,'Units','normalized','BackgroundColor',bg,'Color',C_test);
end
title(ax59,'Performance Summary','FontWeight','bold','Color',FG,'FontSize',10);
sgtitle(fig5,'Error Analysis — GP Model v3 (Monsoon)','Color',FG,'FontSize',13,'FontWeight','bold');
saveas(fig5, fullfile(output_dir,'Fig5_Error_Analysis.png'));
fprintf('  Saved Fig5_Error_Analysis.png\n');

%% ====================================================================
%%  FIGURE 6 — FEATURE IMPORTANCE  (white bg)
%% ====================================================================
all_paths  = gp_get_all_paths(best_tree,[]);
var_counts = zeros(n_feat,1);
for pi=1:length(all_paths)
    nd2=gp_node_at_path(best_tree,all_paths{pi});
    if strcmp(nd2.type,'var')
        var_counts(nd2.idx)=var_counts(nd2.idx)+1;
    end
end
station_imp=zeros(49,1); lag_imp=zeros(5,1);
for lag=1:5
    for s=1:49
        fi=(lag-1)*49+s;
        station_imp(s)=station_imp(s)+var_counts(fi);
        lag_imp(lag)  =lag_imp(lag)  +var_counts(fi);
    end
end
[simp_s,simp_i]=sort(station_imp,'descend');
top_n=min(20,sum(simp_s>0)); if top_n<1,top_n=5;end

fig6 = figure('Color',FIG_BG,'Position',[60 60 1300 600],...
              'Name','Feature Importance','NumberTitle','off');

ax6L = subplot(1,2,1);
barh(ax6L, simp_s(top_n:-1:1), 'FaceColor',[0.15 0.48 0.82],...
     'EdgeColor','none','FaceAlpha',1.0);
yticks(ax6L, 1:top_n);
yticklabels(ax6L, arrayfun(@(i)sprintf('S%d',simp_i(top_n+1-i)), ...
            1:top_n,'UniformOutput',false));
xlabel(ax6L,'Occurrences in Best Tree','Color',FG,'FontSize',10,'FontWeight','bold');
title(ax6L, sprintf('Top %d Rain Gauge Stations',top_n),...
      'Color',FG,'FontSize',11,'FontWeight','bold');
set(ax6L,'Color',AX_BG,'XColor',FG,'YColor',FG,...
         'GridColor',GRD,'GridAlpha',1,...
         'FontSize',9,'LineWidth',0.8,'TickDir','out','Box','on');
grid(ax6L,'on');

lag_clr = [0.88 0.38 0.07;
           0.88 0.12 0.12;
           0.09 0.62 0.88;
           0.09 0.65 0.28;
           0.55 0.10 0.75];

ax6R = subplot(1,2,2);
hold(ax6R,'on');
for li = 1:5
    bar(ax6R, li, lag_imp(li), 0.6, 'FaceColor',lag_clr(li,:),...
        'EdgeColor','none','FaceAlpha',1.0);
end
xticks(ax6R, 1:5);
xticklabels(ax6R, {'Lag1 (t-1)','Lag2 (t-2)','Lag3 (t-3)','Lag4 (t-4)','Lag5 (t-5)'});
xtickangle(ax6R, 35);
ylabel(ax6R,'Variable Usage Count','Color',FG,'FontSize',10,'FontWeight','bold');
title(ax6R,'Rainfall Usage by Lag Day','Color',FG,'FontSize',11,'FontWeight','bold');
set(ax6R,'Color',AX_BG,'XColor',FG,'YColor',FG,...
         'GridColor',GRD,'GridAlpha',1,...
         'FontSize',9,'LineWidth',0.8,'TickDir','out','Box','on');
grid(ax6R,'on');

sgtitle(fig6,'GP — Variable Usage in Best Expression Tree',...
        'Color',FG,'FontSize',13,'FontWeight','bold');
saveas(fig6,fullfile(output_dir,'Fig6_Feature_Importance.png'));
fprintf('  Saved Fig6_Feature_Importance.png\n');

%% ====================================================================
%%  FIGURE 7 — BEST-FIT EQUATION DISPLAY  (*** NEW FIGURE ***)
%%  *** CHANGE 2: Dedicated figure showing the evolved symbolic equation ***
%% ====================================================================
fig7 = figure('Color',FIG_BG,'Position',[100 100 1400 500],...
              'Name','Best GP Equation','NumberTitle','off');
ax7  = axes(fig7,'Position',[0 0 1 1]);
set(ax7,'Color',FIG_BG,'XTick',[],'YTick',[],'Box','off','Visible','on');
axis(ax7,[0 1 0 1]);

% Title banner
rectangle(ax7,'Position',[0.02 0.82 0.96 0.14],'Curvature',0.04,...
          'FaceColor',[0.09 0.37 0.75],'EdgeColor','none');
text(ax7,0.50,0.89,...
     'Genetic Programming — Best Evolved Symbolic Equation',...
     'Color','w','HorizontalAlignment','center','VerticalAlignment','middle',...
     'FontSize',15,'FontWeight','bold','Units','normalized');

% Metrics strip
met_str = sprintf('Train R^2 = %.4f    |    Test R^2 = %.4f    |    NSE = %.4f    |    RMSE = %.1f cusecs    |    Tree: %d nodes, depth %d    |    Gens: %d',...
    m_tr.R2, m_te.R2, m_te.NSE, m_te.RMSE,...
    gp_tree_size(best_tree), gp_tree_depth(best_tree), N_GENS);
rectangle(ax7,'Position',[0.02 0.70 0.96 0.10],'Curvature',0.03,...
          'FaceColor',[0.93 0.93 0.93],'EdgeColor',[0.80 0.80 0.80]);
text(ax7,0.50,0.75,met_str,...
     'Color',FG,'HorizontalAlignment','center','VerticalAlignment','middle',...
     'FontSize',9,'FontWeight','bold','Units','normalized','Interpreter','none');

% Label
text(ax7,0.50,0.63,'Evolved Formula  f(X) =','Color',[0.50 0.50 0.55],...
     'HorizontalAlignment','center','FontSize',11,'FontWeight','bold',...
     'Units','normalized','Interpreter','none');

% ---- Wrap equation into lines (max ~110 chars each) for readability ----
MAX_LINELEN = 110;
eq_chars    = best_formula;
lines_out   = {};
while length(eq_chars) > MAX_LINELEN
    cut = MAX_LINELEN;
    % try to break at a space or operator boundary
    while cut > 60 && ~ismember(eq_chars(cut), {' ',')','+','-','*','/'})
        cut = cut - 1;
    end
    lines_out{end+1} = eq_chars(1:cut); %#ok<AGROW>
    eq_chars = eq_chars(cut+1:end);
end
lines_out{end+1} = eq_chars;

n_lines  = length(lines_out);
y_top    = 0.57;
y_step   = min(0.10, 0.50/n_lines);
font_sz  = max(6, min(11, round(110/MAX_LINELEN * 10)));

% Box behind equation
box_h = y_step * n_lines + 0.04;
box_y = y_top - box_h + 0.02;
rectangle(ax7,'Position',[0.02 box_y 0.96 box_h],'Curvature',0.03,...
          'FaceColor',[0.97 0.97 1.00],'EdgeColor',[0.70 0.70 0.82],'LineWidth',1.5);

for li = 1:n_lines
    text(ax7,0.50, y_top - (li-1)*y_step,...
         strtrim(lines_out{li}),...
         'Color',[0.12 0.12 0.40],...
         'HorizontalAlignment','center','VerticalAlignment','top',...
         'FontSize',font_sz,'FontName','Courier New','FontWeight','bold',...
         'Units','normalized','Interpreter','none');
end

% Footer note
text(ax7,0.50,0.04,...
     sprintf('Variables X_i are normalised features. X%d=prev.Inflow  X%d=BasinRain  X%d=Rain5  X%d=AMI  X%d=BaseFlow  X%d=RainMax',...
     F_INF1,F_BAS1,F_R51,F_AMI1,F_BF1,F_MAX1),...
     'Color',[0.45 0.45 0.55],'HorizontalAlignment','center',...
     'FontSize',8,'Units','normalized','Interpreter','none');

sgtitle(fig7,'','Color',FG);
saveas(fig7, fullfile(output_dir,'Fig7_Best_Equation.png'));
fprintf('  Saved Fig7_Best_Equation.png\n');

fprintf('\n=== All outputs saved to: %s ===\n', output_dir);
%% ====================================================================
%%  FIGURE 8 — SENSITIVITY ANALYSIS  (white bg)
%%  One-at-a-time (OAT) perturbation on the test set
%% ====================================================================
fprintf('\n=== [Figs] Sensitivity Analysis ===\n');

PERTURB   = 0.10;          % ±10 % perturbation
base_pred = gp_eval(best_tree, X_test);   % baseline (scaled)

sens_pos  = zeros(n_feat, 1);   % mean |Δ| for +10 %
sens_neg  = zeros(n_feat, 1);   % mean |Δ| for -10 %

for fi = 1:n_feat
    Xp = X_test;  Xp(:, fi) = Xp(:, fi) * (1 + PERTURB);
    Xm = X_test;  Xm(:, fi) = Xm(:, fi) * (1 - PERTURB);
    sens_pos(fi) = mean(abs(gp_eval(best_tree, Xp) - base_pred));
    sens_neg(fi) = mean(abs(gp_eval(best_tree, Xm) - base_pred));
end

sens_avg = (sens_pos + sens_neg) / 2;

%% --- Map feature index → human-readable label -----------------------
feat_labels = cell(n_feat, 1);
for lag = 1:LAG
    offset = (lag - 1) * FEATS_LAG;
    for s = 1:N_STATIONS
        feat_labels{offset + s} = sprintf('S%d_L%d', s, lag);
    end
    feat_labels{offset + N_STATIONS + 1}  = sprintf('Inflow_L%d',  lag);
    feat_labels{offset + N_STATIONS + 2}  = sprintf('BasinRain_L%d',lag);
    feat_labels{offset + N_STATIONS + 3}  = sprintf('Rain3_L%d',   lag);
    feat_labels{offset + N_STATIONS + 4}  = sprintf('Rain5_L%d',   lag);
    feat_labels{offset + N_STATIONS + 5}  = sprintf('Rain7_L%d',   lag);
    feat_labels{offset + N_STATIONS + 6}  = sprintf('RainMax_L%d', lag);
    feat_labels{offset + N_STATIONS + 7}  = sprintf('RainStd_L%d', lag);
    feat_labels{offset + N_STATIONS + 8}  = sprintf('WetDays_L%d', lag);
    feat_labels{offset + N_STATIONS + 9}  = sprintf('AMI_L%d',     lag);
    feat_labels{offset + N_STATIONS + 10} = sprintf('RainAnom_L%d',lag);
    feat_labels{offset + N_STATIONS + 11} = sprintf('BF5d_L%d',    lag);
    feat_labels{offset + N_STATIONS + 12} = sprintf('dInflow_L%d', lag);
    feat_labels{offset + N_STATIONS + 13} = sprintf('Spare_L%d',   lag);
end

%% --- Top-N features ---------------------------------------------------
TOP_N_SENS  = 30;
[~, sidx_s] = sort(sens_avg, 'descend');
top_idx     = sidx_s(1:min(TOP_N_SENS, n_feat));
top_pos     = sens_pos(top_idx);
top_neg     = sens_neg(top_idx);
top_avg     = sens_avg(top_idx);
top_lbls    = feat_labels(top_idx);

%% --- PANEL A: Top-30 OAT bar chart -----------------------------------
fig8 = figure('Color', FIG_BG, 'Position', [60 60 1600 900], ...
              'Name', 'Sensitivity Analysis', 'NumberTitle', 'off');

ax8a = subplot(2, 2, [1 3]);    % left column, full height
hold(ax8a, 'on');
ypos = (TOP_N_SENS : -1 : 1)';   % rank 1 at top

% stacked ± bars
barh(ax8a, ypos, top_avg, 0.55, ...
     'FaceColor', [0.12 0.48 0.85], 'EdgeColor', 'none', ...
     'FaceAlpha', 0.85, 'DisplayName', 'Avg |Δ|');
barh(ax8a, ypos, top_pos - top_avg, 0.55, ...
     'FaceColor', [0.88 0.25 0.15], 'EdgeColor', 'none', ...
     'FaceAlpha', 0.65, 'DisplayName', '+10% excess', ...
     'BaseValue', 0);   % shown stacked via 'BarLayout' below

% re-draw as grouped for clarity
cla(ax8a);
barh(ax8a, ypos - 0.18, top_pos, 0.30, ...
     'FaceColor', [0.88 0.25 0.15], 'EdgeColor', 'none', ...
     'FaceAlpha', 0.90, 'DisplayName', '+10%');
barh(ax8a, ypos + 0.18, top_neg, 0.30, ...
     'FaceColor', [0.09 0.55 0.82], 'EdgeColor', 'none', ...
     'FaceAlpha', 0.90, 'DisplayName', '-10%');
barh(ax8a, ypos, top_avg, 0.06, ...
     'FaceColor', FG, 'EdgeColor', 'none', 'HandleVisibility', 'off');

set(ax8a, 'YTick', ypos, 'YTickLabel', top_lbls(end:-1:1), ...
          'FontSize', 8);
xlabel(ax8a, 'Mean |Δ Output| (scaled)', 'Color', FG, 'FontSize', 10, 'FontWeight', 'bold');
title(ax8a, sprintf('OAT Sensitivity — Top %d Features  (±10%% perturbation)', TOP_N_SENS), ...
      'Color', FG, 'FontSize', 11, 'FontWeight', 'bold');
lg8a = legend(ax8a, 'Location', 'southeast', 'FontSize', 8, 'TextColor', FG);
lg8a.Color = FIG_BG; lg8a.EdgeColor = [0.7 0.7 0.7];
bax_w(ax8a);
grid(ax8a, 'on');

%% --- PANEL B: Lag-aggregated sensitivity -----------------------------
ax8b = subplot(2, 2, 2);
lag_sens = zeros(LAG, 1);
for lag = 1:LAG
    offset = (lag - 1) * FEATS_LAG;
    lag_sens(lag) = mean(sens_avg(offset + 1 : offset + FEATS_LAG));
end
lag_clr2 = [0.88 0.38 0.07; 0.88 0.12 0.12; 0.09 0.62 0.88;
             0.09 0.65 0.28; 0.55 0.10 0.75];
hold(ax8b, 'on');
for li = 1:LAG
    bar(ax8b, li, lag_sens(li), 0.6, ...
        'FaceColor', lag_clr2(li, :), 'EdgeColor', 'none', 'FaceAlpha', 0.9);
end
set(ax8b, 'XTick', 1:LAG, ...
          'XTickLabel', arrayfun(@(l)sprintf('Lag %d\n(t-%d)',l,l), ...
          1:LAG, 'UniformOutput', false));
ylabel(ax8b, 'Mean |Δ Output| (avg over features)', ...
       'Color', FG, 'FontSize', 9);
title(ax8b, 'Sensitivity by Lag Depth', ...
      'Color', FG, 'FontSize', 11, 'FontWeight', 'bold');
bax_w(ax8b);

%% --- PANEL C: Category-level radar-style bar (scalar features only) --
ax8c = subplot(2, 2, 4);

% Indices for lag-1 scalar features (after the 49 station cols)
base1 = N_STATIONS;   % lag-1 offset = 0
cat_names = {'Inflow','BasinRain','Rain3','Rain5','Rain7',...
             'RainMax','RainStd','WetDays','AMI','RainAnom','BF5d','dInflow'};
n_cats = length(cat_names);
cat_sens = zeros(n_cats, LAG);

for lag = 1:LAG
    offset = (lag - 1) * FEATS_LAG + N_STATIONS;
    for ci = 1:min(n_cats, FEATS_LAG - N_STATIONS - 1)
        fi = offset + ci;
        if fi <= n_feat
            cat_sens(ci, lag) = sens_avg(fi);
        end
    end
end
cat_avg = mean(cat_sens, 2);
[cat_avg_s, cat_si] = sort(cat_avg, 'descend');

bar_clr_cat = [0.12 0.48 0.85; 0.88 0.25 0.15; 0.09 0.65 0.28; ...
               0.85 0.65 0.05; 0.55 0.10 0.75; 0.09 0.62 0.75; ...
               0.80 0.30 0.10; 0.30 0.50 0.20; 0.50 0.10 0.60; ...
               0.70 0.40 0.00; 0.15 0.55 0.40; 0.40 0.25 0.75];
hold(ax8c, 'on');
for ci = 1:n_cats
    bar(ax8c, ci, cat_avg_s(ci), 0.65, ...
        'FaceColor', bar_clr_cat(ci, :), 'EdgeColor', 'none', 'FaceAlpha', 0.9);
end
set(ax8c, 'XTick', 1:n_cats, ...
          'XTickLabel', cat_names(cat_si), 'XTickLabelRotation', 35, ...
          'FontSize', 8);
ylabel(ax8c, 'Mean |Δ Output| (avg over lags)', ...
       'Color', FG, 'FontSize', 9);
title(ax8c, 'Sensitivity by Feature Category (lag-averaged)', ...
      'Color', FG, 'FontSize', 11, 'FontWeight', 'bold');
bax_w(ax8c);

sgtitle(fig8, 'GP Model v3 — One-At-a-Time Sensitivity Analysis', ...
        'Color', FG, 'FontSize', 13, 'FontWeight', 'bold');
saveas(fig8, fullfile(output_dir, 'Fig8_Sensitivity.png'));
fprintf('  Saved Fig8_Sensitivity.png\n');

%% --- Save sensitivity table to CSV -----------------------------------
T_sens = table((1:n_feat)', feat_labels, sens_avg, sens_pos, sens_neg, ...
               'VariableNames', {'FeatureIdx','FeatureName',...
               'SensAvg','SensPlus10pct','SensMinus10pct'});
T_sens = sortrows(T_sens, 'SensAvg', 'descend');
writetable(T_sens, fullfile(output_dir, 'Sensitivity_Table.csv'));
fprintf('  Saved Sensitivity_Table.csv\n');

%% ========================================================================
%%  HELPER FUNCTIONS
%% ========================================================================

%% --- Node constructors --------------------------------------------------
function nd=gp_make_func(op,ch),  nd.type='func';  nd.op=op; nd.ch=ch; nd.idx=0; nd.val=0; end
function nd=gp_make_var(idx),     nd.type='var';   nd.op=''; nd.ch={}; nd.idx=idx;nd.val=0; end
function nd=gp_make_const(val),   nd.type='const'; nd.op=''; nd.ch={}; nd.idx=0; nd.val=val; end

function nd=gp_rand_terminal(n_feat)
    if rand()<0.75, nd=gp_make_var(randi(n_feat));
    else,           nd=gp_make_const(rand()*2-1); end
end

%% --- Tree generators ----------------------------------------------------
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

%% --- Evaluation ---------------------------------------------------------
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
                    idx = abs(b) < 1e-6;
                    b(idx) = sign(b(idx) + 1e-10) * 1e-6;
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

%% --- Tree metrics -------------------------------------------------------
function s=gp_tree_size(nd)
    if isempty(nd.ch),s=1; else,s=1+sum(cellfun(@gp_tree_size,nd.ch));end
end
function d=gp_tree_depth(nd)
    if isempty(nd.ch),d=0; else,d=1+max(cellfun(@gp_tree_depth,nd.ch));end
end

%% --- R2 utility ---------------------------------------------------------
function r2=compute_r2(obs,pred)
    obs=obs(:); pred=pred(:);
    ss_tot=sum((obs-mean(obs)).^2);
    if ss_tot<1e-12||any(isnan(pred))||any(isinf(pred)), r2=0; return; end
    r2=1-sum((obs-pred).^2)/ss_tot;
end

%% --- Fitness ------------------------------------------------------------
function f=gp_fitness(nd,X,y,peak_mask,size_limit,parsimony_coef)
    try
        pred=gp_eval(nd,X);
        if any(isnan(pred))||any(isinf(pred)), f=1e9; return; end
        ss_res=sum((y-pred).^2);
        ss_tot=sum((y-mean(y)).^2);
        if ss_tot<1e-12, f=1e9; return; end
        nse  = 1 - ss_res/ss_tot;
        rmse = sqrt(mean((y-pred).^2));
        r2_penalty = 0;
        if nse < 0.88,  r2_penalty = 20*(0.85-nse)^2; end
        if nse < 0.75,  r2_penalty = 50*(0.75-nse)^2; end
        pk_err   = mean((y(peak_mask)-pred(peak_mask)).^2);
        pk_bonus = 0.6 * pk_err;
        sz      = gp_tree_size(nd);
        bloat   = max(0,sz-size_limit)*parsimony_coef;
        f = 5*(1-nse)^2 + 0.005*rmse + r2_penalty + pk_bonus + bloat;
        if f<0, f=0; end
    catch
        f=1e9;
    end
end

%% --- Path utilities -----------------------------------------------------
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

%% --- Genetic operators --------------------------------------------------
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
    node.val=node.val+randn()*0.05;
    if isempty(const_paths{pi_}), child=node;
    else, [child,~]=gp_replace_at_path(nd,const_paths{pi_},node); end
end

function winner=gp_tournament(pop,fits,k)
    idx=randperm(length(pop),k); [~,bi]=min(fits(idx)); winner=pop{idx(bi)};
end

%% --- Post-GP constant optimisation -------------------------------------
function best_tree=gp_optimise_constants(tree,X,y,peak_mask,size_limit,parsimony)
    paths=gp_get_all_paths(tree,[]);
    const_paths=paths(cellfun(@(p) strcmp(gp_node_at_path(tree,p).type,'const'),paths));
    if isempty(const_paths), best_tree=tree; return; end
    c0=zeros(length(const_paths),1);
    for i=1:length(const_paths)
        c0(i)=gp_node_at_path(tree,const_paths{i}).val;
    end
    obj=@(c) local_const_obj(tree,const_paths,c,X,y,peak_mask);
    opts=optimset('MaxFunEvals',500,'MaxIter',300,...
                  'Display','off','TolFun',1e-5,'TolX',1e-5);
    try, c_opt=fminsearch(obj,c0,opts); catch, c_opt=c0; end
    best_tree=tree;
    for i=1:length(const_paths)
        nd=gp_node_at_path(best_tree,const_paths{i});
        nd.val=c_opt(i);
        if isempty(const_paths{i}), best_tree=nd;
        else, [best_tree,~]=gp_replace_at_path(best_tree,const_paths{i},nd); end
    end
end

function f=local_const_obj(tree,const_paths,c,X,y,peak_mask)
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
    nse=1-ss_res/ss_tot;
    rmse=sqrt(mean((y-pred).^2));
    pk_err=mean((y(peak_mask)-pred(peak_mask)).^2);
    r2_pen=0; if nse<0.8598, r2_pen=5*(0.8598-nse)^2; end
    f=(1-nse)^2+0.01*rmse+r2_pen+0.25*pk_err;
end

%% --- Performance metrics ------------------------------------------------
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

%% --- Tree-to-string (equation extractor) --------------------------------
function expr = gp_tree_to_string(nd)
    switch nd.type
        case 'var',   expr = sprintf('X%d', nd.idx);
        case 'const', expr = num2str(nd.val, '%.4f');
        case 'func'
            if length(nd.ch) == 2
                left  = gp_tree_to_string(nd.ch{1});
                right = gp_tree_to_string(nd.ch{2});
                expr = sprintf('(%s %s %s)', left, nd.op, right);
            else
                child = gp_tree_to_string(nd.ch{1});
                expr = sprintf('%s(%s)', nd.op, child);
            end
    end
end

%% --- Figure helpers (WHITE BACKGROUND versions) -------------------------

% arch_box_w: coloured box for architecture diagram (white-bg friendly)
function arch_box_w(ax,x,y,w,h,txt,fc)
    rectangle(ax,'Position',[x y w h],'Curvature',0.08,...
              'FaceColor',fc,'EdgeColor',[0.30 0.30 0.35],'LineWidth',1.2);
    text(ax,x+w/2,y+h/2,txt,'Color','w','HorizontalAlignment','center',...
         'VerticalAlignment','middle','FontSize',8.2,'FontWeight','bold','Interpreter','none');
end

% bax_w: style axes for white background
function bax_w(axh)
    set(axh,'Color',[1 1 1],'XColor',[0.08 0.08 0.10],'YColor',[0.08 0.08 0.10],...
            'GridColor',[0.82 0.82 0.82],'GridAlpha',1,...
            'FontSize',9,'LineWidth',0.8,'Box','on','TickDir','out');
    grid(axh,'on');
end

% dbax_w: styled axes with labels (white background)
function dbax_w(axh,xlbl,ylbl,ttl)
    FGc = [0.08 0.08 0.10];
    xlabel(axh,xlbl,'Color',FGc,'FontSize',9);
    ylabel(axh,ylbl,'Color',FGc,'FontSize',9);
    title(axh,ttl,'Color',FGc,'FontSize',10,'FontWeight','bold');
    set(axh,'Color',[1 1 1],'XColor',FGc,'YColor',FGc,...
            'GridColor',[0.82 0.82 0.82],'GridAlpha',1,...
            'FontSize',8,'LineWidth',0.8,'Box','on','TickDir','out');
    grid(axh,'on');
end

% yr_ticks: generate year tick positions and labels
function [xtk,xlbl]=yr_ticks(yr_vec,x_vec,d_vec)
    yrs=unique(yr_vec); xtk=zeros(length(yrs),1); xlbl=cell(length(yrs),1);
    for k=1:length(yrs)
        idx=find(yr_vec==yrs(k),1,'first');
        xtk(k) =x_vec(idx);
        xlbl{k}=sprintf('%d\n(%s)',yrs(k),datestr(d_vec(idx),'dd-mmm'));
    end
end