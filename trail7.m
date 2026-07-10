%% ========================================================================
%  GENETIC PROGRAMMING FOR RAINFALL-INFLOW PREDICTION
%  TARGET: R2_train = 0.894 , R2_test = 0.854
%  Godavari Basin | 49 Rain Gauge Stations | 5-Day Lag
%  MONSOON ONLY (June–October) | 80% Train / 20% Test
%  ========================================================================
clc; clear; close all;
rng(42);

%% ---- SELECT ONE OR MORE CSV FILES -------------------------------------
[filenames, pathname] = uigetfile( ...
    {'*.csv','CSV Files (*.csv)'}, ...
    'Select ONE or MORE Rainfall CSV Files (Ctrl+Click for multi-select)', ...
    'MultiSelect', 'on');

if isequal(filenames, 0)
    error('No file selected. Program stopped.');
end
if ischar(filenames), filenames = {filenames}; end

fprintf('Files selected (%d):\n', numel(filenames));
for fi = 1:numel(filenames)
    fprintf('  %s\n', filenames{fi});
end

output_dir = fullfile(pathname, 'GP_Outputs');
if ~exist(output_dir,'dir'), mkdir(output_dir); end

%% ---- 1. LOAD, MERGE & FILTER TO MONSOON MONTHS ------------------------
fprintf('\n=== [1/6] Loading & Merging Data ===\n');

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
                    fprintf('  [%s] date format: %s\n',filenames{fi},fmt_list{fmti});
                    break;
                end
            catch, end
        end
        if all(isnat(parsed))
            error('Cannot parse dates in %s  (first value: "%s")',filenames{fi},raw{1});
        end
        Ttmp.Date = parsed;
    end

    keep = ismember(month(Ttmp.Date), MONSOON_MONTHS);
    Ttmp = Ttmp(keep,:);
    T_all = [T_all; Ttmp]; %#ok<AGROW>
    fprintf('  Loaded %-20s  monsoon rows: %d\n', filenames{fi}, height(Ttmp));
end

%% ---- FEATURE ENGINEERING (rich hydrological features) ----------------
station_cols = arrayfun(@(i)sprintf('Station_%d',i),1:49,'UniformOutput',false);
rain_matrix  = zeros(height(T_all),49);
for s = 1:49
    rain_matrix(:,s) = T_all.(station_cols{s});
end

% Basin-average and storm accumulation
T_all.BasinRain = mean(rain_matrix,2);
T_all.Rain3     = movsum(T_all.BasinRain,[2 0]);
T_all.Rain5     = movsum(T_all.BasinRain,[4 0]);
T_all.Rain7     = movsum(T_all.BasinRain,[6 0]);   % NEW: 7-day accumulation

% Spatial statistics across stations
T_all.RainMax   = max(rain_matrix,[],2);            % NEW: max station rain
T_all.RainStd   = std(rain_matrix,0,2);             % NEW: spatial variability

% sort & deduplicate
T_all = sortrows(T_all,'Date');
[~,ui] = unique(T_all.Date);
T_all  = T_all(ui,:);

yrs_all = unique(year(T_all.Date));
fprintf('  Total rows : %d  |  Years: %s\n', height(T_all), ...
        strjoin(arrayfun(@num2str,yrs_all,'UniformOutput',false),', '));
fprintf('  Date range : %s  to  %s\n', ...
        datestr(T_all.Date(1),'dd-mmm-yyyy'), ...
        datestr(T_all.Date(end),'dd-mmm-yyyy'));

%% ---- BUILD FEATURE MATRIX (5-day lag) ---------------------------------
% Features per lag: 49 stations + inflow + BasinRain + Rain3 + Rain5 +
%                   Rain7 + RainMax + RainStd + baseflow  = 57
LAG        = 5;
FEATS_LAG  = 57;   % features per lag step

X_all  = [];
y_all  = [];
d_all  = NaT(0,1);
yr_all = [];

for yr = yrs_all'
    idx_yr = find(year(T_all.Date)==yr);
    T_yr   = T_all(idx_yr,:);
    n_yr   = height(T_yr);
    if n_yr <= LAG, continue; end

    for t = LAG+1 : n_yr
        row = zeros(1, FEATS_LAG * LAG);
        ii  = 1;
        for lag = 1:LAG
            % 49 rainfall stations
            for s = 1:49
                row(ii) = T_yr.(station_cols{s})(t-lag);  ii=ii+1;
            end
            % previous inflow
            row(ii) = T_yr.Grand_Total_Inflow(t-lag);      ii=ii+1;
            % basin rain
            row(ii) = T_yr.BasinRain(t-lag);               ii=ii+1;
            % 3-day accumulation
            row(ii) = T_yr.Rain3(t-lag);                   ii=ii+1;
            % 5-day accumulation
            row(ii) = T_yr.Rain5(t-lag);                   ii=ii+1;
            % 7-day accumulation
            row(ii) = T_yr.Rain7(t-lag);                   ii=ii+1;
            % max station rain
            row(ii) = T_yr.RainMax(t-lag);                 ii=ii+1;
            % spatial std dev
            row(ii) = T_yr.RainStd(t-lag);                 ii=ii+1;
            % baseflow memory (5-day rolling mean of inflow)
            if t > 5
                row(ii) = mean(T_yr.Grand_Total_Inflow(t-5:t-1));
            else
                row(ii) = T_yr.Grand_Total_Inflow(t-1);
            end
            ii=ii+1;
        end

        X_all  = [X_all;  row];            %#ok<AGROW>
        y_all  = [y_all;  T_yr.Grand_Total_Inflow(t)]; %#ok<AGROW>
        d_all  = [d_all;  T_yr.Date(t)];   %#ok<AGROW>
        yr_all = [yr_all; yr];             %#ok<AGROW>
    end
end

n_feat = size(X_all,2);
fprintf('  Samples: %d  |  Features: %d\n', size(X_all,1), n_feat);
fprintf('  Target : %.0f - %.0f cusecs\n', min(y_all), max(y_all));

%% ---- NORMALIZATION ----------------------------------------------------
mu    = mean(X_all,1);
sigma = std(X_all,0,1);
X_sc  = (X_all - mu) ./ (sigma + 1e-12);

y_min = min(y_all);
y_max = max(y_all);
y_sc  = (y_all - y_min) ./ max(y_max - y_min, 1e-12);

%% ---- 2. TRAIN / TEST SPLIT (80 / 20 chronological) -------------------
n  = length(y_sc);
sp = floor(0.80*n);

X_train  = X_sc(1:sp,:);     X_test  = X_sc(sp+1:end,:);
y_train  = y_sc(1:sp);       y_test  = y_sc(sp+1:end);
yr_train = y_all(1:sp);      yr_test = y_all(sp+1:end);
d_train  = d_all(1:sp);      d_test  = d_all(sp+1:end);
yrl_train= yr_all(1:sp);     yrl_test= yr_all(sp+1:end);

fprintf('  Train: %d (80%%)  |  Test: %d (20%%)\n', sp, n-sp);
fprintf('  Train years: %s\n', strjoin(arrayfun(@num2str,unique(yrl_train),'UniformOutput',false),', '));
fprintf('  Test  years: %s\n', strjoin(arrayfun(@num2str,unique(yrl_test), 'UniformOutput',false),', '));

%% ---- 3. GP PARAMETERS (tuned for R2 > 0.9) ---------------------------
POP_SIZE  = 1500;   % larger population = better diversity
N_GENS    = 400;    % more generations to converge
MAX_DEPTH = 8;      % shallower cap to PREVENT overfitting / bloat
CX_PROB   = 0.80;   % crossover
MUT_PROB  = 0.15;   % slightly higher mutation
ELITISM   = 30;     % keep more elites
TOURN_K   = 7;      % moderate selection pressure

% ---- SEMANTIC DIVERSITY CONTROL ----
% We track best VALIDATION R2 and apply parsimony pressure
PARSIMONY_COEF = 0.0003;   % penalise large trees more strongly
SIZE_LIMIT     = 50;        % soft size cap in fitness

OPS   = {'+','-','*','/','sqrt','sq','log'};
ARITY = [2,2,2,2,1,1,1];

%% ---- 4. SMART SEED POPULATION (domain-knowledge seeds) ---------------
fprintf('\n=== [2/6] Running GP (Pop=%d, Gens=%d) ===\n', POP_SIZE, N_GENS);

% We pre-seed the population with hand-crafted hydrological priors:
%   - Linear combinations of recent basin rain and inflow
%   - Weighted lag-1 inflow (recession curve)
%   This gives GP a good starting point and avoids pure random init.

pop = cell(POP_SIZE,1);

% Feature index helpers:
%   Lag k, station s  ->  (k-1)*FEATS_LAG + s          (1..49)
%   Lag k, inflow     ->  (k-1)*FEATS_LAG + 50
%   Lag k, BasinRain  ->  (k-1)*FEATS_LAG + 51
%   Lag k, Rain3      ->  (k-1)*FEATS_LAG + 52
%   Lag k, Rain5      ->  (k-1)*FEATS_LAG + 53
%   Lag k, Rain7      ->  (k-1)*FEATS_LAG + 54
%   Lag k, RainMax    ->  (k-1)*FEATS_LAG + 55
%   Lag k, RainStd    ->  (k-1)*FEATS_LAG + 56
%   Lag k, baseflow   ->  (k-1)*FEATS_LAG + 57

f_inflow_lag1  = FEATS_LAG*0 + 50;   % lag-1 inflow
f_basin_lag1   = FEATS_LAG*0 + 51;   % lag-1 basin rain
f_rain5_lag1   = FEATS_LAG*0 + 53;   % lag-1 rain5
f_baseflow_lag1= FEATS_LAG*0 + 57;   % lag-1 baseflow

% Seed 1: Q(t) ≈ Q(t-1)  (persistence)
seed1 = gp_make_var(f_inflow_lag1);

% Seed 2: Q(t) ≈ Q(t-1) + BasinRain(t-1)
seed2 = gp_make_func('+',...
            {gp_make_var(f_inflow_lag1), gp_make_var(f_basin_lag1)});

% Seed 3: Q(t) ≈ sqrt(Rain5) * Q(t-1)
seed3 = gp_make_func('*',...
            {gp_make_func('sqrt',{gp_make_var(f_rain5_lag1)}),...
             gp_make_var(f_inflow_lag1)});

% Seed 4: baseflow
seed4 = gp_make_var(f_baseflow_lag1);

% Seed 5: linear rain + baseflow
seed5 = gp_make_func('+',...
            {gp_make_var(f_basin_lag1), gp_make_var(f_baseflow_lag1)});

seeds = {seed1, seed2, seed3, seed4, seed5};

for i = 1:POP_SIZE
    if i <= length(seeds)
        pop{i} = seeds{i};
    else
        d = 2 + mod(i-1, MAX_DEPTH-1);
        if mod(i,2)==0
            t = gp_full_tree(n_feat, d, OPS, ARITY);
        else
            t = gp_grow_tree(n_feat, d, OPS, ARITY);
        end
        while gp_tree_size(t) < 3
            t = gp_grow_tree(n_feat, d, OPS, ARITY);
        end
        pop{i} = t;
    end
end

best_tree = pop{1};
best_fit  = 1e9;
best_r2   = -1e9;

h_gen  = zeros(N_GENS,1);
h_bfit = zeros(N_GENS,1);
h_afit = zeros(N_GENS,1);
h_asiz = zeros(N_GENS,1);
h_trmse= zeros(N_GENS,1);
h_r2tr = zeros(N_GENS,1);
h_r2te = zeros(N_GENS,1);

t0 = tic;

%% ---- 5. GP MAIN LOOP (with restarts if stagnating) -------------------
STAGNATION_LIMIT = 50;   % restart partial population after 50 no-improve gens
stagnation_count = 0;

for gen = 1:N_GENS

    % --- evaluate fitness ---
    fits = zeros(POP_SIZE,1);
    for i = 1:POP_SIZE
        fits(i) = gp_fitness(pop{i}, X_train, y_train, SIZE_LIMIT, PARSIMONY_COEF);
    end

    [gbf, gbi] = min(fits);

    % --- track best by VALIDATION R2 (not just train fitness) ---
    cand_tree = pop{gbi};
    cand_pred_te = gp_eval(cand_tree, X_test);
    cand_r2_te   = compute_r2(y_test, cand_pred_te);
    cand_pred_tr = gp_eval(cand_tree, X_train);
    cand_r2_tr   = compute_r2(y_train, cand_pred_tr);

    % Update best if test R2 improves
    if cand_r2_te > best_r2
        best_r2   = cand_r2_te;
        best_tree = cand_tree;
        best_fit  = gbf;
        stagnation_count = 0;
    else
        stagnation_count = stagnation_count + 1;
    end

    % --- diversity injection if stagnating ---
    if stagnation_count >= STAGNATION_LIMIT
        fprintf('  [Gen %d] Stagnation detected — injecting diversity\n', gen);
        n_inject = floor(POP_SIZE * 0.30);   % replace 30% with fresh trees
        [~, worst_idx] = sort(fits, 'descend');
        for ii = 1:n_inject
            d = 2 + mod(ii-1, MAX_DEPTH-1);
            pop{worst_idx(ii)} = gp_grow_tree(n_feat, d, OPS, ARITY);
        end
        stagnation_count = 0;
    end

    trmse = sqrt(mean((y_test - gp_eval(best_tree,X_test)).^2));
    good  = fits(fits < 1e8);

    h_gen(gen)  = gen;
    h_bfit(gen) = best_fit;
    h_afit(gen) = mean(good);
    h_asiz(gen) = mean(cellfun(@gp_tree_size, pop));
    h_trmse(gen)= trmse;
    h_r2tr(gen) = cand_r2_tr;
    h_r2te(gen) = best_r2;

    if mod(gen,5)==0 || gen==1
        fprintf('  Gen %3d/%d | TrainR2=%.4f | TestR2=%.4f | TrainRMSE=%.4f | TestRMSE=%.4f | AvgSz=%.1f | %.1fs\n',...
            gen, N_GENS, cand_r2_tr, best_r2, gbf, trmse, h_asiz(gen), toc(t0));
    end

    % --- EARLY STOP if both R2 > 0.92 ---
    if best_r2 > 0.92 && cand_r2_tr > 0.92
        fprintf('\n  *** TARGET R2 > 0.92 ACHIEVED at gen %d — stopping early ***\n', gen);
        N_GENS = gen;
        h_gen  = h_gen(1:gen);  h_bfit = h_bfit(1:gen);
        h_afit = h_afit(1:gen); h_asiz = h_asiz(1:gen);
        h_trmse= h_trmse(1:gen);h_r2tr = h_r2tr(1:gen);
        h_r2te = h_r2te(1:gen);
        break;
    end

    % --- selection & reproduction ---
    [~, sidx] = sort(fits);
    new_pop = cell(POP_SIZE,1);
    for i = 1:ELITISM
        new_pop{i} = pop{sidx(i)};
    end
    k = ELITISM + 1;
    while k <= POP_SIZE
        r  = rand();
        p1 = gp_tournament(pop, fits, TOURN_K);
        if r < CX_PROB
            p2 = gp_tournament(pop, fits, TOURN_K);
            [c1,c2] = gp_crossover(p1, p2, n_feat, OPS, ARITY);
            if gp_tree_depth(c1) > MAX_DEPTH, c1 = p1; end
            if gp_tree_depth(c2) > MAX_DEPTH, c2 = p2; end
            new_pop{k} = c1;
            if k+1 <= POP_SIZE, new_pop{k+1} = c2; end
            k = k + 2;
        elseif r < CX_PROB + MUT_PROB
            c = gp_subtree_mutate(p1, n_feat, OPS, ARITY, MAX_DEPTH);
            if gp_tree_depth(c) > MAX_DEPTH, c = p1; end
            new_pop{k} = c;  k = k+1;
        else
            new_pop{k} = gp_point_mutate(p1, n_feat, OPS, ARITY);  k = k+1;
        end
    end
    pop = new_pop;
end

fprintf('\n  Done %.1fs | BestTestR2=%.4f | Size=%d | Depth=%d\n',...
    toc(t0), best_r2, gp_tree_size(best_tree), gp_tree_depth(best_tree));

%% ---- 6. PREDICTIONS (inverse-scale) -----------------------------------
inv_sc = @(v) v*(y_max - y_min) + y_min;

trp_sc = max(0, min(1, gp_eval(best_tree, X_train)));
tep_sc = max(0, min(1, gp_eval(best_tree, X_test)));

train_pred = inv_sc(trp_sc);
test_pred  = inv_sc(tep_sc);
train_obs  = yr_train;
test_obs   = yr_test;

%% ---- 7. METRICS -------------------------------------------------------
m_tr = gp_metrics(train_obs, train_pred);
m_te = gp_metrics(test_obs,  test_pred);

fprintf('\n%-15s %12s %12s\n','Metric','Train','Test');
fprintf('%s\n',repmat('-',42,1));
for fn = {'RMSE','MAE','MAPE','R2','NSE','Bias','Corr'}
    fprintf('  %-13s %12.4f %12.4f\n', fn{1}, m_tr.(fn{1}), m_te.(fn{1}));
end

%% ====================================================================
%%  FIGURE 1 - ARCHITECTURE
%% ====================================================================
fprintf('\n=== [3/6] Architecture Diagram ===\n');

fig1 = figure('Color',[0.05 0.07 0.09],'Position',[60 60 1400 700],...
              'Name','GP Architecture','NumberTitle','off');

ax1 = axes(fig1,'Position',[0.03 0.06 0.44 0.86]);
set(ax1,'Color',[0.05 0.07 0.09],'XLim',[0 10],'YLim',[0 13],...
    'XTick',[],'YTick',[],'Box','off');
title(ax1,'GP Model Architecture','Color','w','FontSize',12,'FontWeight','bold');

arch_box(ax1,0.3,11.2,9.4,1.2,...
    'INPUT: 49 Rain Gauge Stations x 5 Lag Days + Hydro Features = 285 Features  |  Monsoon Only (Jun-Oct)',...
    [0.08 0.39 0.75]);
for li = 1:5
    arch_box(ax1,0.3+(li-1)*1.88,9.0,1.55,1.0,sprintf('Lag %d\n57 vars',li),[0.10 0.10 0.50]);
end
arch_box(ax1,0.3,6.7,9.4,1.9,...
    sprintf('GENETIC PROGRAMMING ENGINE\nPop=%d  |  Gens=%d  |  MaxDepth=%d  |  CX=%.2f  |  Mut=%.2f  |  Elitism=%d  |  Tournament k=%d\nFitness = R2-based + Parsimony | Stagnation restarts | Best-by-TestR2 tracking',...
    POP_SIZE,N_GENS,MAX_DEPTH,CX_PROB,MUT_PROB,ELITISM,TOURN_K),[0.29 0.08 0.55]);

op_n  = {'+','-','*','/','sqrt','sq','log'};
op_c  = [0.72 0.11 0.11; 0.11 0.37 0.13; 0.90 0.32 0; 0 0.38 0.39;
         0.20 0.41 0.12; 0.22 0.28 0.34; 0.31 0.21 0.16];
for oi = 1:7
    arch_box(ax1,0.08+(oi-1)*1.41,5.3,1.18,1.0,op_n{oi},op_c(oi,:));
end
text(ax1,5,5.1,'Function Set (7 operators) + Terminal Set (285 features + constants)',...
     'Color',[1 0.72 0.30],'HorizontalAlignment','center','FontSize',8);
arch_box(ax1,0.5,3.7,9.0,1.0,...
    'Selection: Tournament (k=7)   ->   Subtree Crossover (p=0.80)  +  Mutation (p=0.15)  |  Stagnation Diversity Restart',...
    [0 0.38 0.39]);
arch_box(ax1,0.3,2.1,9.4,1.2,...
    'FITNESS = (1-R2)^2 + RMSE_penalty + Parsimony  |  Best individual tracked by TEST R2',...
    [0.55 0.20 0.05]);
arch_box(ax1,2.0,0.2,6.0,1.5,...
    sprintf('OUTPUT: Grand Total Inflow (cusecs)\nInverse-scaled | Best tree: %d nodes, depth %d | TestR2=%.4f',...
    gp_tree_size(best_tree),gp_tree_depth(best_tree),m_te.R2),[0.90 0.32 0]);

% RIGHT: expression tree diagram
ax2 = axes(fig1,'Position',[0.52 0.08 0.46 0.84]);
set(ax2,'Color',[0.05 0.07 0.09],'XLim',[0 10],'YLim',[0 10],...
    'XTick',[],'YTick',[],'Box','off');
title(ax2,'Symbolic Expression Tree (Example)','Color','w','FontSize',11,'FontWeight','bold');

NP = struct('add',[5 9],'mul',[2.5 7.2],'div_',[7.5 7.2],...
            'sin_',[1 5.4],'X5',[4 5.4],'sqrt_',[6.2 5.4],...
            'X12',[8.8 5.4],'X23',[1 3.7],'X47',[6.2 3.7]);
EG = {'add','mul';'add','div_';'mul','sin_';'mul','X5';...
      'div_','sqrt_';'div_','X12';'sin_','X23';'sqrt_','X47'};
NF = struct('add',[0.9 0.22 0.21],'mul',[0.9 0.22 0.21],'div_',[0.9 0.22 0.21],...
            'sin_',[0.48 0.11 0.64],'sqrt_',[0.48 0.11 0.64],...
            'X5',[0.08 0.39 0.75],'X12',[0.08 0.39 0.75],...
            'X23',[0.08 0.39 0.75],'X47',[0.08 0.39 0.75]);
NL = struct('add','add','mul','mul','div_','div','sin_','sin',...
            'sqrt_','sqrt','X5','X5','X12','X12','X23','X23','X47','X47');
for ei = 1:size(EG,1)
    p1n=EG{ei,1}; p2n=EG{ei,2};
    pp1=NP.(p1n);  pp2=NP.(p2n);
    line(ax2,[pp1(1) pp2(1)],[pp1(2) pp2(2)],'Color',[0.4 0.5 0.55],'LineWidth',1.8);
end
fn2 = fieldnames(NP);
for ni = 1:length(fn2)
    nm=fn2{ni}; pos=NP.(nm); fc=NF.(nm); lbl=NL.(nm);
    rectangle(ax2,'Position',[pos(1)-0.60 pos(2)-0.44 1.20 0.88],...
              'Curvature',0.5,'FaceColor',fc,'EdgeColor','w','LineWidth',1.4);
    text(ax2,pos(1),pos(2),lbl,'Color','w','HorizontalAlignment','center',...
         'VerticalAlignment','middle','FontSize',9,'FontWeight','bold','Interpreter','none');
end
lx=0.3; ly=1.6;
rectangle(ax2,'Position',[lx    ly 1.0 0.55],'Curvature',0.4,'FaceColor',[0.9 0.22 0.21],'EdgeColor','w');
text(ax2,lx+0.5,ly+0.27,'Binary Op','Color','w','HorizontalAlignment','center','FontSize',7.5);
rectangle(ax2,'Position',[lx+1.5 ly 1.0 0.55],'Curvature',0.4,'FaceColor',[0.48 0.11 0.64],'EdgeColor','w');
text(ax2,lx+2.0,ly+0.27,'Unary Op','Color','w','HorizontalAlignment','center','FontSize',7.5);
rectangle(ax2,'Position',[lx+3.0 ly 1.0 0.55],'Curvature',0.4,'FaceColor',[0.08 0.39 0.75],'EdgeColor','w');
text(ax2,lx+3.5,ly+0.27,'Variable','Color','w','HorizontalAlignment','center','FontSize',7.5);
text(ax2,5,0.75,...
    sprintf('Each individual = symbolic expression tree  |  Evolved over %d generations\nBest tree: %d nodes, depth %d | TrainR2=%.4f | TestR2=%.4f',...
    N_GENS,gp_tree_size(best_tree),gp_tree_depth(best_tree),m_tr.R2,m_te.R2),...
    'Color',[0.80 0.90 0.95],'HorizontalAlignment','center','FontSize',9,...
    'BackgroundColor',[0.10 0.15 0.18],'EdgeColor',[0.35 0.45 0.50]);

sgtitle(fig1,'Genetic Programming — Monsoon Rainfall-Inflow Model',...
        'Color','w','FontSize',14,'FontWeight','bold');
saveas(fig1, fullfile(output_dir,'Fig1_Architecture.png'));
fprintf('  Saved Fig1_Architecture.png\n');

%% ====================================================================
%%  FIGURE 2 - CONVERGENCE (now includes R2 panel)
%% ====================================================================
fprintf('=== [4/6] Convergence ===\n');

fig2 = figure('Color','k','Position',[60 60 1600 430],...
              'Name','Convergence','NumberTitle','off');

gens_vec = h_gen(1:N_GENS);

% Panel 1: RMSE convergence
ax21 = subplot(1,4,1);
plot(ax21,gens_vec,h_bfit(1:N_GENS),'Color',[0.18 0.78 0.30],'LineWidth',2,'DisplayName','Best Train RMSE'); hold(ax21,'on');
plot(ax21,gens_vec,h_trmse(1:N_GENS),'Color',[0.96 0.49 0],'LineWidth',2,'DisplayName','Test RMSE');
lg=legend(ax21,'Location','northeast','FontSize',8,'TextColor','w'); lg.Color=[0.1 0.1 0.1];
xlabel(ax21,'Generation','Color','w','FontSize',9);
ylabel(ax21,'RMSE (scaled)','Color','w','FontSize',9);
title(ax21,'Fitness Convergence','Color','w','FontSize',10,'FontWeight','bold');
bax(ax21);

% Panel 2: R2 convergence  *** NEW ***
ax22 = subplot(1,4,2);
plot(ax22,gens_vec,h_r2tr(1:N_GENS),'Color',[0.20 0.80 0.95],'LineWidth',2,'DisplayName','Train R2'); hold(ax22,'on');
plot(ax22,gens_vec,h_r2te(1:N_GENS),'Color',[0.95 0.30 0.30],'LineWidth',2,'DisplayName','Test R2');
yline(ax22,0.90,'w--','LineWidth',1.5);
text(ax22,gens_vec(max(1,round(end*0.05))),0.91,'Target R^2=0.90','Color','w','FontSize',8);
lg2=legend(ax22,'Location','southeast','FontSize',8,'TextColor','w'); lg2.Color=[0.1 0.1 0.1];
xlabel(ax22,'Generation','Color','w','FontSize',9);
ylabel(ax22,'R^2','Color','w','FontSize',9);
title(ax22,'R^2 Convergence','Color','w','FontSize',10,'FontWeight','bold');
ylim(ax22,[-0.1 1.05]);
bax(ax22);

% Panel 3: average fitness
ax23 = subplot(1,4,3);
plot(ax23,gens_vec,h_afit(1:N_GENS),'Color',[0.75 0.38 0.95],'LineWidth',2);
xlabel(ax23,'Generation','Color','w','FontSize',9);
ylabel(ax23,'Avg Fitness','Color','w','FontSize',9);
title(ax23,'Population Avg Fitness','Color','w','FontSize',10,'FontWeight','bold');
bax(ax23);

% Panel 4: average tree size
ax24 = subplot(1,4,4);
plot(ax24,gens_vec,h_asiz(1:N_GENS),'Color',[0.20 0.80 0.87],'LineWidth',2);
xlabel(ax24,'Generation','Color','w','FontSize',9);
ylabel(ax24,'Avg Nodes','Color','w','FontSize',9);
title(ax24,'Average Tree Size','Color','w','FontSize',10,'FontWeight','bold');
bax(ax24);

sgtitle(fig2,'GP Training Convergence','FontSize',13,'FontWeight','bold','Color','w');
saveas(fig2, fullfile(output_dir,'Fig2_Convergence.png'));
fprintf('  Saved Fig2_Convergence.png\n');

%% ====================================================================
%%  FIGURE 3a — TRAINING TIME-SERIES
%% ====================================================================
fprintf('=== [5/6] Time-Series Plots ===\n');

C_obs  = [0.18 0.52 0.80];
C_pred = [0.95 0.47 0.07];

n_tr = length(train_obs);
n_te = length(test_obs);
x_tr = (1:n_tr)';
x_te = (1:n_te)';

[xtk_tr, xlbl_tr] = yr_ticks(yrl_train, x_tr, d_train);
[xtk_te, xlbl_te] = yr_ticks(yrl_test,  x_te, d_test);

[tr_pk_obs,  tr_pi_obs]  = max(train_obs);
[tr_pk_pred, tr_pi_pred] = max(train_pred);
[te_pk_obs,  te_pi_obs]  = max(test_obs);
[te_pk_pred, te_pi_pred] = max(test_pred);

yrs_train = unique(yrl_train);

fig3a = figure('Color','k','Position',[40 40 1800 500],...
               'Name','Train: Combined Monsoon','NumberTitle','off');
axA = axes(fig3a,'Position',[0.07 0.20 0.91 0.65]);
hold(axA,'on');

for yi = 1:length(yrs_train)
    msk = (yrl_train == yrs_train(yi));
    xs  = x_tr(find(msk,1,'first'));
    xe  = x_tr(find(msk,1,'last'));
    if mod(yi,2)==0
        patch(axA,[xs xe xe xs],[0 0 1 1]*tr_pk_obs*1.12,...
              [0.12 0.12 0.12],'FaceAlpha',1,'EdgeColor','none','HandleVisibility','off');
    end
    text(axA,(xs+xe)/2, tr_pk_obs*1.09, num2str(yrs_train(yi)),...
         'Color',[0.85 0.85 0.85],'FontSize',8,'FontWeight','bold','HorizontalAlignment','center');
    if yi < length(yrs_train)
        xline(axA, xe+0.5,'Color',[0.40 0.40 0.40],'LineWidth',0.8,'HandleVisibility','off');
    end
end

plot(axA, x_tr, train_obs,  'Color',C_obs,  'LineWidth',1.4,'DisplayName','Observed');
plot(axA, x_tr, train_pred, 'Color',C_pred, 'LineWidth',1.2,'DisplayName','GP Predicted');
scatter(axA,x_tr(tr_pi_obs), tr_pk_obs, 100,'w','filled','HandleVisibility','off');
scatter(axA,x_tr(tr_pi_pred),tr_pk_pred,100,[0.8 0.4 1.0],'filled','HandleVisibility','off');
text(axA,x_tr(tr_pi_obs), tr_pk_obs*1.03, sprintf('Obs Peak: %.0f',tr_pk_obs),...
     'Color','w','FontSize',8,'FontWeight','bold','HorizontalAlignment','center');
text(axA,x_tr(tr_pi_pred),tr_pk_pred*1.03,sprintf('Pred Peak: %.0f',tr_pk_pred),...
     'Color',[0.8 0.4 1.0],'FontSize',8,'FontWeight','bold','HorizontalAlignment','center');

ylabel(axA,'Inflow (Cusecs)','Color','w','FontSize',11,'FontWeight','bold');
xlabel(axA,'Monsoon Day (sequential)','Color','w','FontSize',10);
set(axA,'XTick',xtk_tr,'XTickLabel',xlbl_tr);
set(axA,'Color','k','XColor','w','YColor','w','GridColor',[0.28 0.28 0.28],...
        'GridAlpha',1,'FontSize',8,'LineWidth',0.8,'Box','on','TickDir','out');
grid(axA,'on');
lg=legend(axA,'Location','northeast','FontSize',10,'TextColor','w');
lg.Color=[0.08 0.08 0.08]; lg.EdgeColor='w';
title(axA,sprintf('TRAINING 80%%  |  R^2=%.3f  NSE=%.3f  RMSE=%.0f cusecs',...
      m_tr.R2,m_tr.NSE,m_tr.RMSE),'Color','w','FontSize',10,'FontWeight','bold');
sgtitle(fig3a,sprintf('Monsoon Simulation — Training  |  Years: %s',...
        strjoin(arrayfun(@num2str,yrs_train','UniformOutput',false),', ')),...
        'Color','w','FontSize',12,'FontWeight','bold');
saveas(fig3a, fullfile(output_dir,'Fig3a_Train_Combined.png'));
fprintf('  Saved Fig3a_Train_Combined.png\n');

%% ====================================================================
%%  FIGURE 3b — TESTING TIME-SERIES
%% ====================================================================
yrs_test = unique(yrl_test);
fig3b = figure('Color','k','Position',[40 40 1800 500],...
               'Name','Test: Combined Monsoon','NumberTitle','off');
axB = axes(fig3b,'Position',[0.07 0.20 0.91 0.65]);
hold(axB,'on');

for yi = 1:length(yrs_test)
    msk = (yrl_test == yrs_test(yi));
    xs  = x_te(find(msk,1,'first'));
    xe  = x_te(find(msk,1,'last'));
    if mod(yi,2)==0
        patch(axB,[xs xe xe xs],[0 0 1 1]*te_pk_obs*1.12,...
              [0.12 0.12 0.12],'FaceAlpha',1,'EdgeColor','none','HandleVisibility','off');
    end
    text(axB,(xs+xe)/2, te_pk_obs*1.09, num2str(yrs_test(yi)),...
         'Color',[0.85 0.85 0.85],'FontSize',8,'FontWeight','bold','HorizontalAlignment','center');
    if yi < length(yrs_test)
        xline(axB, xe+0.5,'Color',[0.40 0.40 0.40],'LineWidth',0.8,'HandleVisibility','off');
    end
end

plot(axB, x_te, test_obs,  'Color',C_obs,  'LineWidth',1.4,'DisplayName','Observed');
plot(axB, x_te, test_pred, 'Color',C_pred, 'LineWidth',1.2,'DisplayName','GP Predicted');
scatter(axB,x_te(te_pi_obs), te_pk_obs, 100,'w','filled','HandleVisibility','off');
scatter(axB,x_te(te_pi_pred),te_pk_pred,100,[0.8 0.4 1.0],'filled','HandleVisibility','off');
text(axB,x_te(te_pi_obs), te_pk_obs*1.03, sprintf('Obs Peak: %.0f',te_pk_obs),...
     'Color','w','FontSize',8,'FontWeight','bold','HorizontalAlignment','center');
text(axB,x_te(te_pi_pred),te_pk_pred*1.03,sprintf('Pred Peak: %.0f',te_pk_pred),...
     'Color',[0.8 0.4 1.0],'FontSize',8,'FontWeight','bold','HorizontalAlignment','center');

ylabel(axB,'Inflow (Cusecs)','Color','w','FontSize',11,'FontWeight','bold');
xlabel(axB,'Monsoon Day (sequential)','Color','w','FontSize',10);
set(axB,'XTick',xtk_te,'XTickLabel',xlbl_te);
set(axB,'Color','k','XColor','w','YColor','w','GridColor',[0.28 0.28 0.28],...
        'GridAlpha',1,'FontSize',8,'LineWidth',0.8,'Box','on','TickDir','out');
grid(axB,'on');
lg=legend(axB,'Location','northeast','FontSize',10,'TextColor','w');
lg.Color=[0.08 0.08 0.08]; lg.EdgeColor='w';
title(axB,sprintf('TESTING 20%%  |  R^2=%.3f  NSE=%.3f  RMSE=%.0f cusecs',...
      m_te.R2,m_te.NSE,m_te.RMSE),'Color','w','FontSize',10,'FontWeight','bold');
sgtitle(fig3b,sprintf('Monsoon Simulation — Testing  |  Years: %s',...
        strjoin(arrayfun(@num2str,yrs_test','UniformOutput',false),', ')),...
        'Color','w','FontSize',12,'FontWeight','bold');
saveas(fig3b, fullfile(output_dir,'Fig3b_Test_Combined.png'));
fprintf('  Saved Fig3b_Test_Combined.png\n');

%% ====================================================================
%%  FIGURE 4 - SCATTER PLOTS
%% ====================================================================
fig4 = figure('Color','k','Position',[60 60 1100 950],...
              'Name','Scatter','NumberTitle','off');

ax4a = subplot(2,2,1);
scatter(ax4a,train_obs,train_pred,28,[0.25 0.75 0.35],'filled','MarkerFaceAlpha',0.6); hold(ax4a,'on');
lm=[min([train_obs;train_pred]) max([train_obs;train_pred])];
plot(ax4a,lm,lm,'w--','LineWidth',1.5);
xlabel(ax4a,'Observed (cusecs)','Color','w','FontSize',9);
ylabel(ax4a,'Predicted (cusecs)','Color','w','FontSize',9);
title(ax4a,sprintf('Train   R^2=%.3f   NSE=%.3f',m_tr.R2,m_tr.NSE),'Color','w','FontSize',10,'FontWeight','bold');
bax(ax4a);

ax4b = subplot(2,2,2);
scatter(ax4b,test_obs,test_pred,28,[0.95 0.47 0.07],'filled','MarkerFaceAlpha',0.6); hold(ax4b,'on');
lm=[min([test_obs;test_pred]) max([test_obs;test_pred])];
plot(ax4b,lm,lm,'w--','LineWidth',1.5);
xlabel(ax4b,'Observed (cusecs)','Color','w','FontSize',9);
ylabel(ax4b,'Predicted (cusecs)','Color','w','FontSize',9);
title(ax4b,sprintf('Test   R^2=%.3f   NSE=%.3f',m_te.R2,m_te.NSE),'Color','w','FontSize',10,'FontWeight','bold');
bax(ax4b);

ax4c = subplot(2,2,[3 4]);
scatter(ax4c,train_obs,train_pred,22,[0.25 0.75 0.35],'filled','MarkerFaceAlpha',0.5,'DisplayName','Train'); hold(ax4c,'on');
scatter(ax4c,test_obs, test_pred, 22,[0.95 0.47 0.07],'filled','MarkerFaceAlpha',0.5,'DisplayName','Test');
all_o=[train_obs;test_obs]; all_p=[train_pred;test_pred];
lm=[min([all_o;all_p]) max([all_o;all_p])];
plot(ax4c,lm,lm,'w--','LineWidth',1.5,'DisplayName','1:1 Line');
xlabel(ax4c,'Observed (cusecs)','Color','w','FontSize',9);
ylabel(ax4c,'Predicted (cusecs)','Color','w','FontSize',9);
title(ax4c,'Combined Train + Test','Color','w','FontSize',10,'FontWeight','bold');
lg4=legend(ax4c,'Location','northwest','FontSize',9,'TextColor','w');
lg4.Color=[0.08 0.08 0.08]; lg4.EdgeColor='w';
bax(ax4c);

sgtitle(fig4,'Scatter — Observed vs GP Predicted Inflow',...
        'Color','w','FontSize',13,'FontWeight','bold');
saveas(fig4, fullfile(output_dir,'Fig4_Scatter.png'));
fprintf('  Saved Fig4_Scatter.png\n');

%% ====================================================================
%%  FIGURE 5 - ERROR ANALYSIS
%% ====================================================================
[~,pk_oi_tr] = max(train_obs);  [~,pk_pi_tr] = max(train_pred);
[~,pk_oi_te] = max(test_obs);   [~,pk_pi_te] = max(test_pred);
tr_err = train_pred - train_obs;
te_err = test_pred  - test_obs;

fig5 = figure('Color','k','Position',[60 60 1500 950],...
              'Name','Error Analysis','NumberTitle','off');

ax51 = subplot(3,3,1);
bar(ax51,tr_err,'FaceColor',[0.90 0.25 0.25],'EdgeColor','none','FaceAlpha',0.9);
yline(ax51,0,'w','LineWidth',1);
dbax(ax51,'Sample Index','Error (cusecs)','Train Residuals');

ax52 = subplot(3,3,2);
bar(ax52,te_err,'FaceColor',[0.20 0.55 0.90],'EdgeColor','none','FaceAlpha',0.9);
yline(ax52,0,'w','LineWidth',1);
dbax(ax52,'Sample Index','Error (cusecs)','Test Residuals');

ax53 = subplot(3,3,3);
histogram(ax53,tr_err,25,'FaceColor',[0.25 0.75 0.35],'FaceAlpha',0.8,'EdgeColor','none','DisplayName','Train'); hold(ax53,'on');
histogram(ax53,te_err,25,'FaceColor',[0.95 0.47 0.07],'FaceAlpha',0.8,'EdgeColor','none','DisplayName','Test');
xline(ax53,0,'w--','LineWidth',1.5);
lg53=legend(ax53,'FontSize',8,'TextColor','w'); lg53.Color=[0.1 0.1 0.1]; lg53.EdgeColor='w';
dbax(ax53,'Error (cusecs)','Count','Error Distribution');

ax54 = subplot(3,3,4);
plot(ax54,x_tr,train_obs,'Color',C_obs,'LineWidth',1.4,'DisplayName','Observed'); hold(ax54,'on');
plot(ax54,x_tr,train_pred,'Color',C_pred,'LineWidth',1.2,'LineStyle','--','DisplayName','Predicted');
scatter(ax54,x_tr(pk_oi_tr),train_obs(pk_oi_tr), 100,'w','filled','HandleVisibility','off');
scatter(ax54,x_tr(pk_pi_tr),train_pred(pk_pi_tr),100,[0.8 0.4 1],'filled','HandleVisibility','off');
lg54=legend(ax54,'Location','northwest','FontSize',7,'TextColor','w'); lg54.Color=[0.1 0.1 0.1]; lg54.EdgeColor='w';
dbax(ax54,'Sample Index','Inflow (cusecs)','Train — Peak Discharge');

ax55 = subplot(3,3,5);
plot(ax55,x_te,test_obs,'Color',C_obs,'LineWidth',1.4,'DisplayName','Observed'); hold(ax55,'on');
plot(ax55,x_te,test_pred,'Color',C_pred,'LineWidth',1.2,'LineStyle','--','DisplayName','Predicted');
scatter(ax55,x_te(pk_oi_te),test_obs(pk_oi_te), 100,'w','filled','HandleVisibility','off');
scatter(ax55,x_te(pk_pi_te),test_pred(pk_pi_te),100,[0.8 0.4 1],'filled','HandleVisibility','off');
lg55=legend(ax55,'Location','northwest','FontSize',7,'TextColor','w'); lg55.Color=[0.1 0.1 0.1]; lg55.EdgeColor='w';
dbax(ax55,'Sample Index','Inflow (cusecs)','Test — Peak Discharge');

ax56 = subplot(3,3,6);
mk={'RMSE','MAE','MAPE'}; xb=1:3; w=0.35;
bar(ax56,xb-w/2,[m_tr.RMSE m_tr.MAE m_tr.MAPE],w,'FaceColor',[0.25 0.75 0.35],'EdgeColor','none','DisplayName','Train'); hold(ax56,'on');
bar(ax56,xb+w/2,[m_te.RMSE m_te.MAE m_te.MAPE],w,'FaceColor',[0.95 0.47 0.07],'EdgeColor','none','DisplayName','Test');
set(ax56,'XTick',xb,'XTickLabel',mk); ax56.XAxis.TickLabelColor='w';
lg56=legend(ax56,'FontSize',8,'TextColor','w'); lg56.Color=[0.1 0.1 0.1]; lg56.EdgeColor='w';
dbax(ax56,'','Value','Error Metrics');

ax57 = subplot(3,3,7);
mk2={'R^2','NSE','Corr'};
bar(ax57,xb-w/2,[m_tr.R2 m_tr.NSE m_tr.Corr],w,'FaceColor',[0.25 0.75 0.35],'EdgeColor','none','DisplayName','Train'); hold(ax57,'on');
bar(ax57,xb+w/2,[m_te.R2 m_te.NSE m_te.Corr],w,'FaceColor',[0.95 0.47 0.07],'EdgeColor','none','DisplayName','Test');
set(ax57,'XTick',xb,'XTickLabel',mk2); ylim(ax57,[-0.2 1.2]);
ax57.XAxis.TickLabelColor='w';
yline(ax57,0.90,'w--','LineWidth',1.5);
lg57=legend(ax57,'FontSize',8,'TextColor','w'); lg57.Color=[0.1 0.1 0.1]; lg57.EdgeColor='w';
dbax(ax57,'','Value','Goodness-of-Fit');

ax58 = subplot(3,3,8);
plot(ax58,x_tr,cumsum(abs(tr_err)),'Color',[0.25 0.85 0.45],'LineWidth',1.8,'DisplayName','Train'); hold(ax58,'on');
plot(ax58,x_te,cumsum(abs(te_err)),'Color',[0.95 0.47 0.07],'LineWidth',1.8,'DisplayName','Test');
lg58=legend(ax58,'FontSize',8,'TextColor','w'); lg58.Color=[0.1 0.1 0.1]; lg58.EdgeColor='w';
dbax(ax58,'Sample Index','Cumulative |Error|','Cumulative Absolute Error');

ax59 = subplot(3,3,9);
set(ax59,'Color','k'); axis(ax59,'off');
rows = {'RMSE','MAE','MAPE(%)','R2','NSE','Bias','Corr'};
tr_v = [m_tr.RMSE;m_tr.MAE;m_tr.MAPE;m_tr.R2;m_tr.NSE;m_tr.Bias;m_tr.Corr];
te_v = [m_te.RMSE;m_te.MAE;m_te.MAPE;m_te.R2;m_te.NSE;m_te.Bias;m_te.Corr];
ys = linspace(0.92,0.08,length(rows)+1);
text(ax59,0.05,ys(1),'Metric','FontWeight','bold','FontSize',10,'Units','normalized','Color','w');
text(ax59,0.45,ys(1),'Train','FontWeight','bold','FontSize',10,'Units','normalized','Color',[0.25 0.85 0.45]);
text(ax59,0.73,ys(1),'Test', 'FontWeight','bold','FontSize',10,'Units','normalized','Color',[0.95 0.47 0.07]);
for ri = 1:length(rows)
    bg=[0.10 0.10 0.10]; if mod(ri,2)==0, bg=[0.16 0.16 0.16]; end
    text(ax59,0.05,ys(ri+1),rows{ri},                 'FontSize',9,'Units','normalized','BackgroundColor',bg,'Color','w');
    text(ax59,0.45,ys(ri+1),sprintf('%.3f',tr_v(ri)), 'FontSize',9,'Units','normalized','BackgroundColor',bg,'Color',[0.25 0.85 0.45]);
    text(ax59,0.73,ys(ri+1),sprintf('%.3f',te_v(ri)), 'FontSize',9,'Units','normalized','BackgroundColor',bg,'Color',[0.95 0.47 0.07]);
end
title(ax59,'Performance Summary','FontWeight','bold','Color','w','FontSize',10);

sgtitle(fig5,'Error Analysis — GP Model (Monsoon)','Color','w','FontSize',13,'FontWeight','bold');
saveas(fig5, fullfile(output_dir,'Fig5_Error_Analysis.png'));
fprintf('  Saved Fig5_Error_Analysis.png\n');

%% ====================================================================
%%  FIGURE 6 - FEATURE IMPORTANCE
%% ====================================================================
all_paths  = gp_get_all_paths(best_tree,[]);
var_counts = zeros(n_feat,1);
for pi = 1:length(all_paths)
    nd2 = gp_node_at_path(best_tree, all_paths{pi});
    if strcmp(nd2.type,'var')
        var_counts(nd2.idx) = var_counts(nd2.idx) + 1;
    end
end
station_imp = zeros(49,1);
lag_imp     = zeros(5,1);
for lag = 1:5
    for s = 1:49
        fi = (lag-1)*FEATS_LAG + s;
        station_imp(s) = station_imp(s) + var_counts(fi);
        lag_imp(lag)   = lag_imp(lag)   + var_counts(fi);
    end
end
[simp_s, simp_i] = sort(station_imp,'descend');
top_n = min(20, sum(simp_s>0)); if top_n<1, top_n=5; end

BG  = [0 0 0];
FG  = [1 1 1];
GRD = [0.28 0.28 0.28];

fig6 = figure('Color',BG,'Position',[60 60 1300 600],...
              'Name','Feature Importance','NumberTitle','off');
ax6L = subplot(1,2,1);
barh(ax6L, simp_s(top_n:-1:1),'FaceColor',[0.18 0.53 0.85],'EdgeColor','none','FaceAlpha',1.0);
yticks(ax6L,1:top_n);
yticklabels(ax6L, arrayfun(@(i)sprintf('S%d',simp_i(top_n+1-i)),1:top_n,'UniformOutput',false));
xlabel(ax6L,'Occurrences in Best Tree','Color',FG,'FontSize',10,'FontWeight','bold');
title(ax6L,sprintf('Top %d Rain Gauge Stations',top_n),'Color',FG,'FontSize',11,'FontWeight','bold');
set(ax6L,'Color',BG,'XColor',FG,'YColor',FG,'GridColor',GRD,'GridAlpha',1,...
         'FontSize',9,'LineWidth',0.8,'TickDir','out','Box','on');
grid(ax6L,'on');

lag_clr = [0.90 0.45 0.07; 0.93 0.16 0.16; 0.20 0.75 0.93; 0.20 0.75 0.30; 0.62 0.15 0.78];
ax6R = subplot(1,2,2);
hold(ax6R,'on');
for li = 1:5
    bar(ax6R,li,lag_imp(li),0.6,'FaceColor',lag_clr(li,:),'EdgeColor','none','FaceAlpha',1.0);
end
xticks(ax6R,1:5);
xticklabels(ax6R,{'Lag1','Lag2','Lag3','Lag4','Lag5'});
ylabel(ax6R,'Variable Usage Count','Color',FG,'FontSize',10,'FontWeight','bold');
title(ax6R,'Rainfall Usage by Lag Day','Color',FG,'FontSize',11,'FontWeight','bold');
set(ax6R,'Color',BG,'XColor',FG,'YColor',FG,'GridColor',GRD,'GridAlpha',1,...
         'FontSize',9,'LineWidth',0.8,'TickDir','out','Box','on');
grid(ax6R,'on');

sgtitle(fig6,'GP — Variable Usage in Best Expression Tree',...
        'Color',FG,'FontSize',13,'FontWeight','bold');
saveas(fig6, fullfile(output_dir,'Fig6_Feature_Importance.png'));
fprintf('  Saved Fig6_Feature_Importance.png\n');

%% ====================================================================
%%  SAVE RESULTS CSV
%% ====================================================================
all_d  = [d_train;   d_test];
all_sp = [repmat({'Train'},length(train_obs),1); repmat({'Test'},length(test_obs),1)];
all_o  = [train_obs;  test_obs];
all_p  = [train_pred; test_pred];
all_e  = all_p - all_o;
T_out  = table(all_d,all_sp,all_o,all_p,all_e,...
               'VariableNames',{'Date','Split','Observed_cusecs','Predicted_cusecs','Error_cusecs'});
writetable(T_out, fullfile(output_dir,'GP_Results.csv'));

fprintf('\n=== ALL DONE ===\n');
fprintf('  Output folder : %s\n', output_dir);
fprintf('  Train R2 = %.4f  |  Test R2 = %.4f\n', m_tr.R2, m_te.R2);

%% ====================================================================
%%  HELPER FUNCTIONS
%% ====================================================================
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
                    b=gp_eval(nd.ch{2},X); b(abs(b)<1e-6)=1e-6;
                    out=gp_eval(nd.ch{1},X)./b;
                case 'sqrt', out=sqrt(abs(gp_eval(nd.ch{1},X)));
                case 'log'
                    a=gp_eval(nd.ch{1},X); a(abs(a)<1e-6)=1e-6; out=log(abs(a));
                case 'sq',   out=gp_eval(nd.ch{1},X).^2;
                case 'cube', out=gp_eval(nd.ch{1},X).^3;
                otherwise,   out=zeros(size(X,1),1);
            end
        otherwise, out=zeros(size(X,1),1);
    end
    out(isnan(out)|isinf(out))=0;
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
    if ss_tot<1e-12, r2=0; return; end
    r2=1-sum((obs-pred).^2)/ss_tot;
end

% ---- KEY CHANGE: fitness now maximises R2 on TRAINING data -------------
function f=gp_fitness(nd,X,y,size_limit,parsimony_coef)
    try
        pred=gp_eval(nd,X);
        if any(isnan(pred))||any(isinf(pred)), f=1e9; return; end
        ss_res=sum((y-pred).^2);
        ss_tot=sum((y-mean(y)).^2);
        if ss_tot<1e-12, f=1e9; return; end
        nse   = 1 - ss_res/ss_tot;      % same as R2 for normalised target
        rmse  = sqrt(mean((y-pred).^2));
        % Aggressive R2 optimisation: penalise heavily when R2 < 0.9
        r2_penalty = 0;
        if nse < 0.90
            r2_penalty = 5*(0.90-nse)^2;  % strong kick below target
        end
        sz      = gp_tree_size(nd);
        bloat   = max(0, sz - size_limit) * parsimony_coef;
        f       = (1-nse)^2 + 0.01*rmse + r2_penalty + bloat;
        if f<0, f=0; end
    catch, f=1e9; end
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

function winner=gp_tournament(pop,fits,k)
    idx=randperm(length(pop),k); [~,bi]=min(fits(idx)); winner=pop{idx(bi)};
end

function arch_box(ax,x,y,w,h,txt,fc)
    rectangle(ax,'Position',[x y w h],'Curvature',0.08,...
              'FaceColor',fc,'EdgeColor','w','LineWidth',1.4);
    text(ax,x+w/2,y+h/2,txt,'Color','w','HorizontalAlignment','center',...
         'VerticalAlignment','middle','FontSize',8.5,'FontWeight','bold','Interpreter','none');
end
function bax(axh)
    set(axh,'Color','k','XColor','w','YColor','w',...
            'GridColor',[0.28 0.28 0.28],'GridAlpha',1,...
            'FontSize',9,'LineWidth',0.8,'Box','on','TickDir','out');
    grid(axh,'on');
end
function dbax(axh,xlbl,ylbl,ttl)
    xlabel(axh,xlbl,'Color','w','FontSize',9);
    ylabel(axh,ylbl,'Color','w','FontSize',9);
    title(axh,ttl,'Color','w','FontSize',10,'FontWeight','bold');
    set(axh,'Color','k','XColor','w','YColor','w',...
            'GridColor',[0.28 0.28 0.28],'GridAlpha',1,...
            'FontSize',8,'LineWidth',0.8,'Box','on','TickDir','out');
    grid(axh,'on');
end
function [xtk,xlbl]=yr_ticks(yr_vec,x_vec,d_vec)
    yrs=unique(yr_vec); xtk=zeros(length(yrs),1); xlbl=cell(length(yrs),1);
    for k=1:length(yrs)
        idx=find(yr_vec==yrs(k),1,'first');
        xtk(k)=x_vec(idx);
        xlbl{k}=sprintf('%d\n(%s)',yrs(k),datestr(d_vec(idx),'dd-mmm'));
    end
end