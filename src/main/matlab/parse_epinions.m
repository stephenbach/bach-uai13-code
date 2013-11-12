clear;

methods = {'quad-mle-100-5.0', 'quad-mple-100-5.0','quad-mm-0.1-none-none-false',...
    'linear-mle-100-5.0', 'linear-mple-100-5.0', 'linear-mm-0.1-none-none-false', ...
    'bool-mle-100-5.0', 'bool-mple-100-5.0', 'bool-mm-0.1-none-none-false'};

allPos = {};
allNeg = {};
allROC = {};

files = {'../../../output/epinions.hlmrf-q.out'
	'../../../output/epinions.hlmrf-l.out'
	'../../../output/epinions.mrf.out'};
for k = 1:length(files)
    fin = fopen(files{k}, 'r');
    while ~feof(fin)
        line = fgetl(fin);
        pattern = 'Method ([^,]+), fold ([^,]+), auprc positive: ([^,]+), negative: ([^,]+), auROC: ([^,]+)';
        matches = regexp(line, pattern, 'tokens');
        if ~isempty(matches)
            matches = matches{1};
            method = matches{1};
            fold = str2num(matches{2})+1;
            pos = str2num(matches{3});
            neg = str2num(matches{4});
            roc = str2num(matches{5});

            i = find(strcmp(method, methods));
            
            if ~isempty(i)
                allPos{i}(fold) = pos;
                allNeg{i}(fold) = neg;
                allROC{i}(fold) = roc;
            end
        end
    end
end

%% compute means

for i = 1:length(methods)
    meanPos(i) = mean(allPos{i});
    meanNeg(i) = mean(allNeg{i});
    meanROC(i) = mean(allROC{i});
end


%% get best method

[~, bestPos] = max(meanPos);
[~, bestNeg] = max(meanNeg);
[~, bestROC] = max(meanROC);

%% sig test all vs best

threshold = 0.01;

for i = 1:length(methods)
    sigPos(i) = ttest(allPos{bestPos}, allPos{i}, 'Alpha', threshold);
    sigNeg(i) = ttest(allNeg{bestNeg}, allNeg{i}, 'Alpha', threshold);
    sigROC(i) = ttest(allROC{bestROC}, allROC{i}, 'Alpha', threshold);
end

%% print latex
latexNames = {'HL-MRF-Q (MLE)', 'HL-MRF-Q (MPLE)', 'HL-MRF-Q (LME)',...
    'HL-MRF-L (MLE)', 'HL-MRF-L (MPLE)', 'HL-MRF-L (LME)',...
    'MRF (MLE)', 'MRF (MPLE)', 'MRF (LME)'};

fprintf('BEGIN SOCIAL TRUST PREDICTION RESULTS TABLE\n\n')
fprintf('\\begin{tabular}{lrrr}\n')
fprintf('\\toprule\n')
fprintf(' & ROC & P-R (+) & P-R (-) \\\\\n')
fprintf('\\midrule\n')

for i = 1:length(latexNames)
    str = latexNames{i}; 
    
    if bestROC ~= i && sigROC(i)
        str = sprintf('%s & %0.3f', str, meanROC(i));
    else
        str = sprintf('%s & \\textbf{%0.3f}', str, meanROC(i));
    end
    
    if bestPos ~= i && sigPos(i)
        str = sprintf('%s & %0.3f', str, meanPos(i));
    else
        str = sprintf('%s & \\textbf{%0.3f}', str, meanPos(i));
    end
        
    if bestNeg ~= i && sigNeg(i)
        str = sprintf('%s & %0.3f', str, meanNeg(i));
    else
        str = sprintf('%s & \\textbf{%0.3f}', str, meanNeg(i));
    end
      
    fprintf('%s\\\\\n', str);

  if (i == 3 || i == 6)
	fprintf('\\addlinespace\n')
  end
end

fprintf('\\bottomrule\n')
fprintf('\\end{tabular}\n')
fprintf('\nEND SOCIAL TRUST PREDICTION RESULTS TABLE\n')

exit;

