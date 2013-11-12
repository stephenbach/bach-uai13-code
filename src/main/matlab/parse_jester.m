clear;

%% Reads HL-MRF results

methods = {'quad-mle-100-1.0', 'quad-mple-100-1.0','quad-mm-0.1-none-none-false',...
    'linear-mle-100-1.0', 'linear-mple-100-1.0', 'linear-mm-0.1-none-none-false'};

allMSE = {};
allMAE = {};
files = {'../../../output/jester.hlmrf-q.out'
	'../../../output/jester.hlmrf-l.out'};
for k = 1:length(files)
    fin = fopen(files{k}, 'r');
    while ~feof(fin)
        line = fgetl(fin);
	pattern = 'INFO  edu.umd.cs.bachuai13.jester.Jester  - Fold ([^,]+) : ([^,]+) : MSE ([^,]+) : MAE ([^,]+)';
        matches = regexp(line, pattern, 'tokens');
        if ~isempty(matches)
            matches = matches{1};
            method = matches{2};
            fold = str2num(matches{1})+1;
            mse = str2num(matches{3});
            mae = str2num(matches{4});

            i = find(strcmp(method, methods));
            
            if ~isempty(i)
                allMSE{i}(fold) = mse;
                allMAE{i}(fold) = mae;
            end
        end
    end
end

%% Reads BPMF results

fin = fopen('../../../output/jester.bpmf.out', 'r');
fold = 1;
while ~feof(fin)
	line = fgetl(fin);
	pattern = 'BPMF: MSE=([^,]+), MAE=([^,]+)';
	matches = regexp(line, pattern, 'tokens');
	if ~isempty(matches)
	    matches = matches{1};
	    mse = str2num(matches{1});
	    mae = str2num(matches{2});

	    allMSE{length(methods)+1}(fold) = mse;
	    allMAE{length(methods)+1}(fold) = mae;

	    fold = fold + 1;
	end
end

%% compute means

for i = 1:length(methods)+1
    meanMSE(i) = mean(allMSE{i});
    meanMAE(i) = mean(allMAE{i});
end


%% get best method

[~, bestMSE] = min(meanMSE);
[~, bestMAE] = min(meanMAE);

%% sig test all vs best

threshold = 0.01;

for i = 1:length(methods)+1
    sigMSE(i) = ttest(allMSE{bestMSE}, allMSE{i}, 'Alpha', threshold);
    sigMAE(i) = ttest(allMAE{bestMAE}, allMAE{i}, 'Alpha', threshold);
end

%% print latex
latexNames = {'HL-MRF-Q (MLE)', 'HL-MRF-Q (MPLE)', 'HL-MRF-Q (LME)',...
    'HL-MRF-L (MLE)', 'HL-MRF-L (MPLE)', 'HL-MRF-L (LME)',...
    'BPMF'};

fprintf('BEGIN PREFERENCE PREDICTION RESULTS TABLE\n\n')
fprintf('\\begin{tabular}{lcc}\n')
fprintf('\\toprule\n')
fprintf(' & NMSE & NMAE \\\\\n')
fprintf('\\midrule\n')

for i = 1:length(latexNames)
    str = latexNames{i}; 
    
    if bestMSE ~= i && sigMSE(i)
        str = sprintf('%s & %0.4f', str, meanMSE(i));
    else
        str = sprintf('%s & \\textbf{%0.4f}', str, meanMSE(i));
    end
    
    if bestMAE ~= i && sigMAE(i)
        str = sprintf('%s & %0.4f', str, meanMAE(i));
    else
        str = sprintf('%s & \\textbf{%0.4f}', str, meanMAE(i));
    end
      
    fprintf('%s\\\\\n', str);

  if (i == 3 || i == 6)
	fprintf('\\addlinespace\n')
  end
end

fprintf('\\bottomrule\n')
fprintf('\\end{tabular}\n')
fprintf('\nEND PREFERENCE PREDICTION RESULTS TABLE\n')

exit;

