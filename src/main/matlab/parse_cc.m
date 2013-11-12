clear;

methods = {'quad-mle-100-5.0', 'quad-mple-100-5.0', 'quad-mm-0.1-none-none-false', ...
    'linear-mle-100-5.0', 'linear-mple-100-5.0', 'linear-mm-0.1-none-none-false', ...
    'bool-mle-100-5.0', 'bool-mple-100-5.0', 'bool-mm-0.1-none-none-false'};

allAcc = {};
allPrec = {};
allRec = {};
allF1 = {};

%% read prediction output

files{1} = {'../../../output/citeseer.hlmrf-q.out'
	'../../../output/citeseer.hlmrf-l.out'
	'../../../output/citeseer.mrf.out'};
files{2} = {'../../../output/cora.hlmrf-q.out'
	'../../../output/cora.hlmrf-l.out'
	'../../../output/cora.mrf.out'};

for k = 1:length(files)
  for j = 1:length(files{k})
    fin = fopen(files{k}{j}, 'r');
    while ~feof(fin)
        line = fgetl(fin);
        pattern = 'Method ([^,]+), fold ([^,]+), acc ([^,]+), prec([^,]+), rec ([^,]+), F1 ([^,]+)';
        matches = regexp(line, pattern, 'tokens');
        if ~isempty(matches)
            matches = matches{1};
            method = matches{1};
            fold = str2num(matches{2})+1;
            acc = str2num(matches{3});
            prec = str2num(matches{4});
            rec = str2num(matches{5});
            f1 = str2num(matches{6});
            
            i = find(strcmp(method, methods));
            
            if ~isempty(i)
                allAcc{k}{i}(fold) = acc;
                allPrec{k}{i}(fold) = prec;
                allRec{k}{i}(fold) = rec;
                allF1{k}{i}(fold) = f1;
            end
        end
    end
  end
end


%% compute means
for k = 1:length(files)
    for i = 1:length(methods)
        meanPrec(k,i) = mean(allPrec{k}{i});
        meanRec(k,i) = mean(allRec{k}{i});
        meanF1(k,i) = mean(allF1{k}{i});
        meanAcc(k,i) = mean(allAcc{k}{i});
    end
end


%% get best method

for k = 1:length(files)
    [~, bestPrec(k)] = max(meanPrec(k,:));
    [~, bestRec(k)] = max(meanRec(k,:));
    [~, bestF1(k)] = max(meanF1(k,:));
    [~, bestAcc(k)] = max(meanAcc(k,:));
end

%% sig test all vs best

threshold = 0.01;
for k = 1:length(files)
    for i = 1:length(methods)
        sigPrec(k,i) = ttest(allPrec{k}{bestPrec(k)}, allPrec{k}{i}, 'Alpha', threshold);
        sigRec(k,i) = ttest(allRec{k}{bestRec(k)}, allRec{k}{i}, 'Alpha', threshold);
        sigF1(k,i) = ttest(allF1{k}{bestF1(k)}, allF1{k}{i}, 'Alpha', threshold);
        sigAcc(k,i) = ttest(allAcc{k}{bestAcc(k)}, allAcc{k}{i}, 'Alpha', threshold);
    end
end

% Here precision is multiclass classification accuracy

%% print latex
latexNames = {'HL-MRF-Q (MLE)', 'HL-MRF-Q (MPLE)', 'HL-MRF-Q (LME)',...
    'HL-MRF-L (MLE)', 'HL-MRF-L (MPLE)', 'HL-MRF-L (LME)', ...
    'MRF (MLE)', 'MRF (MPLE)', 'MRF (LME)'};

fprintf('BEGIN COLLECTIVE CLASSIFICATION RESULTS TABLE\n\n')
fprintf('\\begin{tabular}{lrrr}\n')
fprintf('\\toprule\n')
fprintf('  & Citeseer & Cora \\\\\n')
fprintf('\\midrule\n')

for i = 1:length(latexNames)
  str = latexNames{i};
    
  for k = 1:length(files)

    if bestPrec(k) ~= i && sigPrec(k,i)
        str = sprintf('%s & %0.3f', str, meanPrec(k,i));
    else
        str = sprintf('%s & \\textbf{%0.3f}', str, meanPrec(k,i));
    end
  end

  fprintf('%s\\\\\n', str);

  if (i == 3 || i == 6)
	fprintf('\\addlinespace\n')
  end
end

fprintf('\\bottomrule\n')
fprintf('\\end{tabular}\n')
fprintf('\nEND COLLECTIVE CLASSIFICATION RESULTS TABLE\n')

exit;

