
clear

% add BPTF to path, just in case
addpath('export_bptf');
addpath('export_bptf/lib');

output = fopen('../../../output/jester.bpmf.out', 'w');

nfolds = 10;
stats = zeros(nfolds,2);
for f=1:nfolds
	
	% load the data from file
	x_tr_o = load(sprintf('../../../data/jester/ratings/jester-1-tr-obs-%d.txt', f-1));
	x_tr_u = load(sprintf('../../../data/jester/ratings/jester-1-tr-uno-%d.txt', f-1));
	x_te_o = load(sprintf('../../../data/jester/ratings/jester-1-te-obs-%d.txt', f-1));
	x_te_u = load(sprintf('../../../data/jester/ratings/jester-1-te-uno-%d.txt', f-1));
	
	% convert to sparse format
	X_tr = spTensor([spconvert([x_tr_o ; x_tr_u]) ; spconvert(x_te_o)]);
	X_te = spTensor([sparse(1000,100) ; spconvert(x_te_u)]);
	
	% config parameters
	d = 30;
	maxiter = 200;
	nsamp = 100;
	pn = 50e-3;
	learnrate = 1e-3;
	alpha = 2;
	fprintf('test')
	
	% init factorization using PMF gradient descent
	pars = struct('ridge',pn,'learn_rate',learnrate,'range',[0,1],'max_iter',maxiter);
	[U, V, dummy, r_pmf] = PMF_Grad(X_tr, X_te, d, pars);
	fprintf(output, 'PMF: %.4f\n', r_pmf);
	% run BPMF
	pars = struct('max_iter',maxiter,'n_sample',nsamp,'save_sample',false);
	[Us, Vs] = BPMF(X_tr, X_te, d, alpha, [], {U,V}, pars);
	Y = BPMF_Predict(Us, Vs, d, X_te, [0,1]);
	diff = Y.vals - X_te.vals;
	stats(f,1) = mean(diff(:).^2);
	stats(f,2) = mean(abs(diff(:)));
	fprintf(output, 'BPMF: MSE=%.4f, MAE=%.4f\n', stats(f,1), stats(f,2));
	
end

fclose(output);
exit
