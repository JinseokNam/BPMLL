addpath('mex/');

useGPU = true;
debug = true;

hiddenSize = 5;
visibleSize = 20;
labelSize = 20;
layers = [hiddenSize labelSize];
stack = cell(length(layers),1);
for i=1:numel(stack)
	if i==1
		in_units = visibleSize;
	else
		in_units = layers(i-1);
	end
	out_units = layers(i);

	r = sqrt(6)/sqrt(in_units+out_units+1);
	stack{i}.w = rand(out_units, in_units)*2*r-r;
	stack{i}.b = zeros(out_units,1);
end

theta = [stack{1}.w(:); stack{1}.b(:); stack{2}.w(:); stack{2}.b(:)];

M=12;
data = rand(visibleSize, M);
labels = binornd(1,.5,labelSize,M);
skewed_example_index = all(labels==1,1) | all(labels==0,1);
data(:,skewed_example_index) = [];
labels(:,skewed_example_index) = [];
labels(labels ==0) = -1;

options.l2_lambda = 1e-4;
options.useGPU = useGPU;
options.debug = debug;

if useGPU
	theta = gpuArray(theta);
	data = gpuArray(data);
	labels = gpuArray(labels);
end

[cost, grad] = BPMLLCost(theta, data, labels, visibleSize, hiddenSize, labelSize, options);

numgrad = computeNumericalGradient( @(p) BPMLLCost(p, data, labels, visibleSize, hiddenSize, labelSize, options), theta);

disp([gather(full(numgrad)) gather(full(grad))]);
diff = norm(numgrad-grad)/norm(numgrad+grad);
disp('Difference between numerical gradients and actual graidents should be close to zero (e.g. 1e-9).');
disp(diff);
