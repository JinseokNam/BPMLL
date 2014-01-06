function [ cost, grad ] = BPMLLCost(theta, X, T, visibleSize, hiddenSize, labelSize, options)
                                         
l2_lambda = options.l2_lambda;
useGPU = false;
debug = false;
if isfield(options,'useGPU')
    useGPU = options.useGPU;
end
if isfield(options,'debug')
    debug = options.debug;
end

%% Unroll parameter
% Extract out the "stack"
W1 = reshape(theta(1:visibleSize*hiddenSize), hiddenSize, visibleSize);
b1 = theta(visibleSize*hiddenSize+1:visibleSize*hiddenSize+hiddenSize);
W2 = reshape(theta(visibleSize*hiddenSize+hiddenSize+1:visibleSize*hiddenSize+hiddenSize+hiddenSize*labelSize), labelSize, hiddenSize);
b2 = theta(visibleSize*hiddenSize+hiddenSize+hiddenSize*labelSize+1:end);

[D, M] = size(X);

% forward pass for autoencoder
[H, dH] = tanh_act(bsxfun(@plus, W1*X, b1));
[O, dO] = tanh_act(bsxfun(@plus, W2*H, b2));

if useGPU
    [delta] = eval_bpmll_loss_gpumex(T,O);
    if debug
        [err,~] = computePW(gather(T),gather(O));
        err = gpuArray(err);
    end
else
    [delta] = eval_bpmll_loss_cpumex(T,O);
    if debug
        [err,~] = computePW(T,O);
    end
end

if debug
    cost = 1/M*sum(err) + .5*l2_lambda*sum(sum(W1(:).^2) + sum(W2(:).^2));
else
    cost = 0;
end

if nargout > 1

    delta = (1/M)*delta .* dO; 
    W2grad = delta*H' + l2_lambda*W2;
    b2grad = sum(delta,2);

    delta = (W2'*delta) .* dH;

    W1grad = delta*X' + l2_lambda*W1;
    b1grad = sum(delta,2);

    %% Roll gradient vector
    grad = [W1grad(:) ; b1grad(:) ; W2grad(:) ; b2grad(:)]; 
else
    grad = zeros(size(theta));
end


end

function [F,dF] = sigm_act(X)
    F = 1./(1+exp(-X));
    dF = F.*(1-F);
end

function [F, dF] = tanh_act(X)
    F = tanh(X);
    dF = 1-F.^2;
end
