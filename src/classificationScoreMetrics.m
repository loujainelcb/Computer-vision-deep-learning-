function metrics = classificationScoreMetrics(scores, labels, classNames, weights)
% Deterministic, whole-dataset metrics. Accuracy values are fractions [0,1].
% Uses exactly runTrainingFineTuning's weighted crossentropy formulation over all images.
if isa(scores,'dlarray'), scores=extractdata(scores); end
scores = gather(scores);
K = numel(classNames);
N = numel(labels);
assert(isequal(size(scores),[N K]), 'Expected N-by-K minibatchpredict scores.');
[ok,truth] = ismember(string(labels(:)),string(classNames(:)));
assert(all(ok),'Unknown class in evaluation labels.');
[~,pred] = max(scores,[],2);
C = accumarray([truth,pred],1,[K K]);
tp = diag(C);
precision = tp./max(sum(C,1)',1);
recall = tp./max(sum(C,2),1);
f1 = 2*precision.*recall./max(precision+recall,eps);
targets = zeros(K,N,'like',scores);
targets(sub2ind([K N],truth',(1:N))) = 1;
Y = dlarray(scores','CB');
T = dlarray(targets,'CB');
if isempty(weights)
    loss = crossentropy(Y,T);
else
    loss = crossentropy(Y,T,weights,WeightsFormat="C");
end
metrics.accuracy = sum(tp)/N;
metrics.macroPrecision = mean(precision);
metrics.macroRecall = mean(recall);
metrics.macroF1 = mean(f1);
metrics.loss = double(extractdata(loss));
metrics.confusionMatrix = C;
metrics.perClass = table(string(classNames(:)),sum(C,2),precision,recall,f1,recall, ...
    'VariableNames',{'Class','Support','Precision','Recall','F1','Accuracy'});
end
