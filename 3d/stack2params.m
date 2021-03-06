function [params, netconfig] = stack2params(stack, poolMethod, convMethod)

% Converts a "stack" structure into a flattened parameter vector and also
% stores the network configuration. This is useful when working with
% optimization toolboxes such as minFunc.
%
% [params, netconfig] = stack2params(stack)
%
% stack - the stack structure, where stack{1}.w = weights of first layer
%                                    stack{1}.b = weights of first layer
%                                    stack{2}.w = weights of second layer
%                                    stack{2}.b = weights of second layer
%                                    ... etc.


% Setup the compressed param vector
params = [];
for d = 1:numel(stack)
    
    % This can be optimized. But since our stacks are relatively short, it
    % is okay
    params = [params ; stack{d}.w(:) ; stack{d}.b(:) ];
    
    % Check that stack is of the correct form
    assert(size(stack{d}.w, 4) == size(stack{d}.b, 1), ...
        ['The bias should be a *column* vector of ' ...
         int2str(size(stack{d}.w, 4)) 'x1']);
%     if d < numel(stack)
%         assert(size(stack{d}.w, 4) == size(stack{d+1}.w, 2), ...
%             ['The adjacent layers L' int2str(d) ' and L' int2str(d+1) ...
%              ' should have matching sizes.']);
%     end
    
end

if nargout > 1
    % Setup netconfig
    if numel(stack) == 0
        netconfig.inputsize = 0;
        netconfig.layersizes = {};
    else
        netconfig.poolMethod = poolMethod;
        netconfig.convMethod = convMethod;
        netconfig.layersizes = {};
        for d = 1:numel(stack)
            netconfig.layersizes = [netconfig.layersizes prod(prod(size(stack{d}.w)))];
            if isfield(stack{d}, 'pd')
                netconfig.layerInfo{d}.pd = stack{d}.pd;
                netconfig.layerInfo{d}.pi = stack{d}.pi;
                netconfig.layerInfo{d}.wsize = size(stack{d}.w);
                netconfig.layerInfo{d}.fd = stack{d}.fd;
            end
        end
    end
end

end