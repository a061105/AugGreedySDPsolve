seed = 10;
n=100;
m=7;
[A,b,C] = generate_SDP_syn_data(n,m,seed);
%C = zeros(n);
conjA = @(AA,yy) sum(bsxfun(@times,AA,reshape(yy,[1 1 m])),3);
funA = @(AA,XX) squeeze(sum(sum(bsxfun(@times,AA,XX))));

y = zeros(m,1);
eta = 1;
step_size = 1;
inner_iter_max = 10000;
outer_iter_max = 100;
yt_iter_max = 1000;
%c0 = 100000;
epsilon = 1e-8;
%new active set size
newk = 2;
for yt_iter = 1:yt_iter_max
    %previous y
    yt = y;
    bhat = b - 1/eta*yt;
    % initialize X
    %X = zeros(n,n);
    
    %% Primal: min b'y + 1/(2*eta)||y - y_t||_2^2, s.t. C+A^*(y) >= 0
    %% Dual: min <C,X>+eta/2*||A(X) - b - 1/eta*y_t||_2^2, s.t. X >= 0
    %grad = C + eta*conjA(A,funA(A,X)-b-1/eta*yt);
    UAUS = zeros(m,1);
    U = zeros(n,0);
    UCUS = 0;
    
    for outer_iter = 1:outer_iter_max
        %% Gradient: C + eta*A^*(A(X)-b+1/eta*y_t)
        grad = C + eta*conjA(A,UAUS-bhat);
        
        %new active set
        [U_new,D] = eigs(-grad,newk,'LA');
        U_new = U_new(:,diag(D)>0);
        if size(U_new,2)==0
            break;
        end
        U = [U,U_new];
        %% solve subproblem: min_S g(S) = <C,USU^T> + eta/2*||A(USU^T) - b - 1/eta*y_t||_2^2, s.t. S >= 0, tr(S)<=c0
        k = size(U,2);
        if k > n
            fprintf('k larger than n\n');
            %break;
        end
        UCU = U'*C*U; % TODO: speedup
        
        UAU = zeros(k,k,m);
        for i=1:m
            UAU(:,:,i) = U'*A(:,:,i)*U; % TODO: speedup
        end
        % S = zeros(k);
        %% frank-wolfe: \nabla g^TS_new, s.t. S_new >= 0, tr(S_new)<=c0, rank(S_new) = 1
        
        ini_obj = UCUS + eta/2*norm(UAUS-bhat)^2;
        old_obj = ini_obj;
        for inner_iter = 1:inner_iter_max
            grad_subproblem = UCU + eta*(conjA(UAU,UAUS - bhat));
            [vsub,dsub] = eigs(-(grad_subproblem+grad_subproblem')/2,1,'LA');
            if dsub<0
                alpha = (eta*UAUS'*bhat - UCUS)/(eta*norm(UAUS)^2);
                alpha = max(alpha,0);
                alpha = min(alpha,1);
                UAUS = alpha*UAUS;
                UCUS = alpha*UCUS;
            else
                vUCUv = vsub'*UCU*vsub;
                vUAUv = zeros(m,1);
                for ii = 1:m
                    vUAUv(ii) = vsub'*UAU(:,:,ii)*vsub;
                end                
                alpha = -(eta*vUAUv'*(UAUS-bhat)+vUCUv)/(eta*norm(vUAUv)^2);
                UAUS = UAUS + alpha*vUAUv;
                UCUS = UCUS + alpha*vUCUv;
            end

            %alpha = (UCUS - c0*vUCUv + eta*(UAUS-c0*vUAUv)'*(UAUS-bhat))/(eta*(norm(UAUS-c0*vUAUv)^2));
            
            obj = UCUS + eta/2*norm(UAUS-bhat)^2;
            if obj > old_obj
                fprintf('In inner iterations, obj is increasing\n');
            end
            if (old_obj - obj)/ini_obj < epsilon
                break;
            end
            old_obj = obj;
            %fprintf('inner iter=%d, obj=%f\n',inner_iter,obj);
        end
        fprintf('outer iter=%d, obj=%f, #inner iters=%d \n',outer_iter,obj,inner_iter);
    end
    % y = yt + eta*(A(X)-b)
    y = yt + step_size*eta*(UAUS-b);
    
    fprintf('%f\n',norm(UAUS-b));
    
end
    
