
function [fMin , bestX,Convergence_curve ] = SSA(pop, M,c,d,dim,fobj  )

%%参数初始化
P_percent = 0.2;    % The population size of producers accounts for "P_percent" percent of the total population size生产者数量占总种群的比例
pNum = round( pop *  P_percent );    % The population size of the producers生产者数量


lb= c.*ones( 1,dim );    % Lower limit/bounds/     a vector   上下界
ub= d.*ones( 1,dim );    % Upper limit/bounds/     a vector
%Initialization
for i = 1 : pop

    x( i, : ) = lb + (ub - lb) .* rand( 1, dim );    %种群位置
    fit( i ) = fobj( x( i, : ) ) ;                  %种群适应度
end
pFit = fit;
pX = x;                            % The individual's best position corresponding to the pFit
[ fMin, bestI ] = min( fit );      % fMin denotes the global optimum fitness value
bestX = x( bestI, : );             % bestX denotes the global optimum position corresponding to fMin





% 收敛曲线初始化
Convergence_curve = zeros(M, 1);
% Start updating the solutions.


for t = 1 : M


    [ ans, sortIndex ] = sort( pFit );% Sort.

    [fmax,B]=max( pFit );
    worse= x(B,:);

    r2=rand(1);
    if(r2<0.8)

        for i = 1 : pNum                                                   % Equation (3)
            r1=rand(1);
            x( sortIndex( i ), : ) = pX( sortIndex( i ), : )*exp(-(i)/(r1*M));
            x( sortIndex( i ), : ) = Bounds( x( sortIndex( i ), : ), lb, ub );
            fit( sortIndex( i ) ) = fobj( x( sortIndex( i ), : ) );
        end
    else
        for i = 1 : pNum

            x( sortIndex( i ), : ) = pX( sortIndex( i ), : )+randn(1)*ones(1,dim);
            x( sortIndex( i ), : ) = Bounds( x( sortIndex( i ), : ), lb, ub );
            fit( sortIndex( i ) ) = fobj( x( sortIndex( i ), : ) );

        end

    end


    [ fMMin, bestII ] = min( fit );
    bestXX = x( bestII, : );


    for i = ( pNum + 1 ) : pop                     % Equation (4)

        A=floor(rand(1,dim)*2)*2-1;

        if( i>(pop/2))
            x( sortIndex(i ), : )=randn(1)*exp((worse-pX( sortIndex( i ), : ))/(i)^2);
        else
            x( sortIndex( i ), : )=bestXX+(abs(( pX( sortIndex( i ), : )-bestXX)))*(A'*(A*A')^(-1))*ones(1,dim);

        end
        x( sortIndex( i ), : ) = Bounds( x( sortIndex( i ), : ), lb, ub );
        fit( sortIndex( i ) ) = fobj( x( sortIndex( i ), : ) );

    end
    c=randperm(numel(sortIndex));
    b=sortIndex(c(1:round(pop*0.3)));
    for j =  1  : length(b)      % Equation (5)

        if( pFit( sortIndex( b(j) ) )>(fMin) )
            x( sortIndex( b(j) ), : )=bestX+(randn(1,dim)).*(abs(( pX( sortIndex( b(j) ), : ) -bestX)));
        else
            x( sortIndex( b(j) ), : ) =pX( sortIndex( b(j) ), : )+(2*rand(1)-1)*(abs(pX( sortIndex( b(j) ), : )-worse))/ ( pFit( sortIndex( b(j) ) )-fmax+1e-50);
        end
        x( sortIndex(b(j) ), : ) = Bounds( x( sortIndex(b(j) ), : ), lb, ub );

        fit( sortIndex( b(j) ) ) = fobj( x( sortIndex( b(j) ), : ) );
    end
    for i = 1 : pop
        if ( fit( i ) < pFit( i ) )
            pFit( i ) = fit( i );
            pX( i, : ) = x( i, : );
        end

        if( pFit( i ) < fMin )
            fMin= pFit( i );
            bestX = pX( i, : );


        end
    end
  
 

    Convergence_curve(t)=fMin;
    disp([num2str(t),'次迭代的RMSE为：',num2str(fMin)])
end




% Application of simple limits/bounds
function s = Bounds( s, Lb, Ub)
% Apply the lower bound vector
temp = s;
I = temp < Lb;
temp(I) = Lb(I);

% Apply the upper bound vector
J = temp > Ub;
temp(J) = Ub(J);
% Update this new move
s = temp;

%---------------------------------------------------------------------------------------------------------------------------
