
%xo: puntos iniciales

function x = neldermead(func,xo,tol,iter)

xo = xo';

%dimensión del problema
d  = size(xo,2); % Cantidad de columnas = dimension del problema
%número de puntos originales
n = size(xo,1); % Cantidad de filas = puntos diferentes

if (n ~= d+1)
    x = xo(1,:);
    disp('Tiene que haber tantos puntos como dimension+1');
    return;
end

f = zeros(n,1);
for k = 1:n
    f(k) = feval(func,xo(k,:)');
end

%ordeno los puntos
[f,i] = sort(f);    
xo = xo(i,:);
x  = xo(1,:);
o = 1;      %óptimo
b = n-1;    %bueno
p = n;      %peor

S = sum(xo(1:d,:));
xo;
for k = 1:iter

	if norm(xo(o,:)-xo(p,:)) < tol
		break;
	end
    %baricentro
    M = S/d;
    
    %%% REFLEXIÓN
    R = 2*M-xo(p,:);    %reflejado
    fR= feval(func,R');
    if(fR < f(b))
        if( f(o) < fR )
            %Reemplazo P con R
            l = 2;          %ya sé que R no es el nuevo óptimo
            while (fR >= f(l))
                l = l + 1;
            end
            xo = [xo(1:l-1,:); R;xo(l:b,:)];
            f  = [ f(1:l-1)  ;fR; f(l:b)  ];            
            S = S - xo(p,:) + R;
        else
            %%% EXPANDO
            E = 3*M-2*xo(p,:);
            fE= feval(func,E');
            if( fE < f(o) )
                %Reemplazo P con E
                xo = [ E; xo(1:b,:)];    %ya sé que E es el nuevo óptimo
                f  = [fE;  f(1:b)  ];
                S = S - xo(p,:) + E;
            else
                %Reemplazo P con R
                xo = [ R; xo(1:b,:)];    %ya sé que R es el nuevo óptimo
                f  = [fR;  f(1:b)  ];    
                S = S - xo(p,:) + R;
            end
        end        
    else
        if( fR < f(p) )
            %Reemplazo P con R
            xo = [xo(1:b,:); R];    %ya sé que R es el nuevo peor
            f  = [ f(1:b)  ;fR];    
            S = S - xo(p,:) + R;           
        else
            %%% CONTRAIGO
            C1 = (xo(p,:)+M)/2;
            C2 = (R+M)/2;
            fC1= feval(func,C1');
            fC2= feval(func,C2');
            if( fC1 < fC2 )
                C  = C1;
                fC = fC1;
            else
                C = C2;
                fC = fC2;                
            end
            if( fC < f(p) )
                %Reemplazo P con C
                l = 1;
                while (fC >= f(l))
                    l = l+1;
                end
                if( l == 1 )
                    xo = [ C; xo(1:b,:)];    %ya sé que C es el nuevo óptimo
                    f  = [fC;  f(1:b)  ];
                else
                    xo = [xo(1:l-1,:); C;xo(l:b,:)];
                    f  = [ f(1:l-1)  ;fC; f(l:b)  ];            
                end
                S = S - xo(p,:) + C;                
            else
                %%% ENCOJO                
                for l = 2:n
                    xo(l,:) = (xo(l,:)+x(1,:))/2;
                    f(l) = feval(func,xo(l,:)');
                end
                [f,i] = sort(f);
                xo = xo(i,:);
                S  = sum(xo(1:d,:));
            end
        end               
    end
    x = xo(o,:);
    xo;
end
k
