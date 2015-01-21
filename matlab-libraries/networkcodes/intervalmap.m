function y=intervalmap(a,b,c,d,x)

% just a map form [a b] to [c d]

if (b==1)
    y=1;
else
    if (d==1)
    y=1;
    else
y = (x-a)./(b-a).*(d-c) + c;



    end
end
end