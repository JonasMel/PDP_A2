x = linspace(0,1,100);
y = linspace(0,1,100);
c = zeros(100,100);
for i  = 1:length(x)
    for j = 1:length(y)
        a = (pi*sqrt((x(i)-0.25)^2 + (y(j)-0.5)^2))/(2*0.1);
        c(i,j) = cos(a);
        if sqrt((x(i)-0.25)^2 + (y(j)-0.5)^2) > 0.1
            c(i,j) = 0;
        end
    end
end
surf(x,y,c);