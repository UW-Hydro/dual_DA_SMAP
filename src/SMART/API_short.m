function [out] = API_short(start,API_COEFF,bb,time_steps)
scale_factor = 1/time_steps;
sub_start = start;
for kk=1:time_steps
        sub_end = scale_factor*sign(sub_start)*(API_COEFF-1)*abs(sub_start)^bb + sub_start;
        sub_start = sub_end;
end
out=sub_end;
end

